import os
import json
import argparse
import unicodedata
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from elasticsearch import Elasticsearch


# ---------- helpers ----------
def strip_accents(s: str) -> str:
    """Fold accents to ASCII (aligns with your vi_folded analyzer)."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def split_query(q: str) -> List[str]:
    """Light tokenizer (Elasticsearch analyzer will tokenize better)."""
    return [t.strip() for t in q.strip().split() if t.strip()]


def get_es_client() -> Elasticsearch:
    """Connect to Elasticsearch using environment variables."""
    load_dotenv()
    es_url = os.getenv("ES_LOCAL_URL", "http://localhost:9200")
    api_key_raw = os.getenv("ES_LOCAL_API_KEY", "")
    api_key = tuple(api_key_raw.split(":", 1)) if ":" in api_key_raw else api_key_raw or None

    if api_key:
        return Elasticsearch(hosts=[es_url], api_key=api_key, request_timeout=60)
    else:
        # Fallback to username/password if no API key
        user = os.getenv("ES_USER", "elastic")
        pwd = os.getenv("ES_LOCAL_PASSWORD", "changeme")
        return Elasticsearch(hosts=[es_url], basic_auth=(user, pwd), request_timeout=60)


# ---------- query builder ----------
def build_frame_agg_query(
    query_text: str,
    *,
    index_field: str = "text.folded",
    min_terms: Optional[int] = None,
    min_rec_conf: float = 0.6,
    min_det_score: float = 0.0,
    language: Optional[str] = None,
    page_size: int = 200,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Composite aggregation grouping by (video_name, frame_id):
      - Only returns frames that meet `min_terms` matched words.
      - Filters by OCR confidence, detection score, and language.
    """
    raw_tokens = split_query(query_text)
    folded_tokens = [strip_accents(t.lower()) for t in raw_tokens] if index_field == "text.folded" else raw_tokens

    if not folded_tokens:
        raise ValueError("Empty query")

    if min_terms is None:
        min_terms = len(folded_tokens)

    # --- Query section ---
    should_clauses = [{"match": {index_field: {"query": t}}} for t in folded_tokens]
    filters = [
        {"range": {"rec_conf": {"gte": float(min_rec_conf)}}},
        {"range": {"det_score": {"gte": float(min_det_score)}}}
    ]
    if language:
        filters.append({"term": {"language": language}})

    # --- Aggregation section ---
    term_aggs = {}
    buckets_path = {}
    indicator_exprs = []

    for i, tok in enumerate(folded_tokens):
        name = f"t{i}"
        term_aggs[name] = {"filter": {"match": {index_field: {"query": tok}}}}
        buckets_path[name] = f"{name}>_count"
        indicator_exprs.append(f"(params.{name} > 0 ? 1 : 0)")

    matched_terms_script = " + ".join(indicator_exprs)

    body = {
        "size": 0,
        "track_total_hits": False,  # saves time for large datasets
        "_source": False,
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
                "filter": filters
            }
        },
        "aggs": {
            "by_frame": {
                "composite": {
                    "size": page_size,
                    "sources": [
                        {"video_name": {"terms": {"field": "video_name"}}},
                        {"frame_id": {"terms": {"field": "frame_id"}}},
                    ],
                },
                "aggs": {
                    **term_aggs,
                    "matched_terms": {
                        "bucket_script": {
                            "buckets_path": buckets_path,
                            "script": matched_terms_script,
                        }
                    },
                    "sum_rec_conf": {"sum": {"field": "rec_conf"}},
                    "max_rec_conf": {"max": {"field": "rec_conf"}},
                    "examples": {
                        "top_hits": {
                            "size": 2,
                            "_source": {
                                "includes": ["text", "rec_conf", "det_score"]
                            },
                            "sort": [{"rec_conf": {"order": "desc"}}],
                        }
                    },
                    "keep": {
                        "bucket_selector": {
                            "buckets_path": {"m": "matched_terms"},
                            "script": f"params.m >= {int(min_terms)}",
                        }
                    },
                    "order": {
                        "bucket_sort": {
                            "sort": [
                                {"matched_terms": {"order": "desc"}},
                                {"sum_rec_conf": {"order": "desc"}},
                                {"max_rec_conf": {"order": "desc"}},
                            ],
                            "size": page_size,
                        }
                    },
                },
            }
        },
    }

    return body, folded_tokens


# ---------- search driver ----------
def search_frames(
    query_text: str,
    *,
    index_name: str = "ocr_index_v2",
    min_terms: Optional[int] = None,
    use_folded: bool = True,
    top_n: int = 100,
    min_rec_conf: float = 0.6,
    min_det_score: float = 0.0,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    es = get_es_client()
    index_field = "text.folded" if use_folded else "text"

    body, toks = build_frame_agg_query(
        query_text=query_text,
        index_field=index_field,
        min_terms=min_terms,
        min_rec_conf=min_rec_conf,
        min_det_score=min_det_score,
        language=language,
        page_size=min(300, max(50, top_n)),
    )

    results: List[Dict[str, Any]] = []
    after_key = None
    while True:
        if after_key:
            body["aggs"]["by_frame"]["composite"]["after"] = after_key

        resp = es.search(index=index_name, body=body)

        buckets = resp.get("aggregations", {}).get("by_frame", {}).get("buckets", [])
        for b in buckets:
            results.append({
                "video_name": b["key"]["video_name"],
                "frame_id": b["key"]["frame_id"],
                "matched_terms": int(b["matched_terms"]["value"]),
                "sum_rec_conf": float(b["sum_rec_conf"]["value"]),
                "max_rec_conf": float(b["max_rec_conf"]["value"]),
                "examples": [hit["_source"] for hit in b["examples"]["hits"]["hits"]],
            })
            if len(results) >= top_n:
                return results

        after_key = resp.get("aggregations", {}).get("by_frame", {}).get("after_key")
        if not after_key:
            break

    return results


# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=os.getenv("OCR_INDEX_NAME", "ocr_index_v2"))
    p.add_argument("--query", required=True, help='e.g. "thịt nạc xay"')
    p.add_argument("--min-terms", type=int, default=None)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--no-folded", action="store_true")
    p.add_argument("--min-rec-conf", type=float, default=float(os.getenv("OCR_MIN_REC_CONF", "0.6")))
    p.add_argument("--min-det-score", type=float, default=0.0)
    p.add_argument("--language", default=None)
    args = p.parse_args()

    frames = search_frames(
        args.query,
        index_name=args.index,
        min_terms=args.min_terms,
        use_folded=not args.no_folded,
        top_n=args.top_n,
        min_rec_conf=args.min_rec_conf,
        min_det_score=args.min_det_score,
        language=args.language,
    )
    print(json.dumps({"query": args.query, "results": frames}, ensure_ascii=False, indent=2))
