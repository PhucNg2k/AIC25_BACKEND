import os
import json
import unicodedata
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# ---------- helpers ----------
def strip_accents(s: str) -> str:
    # fold accents to ASCII (matches your vi_folded analyzer)
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def split_query(q: str) -> List[str]:
    # simple tokenization on whitespace/punct
    # (Elasticsearch analyzer will do a better job; this is just to build sub-aggs)
    parts = []
    for token in q.strip().split():
        t = token.strip()
        if t:
            parts.append(t)
    return parts

def get_es_client() -> Elasticsearch:
    load_dotenv()
    es_url = os.getenv("ES_LOCAL_URL", "http://localhost:9200")
    api_key_raw = os.getenv("ES_LOCAL_API_KEY", "")
    api_key = tuple(api_key_raw.split(":", 1)) if ":" in api_key_raw else api_key_raw or None
    if not api_key:
        raise RuntimeError("ES_LOCAL_API_KEY is not set")
    return Elasticsearch(hosts=[es_url], api_key=api_key, request_timeout=60)

# ---------- query builder ----------
def build_asr_chunk_agg_query(
    query_text: str,
    *,
    index_field: str = "text.folded",
    min_terms: Optional[int] = None,
    min_asr_conf: Optional[float] = 0.0,
    language: Optional[str] = None,
    page_size: int = 1000
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Build a composite aggregation that groups by (video_name, chunk_id) and
    keeps only chunks that match >= min_terms tokens.
    Returns (body, folded_tokens).
    """
    raw_tokens = split_query(query_text)
    # fold accents if we query text.folded
    folded_tokens = [strip_accents(t.lower()) for t in raw_tokens] if index_field == "text.folded" else raw_tokens

    if not folded_tokens:
        raise ValueError("Empty query")

    if min_terms is None:
        min_terms = len(folded_tokens)  # require ALL terms by default

    # main query: filter to docs matching ANY of the tokens to constrain the agg
    should_clauses = [{"match": {index_field: {"query": t}}} for t in folded_tokens]
    filters = []
    if min_asr_conf is not None:
        filters.append({"range": {"asr_conf": {"gte": float(min_asr_conf)}}})
    if language:
        filters.append({"term": {"language": language}})

    # per-term filter aggs to count presence inside each chunk bucket
    term_aggs = {}
    buckets_path = {}
    indicator_exprs = []
    for i, tok in enumerate(folded_tokens):
        name = f"t{i}"
        term_aggs[name] = {"filter": {"match": {index_field: {"query": tok}}}}
        # In pipeline aggs, filter doc_count is addressed as 'aggname>_count'
        buckets_path[name] = f"{name}>_count"
        indicator_exprs.append(f"(params.{name} > 0 ? 1 : 0)")

    matched_terms_script = " + ".join(indicator_exprs)

    body = {
        "size": 0,
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
                "filter": filters
            }
        },
        "aggs": {
            "by_chunk": {
                "composite": {
                    "size": page_size,
                    "sources": [
                        {"video_name": {"terms": {"field": "video_name"}}},
                        {"chunk_id": {"terms": {"field": "chunk_id"}}}
                    ]
                },
                "aggs": {
                    # per-term presence counters
                    **term_aggs,
                    # how many distinct terms matched in this chunk?
                    "matched_terms": {
                        "bucket_script": {
                            "buckets_path": buckets_path,
                            "script": matched_terms_script
                        }
                    },
                    # confidence-based ranking helpers
                    "sum_asr_conf": {"sum": {"field": "asr_conf"}},
                    "max_asr_conf": {"max": {"field": "asr_conf"}},
                    "avg_asr_conf": {"avg": {"field": "asr_conf"}},
                    # show a few example matched tokens from this chunk
                    "examples": {
                        "top_hits": {
                            "size": 3,
                            "_source": {"includes": ["text", "asr_conf", "chunk_idx"]},
                            "sort": [{"asr_conf": {"order": "desc"}}]
                        }
                    },
                    # keep only chunks that meet the min_terms threshold
                    "keep": {
                        "bucket_selector": {
                            "buckets_path": {"m": "matched_terms"},
                            "script": f"params.m >= {int(min_terms)}"
                        }
                    },
                    # final ordering inside each page
                    "order": {
                        "bucket_sort": {
                            "sort": [
                                {"matched_terms": {"order": "desc"}},
                                {"sum_asr_conf": {"order": "desc"}},
                                {"max_asr_conf": {"order": "desc"}}
                            ],
                            "size": page_size
                        }
                    }
                }
            }
        }
    }
    return body, folded_tokens

# ---------- search driver ----------
def search_asr_chunks(
    query_text: str,
    *,
    index_name: str = "asr_index_chunked",
    min_terms: Optional[int] = None,
    use_folded: bool = True,
    top_n: int = 100,
    min_asr_conf: float = 0.0,
    language: Optional[str] = None
) -> List[Dict[str, Any]]:
    es = get_es_client()
    index_field = "text.folded" if use_folded else "text"
    body, toks = build_asr_chunk_agg_query(
        query_text,
        index_field=index_field,
        min_terms=min_terms,
        min_asr_conf=min_asr_conf,
        language=language,
        page_size=min(1000, max(10, top_n))
    )

    results: List[Dict[str, Any]] = []
    after_key = None
    while True:
        if after_key:
            body["aggs"]["by_chunk"]["composite"]["after"] = after_key
        
        resp = es.search(index=index_name, body=body)
        
        buckets = resp.get("aggregations", {}).get("by_chunk", {}).get("buckets", [])
        for b in buckets:
            item = {
                "video_name": b["key"]["video_name"],
                "chunk_id": b["key"]["chunk_id"],
                "matched_terms": int(b["matched_terms"]["value"]),
                "sum_asr_conf": float(b["sum_asr_conf"]["value"]), # -> score
                "max_asr_conf": float(b["max_asr_conf"]["value"]),
                "avg_asr_conf": float(b["avg_asr_conf"]["value"]),
                "examples": [hit["_source"] for hit in b["examples"]["hits"]["hits"]],
            }
            results.append(item)
            if len(results) >= top_n:
                return results

        after_key = resp.get("aggregations", {}).get("by_chunk", {}).get("after_key")
        if not after_key:
            break
    return results

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=os.getenv("ASR_INDEX_NAME", "asr_index_chunked"))
    p.add_argument("--query", required=True, help='e.g. "cooking recipe"')
    p.add_argument("--min-terms", type=int, default=None, help="Require at least this many query tokens in a chunk (default: ALL)")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--no-folded", action="store_true", help="Search raw text (diacritics-sensitive)")
    p.add_argument("--min-asr-conf", type=float, default=0.0)
    p.add_argument("--language", default=None, help='Filter e.g. "vi"')
    args = p.parse_args()

    chunks = search_asr_chunks(
        args.query,
        index_name=args.index,
        min_terms=args.min_terms,
        use_folded=not args.no_folded,
        top_n=args.top_n,
        min_asr_conf=args.min_asr_conf,
        language=args.language
    )
    print(json.dumps({"query": args.query, "results": chunks}, ensure_ascii=False, indent=2))
