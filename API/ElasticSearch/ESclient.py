# AIC25_BACKEND/API/ElasticSearch/ESclient.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch import helpers
import math
import os
import sys


API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if API_DIR not in sys.path:
    sys.path.append(API_DIR)
    
# from load_embed_model import get_asr_embedding


from frame_utils import get_metakey, get_pts_time, get_frame_path
DATA_SOURCE = '/REAL_DATA/keyframes_beit3/keyframes'

class ESClientBase(ABC):
    def __init__(
        self,
        hosts,
        api_key,
        index_name: str,
        request_timeout: int = 30,
    ) -> None:
        self.es = Elasticsearch(hosts=hosts, api_key=api_key)
        self.index_name = index_name
        self.request_timeout = request_timeout

    # -------- core request helpers --------
    def search(
        self,
        body: Dict[str, Any],
        top_k: Optional[int] = None,
        from_: Optional[int] = None,
        search_after: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        body = dict(body)
        if top_k is not None:
            body["size"] = top_k
        if from_ is not None:
            body["from"] = from_
        if search_after is not None:
            body["search_after"] = search_after

        return self.es.search(
            index=self.index_name,
            body=body,
        )

    def index_exists(self) -> bool:
        return self.es.indices.exists(index=self.index_name)

    def get(self, doc_id: str) -> Dict[str, Any]:
        try:
            return self.es.get(index=self.index_name, id=doc_id)
        except NotFoundError:
            return {}

    # -------- Index Management --------
    def create_index(self, force: bool = False) -> bool:
        """Create index with mapping defined by child class"""
        try:
            if self.index_exists():
                if force:
                    self.delete_index()
                else:
                    print(f"⚠️ Index '{self.index_name}' already exists")
                    return False
            
            mapping = self.get_mapping()
            self.es.indices.create(index=self.index_name, body=mapping)
            print(f"✅ Created index '{self.index_name}'")
            return True
        except Exception as e:
            print(f"❌ Error creating index '{self.index_name}': {e}")
            return False

    def delete_index(self) -> bool:
        """Delete the index"""
        try:
            if not self.index_exists():
                print(f"⚠️ Index '{self.index_name}' does not exist")
                return False
            
            self.es.indices.delete(index=self.index_name)
            print(f"✅ Deleted index '{self.index_name}'")
            return True
        except Exception as e:
            print(f"❌ Error deleting index '{self.index_name}': {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            if not self.index_exists():
                return {}
            
            stats = self.es.indices.stats(index=self.index_name)
            index_stats = stats['indices'][self.index_name]
            return {
                'document_count': index_stats['total']['docs']['count'],
                'index_size_bytes': index_stats['total']['store']['size_in_bytes'],
                'index_size_mb': round(index_stats['total']['store']['size_in_bytes'] / (1024 * 1024), 2)
            }
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")
            return {}

    @abstractmethod
    def get_mapping(self) -> Dict[str, Any]:
        """Return the mapping configuration for this index type"""
        raise NotImplementedError

    # -------- Bulk Operations --------
    # --- in ESClientBase.bulk_index (replace the method) ---
    def bulk_index(self, documents: List[Dict[str, Any]], batch_size: int = 1000) -> bool:
        """Bulk index documents using helpers.bulk"""
        try:
            if not self.index_exists():
                print(f"❌ Index '{self.index_name}' does not exist. Create it first.")
                return False

            actions = []
            for doc in documents:
                # Allow caller to pass a stable _id; keep the rest in _source
                _id = doc.pop("_id", None)
                action = {
                    "_index": self.index_name,
                    "_source": doc
                }
                if _id is not None:
                    action["_id"] = _id

                actions.append(action)

                if len(actions) >= batch_size:
                    helpers.bulk(self.es, actions, request_timeout=self.request_timeout)
                    print(f"✅ Indexed {len(actions)} documents")
                    actions = []

            if actions:
                helpers.bulk(self.es, actions, request_timeout=self.request_timeout)
                print(f"✅ Indexed final {len(actions)} documents")

            print(f"🎉 Successfully indexed all documents to '{self.index_name}'")
            return True

        except Exception as e:
            print(f"❌ Error during bulk indexing: {e}")
            return False


    def bulk_index_from_dataframe(self, df, batch_size: int = 1000) -> bool:
        """Bulk index from pandas DataFrame using the child's document converter"""
        try:
            documents = []
            print("DF columns: ", df.columns)
            for _, row in df.iterrows():
                doc = self.convert_row_to_document(row)
                if doc:  # Skip invalid documents
                    documents.append(doc)
            print('DOCUMENT: ', len(documents))
            return self.bulk_index(documents, batch_size)
        except Exception as e:
            print(f"❌ Error converting DataFrame to documents: {e}")
            return False

    @abstractmethod
    def convert_row_to_document(self, row) -> Optional[Dict[str, Any]]:
        """Convert a DataFrame row to a document for indexing"""
        raise NotImplementedError

    # -------- high-level parsed search --------
    def search_parsed(
        self,
        body: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        resp = self.search(body=body, top_k=top_k)
        results = self.parse_hits(resp)
        
        return results

    @abstractmethod
    def parse_hits(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert ES hits into app-level records."""
        raise NotImplementedError
    
    @staticmethod
    def _is_invalid(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in ("", "none", "nan")
        if isinstance(value, float):
            return math.isnan(value)
        return False

class OCRClient(ESClientBase):
    def __init__(self, hosts, api_key, index_name: str = "ocr_index_v2"):
        super().__init__(hosts, api_key, index_name)

    def get_mapping(self) -> Dict[str, Any]:
        """Return OCR index mapping (no ICU plugin required)"""
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "char_filter": {
                        "junk_strip": {
                            "type": "pattern_replace",
                            "pattern": r"[^\p{L}\p{Nd}\s]",  # drop OCR junk
                            "replacement": " "
                        }
                    },
                    "filter": {
                        # keep max_gram - min_gram <= 1 to avoid index.max_ngram_diff errors
                        "ng34": { "type": "ngram", "min_gram": 3, "max_gram": 4 }
                    },
                    "analyzer": {
                        "vi_clean": {       # keep diacritics; just clean junk + lowercase
                            "type": "custom",
                            "char_filter": ["html_strip", "junk_strip"],
                            "tokenizer": "standard",
                            "filter": ["lowercase"]
                        },
                        "vi_folded": {      # accent-insensitive
                            "type": "custom",
                            "char_filter": ["html_strip", "junk_strip"],
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"]
                        },
                        "char_ngrams": {    # robust char n-grams
                            "type": "custom",
                            "tokenizer": "keyword",
                            "filter": ["lowercase", "asciifolding", "ng34"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "batch":        {"type": "keyword"},
                    "inter_folder": {"type": "keyword"},
                    "leaf_folder":  {"type": "keyword"},
                    "file_name":    {"type": "keyword"},

                    "video_id":     {"type": "keyword"},
                    "video_name":   {"type": "keyword"},
                    "frame_id":     {"type": "keyword"},
                    "frame_idx":    {"type": "integer"},

                    "text_raw":     {"type": "text"},
                    "text": {
                        "type": "text",
                        "analyzer": "vi_clean",
                        "search_analyzer": "vi_clean",
                        "fields": {
                            "folded":  { "type": "text", "analyzer": "vi_folded" },
                            "ngrams":  { "type": "text", "analyzer": "char_ngrams" },
                            "keyword": { "type": "keyword", "ignore_above": 256 }
                        }
                    },
                    "text_folded":  {"type": "keyword"},

                    "language":     {"type": "keyword"},
                    "source":       {"type": "keyword"},
                    "det_score":    {"type": "float"},
                    "rec_conf":     {"type": "float"},

                    "bbox": {
                        "properties": {
                            "x1": {"type": "float"},
                            "y1": {"type": "float"},
                            "x2": {"type": "float"},
                            "y2": {"type": "float"}
                        }
                    },
                    "poly":         {"type": "float"}   # array OK
                }
            }
        }


    def convert_row_to_document(self, row) -> Optional[Dict[str, Any]]:
        """Convert DataFrame row to OCR document"""
        try:
            # Handle different column layouts
            df_columns = row.index.tolist()
            
            required = ['video_folder', 'video_name', 'frame_id', 'text']
            if all(col in df_columns for col in required):
                doc = {
                    "video_folder": str(row.get('video_folder', '')),
                    "video_name": str(row.get('video_name', '')),
                    "frame_id": str(row.get('frame_id', '')),
                    "text": str(row.get('text', ''))
                }
            elif 'group' not in df_columns:
                doc = {
                    "video_folder": str(row.get('video', '')),
                    "video_name": str(row.get('image', '')),
                    "frame_id": str(row.get('frame_id', '')),
                    "text": str(row.get('text.1', ''))
                }
            else:
                doc = {
                    "video_folder": str(row.get('group', '')),
                    "video_name": str(row.get('video', '')),
                    "frame_id": str(row.get('frame_id', '')),
                    "text": str(row.get('text', ''))
                }
            
            # Validate required fields
            if any(self._is_invalid(v) for v in [doc["video_folder"], doc["video_name"], doc["frame_id"]]):
                return None
                
            return doc
        except Exception as e:
            print(f"❌ Error converting row to document: {e}")
            return None

    def search(self, body: Dict[str, Any], top_k: Optional[int] = None, from_: Optional[int] = None, search_after: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        after_key = None

        while True:
            if after_key:
                body["aggs"]["by_frame"]["composite"]["after"] = after_key
            
            resp = self.es.search(index=self.index_name, body=body)
            
            buckets = resp.get("aggregations", {}).get("by_frame", {}).get("buckets", [])
            for b in buckets:
                item = {
                    "video_name": b["key"]["video_name"],
                    "frame_id":   b["key"]["frame_id"],
                    "matched_terms": int(b["matched_terms"]["value"]),
                    "sum_rec_conf": float(b["sum_rec_conf"]["value"]), # -> score
                    "max_rec_conf": float(b["max_rec_conf"]["value"]),
                    "examples": [hit["_source"] for hit in b["examples"]["hits"]["hits"]],
                }
                results.append(item)
                if len(results) >= top_k:
                    return results

            after_key = resp.get("aggregations", {}).get("by_frame", {}).get("after_key")
            if not after_key:
                break
        
        return results

    def parse_hits(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        
        for r in rows:
            video_name = r["video_name"]

            try:
                frame_idx = int(float(r["frame_id"][1:]))
            except Exception:
                continue

            metakey  = get_metakey(video_name, frame_idx)
            image    = get_frame_path(metakey)
            pts_time = get_pts_time(metakey)

            out.append({
                "video_name": video_name,
                "frame_idx": frame_idx,
                "score": r.get("sum_rec_conf", 0.0),
                "image_path": image,
                "pts_time": pts_time,
            })
        
        return out


class ASRClient(ESClientBase):
    def __init__(self, hosts, api_key, index_name: str = "asr_index", embedding_dims: int = 768):
        super().__init__(hosts, api_key, index_name)
        self.embedding_dims = embedding_dims
    
    
    def get_mapping(self) -> Dict[str, Any]:
        return {
            "mappings": {
                "properties": {
                    "video_name":   {"type": "keyword"},
                    "text":     {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dims,
                        "index": False   # script_score requires index:false
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
    
    """
    def get_mapping(self) -> Dict[str, Any]:
        return {
            "mappings": {
                "properties": {
                    "video_name":   {"type": "keyword"},
                    "text":     {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dims,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
    """

    
    def convert_row_to_document(self, row) -> Optional[Dict[str, Any]]:
        try:
            doc = {
                "video_name": str(row.get('video_name', '')),
                "start_ms": int(row.get('start_ms', 0)),
                "end_ms": int(row.get('end_ms', 0)),
                "text": str(row.get('text', '')),
                "transcript": str(row.get('transcript', ''))
            }
            
            # Validate required fields
            if any(self._is_invalid(v) for v in [doc["video_name"], doc["start_ms"], doc["end_ms"]]):
                return None
                
            return doc
        except Exception as e:
            print(f"❌ Error converting row to document: {e}")
            return None
    
    """
    def convert_row_to_document(self, row) -> Optional[Dict[str, Any]]:
        try:
            doc = {
                "video_name": str(row.get('video_name', '')),
                "start_ms": int(row.get('start_ms', 0)),
                "end_ms": int(row.get('end_ms', 0)),
                "text": str(row.get('text', '')),
                "transcript": str(row.get('transcript', ''))
            }

            if any(self._is_invalid(v) for v in [doc["video_name"], doc["start_ms"], doc["end_ms"]]):
                return None

            # Prefer precomputed vector in the row; otherwise compute from text
            emb = row.get('embedding', None)
            if emb is not None:
                doc["embedding"] = list(emb)
            else:
                doc["embedding"] = get_asr_embedding(doc["text"])

            return doc
        except Exception as e:
            print(f"❌ Error converting row to document: {e}")
            return None
    """

    def parse_hits(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            score = hit.get("_score", 0.0)

            video_name = src.get("video_name")
            text_conent = src.get("text")
    
            if video_name is None or text_conent is None:
                continue

            out.append({
                "video_name": video_name,
                "text": text_conent,
                "score": score,
                "raw": src,
            })
            
        return out
    
