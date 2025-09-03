from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch import helpers
import math


DATA_SOURCE = '/REAL_DATA/keyframes_b1/keyframes'

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
    def bulk_index(self, documents: List[Dict[str, Any]], batch_size: int = 1000) -> bool:
        """Bulk index documents using helpers.bulk"""
        try:
            if not self.index_exists():
                print(f"❌ Index '{self.index_name}' does not exist. Create it first.")
                return False

            actions = []
            for doc in documents:
                action = {
                    "_index": self.index_name,
                    "_source": doc
                }
                actions.append(action)

                # Bulk insert when batch size is reached
                if len(actions) >= batch_size:
                    helpers.bulk(self.es, actions)
                    print(f"✅ Indexed {len(actions)} documents")
                    actions = []

            # Index remaining documents
            if actions:
                helpers.bulk(self.es, actions)
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
            for _, row in df.iterrows():
                doc = self.convert_row_to_document(row)
                if doc:  # Skip invalid documents
                    documents.append(doc)

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
    def __init__(self, hosts, api_key, index_name: str = "ocr_index"):
        super().__init__(hosts, api_key, index_name)

    def get_mapping(self) -> Dict[str, Any]:
        """Return OCR index mapping"""
        return {
            "mappings": {
                "properties": {
                    "video_folder": {"type": "keyword"},
                    "video_name": {"type": "keyword"},
                    "frame_id": {"type": "keyword"},
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                        "search_analyzer": "standard",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }

    def convert_row_to_document(self, row) -> Optional[Dict[str, Any]]:
        """Convert DataFrame row to OCR document"""
        try:
            # Handle different column layouts
            df_columns = row.index.tolist()
            
            if 'group' not in df_columns:
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

    def parse_hits(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            score = hit.get("_score", 0.0)

            video_folder = src.get("video_folder")
            video_name = src.get("video_name")
            frame_id = src.get("frame_id")

            if any(self._is_invalid(v) for v in [video_folder, video_name, frame_id]):
                continue

            try:
                frame_idx = int(float(frame_id))
            except (TypeError, ValueError):
                continue

            out.append({
                "video_name": video_name,
                "frame_idx": frame_idx,
                "score": score,
                "image_path": f"{DATA_SOURCE}/{video_folder}/{video_name}/f{frame_idx:06d}.webp",
                "raw": src,
            })
        return out


class ASRClient(ESClientBase):
    def __init__(self, hosts, api_key, index_name: str = "asr_index"):
        super().__init__(hosts, api_key, index_name)

    def get_mapping(self) -> Dict[str, Any]:
        """Return ASR index mapping"""
        return {
            "mappings": {
                "properties": {
                    "video_name": {"type": "keyword"},
                    "start_ms": {"type": "long"},
                    "end_ms": {"type": "long"},
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                        "search_analyzer": "standard",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "transcript": {
                        "type": "text",
                        "analyzer": "standard",
                        "search_analyzer": "standard"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }

    def convert_row_to_document(self, row) -> Optional[Dict[str, Any]]:
        """Convert DataFrame row to ASR document"""
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

    def parse_hits(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            score = hit.get("_score", 0.0)

            video_name = src.get("video_name")
            start_ms = src.get("start_ms")
            end_ms = src.get("end_ms")
            transcript = src.get("text") or src.get("transcript")

            if video_name is None or start_ms is None or end_ms is None:
                continue

            out.append({
                "video_name": video_name,
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "text": transcript,
                "score": score,
                "raw": src,
            })
        return out
    

# Usage Example:
"""
from elasticsearch import Elasticsearch
import pandas as pd

# Initialize ES client
es = Elasticsearch(hosts=[ES_LOCAL_URL], api_key=ES_LOCAL_API_KEY)

# Create OCR client
ocr_client = OCRClient(es, index_name="ocr_index")

# Create index
ocr_client.create_index()

# Index data from DataFrame
df = pd.read_excel("ocr_data.xlsx")
ocr_client.bulk_index_from_dataframe(df)

# Search
search_body = {
    "query": {
        "fuzzy": {
            "text": {
                "value": "receipt total",
                "fuzziness": "AUTO"
            }
        }
    },
    "_source": ["video_folder", "video_name", "frame_id", "text"]
}
results, total = ocr_client.search_parsed(search_body, top_k=100)

# Get stats
stats = ocr_client.get_index_stats()
print(f"Index has {stats['document_count']} documents")
""" 