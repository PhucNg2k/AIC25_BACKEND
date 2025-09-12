from dotenv import load_dotenv
import os
from API.ElasticSearch.ESclient import OCRClient

load_dotenv()
es_url = os.getenv("ES_LOCAL_URL", "http://localhost:9200")

# ES accepts either a base64 "id:secret" token or a tuple ("id", "secret").
raw = os.getenv("ES_LOCAL_API_KEY")
# print(raw)
api_key = tuple(raw.split(":", 1)) if raw and ":" in raw else raw
# print(api_key)
if not api_key:
    raise RuntimeError("ES_LOCAL_API_KEY is missing")

index_name = os.getenv("OCR_INDEX_NAME", "ocr_index_v2")  # match the index you created

client = OCRClient(hosts=[es_url], api_key=api_key, index_name=index_name)

# stats = client.get_index_stats()
# print("Index stats:", stats)  # expect non-empty dict


# TESTING NOW
body = {
  "size": 5,
  "_source": ["video_name","frame_id","text"],
  "query": {"match": {"text": {"query": "thịt nạc xay", "fuzziness": "AUTO"}}}
}
print(client.search(body))
