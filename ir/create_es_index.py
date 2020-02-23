from elasticsearch import Elasticsearch
import pymongo

index_mappings = {
  "mappings": {
      "properties": {
          "content": {
              "type": "text",
              "analyzer": "ik_max_word",
              "search_analyzer": "ik_smart"
          }
      }
  }
}

es = Elasticsearch()
client = pymongo.MongoClient()
db = client["2019-nCov"]
news = db.news

if not es.indices.exists(index='news_index') is not True:
    print("create news_index")
    es.indices.delete('news_index')

es.indices.create(index='news_index', body=index_mappings)

for item in news.find({}):
    res = es.index(index="news_index", id=item["id"], body={
        "id": item["id"],
        "content": item["content"],
    })
    print(res)

