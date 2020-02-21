import csv
import pymongo

client = pymongo.MongoClient()
db = client["2019-nCov"]
news = db.news

with open("DXY-COVID-19-Data/csv/DXYNews.csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i > 0:
            print(row[0], row[2], row[4])
            if news.find({"id": row[0]}).count() == 0:
                news.insert({
                    "id": row[0],
                    "time": row[2],
                    "content": row[4]
                })

with open("DXY-COVID-19-Data/csv/DXYRumors.csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i > 0:
            print(row[0], row[8], row[2] + row[3] + row[5])
            if news.find({"id": row[0]}).count() == 0:
                news.insert({
                    "id": row[0],
                    "time": row[8],
                    "content": row[2] + row[3] + row[5]
                })
