from fastapi import FastAPI
from tvdq.cleaning import basic_clean
from tvdq.enrichment import enrich_record
from tvdq.linkage import cluster_predict

app = FastAPI(title="TV Data Quality API")

@app.post("/enrich")
def enrich(record: dict):
    record = basic_clean(record)
    return enrich_record(record)

@app.post("/dedupe")
def dedupe(records: list[dict]):
    return cluster_predict(records)
