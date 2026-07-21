from __future__ import annotations
import logging
import bm25s
import torch
from . import ingest
import numpy as np
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

from .schemas import Candidate, Trial

def retrieve(note: str, config: dict, k: int) -> list[Candidate]:
    bmRet = bm25Search(note, config=config, topN=k)
    dnRet = denseSearch(note, config=config, topN=k)
    fusedRet = hybrid(bmRet, dnRet, config)
    if config["retrieval"]["useRerank"]:
        fusedRet = rerank(note, fusedRet, config)
    logger.info(f"fusedRet: {fusedRet}")
    logger.info("--end of retrieval--")
    return fusedRet
    

def bm25Search(note: str, config: dict, topN: int) -> list[Candidate]:
    #https://pypi.org/project/BM25/
    logger.info("--bm25 search in progress--")
    logger.info(f"..{note}..{config}..{topN}")
    # corpus = BM25.load(ingest.loadTrials(), document_column="text")
    # retriever = BM25.index(corpus)
    # results = retriever.retrieve(note, k=topN)
    trials = list(ingest.loadTrials())
    corpus_tokens = bm25s.tokenize([t.searchText() for t in trials], stopwords = "en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    query_tokens = bm25s.tokenize(note, stopwords = "en")
    results, scores = retriever.retrieve(query_tokens, k=topN)
    # for doc_idx, score in zip(results[0], scores[0]):
    #     logger.info(f"Document: {corpus_tokens[doc_idx]} | Score: {score:.4f}")
    return [Candidate(nctId=trials[i].nctId, score=float(s), retrieverBreakdown={"bm25": float(s)})
        for i, s in zip(results[0], scores[0])]
    #return results

def denseSearch(note: str, config: dict, topN: int) -> list[Candidate]:
    #https://github.com/ncbi/MedCPT
    tokenizer_query = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    model_query = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")

    # tokenizer_article = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    # model_article = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    
    # this shouldnt work here
    # trials = list(ingest.loadTrials())
    # corpus_tokens = [t.searchText() for t in trials]

    # encoded_articles = tokenizer_article(corpus_tokens, padding=True, truncation=True, return_tensors='pt')
    # with torch.no_grad():
    #     article_embeddings = model_article(**encoded_articles)[0][:, 0].cpu().numpy()

    encoded_query = tokenizer_query(note, return_tensors='pt')
    with torch.no_grad():
        query_embedding = model_query(**encoded_query)[0][0].cpu().numpy()
    
    try:
        client = QdrantClient(path=config["retrieval"]["qdrant"]["location"]) 
    except:
        logger.debug("--qdrant connect failure--")
        return []
    collection_name = "trials2021"

    # below should not be here
    # client.create_collection(
    #     collection_name=collection_name,
    #     vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    # )

    # points = [
    #     PointStruct(
    #         id=str(uuid.uuid4()),
    #         vector=embedding.tolist(),
    #         payload={"text": text}
    #     )
    #     for embedding, text in zip(article_embeddings, corpus_tokens)
    # ]

    # client.upsert(
    #     collection_name=collection_name,
    #     points=points
    # )
    
    query_vector = query_embedding.tolist()

    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=topN 
    ).points

    for result in search_results:
        logger.info(f"Score: {result.score:.4f} | Text: {result.payload['nctId']}")
        
    return [Candidate(nctId=i.payload["nctId"], score=float(i.score), retrieverBreakdown={"dense": float(i.score)})
        for i in search_results]

# https://mbrenndoerfer.com/writing/hybrid-search-bm25-dense-retrieval-fusion
def normalizeScores(ranked_list: list) -> dict:
    """Min-max normalize scores from a ranked list. Returns {doc_id: normalized_score}."""
    if not ranked_list:
        return {}
    scores = [c.score for c in ranked_list]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    if score_range == 0:
        return {c.nctId: 1.0 for c in ranked_list}

    return {c.nctId: (c.score - min_score) / score_range for c in ranked_list}
    
def hybrid(bm25Hits: list[Candidate], denseHits: list[Candidate], config: dict) -> list[Candidate]:
    bm25Nor = normalizeScores(bm25Hits)
    denseNor = normalizeScores(denseHits)
    logger.info(f"Hybrid retrieval -- bm25Nor: {bm25Nor}, denseNor: {denseNor}")
    docList = set(bm25Nor)|set(denseNor)
    combined = {}
    for doc_id in docList:
        b_score = bm25Nor.get(doc_id, 0.0)
        d_score = denseNor.get(doc_id, 0.0)
        combined[doc_id] = config["retrieval"]["alpha"] * d_score + (1 - config["retrieval"]["alpha"]) * b_score
    logger.info(f"Combined: {combined}")
    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [Candidate(nctId=d, score=s, retrieverBreakdown={"hybrid": s})
        for d, s in sorted_docs]

from transformers import AutoModelForSequenceClassification

#https://github.com/ncbi/MedCPT/blob/main/reranker/main.py

def rerank(note: str, candidates: list[Candidate], config: dict) -> list[Candidate]:
    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")

    trials = fetchTrials([c.nctId for c in candidates], config)   
    pairs = [[note, trials[c.nctId].searchText()] for c in candidates]

    # tokenise
    enc = tok(pairs, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    # scoring, bc isme classification kyun hai bc, gandu ek ghanta kah gaya
    with torch.no_grad():
        scores = model(**enc).logits.squeeze(dim=1).tolist()

    for c, s in zip(candidates, scores):
        c.retrieverBreakdown["rerank"] = float(s)

    return sorted(candidates, key=lambda c: c.retrieverBreakdown["rerank"], reverse=True)

def fetchTrials(nctIds: list[str], config: dict) -> dict[str, Trial]:
    wanted = set(nctIds)
    logger.info("---Fetch Trials---")
    logger.info(f"{wanted}")
    return {t.nctId: t for t in ingest.loadTrials() if t.nctId in wanted}
