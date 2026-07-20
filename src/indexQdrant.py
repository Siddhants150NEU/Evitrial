from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModel
import logging
from . import ingest
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm 
import numpy as np
logger = logging.getLogger(__name__)


# https://huggingface.co/ncbi/MedCPT-Article-Encoder
# https://qdrant.tech/documentation/manage-data/indexing/
def buildIndex(config: dict) -> None:
    logger.info("-----------------------------------------------------")
    logger.info("building index")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    trials = list(ingest.loadTrials())
    # logger.info(f"trials: {trials}")
    medicalTexts = [t.searchText() for t in trials]
    # logger.info(f"medicalTexts: {medicalTexts}")
    embeddings = []
    B = 64
    for start in tqdm(range(0, len(medicalTexts), B)):
        chunk = medicalTexts[start:start + B]
        
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            vecs = model(**enc)[0][:, 0].cpu().numpy()   
        embeddings.append(vecs)
    embeddings = np.concatenate(embeddings)
    
    client = QdrantClient(path=config["retrieval"]["qdrant"]["location"])
    client.create_collection(
    collection_name = config["retrieval"]["qdrant"]["collection"],
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )

    # Prepare points for upsert
    points = []
    for i, (text, embedding) in enumerate(zip(trials, embeddings)):
        points.append(
            models.PointStruct(
                id=i, 
                vector=embedding.tolist(),
                payload={"nctId": text.nctId}  # Storing the text as metadata payload
            )
        )

    # Upsert the vectors into Qdrant
    client.upsert(
        collection_name =  config["retrieval"]["qdrant"]["collection"],
        points=points
    )


if __name__ == "__main__":
    from .config import loadConfig, setSeeds

    cfg = loadConfig()
    setSeeds(cfg["seed"])
    buildIndex(cfg)
