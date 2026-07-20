from __future__ import annotations

import ast
import re
import os
os.environ.setdefault("IR_DATASETS_HOME", os.path.abspath("data/ir_datasets"))
from .schemas import PatientCriterionPair, Trial


TREC_2021 = "clinicaltrials/2021/trec-ct-2021"
TREC_2022 = "clinicaltrials/2021/trec-ct-2022"
ANNOTATIONS_REPO = "ncbi/TrialGPT-Criterion-Annotations"

def _dataset(name: str = TREC_2021):
    import ir_datasets

    return ir_datasets.load(name)

def loadTrials(name: str = TREC_2021):
    for doc in _dataset(name).docs_iter():
        inclusion, exclusion = splitEligibility(doc.eligibility)
        yield Trial(
            nctId=doc.doc_id,
            title=doc.title,
            condition=doc.condition,
            summary=doc.summary,
            detailedDescription=doc.detailed_description,
            eligibility=doc.eligibility,
            inclusionCriteria=inclusion,
            exclusionCriteria=exclusion,
        )

def loadTopics(name: str = TREC_2021):
    for query in _dataset(name).queries_iter():
        yield query.query_id, query.text

def loadQrels(name: str = TREC_2021):
    for qrel in _dataset(name).qrels_iter():
        yield qrel.query_id, qrel.doc_id, qrel.relevance

def splitEligibility(text: str) -> tuple[list[str], list[str]]:
    if not text:
        return [], []

    lowered = text.lower()
    excStart = lowered.find("exclusion criteria")
    if excStart == -1:
        inclusionPart, exclusionPart = text, ""
    else:
        inclusionPart, exclusionPart = text[:excStart], text[excStart:]

    incStart = inclusionPart.lower().find("inclusion criteria")
    if incStart != -1:
        inclusionPart = inclusionPart[incStart + len("inclusion criteria"):]

    exclusionPart = re.sub(r"(?i)^\s*exclusion criteria:?", "", exclusionPart).strip()
    return _splitLines(inclusionPart), _splitLines(exclusionPart)

def _splitLines(block: str) -> list[str]:
    lines = []
    for raw in re.split(r"[\n\r]+|(?:\s-\s)", block):
        cleaned = raw.strip(" \t-*.:;")
        if len(cleaned) > 3:
            lines.append(cleaned)
    return lines

_LABEL_MAP = {
    "included": "MET",
    "excluded": "MET",
    "not included": "NOT_MET",
    "not excluded": "NOT_MET",
    "not enough information": "UNKNOWN",
    "not applicable": "UNKNOWN",
}

def mapToEviLabels(expertEligibility: str) -> str:
    return _LABEL_MAP[expertEligibility.strip().lower()]

def loadAnnotations(repo: str = ANNOTATIONS_REPO, split: str = "train") -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(repo, split=split)
    return [_rowToDict(row) for row in dataset]

def _rowToDict(row: dict) -> dict:
    return {
        "annotationId": row["annotation_id"],
        "patientId": row["patient_id"],
        "note": row["note"],
        "nctId": row["trial_id"],
        "trialTitle": row["trial_title"],
        "criterionType": row["criterion_type"],
        "criterionText": row["criterion_text"],
        "expertEligibility": row["expert_eligibility"],
        "expertSentences": _parseIndexList(row["expert_sentences"]),
        "gpt4Eligibility": row["gpt4_eligibility"],
        "gpt4Sentences": _parseIndexList(row["gpt4_sentences"]),
        "training": bool(row["training"]),
    }

def toEvalPairs(rows: list[dict]) -> list[PatientCriterionPair]:
    pairs = []
    for row in rows:
        sentences = splitNumberedNote(row["note"])
        chosen = [sentences[i] for i in row["expertSentences"] if 0 <= i < len(sentences)]
        goldSpan = " ".join(chosen) if chosen else None
        pairs.append(
            PatientCriterionPair(
                patientId=row["patientId"],
                nctId=row["nctId"],
                criterionId=str(row["annotationId"]),
                note=row["note"],
                criterionText=row["criterionText"],
                criterionType=row["criterionType"],
                label=mapToEviLabels(row["expertEligibility"]),
                patientSpan=goldSpan,
            )
        )
    return pairs

def splitNumberedNote(note: str) -> list[str]:
    parts = re.split(r"(?:^|\s)(\d+)\.\s", note)
    sentences: dict[int, str] = {}
    remainder = iter(parts[1:])
    for number, chunk in zip(remainder, remainder):
        sentences[int(number)] = chunk.strip()
    if not sentences:
        return [note.strip()]
    return [sentences.get(i, "") for i in range(max(sentences) + 1)]

def _parseIndexList(value: str):
    try:
        return list(ast.literal_eval(value)) if value else []
    except Exception:
        return []
