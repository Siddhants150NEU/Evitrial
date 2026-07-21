from __future__ import annotations
# import ahocorasick
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from .schemas import Criterion, Decision
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import pipeline
import torch

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def match(note: str, criterion: Criterion, config: dict) -> Decision:
    if config["matcher"]["rung"] == "rules":
        return ruleMatch(note, criterion=criterion, config=config)
    elif config["matcher"]["rung"] == "zeroShot":
        return zeroShotMatch(note, criterion, config)
    else:
        return loraMatch(note, criterion, config)
    raise NotImplementedError("dispatch on config['matcher']['rung'] to the rungs below")

def ruleMatch(note: str, criterion: Criterion, config: dict) -> Decision:
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(criterion.text)
    words_lower = [word.lower() for word in words]
    filtered_words = [word for word in words_lower if word.isalnum() and word.lower() not in stop_words]
    keywords = set(filtered_words)
    # automaton = ahocorasick.Automaton()
    # for index, keyword in enumerate(keywords):
    #     automaton.add_word(keyword, (index, keyword))
    
    # count = 0
    # for i in keywords:
    #     if i in note:
    #         count+=1
    # if count>0:
    #     decision = "MET"
    # else:
    #     decision = "NON-MET"
    if not keywords:
        return Decision(
            label="UNKNOWN",
            confidence=0.0,
            trialSpan=criterion.text,
            patientSpan=None,
            criterionId=criterion.criterionId,
            criterionType=criterion.criterionType,
            verified=False,
        )
        
    sentences = sent_tokenize(note) if note and note.strip() else []
    bestSentence = ""
    max_matches = 0
    
    for sentence in sentences:
        sentenceWords = {
            word.lower() for word in word_tokenize(sentence) if word.isalnum()
        }
        matches = keywords.intersection(sentenceWords)
        match_count = len(matches)

        if match_count > max_matches:
            max_matches = match_count
            bestSentence = sentence.strip()
    
    hasMatch = max_matches > 0
    isCriterionNegated = getattr(criterion, "negation", False)
    negation_cues = {
        "no",
        "not",
        "denies",
        "without",
        "absent",
        "negative",
        "never",
    }
    sentenceHasNegation = False
    if bestSentence:
        bestSentenceWords = {w.lower() for w in word_tokenize(bestSentence)}
        sentenceHasNegation = bool(
            negation_cues.intersection(bestSentenceWords)
        )
        
    # decision = "MET" if has_match else "NON-MET"
    
    if hasMatch:
        if sentenceHasNegation or isCriterionNegated:
            decision = "NOT_MET"
        else:
            decision = "MET"
    else:
        decision = "UNKNOWN"
    confidence = max_matches / len(keywords)

    return Decision(
        label=decision,
        confidence=round(confidence, 4),
        trialSpan=criterion.text,
        patientSpan=bestSentence or None,
        criterionId=criterion.criterionId,
        criterionType=criterion.criterionType,
        verified=False,
    )
    # return Decision(
    #     label=decision,                       
    #     confidence=count / len(keywords), 
    #     trialSpan=criterion.text,
    #     patientSpan=bestSentence,        
    #     criterionId=criterion.criterionId,
    #     criterionType=criterion.criterionType,
    #     verified=False,
    # )
    # raise NotImplementedError("implement the lexical rule baseline")

def zeroShotMatch(note: str, criterion: Criterion, config: dict) -> Decision:
    model_name = "pritamdeka/PubMedBERT-MNLI-MedNLI" #config["matcher"]["nliModel"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    sentences = sent_tokenize(note) if note and note.strip() else []
    
    if not sentences:
        return Decision(
            label="UNKNOWN",
            confidence=0.0,
            trialSpan=criterion.text,
            patientSpan=None,
            criterionId=criterion.criterionId,
            criterionType=criterion.criterionType,
            verified=False,
        )
    
    id2label = getattr(model.config, "id2label", {})
    
    bestSentence = None
    bestLabel = "UNKNOWN"
    bestConfidence = 0.0
    
    for sentence in sentences:
        pairs = [[sentence, criterion.text]]
        with torch.no_grad():
            encoded = tokenizer(pairs, truncation=True, padding=True, return_tensors="pt", max_length=512)
            # logits = model(**encoded).logits.squeeze(dim=1)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]
            
            max_prob, max_idx = torch.max(probs, dim=-1)
            predictedIdx = max_idx.item()
            confidenceVal = max_prob.item()
            rawLabel = id2label.get(predictedIdx, "").upper()
            
            if "ENTAIL" in rawLabel:
                currentLabel = "MET"
            elif "CONTRADICT" in rawLabel:
                currentLabel = "NOT_MET"
            else:
                currentLabel = "UNKNOWN"
            
            if currentLabel != "UNKNOWN" and confidenceVal > bestConfidence:
                bestConfidence = confidenceVal
                bestLabel = currentLabel
                bestSentence = sentence.strip()
    
    return Decision(
        label=bestLabel,
        confidence=round(bestConfidence, 4),
        trialSpan=criterion.text,
        patientSpan=bestSentence,
        criterionId=criterion.criterionId,
        criterionType=criterion.criterionType,
        verified=False,
    )
    

def loraMatch(note: str, criterion: Criterion, config: dict) -> Decision:
    raise NotImplementedError("implement the LoRA-fine-tuned matcher")
