from __future__ import annotations
# import ahocorasick
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from .schemas import Criterion, Decision

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
    raise NotImplementedError("implement the zero-shot NLI baseline")

def loraMatch(note: str, criterion: Criterion, config: dict) -> Decision:
    raise NotImplementedError("implement the LoRA-fine-tuned matcher")
