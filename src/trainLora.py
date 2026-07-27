from __future__ import annotations
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from . import ingest
from .schemas import PatientCriterionPair

_LABEL2ID = {"MET": 0, "NOT_MET": 1, "UNKNOWN": 2}

# https://huggingface.co/docs/peft/en/package_reference/lora
# https://medium.com/@ichigo.v.gen12/understanding-lora-with-python-implementation-31375d2d1c10
# https://lightning.ai/lightning-ai/templates/code-lora-from-scratch?section=featured
def _buildDataset(pairs: list[PatientCriterionPair], tokenizer, maxLength: int) -> Dataset:
    texts = [p.patientSpan if p.patientSpan else p.note for p in pairs]
    encodings = tokenizer(
        texts,
        [p.criterionText for p in pairs],
        truncation=True,
        max_length=maxLength,
    )
    encodings["labels"] = [_LABEL2ID[p.label] for p in pairs]
    return Dataset.from_dict(encodings)

def train(config: dict) -> None:
    rows = ingest.loadAnnotations()
    pairs = ingest.toEvalPairs(rows)
    trainPairs = ingest.splitPairs(pairs, config)["train"]
    before = len(trainPairs)
    trainPairs = [p for p in trainPairs if p.note and p.criterionText]
    if len(trainPairs) < before:
        print(f"dropped {before - len(trainPairs)} train pairs with missing text")
    tokenizer = AutoTokenizer.from_pretrained(config["matcher"]["lora"]["baseModel"])
    baseModel = AutoModelForSequenceClassification.from_pretrained(config["matcher"]["lora"]["baseModel"], num_labels = len(_LABEL2ID))
    loraConfig = LoraConfig(
        r = config["matcher"]["lora"]["rank"],
        lora_alpha = config["matcher"]["lora"]["alpha"],
        lora_dropout = config["matcher"]["lora"]["dropout"],
        task_type = TaskType.SEQ_CLS,
    )
    model = get_peft_model(baseModel, loraConfig)
    
    trainDataset = _buildDataset(trainPairs, tokenizer, config["matcher"]["lora"]["maxLength"])
    collator = DataCollatorWithPadding(tokenizer)
    trainingArgs = TrainingArguments(
        output_dir = config["paths"]["loraAdapter"],
        num_train_epochs = config["matcher"]["lora"]["epochs"],
        learning_rate = config["matcher"]["lora"]["learningRate"],
        per_device_train_batch_size = 8,
        save_strategy = "no",
        report_to = [],
    )
    
    trainer = Trainer(
        model = model,
        args = trainingArgs,
        train_dataset = trainDataset,
        data_collator = collator
    )
    trainer.train()
    model.save_pretrained(config["paths"]["loraAdapter"])
    # return trainer

if __name__ == "__main__":
    from .config import loadConfig, setSeeds

    cfg = loadConfig()
    setSeeds(cfg["seed"])
    train(cfg)
