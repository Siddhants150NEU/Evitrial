from __future__ import annotations

def train(config: dict) -> None:
    raise NotImplementedError("implement LoRA fine-tuning on the train split only")

if __name__ == "__main__":
    from .config import loadConfig, setSeeds

    cfg = loadConfig()
    setSeeds(cfg["seed"])
    train(cfg)
