import hashlib
import os
import time
import anthropic
import urllib.error
import json
import urllib.request

def callModel(prompt: str, config: dict) -> tuple[str, dict]:
    genConfig = config.get("matcher", {}).get("generative", {})
    provider = genConfig.get("provider", "ollama")
    model = genConfig[provider]["model"]
    
    hash_input = f"{provider}:{model}:{prompt}".encode("utf-8")
    prompt_hash = hashlib.sha256(hash_input).hexdigest()
    
    # cacheDir = os.path.join("data", "llmCache")
    cacheDir = genConfig["cachePath"]
    os.makedirs(cacheDir, exist_ok=True)
    cache_path = os.path.join(cacheDir, f"{prompt_hash}.json")
    
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            meta = cached_data.get("meta", {})
            meta["cached"] = True
            return cached_data.get("rawText", ""), meta

    maxAttempts = genConfig["maxRetries"] 
    rawText = None
    startTime = time.time()
    
    for attempt in range(1, maxAttempts + 1):
        try:
            if provider == "claude":
                client = anthropic.Anthropic()
                
                response = client.messages.create(
                    model=model,
                    max_tokens=genConfig["claude"]["maxTokens"],
                    #temperature=0.0,
                    messages=[{"role": "user", "content": prompt}]
                )
                # while(type=="text"): rawText = response.content[0].text
                rawText = next(b.text for b in response.content if b.type == "text")
                break
                
            elif provider == "ollama":
                req = urllib.request.Request(
                    # "http://localhost:11434/api/generate",
                    f'{genConfig["ollama"]["host"]}/api/generate',
                    data=json.dumps({
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        # "options": {"temperature": 0.0}
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                
                with urllib.request.urlopen(req, timeout=genConfig["ollama"]["timeoutSec"]) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    rawText = result.get("response", "")
                break
                
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            isTransportError = False
            
            if provider == "claude":
                if isinstance(e, (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)):
                    isTransportError = True
            elif provider == "ollama":
                if isinstance(e, (urllib.error.URLError, TimeoutError)):
                    isTransportError = True
                    
            if isTransportError and attempt < maxAttempts:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            
            raise e

    latency_ms = int((time.time() - startTime) * 1000)

    meta = {
        "provider": provider,
        "model": model,
        "promptHash": prompt_hash,
        "latencyMs": latency_ms,
        "cached": False,
        "attempts": attempt
    }

    # with open(cache_path, "w", encoding="utf-8") as f:
    #     json.dump({
    #         # "rawText": rawText,
    #         "rawText": rawText.strip() if rawText else None,
    #         "meta": meta
    #     }, f, indent=2)
    if rawText and rawText.strip():
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"rawText": rawText, "meta": meta}, f, indent=2)

    return rawText, meta