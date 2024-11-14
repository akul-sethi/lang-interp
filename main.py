from datasets import load_dataset
from nnsight import CONFIG, LanguageModel
import os
from dotenv import load_dotenv
import torch
import numpy as np
from transformers import AutoTokenizer

load_dotenv()
CONFIG.set_default_api_key(os.environ["API_KEY"])

llm = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
ds = load_dataset("bigcode/humanevalpack", lang, trust_remote_code=True)["test"]
LANGS = ["python", "js", "java", "go", "cpp", "rust"]


def ma_across_prompt_type(lang):
    """Assume data set takes from rows=(prompt type) cols=(language). returns dataset: rows=(layer) cols=(language)"""
    LAYERS = 32
    DIMS = 4096
    PROMPTS = len(ds)

    # [prompt, layer, lang, dims]
    layers = torch.tensor([LAYERS, DIMS])
    states = [[None for _ in range(LAYERS)] for _ in range(PROMPTS)]
    for prompt in range(PROMPTS):
        with llm.trace(ds[prompt]["prompt"]) as tracer:
            for layer in range(LAYERS):
                states[prompt][layer] = llm.model.layers[layer].output.save()

    for prompt in range(PROMPTS):
        for layer in range(LAYERS):
            layers[layer, :] += states[prompt][layer]

    return layers / PROMPTS


for lang in LANGS:
    torch.save(ma_across_prompt_type(lang), f"{lang}.pt")
