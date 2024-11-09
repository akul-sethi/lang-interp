from datasets import load_dataset
from nnsight import CONFIG, LanguageModel
import os
from dotenv import load_dotenv
import torch

load_dotenv()
CONFIG.set_default_api_key(os.environ["API_KEY"])


def ma_across_prompt_type(dataset):
    """Assume data set takes from rows=(prompt type) cols=(language). returns dataset: rows=(layer) cols=(language)"""
    llm = LanguageModel("bigcode/starcoder2-15b")
    LAYERS = 40
    DIMS = 6144
    PROMPTS, LANGS = dataset.shape

    # [prompt, layer, lang, dims]
    layers = torch.tensor([PROMPTS, LAYERS, LANGS, DIMS])
    for prompt in range(PROMPTS):
        for lang in range(LANGS):
            with llm.trace(dataset[prompt][lang]) as tracer:
                for layer in range(LAYERS):
                    layers[prompt, layer, lang, :] = llm.model.layers[
                        layer
                    ].output.save()


ma_across_prompt_type(None)
