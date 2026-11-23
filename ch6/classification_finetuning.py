import json
from pathlib import Path

import tensorflow as tf
import tiktoken

from ch4.gpt_model import GPTModel, generate_text_simple
from ch5.evaluating_model import text_to_token_ids, token_ids_to_text
from ch5.get_device import get_device
from ch5.gpt_download import load_gpt2_params_from_tf_ckpt
from ch5.load_weights_from_openai import load_weights_into_gpt

if __name__ == "__main__":
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    gpt = GPTModel(BASE_CONFIG)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    model_size = CHOOSE_MODEL.split("(")[-1].rstrip(")")
    model_dir = Path(__file__).resolve().parents[1] / "ch5" / "gpt2" / model_size
    settings = json.loads((model_dir / "hparams.json").read_text(encoding="utf-8"))
    checkpoint = tf.train.latest_checkpoint(str(model_dir))
    params = load_gpt2_params_from_tf_ckpt(checkpoint, settings)
    load_weights_into_gpt(gpt, params)
    device = get_device()
    gpt.to(device)

    # text_1 = "Every effort moves you"
    # token_ids = generate_text_simple(
    #     model=gpt,
    #     idx=text_to_token_ids(text_1, tokenizer).to(device),
    #     max_new_tokens=15,
    #     context_size=BASE_CONFIG["context_length"],
    # )
    # print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text_simple(
        model=gpt,
        idx=text_to_token_ids(text_2, tokenizer).to(device),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))