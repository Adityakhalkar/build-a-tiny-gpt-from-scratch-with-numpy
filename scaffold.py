"""
Build a Tiny GPT From Scratch with NumPy scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo of a tiny character-level GPT built with NumPy."""

import numpy as np
from solution import (
    build_char_vocab, encode_text, decode_ids, sample_batch,
    init_gpt_params, gpt_forward, cross_entropy_loss,
    cross_entropy_softmax_backward, init_adam_state, train_step,
    generate_text,
)


if __name__ == "__main__":
    np.random.seed(0)

    # ---- 1) Toy training corpus + tokenizer ----
    text = (
        "hello world. this is a tiny gpt built from scratch with numpy. "
        "the quick brown fox jumps over the lazy dog. "
        "characters become tokens, tokens become embeddings, "
        "embeddings flow through transformer blocks, and logits come out. "
    ) * 8

    char_to_id, id_to_char = build_char_vocab(text)
    vocab_size = len(char_to_id)
    print(f"[vocab] size={vocab_size} sample chars={list(char_to_id)[:12]}")

    data_ids = encode_text(text, char_to_id)
    print(f"[encode] len={len(data_ids)} first 20 ids={data_ids[:20].tolist()}")
    print(f"[decode] round-trip ok: {decode_ids(data_ids[:30], id_to_char)!r}")

    # ---- 2) Model hyperparameters ----
    d_model = 32
    d_ff = 64
    n_layers = 2
    max_len = 32
    batch_size = 8
    seq_len = 24
    lr = 3e-3
    n_steps = 80

    params = init_gpt_params(vocab_size, d_model, n_layers, max_len, d_ff)
    state = init_adam_state(params)
    print(f"[init] params keys={list(params.keys())}  n_blocks={len(params['blocks'])}")

    # ---- 3) Sanity-check a single forward pass + loss + grad_logits ----
    x, y = sample_batch(data_ids, batch_size, seq_len)
    logits, cache = gpt_forward(x, params)
    loss, probs = cross_entropy_loss(logits, y)
    grad_logits = cross_entropy_softmax_backward(probs, y)
    print(f"[forward] logits shape={logits.shape}  initial loss={loss:.4f}")
    print(f"[backward] grad_logits shape={grad_logits.shape}")

    # ---- 4) Training loop ----
    print("[train] starting...")
    for step in range(1, n_steps + 1):
        x, y = sample_batch(data_ids, batch_size, seq_len)
        out = train_step(params, x, y, state, lr)
        # train_step returns (loss, params, state) in this codebase
        if isinstance(out[0], float):
            loss, params, state = out
        else:
            params, state, loss = out
        if step == 1 or step % 10 == 0:
            print(f"  step {step:3d}  loss={loss:.4f}")

    # ---- 5) Generate text from a prompt ----
    prompt = "the quick "
    prompt_ids = encode_text(prompt, char_to_id)
    full_ids = generate_text(
        params, prompt_ids, n_new_tokens=80, max_len=max_len, temperature=0.8
    )
    generated = decode_ids(full_ids, id_to_char)
    print(f"[generate] prompt={prompt!r}")
    print(f"[generate] output={generated!r}")
