"""
Build a Tiny GPT From Scratch with NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_char_vocab
def build_char_vocab(text):
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_id, id_to_char

# Step 2 - encode_text
import numpy as np

def encode_text(text, char_to_id):
    return np.array([char_to_id[c] for c in text], dtype=np.int64)

# Step 3 - decode_ids
def decode_ids(ids, id_to_char):
    return ''.join(id_to_char[int(i)] for i in ids)

# Step 4 - sample_batch
import numpy as np

def sample_batch(data_ids, batch_size, seq_len):
    data_ids = np.asarray(data_ids)
    n = len(data_ids)
    high = n - seq_len
    ix = np.random.randint(0, high, size=batch_size)
    x = np.stack([data_ids[i:i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data_ids[i + 1:i + 1 + seq_len] for i in ix]).astype(np.int64)
    return x, y

# Step 5 - init_linear
import numpy as np

def init_linear(in_dim, out_dim):
    std = np.sqrt(1.0 / in_dim)
    W = np.random.normal(0.0, std, size=(in_dim, out_dim))
    b = np.zeros((out_dim,))
    return {'W': W, 'b': b}

# Step 6 - init_embedding
import numpy as np

def init_embedding(num_embeddings, dim):
    return np.random.normal(loc=0.0, scale=0.02, size=(num_embeddings, dim))

# Step 7 - init_layer_norm
import numpy as np

def init_layer_norm(dim):
    return {
        'gamma': np.ones((dim,), dtype=np.float64),
        'beta': np.zeros((dim,), dtype=np.float64),
    }

# Step 8 - init_attention_params
def init_attention_params(d_model):
    return {
        'q': init_linear(d_model, d_model),
        'k': init_linear(d_model, d_model),
        'v': init_linear(d_model, d_model),
        'o': init_linear(d_model, d_model),
    }

# Step 9 - init_mlp_params
def init_mlp_params(d_model, d_ff):
    return {
        'fc1': init_linear(d_model, d_ff),
        'fc2': init_linear(d_ff, d_model),
    }

# Step 10 - init_transformer_block_params
def init_transformer_block_params(d_model, d_ff):
    return {
        'ln1': init_layer_norm(d_model),
        'attn': init_attention_params(d_model),
        'ln2': init_layer_norm(d_model),
        'mlp': init_mlp_params(d_model, d_ff),
    }

# Step 11 - init_gpt_params
import numpy as np

def init_gpt_params(vocab_size, d_model, n_layers, max_len, d_ff):
    params = {
        'tok_emb': init_embedding(vocab_size, d_model),
        'pos_emb': init_embedding(max_len, d_model),
        'blocks': [init_transformer_block_params(d_model, d_ff) for _ in range(n_layers)],
        'ln_f': init_layer_norm(d_model),
        'head': init_linear(d_model, vocab_size),
    }
    return params

# Step 12 - softmax
import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

# Step 13 - linear_forward
import numpy as np

def linear_forward(x, W, b):
    return x @ W + b

# Step 14 - relu_forward
import numpy as np

def relu_forward(x):
    return np.maximum(x, 0)

# Step 15 - layer_norm_forward
import numpy as np

def layer_norm_forward(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    out = gamma * x_hat + beta
    cache = {'x_hat': x_hat, 'mean': mean, 'var': var, 'inv_std': inv_std, 'gamma': gamma}
    return out, cache

# Step 16 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

# Step 17 - embed_token_ids
import numpy as np

def embed_token_ids(ids, token_embedding):
    return token_embedding[ids]

# Step 18 - get_positional_embedding
import numpy as np

def get_positional_embedding(pos_embedding, seq_len):
    return pos_embedding[:seq_len]

# Step 19 - scaled_dot_product_attention
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    scale = 1.0 / np.sqrt(d_k)
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    attn = softmax(scores, axis=-1)
    out = np.matmul(attn, V)
    cache = {
        'Q': Q,
        'K': K,
        'V': V,
        'attn_weights': attn,
        'scale': scale,
        'mask': mask,
    }
    return out, cache

# Step 20 - self_attention_forward
def self_attention_forward(x, attn_params, mask):
    q_p = attn_params['q']
    k_p = attn_params['k']
    v_p = attn_params['v']
    o_p = attn_params['o']
    Q = linear_forward(x, q_p['W'], q_p['b'])
    K = linear_forward(x, k_p['W'], k_p['b'])
    V = linear_forward(x, v_p['W'], v_p['b'])
    attn_out, attn_cache = scaled_dot_product_attention(Q, K, V, mask)
    out = linear_forward(attn_out, o_p['W'], o_p['b'])
    cache = {
        'x': x,
        'Q': Q,
        'K': K,
        'V': V,
        'attn_out': attn_out,
        'attn_cache': attn_cache,
        'mask': mask,
    }
    return out, cache

# Step 21 - mlp_forward
def mlp_forward(x, mlp_params):
    fc1 = mlp_params['fc1']
    fc2 = mlp_params['fc2']
    h_pre = linear_forward(x, fc1['W'], fc1['b'])
    h = relu_forward(h_pre)
    out = linear_forward(h, fc2['W'], fc2['b'])
    cache = {'x': x, 'h_pre': h_pre, 'h': h}
    return out, cache

# Step 22 - transformer_block_forward
def transformer_block_forward(x, block_params, mask):
    ln1 = block_params['ln1']
    ln2 = block_params['ln2']
    ln1_out, ln1_cache = layer_norm_forward(x, ln1['gamma'], ln1['beta'])
    attn_out, attn_cache = self_attention_forward(ln1_out, block_params['attn'], mask)
    h1 = x + attn_out
    ln2_out, ln2_cache = layer_norm_forward(h1, ln2['gamma'], ln2['beta'])
    mlp_out, mlp_cache = mlp_forward(ln2_out, block_params['mlp'])
    h2 = h1 + mlp_out
    cache = {
        'x': x,
        'ln1_cache': ln1_cache,
        'attn_cache': attn_cache,
        'h1': h1,
        'ln2_cache': ln2_cache,
        'mlp_cache': mlp_cache,
    }
    return h2, cache

# Step 23 - gpt_input_embeddings
def gpt_input_embeddings(ids, params):
    import numpy as np
    tok = embed_token_ids(ids, params['token_embedding'])
    T = ids.shape[1]
    pos = get_positional_embedding(params['pos_embedding'], T)
    return tok + pos

# Step 24 - gpt_apply_blocks (not yet solved)
# TODO: implement

# Step 25 - gpt_final_projection (not yet solved)
# TODO: implement

# Step 26 - gpt_forward (not yet solved)
# TODO: implement

# Step 27 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 28 - cross_entropy_softmax_backward (not yet solved)
# TODO: implement

# Step 29 - linear_backward (not yet solved)
# TODO: implement

# Step 30 - relu_backward (not yet solved)
# TODO: implement

# Step 31 - layer_norm_grad_beta (not yet solved)
# TODO: implement

# Step 32 - layer_norm_grad_gamma (not yet solved)
# TODO: implement

# Step 33 - layer_norm_grad_xhat (not yet solved)
# TODO: implement

# Step 34 - layer_norm_grad_input (not yet solved)
# TODO: implement

# Step 35 - layer_norm_backward (not yet solved)
# TODO: implement

# Step 36 - attention_backward_v (not yet solved)
# TODO: implement

# Step 37 - attention_backward_weights (not yet solved)
# TODO: implement

# Step 38 - softmax_backward (not yet solved)
# TODO: implement

# Step 39 - attention_backward_qk (not yet solved)
# TODO: implement

# Step 40 - scaled_dot_product_attention_backward (not yet solved)
# TODO: implement

# Step 41 - output_projection_backward (not yet solved)
# TODO: implement

# Step 42 - qkv_projection_backward (not yet solved)
# TODO: implement

# Step 43 - self_attention_backward (not yet solved)
# TODO: implement

# Step 44 - mlp_backward (not yet solved)
# TODO: implement

# Step 45 - transformer_block_backward_mlp_path (not yet solved)
# TODO: implement

# Step 46 - transformer_block_backward_attn_path (not yet solved)
# TODO: implement

# Step 47 - transformer_block_backward (not yet solved)
# TODO: implement

# Step 48 - embedding_backward (not yet solved)
# TODO: implement

# Step 49 - gpt_head_backward (not yet solved)
# TODO: implement

# Step 50 - gpt_final_ln_backward (not yet solved)
# TODO: implement

# Step 51 - gpt_blocks_backward (not yet solved)
# TODO: implement

# Step 52 - gpt_embeddings_backward (not yet solved)
# TODO: implement

# Step 53 - gpt_backward (not yet solved)
# TODO: implement

# Step 54 - init_adam_state (not yet solved)
# TODO: implement

# Step 55 - adam_update_array (not yet solved)
# TODO: implement

# Step 56 - apply_adam_to_leaf (not yet solved)
# TODO: implement

# Step 57 - apply_adam_to_params (not yet solved)
# TODO: implement

# Step 58 - train_step (not yet solved)
# TODO: implement

# Step 59 - sample_next_token (not yet solved)
# TODO: implement

# Step 60 - crop_context (not yet solved)
# TODO: implement

# Step 61 - next_token_logits (not yet solved)
# TODO: implement

# Step 62 - generate_one_token (not yet solved)
# TODO: implement

# Step 63 - generate_text (not yet solved)
# TODO: implement

