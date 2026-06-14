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

# Step 24 - gpt_apply_blocks
def gpt_apply_blocks(x, mask, params):
    block_caches = []
    for block in params['blocks']:
        x, cache = transformer_block_forward(x, block, mask)
        block_caches.append(cache)
    return x, block_caches

# Step 25 - gpt_final_projection
def gpt_final_projection(x, params):
    x_norm, ln_cache = layer_norm_forward(x, params['ln_f']['gamma'], params['ln_f']['beta'])
    logits = linear_forward(x_norm, params['head']['W'], params['head']['b'])
    head_cache = x_norm
    return logits, ln_cache, head_cache

# Step 26 - gpt_forward
def gpt_forward(ids, params):
    import numpy as np
    T = ids.shape[1]
    tok = embed_token_ids(ids, params['tok_emb'])
    pos = get_positional_embedding(params['pos_emb'], T)
    x = tok + pos
    mask = build_causal_mask(T)
    x, block_caches = gpt_apply_blocks(x, mask, params)
    logits, ln_cache, head_cache = gpt_final_projection(x, params)
    cache = {
        'ids': ids,
        'block_caches': block_caches,
        'final_cache': {'ln_cache': ln_cache, 'head_cache': head_cache},
    }
    return logits, cache

# Step 27 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(logits, targets):
    probs = softmax(logits, axis=-1)
    B, T, V = logits.shape
    flat_probs = probs.reshape(B * T, V)
    flat_targets = targets.reshape(B * T).astype(np.int64)
    target_probs = flat_probs[np.arange(B * T), flat_targets]
    target_probs = np.clip(target_probs, 1e-12, 1.0)
    loss = float(-np.mean(np.log(target_probs)) + 0.0)
    return loss, probs

# Step 28 - cross_entropy_softmax_backward
import numpy as np

def cross_entropy_softmax_backward(probs, targets):
    B, T, V = probs.shape
    grad = probs.copy()
    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]
    grad[b_idx, t_idx, targets] -= 1.0
    grad /= (B * T)
    return grad

# Step 29 - linear_backward
import numpy as np

def linear_backward(grad_y, x, W):
    in_dim = x.shape[-1]
    out_dim = grad_y.shape[-1]
    x_flat = x.reshape(-1, in_dim)
    g_flat = grad_y.reshape(-1, out_dim)
    grad_W = x_flat.T @ g_flat
    grad_b = g_flat.sum(axis=0)
    grad_x = grad_y @ W.T
    return grad_x, grad_W, grad_b

# Step 30 - relu_backward
import numpy as np

def relu_backward(grad_y, x):
    return grad_y * (x > 0)

# Step 31 - layer_norm_grad_beta
import numpy as np

def layer_norm_grad_beta(grad_y):
    grad_y = np.asarray(grad_y)
    D = grad_y.shape[-1]
    return grad_y.reshape(-1, D).sum(axis=0)

# Step 32 - layer_norm_grad_gamma
import numpy as np

def layer_norm_grad_gamma(grad_y, x_hat):
    grad_y = np.asarray(grad_y, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    prod = grad_y * x_hat
    if prod.ndim == 1:
        return prod
    axes = tuple(range(prod.ndim - 1))
    return prod.sum(axis=axes)

# Step 33 - layer_norm_grad_xhat
import numpy as np

def layer_norm_grad_xhat(grad_y, gamma):
    return grad_y * gamma

# Step 34 - layer_norm_grad_input
import numpy as np

def layer_norm_grad_input(grad_xhat, x_hat, inv_std):
    mean_g = np.mean(grad_xhat, axis=-1, keepdims=True)
    mean_gx = np.mean(grad_xhat * x_hat, axis=-1, keepdims=True)
    return inv_std * (grad_xhat - mean_g - x_hat * mean_gx)

# Step 35 - layer_norm_backward
import numpy as np

def layer_norm_backward(grad_y, cache):
    x_hat = cache['x_hat']
    inv_std = cache['inv_std']
    gamma = cache['gamma']
    grad_y = np.asarray(grad_y, dtype=float)
    grad_beta = layer_norm_grad_beta(grad_y)
    grad_gamma = layer_norm_grad_gamma(grad_y, x_hat)
    grad_xhat = layer_norm_grad_xhat(grad_y, gamma)
    grad_x = layer_norm_grad_input(grad_xhat, x_hat, inv_std)
    return grad_x, grad_gamma, grad_beta

# Step 36 - attention_backward_v
import numpy as np

def attention_backward_v(grad_out, attn_weights):
    return np.matmul(np.swapaxes(attn_weights, -1, -2), grad_out)

# Step 37 - attention_backward_weights
import numpy as np

def attention_backward_weights(grad_out, V):
    return grad_out @ V.transpose(0, 2, 1)

# Step 38 - softmax_backward
import numpy as np

def softmax_backward(grad_attn, attn_weights):
    dot = np.sum(grad_attn * attn_weights, axis=-1, keepdims=True)
    grad_scores = attn_weights * (grad_attn - dot)
    return grad_scores

# Step 39 - attention_backward_qk
import numpy as np

def attention_backward_qk(grad_scores, Q, K, scale):
    grad_qkT = grad_scores * scale
    grad_Q = np.matmul(grad_qkT, K)
    grad_K = np.matmul(grad_qkT.swapaxes(-1, -2), Q)
    return grad_Q, grad_K

# Step 40 - scaled_dot_product_attention_backward
def scaled_dot_product_attention_backward(grad_out, cache):
    Q = cache['Q']
    K = cache['K']
    V = cache['V']
    attn_weights = cache['attn_weights']
    scale = cache['scale']
    grad_V = attention_backward_v(grad_out, attn_weights)
    grad_attn = attention_backward_weights(grad_out, V)
    grad_scores = softmax_backward(grad_attn, attn_weights)
    grad_Q, grad_K = attention_backward_qk(grad_scores, Q, K, scale)
    return grad_Q, grad_K, grad_V

# Step 41 - output_projection_backward
import numpy as np

def output_projection_backward(grad_out, cache, attn_params):
    attn_out = cache['attn_out']
    W_o = attn_params['o']['W']
    grad_attn_out, grad_Wo, grad_bo = linear_backward(grad_out, attn_out, W_o)
    return grad_attn_out, grad_Wo, grad_bo

# Step 42 - qkv_projection_backward
import numpy as np

def qkv_projection_backward(grad_Q, grad_K, grad_V, cache, attn_params):
    x = cache['x']
    grad_x_q, grad_W_q, grad_b_q = linear_backward(grad_Q, x, attn_params['q']['W'])
    grad_x_k, grad_W_k, grad_b_k = linear_backward(grad_K, x, attn_params['k']['W'])
    grad_x_v, grad_W_v, grad_b_v = linear_backward(grad_V, x, attn_params['v']['W'])
    grad_x = grad_x_q + grad_x_k + grad_x_v
    grads = {
        'q': {'W': grad_W_q, 'b': grad_b_q},
        'k': {'W': grad_W_k, 'b': grad_b_k},
        'v': {'W': grad_W_v, 'b': grad_b_v},
    }
    return grad_x, grads

# Step 43 - self_attention_backward
def self_attention_backward(grad_out, cache, attn_params):
    grad_attn_out, grad_Wo, grad_bo = output_projection_backward(
        grad_out, cache, attn_params
    )
    grad_Q, grad_K, grad_V = scaled_dot_product_attention_backward(
        grad_attn_out, cache['attn_cache']
    )
    grad_x, qkv_grads = qkv_projection_backward(
        grad_Q, grad_K, grad_V, cache, attn_params
    )
    grads = {**qkv_grads, 'o': {'W': grad_Wo, 'b': grad_bo}}
    return grad_x, grads

# Step 44 - mlp_backward
import numpy as np

def mlp_backward(grad_out, cache, mlp_params):
    x = cache['x']
    h_pre = cache['h_pre']
    h = cache['h']
    fc1 = mlp_params['fc1']
    fc2 = mlp_params['fc2']

    grad_h, grad_W2, grad_b2 = linear_backward(grad_out, h, fc2['W'])
    grad_h_pre = relu_backward(grad_h, h_pre)
    grad_x, grad_W1, grad_b1 = linear_backward(grad_h_pre, x, fc1['W'])

    grads = {
        'fc1': {'W': grad_W1, 'b': grad_b1},
        'fc2': {'W': grad_W2, 'b': grad_b2},
    }
    return grad_x, grads

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

