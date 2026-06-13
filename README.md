# Build a Tiny GPT From Scratch with NumPy

Implement a character-level GPT-style transformer end-to-end using only NumPy: tokenization, embeddings, single-head causal self-attention, transformer blocks, manual backprop, Adam, and text generation. Builds deep intuition for how modern language models actually work under the hood.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** build_char_vocab
- [x] **2.** encode_text
- [x] **3.** decode_ids
- [x] **4.** sample_batch
- [x] **5.** init_linear
- [x] **6.** init_embedding
- [x] **7.** init_layer_norm
- [x] **8.** init_attention_params
- [x] **9.** init_mlp_params
- [x] **10.** init_transformer_block_params
- [x] **11.** init_gpt_params
- [x] **12.** softmax
- [x] **13.** linear_forward
- [x] **14.** relu_forward
- [x] **15.** layer_norm_forward
- [x] **16.** build_causal_mask
- [x] **17.** embed_token_ids
- [x] **18.** get_positional_embedding
- [x] **19.** scaled_dot_product_attention
- [x] **20.** self_attention_forward
- [x] **21.** mlp_forward
- [x] **22.** transformer_block_forward
- [x] **23.** gpt_input_embeddings
- [x] **24.** gpt_apply_blocks
- [x] **25.** gpt_final_projection
- [x] **26.** gpt_forward
- [x] **27.** cross_entropy_loss
- [x] **28.** cross_entropy_softmax_backward
- [x] **29.** linear_backward
- [x] **30.** relu_backward
- [x] **31.** layer_norm_grad_beta
- [ ] **32.** layer_norm_grad_gamma
- [ ] **33.** layer_norm_grad_xhat
- [ ] **34.** layer_norm_grad_input
- [ ] **35.** layer_norm_backward
- [ ] **36.** attention_backward_v
- [ ] **37.** attention_backward_weights
- [ ] **38.** softmax_backward
- [ ] **39.** attention_backward_qk
- [ ] **40.** scaled_dot_product_attention_backward
- [ ] **41.** output_projection_backward
- [ ] **42.** qkv_projection_backward
- [ ] **43.** self_attention_backward
- [ ] **44.** mlp_backward
- [ ] **45.** transformer_block_backward_mlp_path
- [ ] **46.** transformer_block_backward_attn_path
- [ ] **47.** transformer_block_backward
- [ ] **48.** embedding_backward
- [ ] **49.** gpt_head_backward
- [ ] **50.** gpt_final_ln_backward
- [ ] **51.** gpt_blocks_backward
- [ ] **52.** gpt_embeddings_backward
- [ ] **53.** gpt_backward
- [ ] **54.** init_adam_state
- [ ] **55.** adam_update_array
- [ ] **56.** apply_adam_to_leaf
- [ ] **57.** apply_adam_to_params
- [ ] **58.** train_step
- [ ] **59.** sample_next_token
- [ ] **60.** crop_context
- [ ] **61.** next_token_logits
- [ ] **62.** generate_one_token
- [ ] **63.** generate_text

---

Built on Deep-ML.
