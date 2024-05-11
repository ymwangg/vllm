"""Attention layer for speculative decoding.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from vllm.attention.backends.flash_attn import (FlashAttentionBackend,
                                                FlashAttentionImpl,
                                                FlashAttentionMetadata)
from vllm import _custom_ops as ops


class SpeculateAttnBackend(FlashAttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["SpeculateAttnImpl"]:
        return SpeculateAttnImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "SpeculateAttnMetadata":
        return SpeculateAttnMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)


@dataclass
class SpeculateAttnMetadata(FlashAttentionMetadata):
    # Record real batch size in case of using cudagraph
    real_batch_size: Optional[int] = None

    # The sequence length per request. For normal decoding step, seq_len = 1.
    # For target evaluation step, seq_len = speculation_length + 1.
    seq_len: Optional[int] = 1

    # Indicate if running target evaluation step.
    is_multi_query_mode: Optional[bool] = False


class SpeculateAttnImpl(FlashAttentionImpl):

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SpeculateAttnMetadata,
        kv_scale: float,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        hidden_size = query.shape[-1]
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping.flatten(),
                attn_metadata.kv_cache_dtype,
            )
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        if prefill_meta := attn_metadata.prefill_metadata:
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.
            assert num_decode_tokens == 0, "Mixing prefill and decode is not allowed"
            output = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=prefill_meta.seq_start_loc,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                max_seqlen_q=prefill_meta.max_seq_len,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
            )
        else:
            assert num_prefill_tokens == 0, "Mixing prefill and decode is not allowed"
            assert len(query.shape) == 3
            decode_meta = attn_metadata.decode_metadata
            assert decode_meta.seq_lens_tensor is not None
            num_tokens = query.shape[0]
            seq_len = getattr(decode_meta, "seq_len", 1)
            batch_size = num_tokens // seq_len
            # Run multi-query attention.
            output = flash_attn_with_kvcache(
                query.view(batch_size, seq_len, self.num_heads,
                           self.head_size),
                key_cache,
                value_cache,
                cache_seqlens=decode_meta.seq_lens_tensor,
                block_table=decode_meta.block_tables,
                causal=True,
                alibi_slopes=self.alibi_slopes,
            )
        # Reshape the output tensor.
        return output.view(-1, hidden_size)
