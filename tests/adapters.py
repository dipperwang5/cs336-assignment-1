from __future__ import annotations

import os
from typing import IO, Any, BinaryIO, Iterator
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor, nn

from dataclasses import dataclass
from collections import defaultdict
import regex as re
from multiprocessing import Pool, cpu_count
import pickle
import math
import pdb

from einops import rearrange, einsum, reduce, repeat


class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int,
        out_features: int, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None):
        """Construct a linear transformation module."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        mean = 0.0
        std = math.sqrt(2.0 / (self.in_features + self.out_features))

        nn.init.trunc_normal_(self.W, mean, std, -3*std, 3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return einsum(x, self.W, "... input_dim, output_dim input_dim -> ... output_dim")


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(d_in, d_out)
    state_dict_to_load = {"W": weights}
    linear.load_state_dict(state_dict_to_load)
    return linear(in_features)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None):
        """Construct an embedding module."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Parameter(torch.empty(self.num_embeddings,
                                      self.embedding_dim,
                                      device=device,
                                      dtype=dtype))
        nn.init.trunc_normal_(self.embeddings, 0, 1, -3, 3)

    def forward(
        self,
        token_ids: torch.Tensor
        ) -> torch.Tensor:
        return self.embeddings[token_ids]

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    embedding = Embedding(vocab_size, d_model)
    state_dict_to_load = {"embeddings": weights}
    embedding.load_state_dict(state_dict_to_load)
    return embedding(token_ids)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ):
        super().__init__()
        d_ff = (int(8.0 / 3.0 * d_model) // 64 * 64)
        
        self.ffn_1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.ffn_2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.ffn_3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(
        self,
        x: torch.Tensor):
        return self.ffn_2((self.silu(self.ffn_1(x)) * self.ffn_3(x)))



def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = SwiGLU(d_model, d_ff)
    swiglu.ffn_1.W.data = w1_weight
    swiglu.ffn_2.W.data = w2_weight
    swiglu.ffn_3.W.data = w3_weight

    return swiglu(in_features)


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = SoftMax()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None) -> torch.Tensor:
        
        d_k = Q.shape[-1]
        pre_softmax_score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
        if mask is not None:
            pre_softmax_score.masked_fill_(mask==0.0, float("-inf"))
        softmax_score = self.softmax(pre_softmax_score, -1)
        return einsum(softmax_score, V, "... queries keys, ... keys d_v -> ... queries d_v")

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    scaled_dot_product_attention = DotProductAttention()
    return scaled_dot_product_attention(Q, K, V, mask)

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None):

        super().__init__()

        self.d_model = d_model
        self.head = num_heads
        self.scaled_dot_product_attention = DotProductAttention()

        self.WQ = Linear(self.d_model, self.d_model)
        self.WK = Linear(self.d_model, self.d_model)
        self.WV = Linear(self.d_model, self.d_model)
        self.WO = Linear(self.d_model, self.d_model)

        self.pos_encode = rope

    def forward(
        self,
        in_features: torch.Tensor,
        token_positions: torch.Tensor | None = None):
        
        Q = self.WQ(in_features)
        K = self.WK(in_features)
        V = self.WV(in_features)

        sequence_length = Q.shape[-2]
        causal_mask = ~torch.triu(torch.ones_like(torch.empty(sequence_length,sequence_length)), diagonal=1).bool()

        multi_head_Q = rearrange(Q, "... sequence_length (h d_k) -> h ... sequence_length d_k", h = self.head)
        multi_head_K = rearrange(K, "... sequence_length (h d_k) -> h ... sequence_length d_k", h = self.head)
        multi_head_V = rearrange(V, "... sequence_length (h d_v) -> h ... sequence_length d_v", h = self.head)

        if self.pos_encode is not None:
            multi_head_Q = self.pos_encode(multi_head_Q, token_positions)
            multi_head_K = self.pos_encode(multi_head_K, token_positions)

        multi_head = self.scaled_dot_product_attention(multi_head_Q, multi_head_K, multi_head_V, causal_mask)
        multi_head = rearrange(multi_head, "h ... sequence_length d_v -> ... sequence_length (h d_v)")
        multi_head_self_attention = self.WO(multi_head)

        return multi_head_self_attention
    

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multi_head_self_attention = MultiheadSelfAttention(d_model, num_heads)
    multi_head_self_attention.WQ.W.data = q_proj_weight
    multi_head_self_attention.WK.W.data = k_proj_weight
    multi_head_self_attention.WV.W.data = v_proj_weight
    multi_head_self_attention.WO.W.data = o_proj_weight

    return multi_head_self_attention(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    rope = RoPE(theta, d_model // num_heads, max_seq_len)

    multi_head_self_attention = MultiheadSelfAttention(d_model, num_heads, rope)
    multi_head_self_attention.WQ.W.data = q_proj_weight
    multi_head_self_attention.WK.W.data = k_proj_weight
    multi_head_self_attention.WV.W.data = v_proj_weight
    multi_head_self_attention.WO.W.data = o_proj_weight

    return multi_head_self_attention(in_features, token_positions)


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_leng: int,
        device: torch.device | None = None):
        super().__init__()

        theta_kd = theta ** (-2 * torch.arange(0, int(d_k//2), device=device) / d_k) #d_k / 2
        pos = torch.arange(0, max_seq_leng, device=device)
        angle = einsum(pos, theta_kd, "i, k -> i k") #max_seq_len d_k/2
        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor) -> torch.Tensor:
        # x (... seq_len, d_k)
        # token_positions (... seq_len)
        cos = self.cos[token_positions] # (... seq_len d_k/2)
        sin = self.sin[token_positions]

        x1 = x[..., 0::2] #(... seq_len d_k/2)
        x2 = x[..., 1::2]

        rotated_x1 = cos * x1 - sin * x2 # (... seq_len d_k/2)
        rotated_x2 = sin * x1 + cos * x2

        rotated_x = rearrange([rotated_x1, rotated_x2], 's ... d_half -> ... (d_half s)')
        return rotated_x

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RoPE(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int | None,
        max_seq_len: int | None):
        super().__init__()

        self.rms_norm_1 = RMSNorm(d_model)
        self.rms_norm_2 = RMSNorm(d_model)
        
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, d_model // num_heads, max_seq_len)
        else:
            self.rope = None
        self.multi_head_self_attention = MultiheadSelfAttention(d_model, num_heads, self.rope)

        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor):
        # x: " batch sequence_length d_model"
        token_positions = repeat(torch.arange(x.shape[-2]), "sequence_length -> batch sequence_length", batch=x.shape[0]) # batch sequence_length
        attention_x = self.multi_head_self_attention(self.rms_norm_1(x), token_positions) # " batch sequence_length d_model"
         
        x = x + attention_x # " batch sequence_length d_model"

        ffn_x = self.ffn(self.rms_norm_2(x)) # " batch sequence_length d_model"

        return ffn_x + x



def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    transformer_block = TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len)
    transformer_block.rms_norm_1.g.data = weights["ln1.weight"]
    transformer_block.multi_head_self_attention.WK.W.data = weights["attn.k_proj.weight"]
    transformer_block.multi_head_self_attention.WQ.W.data = weights["attn.q_proj.weight"]
    transformer_block.multi_head_self_attention.WV.W.data = weights["attn.v_proj.weight"]
    transformer_block.multi_head_self_attention.WO.W.data = weights["attn.output_proj.weight"]
    transformer_block.rms_norm_2.g.data = weights["ln2.weight"]
    transformer_block.ffn.ffn_1.W.data = weights["ffn.w1.weight"]
    transformer_block.ffn.ffn_2.W.data = weights["ffn.w2.weight"]
    transformer_block.ffn.ffn_3.W.data = weights["ffn.w3.weight"]

    return transformer_block(in_features)


class Transformer_LM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int | None,
        context_length: int | None,
        vocab_size: int,
        num_layers: int):
        super().__init__()

        self.num_layers = num_layers

        self.blocks = nn.ModuleList(
                    [TransformerBlock(d_model, num_heads, d_ff, theta, context_length)
                    for _ in range(self.num_layers)])
        
        self.embedding = Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.ffn = Linear(d_model, vocab_size)
        self.softmax = SoftMax()

    def forward(
        self,
        x: torch.Tensor):
        # x (batch_size sequence_length)
        x = self.embedding(x) #(batch_size sequence_length d_model)
        for block in self.blocks:
            x = block(x) #(batch_size sequence_length d_model)
        x = self.norm(x) #(batch_size sequence_length d_model)
        x = self.ffn(x) #(batch_size sequence_length vocab_size)
        # x = self.softmax(x, dim=-1) #(batch_size sequence_length vocab_size)

        return x
    

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer_lm = Transformer_LM(d_model, num_heads, d_ff, rope_theta, context_length, vocab_size, num_layers)
    transformer_lm.embedding.embeddings.data = weights["token_embeddings.weight"]
    for layer in range(num_layers):
        transformer_lm.blocks[layer].rms_norm_1.g.data = weights[f"layers.{layer}.ln1.weight"]
        transformer_lm.blocks[layer].multi_head_self_attention.WK.W.data = weights[f"layers.{layer}.attn.k_proj.weight"]
        transformer_lm.blocks[layer].multi_head_self_attention.WQ.W.data = weights[f"layers.{layer}.attn.q_proj.weight"]
        transformer_lm.blocks[layer].multi_head_self_attention.WV.W.data = weights[f"layers.{layer}.attn.v_proj.weight"]
        transformer_lm.blocks[layer].multi_head_self_attention.WO.W.data = weights[f"layers.{layer}.attn.output_proj.weight"]
        transformer_lm.blocks[layer].rms_norm_2.g.data = weights[f"layers.{layer}.ln2.weight"]
        transformer_lm.blocks[layer].ffn.ffn_1.W.data = weights[f"layers.{layer}.ffn.w1.weight"]
        transformer_lm.blocks[layer].ffn.ffn_2.W.data = weights[f"layers.{layer}.ffn.w2.weight"]
        transformer_lm.blocks[layer].ffn.ffn_3.W.data = weights[f"layers.{layer}.ffn.w3.weight"]
    transformer_lm.norm.g.data = weights["ln_final.weight"]
    transformer_lm.ffn.W.data = weights["lm_head.weight"]
    
    return transformer_lm(in_indices)


class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device=None, 
        dtype=None):
        """Construct the RMSNorm module."""
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        g = torch.ones(self.d_model, device=device, dtype=dtype)
        self.g = nn.Parameter(g)

    def forward(
        self, 
        x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS_x = torch.sqrt((1 / self.d_model) * reduce(x**2, "... d_model -> ... 1", "sum") + self.eps)
        RMS_norm = x / RMS_x * rearrange(self.g, "d_model -> 1 1 d_model")

        return RMS_norm.to(in_dtype)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RMSNorm(d_model, eps)
    state_dict_to_load = {"g": weights}
    rmsnorm.load_state_dict(state_dict_to_load)

    return rmsnorm(in_features)


class SiLU(nn.Module):
    def __init__(
        self):
        super().__init__()

    def forward(
        self, 
        x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
        
def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    silu = SiLU()
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError

class SoftMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        dim: int):
        
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_stable = x - x_max
        
        exp_x = torch.exp(x_stable)
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
        
        return exp_x / sum_exp_x
        


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    softmax = SoftMax()
    return softmax(in_features, dim)

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = SoftMax()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor):

        #logits (batch_size vocab_size)
        # targets (batch_size)
        logit_max = reduce(logits, "batch_size vocab_size -> batch_size 1", "max")
        logits_shift = logits - logit_max #(batch_size vocab_size)
        logit_target = logits_shift[torch.arange(logits_shift.shape[0]), targets]
        
        log_sum_exp_logits = torch.log(reduce(torch.exp(logits_shift), "batch_size vocab_size -> batch_size", "sum"))

        return reduce(log_sum_exp_logits - logit_target, "batch_size -> 1", "mean")
        


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    cross_entropy = CrossEntropy()
    return cross_entropy(inputs, targets)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    grads = []
    grads_l2_norm = 0
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad)
            grads_l2_norm = grads_l2_norm + reduce(p.grad**2, "... -> 1", "sum")
    
    grads_l2_norm = torch.sqrt(grads_l2_norm)

    if grads_l2_norm >= max_l2_norm:
        clip_coef = max_l2_norm / (grads_l2_norm + eps)
        for grad in grads:
            grad.mul_(clip_coef)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    gradient_clipping(parameters, max_l2_norm)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: Float = 1e-3,
        weight_decay: Float = 0.01,
        betas: tuple[Float, Float] = (0.9, 0.999),
        eps: Float = 1e-8,
        ):

        defaults = {"lr": lr, "weight_decay": weight_decay, 
                    "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]

            beta_1, beta_2 = betas

            for p in group["params"]: 
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                m = state.get("m", torch.zeros(p.shape)) # Get m, first moment vector
                v = state.get("v", torch.zeros(p.shape)) # Get v, second moment vector
                t = state.get("t", 1)

                grad = p.grad.data # Get the gradient of loss with respect to p.
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad**2

                lr_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)
                
                p.data -= lr_t * m / (torch.sqrt(v) + eps) # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v

        return loss

def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


class CosineSchedule(nn.Module):
    def __init__(
        self,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
        ) -> float:
        super().__init__()
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def forward(
        self,
        it: int):
        if it < self.warmup_iters:
            return it / self.warmup_iters * self.max_learning_rate
        elif it >= self.warmup_iters and it <= self.cosine_cycle_iters:
            return self.min_learning_rate + 1/2 * (1 + math.cos((it - self.warmup_iters)
                                            / (self.cosine_cycle_iters - self.warmup_iters) * math.pi))\
                                            * (self.max_learning_rate - self.min_learning_rate)
        else:
            return self.min_learning_rate


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    cosine_schedule = CosineSchedule(max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    return cosine_schedule(it)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        if not special_tokens:
            special_tokens = []

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        if self.special_tokens:
            for token_str in self.special_tokens:
                token_bytes = token_str.encode("utf-8")
                if token_bytes not in self.vocab.values():
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token_bytes
        
        self.bytes_to_int = {v: k for k, v in self.vocab.items()}
        self.int_to_bytes = {k: v for k, v in self.vocab.items()}
        self.special_token_bytes = {s.encode("utf-8") for s in self.special_tokens}

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def _bpe_merge(self, word_bytes: bytes) -> list[int]:
        """Helper function to perform BPE on a single word."""
        parts = [bytes([b]) for b in word_bytes]

        while len(parts) > 1:
            pairs = [(parts[i], parts[i+1]) for i in range(len(parts) - 1)]
            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))

            if best_pair not in self.merge_ranks:
                break

            for i in range(len(parts) - 1):
                if (parts[i], parts[i+1]) == best_pair:
                    parts = parts[:i] + [parts[i] + parts[i+1]] + parts[i+2:]
                    break
        
        return [self.bytes_to_int[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # split the text based on special tokens
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        split_pattern = "|".join(re.escape(s) for s in sorted_special_tokens)

        if split_pattern:
            sentences = re.split(f"({split_pattern})", text)
        else:
            sentences = [text]

        # pretokenize the text
        text_pretokenized = []
        for sentence in sentences:
            if sentence in self.special_tokens:
                text_pretokenized.append((sentence.encode("utf-8")))
                continue
            for word in re.finditer(PAT, sentence):
                text_pretokenized.append(word.group().encode("utf-8"))

        # merge the text tokens
        final_token_ids = []
        for word_pretokenized in text_pretokenized:
            if word_pretokenized in self.special_token_bytes:
                final_token_ids.append(self.bytes_to_int[word_pretokenized])
            else:
                final_token_ids.extend(self._bpe_merge(word_pretokenized))

        return final_token_ids



    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        for line in iterable:
            yield from self.encode(line)


    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        id_decodes = b"".join(self.int_to_bytes.get(token_id, b'') for token_id in ids)
        text = id_decodes.decode("utf-8", errors='replace')

        return text





def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_chunk(
    input_path: str, 
    start: int, 
    end: int, 
    special_tokens: list
    ) -> dict[tuple[bytes, ...], int]:
        
        with open(input_path, "rb") as f:
            f.seek(start)
            corpus_chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        split_pattern = "|".join(re.escape(s) for s in special_tokens)
        corpus_chunk = re.split(split_pattern, corpus_chunk)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        chunk_frequency_table = defaultdict(int)
        for document in corpus_chunk:
            if not document: 
                continue
            document_pretokenized = re.finditer(PAT, document)
            for word in document_pretokenized:
                word_pretokenized = tuple(bytes([b]) for b in word.group().encode('utf-8'))
                if word_pretokenized:
                    chunk_frequency_table[word_pretokenized] += 1

        return chunk_frequency_table

def get_pair_counts(frequency_table: dict[tuple[bytes, ...], int]) -> defaultdict[tuple[bytes, bytes], int]:
    """Helper function to calculate initial pair counts."""
    pair_counts = defaultdict(int)
    for word, count in frequency_table.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i+1])] += count
    return pair_counts


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    """

    ### initialize the vocab
    vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
    token_id = 256
    for special_token in special_tokens:
        vocab[token_id] = special_token.encode("utf-8")
        token_id += 1

    ### pre-tokenization
    num_cpu = cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_cpu, "".join(special_tokens).encode("utf-8"))

    task_args = []
    for i in range(len(boundaries) - 1):
        task_args.append((input_path, boundaries[i], boundaries[i+1], special_tokens))
    
    with Pool(num_cpu) as p:
        chunk_frequency_tables = p.starmap(pretokenize_chunk, task_args)
    # chunk_frequency_tables = [pretokenize_chunk(*args) for args in task_args]

    frequency_table = defaultdict(int)
    for chunk_frequency_table in chunk_frequency_tables:
        for word_pretokenized, cnt in chunk_frequency_table.items():
            frequency_table[word_pretokenized] += cnt

    ### merges
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        # Calculate pair counts in every loop.
        pair_counts = get_pair_counts(frequency_table)

        if not pair_counts:
            break

        # This preserves the tie-breaking logic.
        max_pair = max(pair_counts, key=lambda k: (pair_counts[k], k))

        # Add the new merge and token to our vocabulary
        merges.append(max_pair)
        new_token = max_pair[0] + max_pair[1]
        vocab[len(vocab)] = new_token

        new_frequency_table = defaultdict(int)
        for word, count in frequency_table.items():
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == max_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_frequency_table[tuple(new_word)] += count
        
        frequency_table = new_frequency_table

    return vocab, merges

