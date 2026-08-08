import math
import os
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.types import Device

import dnnlpy.cs336.assignment1 as assignment1
import dnnlpy.cs336.utils as utils
import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF
import dnnlpy.optim as dopt
import dnnlpy.tokenizers as dltk

__all__ = [
    'get_adamw_cls',
    'get_tokenizer',
    'run_cross_entropy',
    'run_embedding',
    'run_get_batch',
    'run_get_lr_cosine_schedule',
    'run_gradient_clipping',
    'run_linear',
    'run_load_checkpoint',
    'run_multihead_self_attention',
    'run_multihead_self_attention_with_rope',
    'run_rmsnorm',
    'run_rope',
    'run_save_checkpoint',
    'run_scaled_dot_product_attention',
    'run_silu',
    'run_softmax',
    'run_swiglu',
    'run_train_bpe',
    'run_transformer_block',
    'run_transformer_lm',
]


def run_linear(x: Tensor, weight: Tensor) -> Tensor:
    """Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        x (Tensor): The linear weights to use.
        weight (Tensor): The output tensor to apply the function to.

    Returns:
        output (Tensor): The transformed output of your linear module.
    """
    output = dF.linear(x, weight)
    return output


def run_embedding(token_ids: Tensor, weight: Tensor) -> Tensor:
    """Given the weights of an Embedding layer, get the embeddings for a batch
    of token ids.

    Args:
        token_ids (Tensor): The set of token ids to fetch from the Embedding layer.
        weight (Tensor): The embedding vectors to fetch from.

    Returns:
        output (Tensor): Batch of embeddings returned by your Embedding layer.
    """
    output = dF.embedding(token_ids, weight)
    return output


def run_swiglu(
    x: Tensor,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
) -> Tensor:
    """Given the weights of a SwiGLU network, return the output of your implementation
    with these weights.

    Args:
        x (Tensor): Input embeddings to the feed-forward layer.
        w1_weight (Tensor): Stored weights for w1.
        w2_weight (Tensor): Stored weights for w2.
        w3_weight (Tensor): Stored weights for w3.

    Returns:
        output (Tensor): Output embeddings of the same shape as the input embeddings.
    """
    x1 = dF.linear(x, w1_weight)
    x3 = dF.linear(x, w3_weight)
    x2 = dF.silu(x1) * x3
    x2 = dF.linear(x2, w2_weight)
    return x2


def run_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
) -> Tensor:
    """Given key (K), query (Q), and value (V) tensors, return the output of your
    scaled dot product attention implementation.

    Args:
        query (Tensor): The query tensor.
        key (Tensor): The key tensor.
        value (Tensor): The value tensor.
        sttn_mask (Tensor | None): An optional attention mask to apply to the attention
            scores before softmax.

    Returns:
        attn_output (Tensor): Output of scaled dot product attention.
    """
    if attn_mask is not None:
        attn_mask = ~attn_mask

    output, _ = dF.scaled_dot_product_attention(query, key, value, attn_mask)
    return output


def run_multihead_self_attention(
    x: Tensor,
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
) -> Tensor:
    """Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.

    This function should not use RoPE.

    Args:
        x (Tensor): Input features to the multi-head self-attention layer.
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        q_proj_weight (Tensor): Weights for the query projection.
        k_proj_weight (Tensor): Weights for the key projection.
        v_proj_weight (Tensor): Weights for the value projection.
        o_proj_weight (Tensor): Weights for the output projection.
        in_features (Tensor): Tensor to run your implementation on.

    Returns:
        output (Tensor): Tensor with the output of running your optimized, batched
            multi-headed attention implementation with the given QKV projection weights
            and input features.
    """
    attention = assignment1.MultiheadSelfAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        bias=False,
    )

    params = {
        'q_proj.weight': q_proj_weight,
        'k_proj.weight': k_proj_weight,
        'v_proj.weight': v_proj_weight,
        'out_proj.weight': o_proj_weight,
    }
    attention.load_state_dict(params)

    output = attention(x)
    return output


def run_multihead_self_attention_with_rope(
    x: Tensor,
    token_pos: Tensor | None,
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
) -> Tensor:
    """Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.

    This function should include RoPE.

    Args:
        x (Tensor): Tensor to run your implementation on.
        token_pos (Tensor | None): Optional tensor with the positions of the tokens.
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE theta parameter.
        q_proj_weight (Tensor): Weights for the query projection.
        k_proj_weight (Tensor): Weights for the key projection.
        v_proj_weight (Tensor): Weights for the value projection.
        o_proj_weight (Tensor): Weights for the output projection.

    Returns:
        output (Tensor): Tensor with the output of running your optimized, batched
            multi-headed attention implementation with the given QKV projection weights
            and input features.
    """
    rope = assignment1.RotaryPositionalEmbedding(
        embed_dim=d_model // num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
    )
    attention = assignment1.MultiheadSelfAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        bias=False,
        rope=rope,
    )

    params = {
        'q_proj.weight': q_proj_weight,
        'k_proj.weight': k_proj_weight,
        'v_proj.weight': v_proj_weight,
        'out_proj.weight': o_proj_weight,
    }
    attention.load_state_dict(params)

    output = attention(x, token_pos)
    return output


def run_rope(
    x: Tensor,
    token_pos: Tensor,
    embed_dim: int,
    theta: float,
    max_seq_len: int,
) -> Tensor:
    """Run RoPE for a given input tensor.

    Args:
        x (Tensor): Input tensor to run RoPE on.
        token_pos (Tensor): Tensor of shape (B, T) with token positions.
        embed_dim (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE theta parameter.
        max_seq_len (int): Maximum sequence length to pre-cache.

    Returns:
        output (Tensor): Tensor with RoPE applied to the input tensor.
    """
    rope = assignment1.RotaryPositionalEmbedding(
        embed_dim=embed_dim,
        theta=theta,
        max_seq_len=max_seq_len,
    )

    output = rope(x, token_pos)
    return output


def run_transformer_block(
    x: Tensor,
    d_model: int,
    num_heads: int,
    d_ff: int,
    theta: float,
    max_seq_len: int,
    weights: dict[str, Tensor],
) -> Tensor:
    """Given the weights of a pre-norm Transformer block and input features, return the
    output of running the Transformer block on the input features.

    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        x (Tensor): Tensor to run your implementation on. The shape of `x` is (batch_size,
            sequence_length), where `sequence_length` is at most `context_length`.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        theta (float): The RoPE's theta parameter.
        max_seq_len (int): The maximum sequence length to pre-cache.
        weights (dict[str, Tensor]): Provided model weights of our reference implementation.

    Returns:
        output (Tensor) Tensor with the output of running the Transformer block on the input
            features while using RoPE.
    """
    rope = assignment1.RotaryPositionalEmbedding(
        embed_dim=d_model // num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
    )
    block = assignment1.TransformerBlock(
        embed_dim=d_model,
        num_heads=num_heads,
        hidden_dim=d_ff,
        bias=False,
        rope=rope,
    )
    block.load_state_dict(weights)

    output = block(x)
    return output


def run_transformer_lm(
    x: Tensor,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
) -> Tensor:
    """Given the weights of a Transformer language model and input indices, return the
    output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        x (Tensor): Tensor to run your implementation on. The shape of `x` is (batch_size,
            sequence_length), where `sequence_length` is at most `context_length`.
        vocab_size (int): The number of unique items in the output vocabulary.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE's theta parameter.
        weights (dict[str, Tensor]): Provided model weights of our reference implementation.

    Returns:
        output (Tensor): Tensor with the predicted unnormalized next-word distribution
            for each token.
    """
    model = assignment1.TransformerLM(
        vocab_size=vocab_size,
        block_size=context_length,
        embed_dim=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=d_ff,
        bias=False,
        rope_theta=rope_theta,
    )
    model.load_state_dict(weights)

    output = model(x)
    return output


def run_rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Given the weights of a RMSNorm affine transform, return the output of running
    RMSNorm on the input features.

    Args:
        x (Tensor): Input features to run RMSNorm on. The shape of `x` can be arbitrary.
        weight (Tensor): RMSNorm weights. The shape of `weight` is (d_model,).
        eps: (float): A value added to the denominator for numerical stability.

    Returns:
        output (Tensor): The output tensor after applying RMSNorm to the input features.
            The shape of `output` is the same as the shape of `x`.
    """
    normalized_shape = x.size(-1)
    output = dF.rms_norm(x, normalized_shape, weight, eps=eps)
    return output


def run_silu(x: Tensor) -> Tensor:
    """Given a tensor of inputs, return the output of applying SiLU to each element.

    Args:
        x (Tensor): Input features to run SiLU on. The shape of `x` can be arbitrary.

    Returns:
        output (Tensor): The output tensor after applying SiLU to each element of `x`.
            The shape of `output` is the same as the shape of `x`.
    """
    output = dF.silu(x)
    return output


def run_get_batch(
    dataset: Tensor,
    context_length: int,
    batch_size: int,
    device: Device,
) -> tuple[Tensor, Tensor]:
    """Given a dataset (a 1D Tensor of integers) and a desired batch size and context
    length, sample language modeling input sequences and their corresponding labels
    from the dataset.

    Args:
        dataset (Tensor): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (Device): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the
            device to place the sampled input sequences and labels on.

    Returns:
        output (tuple[Tensor, Tensor]): Tuple of Tensors of shape (batch_size,
            context_length). The first item is the sampled input sequences, and the
            second item is the corresponding language modeling labels.
    """
    output = utils.get_batch(dataset, context_length, batch_size, device=device)
    return output


def run_softmax(x: Tensor, dim: int) -> Tensor:
    """Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        x (Tensor): Input features to softmax. The shape of `x` can be arbitrary.
        dim (int): Dimension of the `x` to apply softmax to.

    Returns:
        output (Tensor): The output tensor after applying softmax to the given `dim`.
            The shape of `output` is the same as the shape of `x`.
    """
    output = dF.softmax(x, dim=dim)
    return output


def run_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Given a tensor of inputs and targets, compute the average cross-entropy loss
    across examples.

    Args:
        inputs (Tensor): inputs[i][j] is the unnormalized logit of jth class for
            the i-th example.
        targets (Tensor): Tensor of shape (batch_size,) with the index of the correct
            class. Each value must be between 0 and `num_classes-1`.

    Returns:
        output (Tensor): The average cross-entropy loss across samples.
    """
    output = dF.cross_entropy_loss(inputs, targets)
    return output


def run_gradient_clipping(params: Iterable[Tensor], max_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at
    most `max_norm`. The gradients of the parameters (params.grad) should be modified
    in-place.

    Args:
        params (Iterable[Tensor]): collection of trainable parameters.
        max_norm (float): a positive value containing the maximum l2-norm.
    """
    dnn.utils.clip_grad_norm_(params, max_norm)


def get_adamw_cls() -> type[optim.Optimizer]:
    """Returns a torch.optim.Optimizer that implements AdamW."""
    return dopt.AdamW


def run_get_lr_cosine_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Given the parameters of a cosine learning rate decay schedule (with linear warmup)
    and an iteration number, return the learning rate at the given iteration under the
    specified schedule.

    Args:
        iteration (int): Iteration number to get learning rate for.
        max_lr (float): alpha_max, the maximum learning rate for the cosine learning
            rate schedule (with warmup).
        min_lr (float): alpha_min, the minimum / final learning rate for the cosine
            learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up the
            learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if iteration < 0:
        raise AssertionError('`iteration` must be non-negative.')
    if warmup_iters < 0:
        raise AssertionError('`warmup_iters` must be non-negative.')
    if cosine_cycle_iters <= warmup_iters:
        raise AssertionError('`cosine_cycle_iters` must exceed `warmup_iters`.')

    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters
    if iteration > cosine_cycle_iters:
        return min_learning_rate

    progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)


def run_save_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
) -> None:
    """Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        path (str | os.PathLike[str]): Path or file-like object to serialize the model,
            optimizer, and iteration to.
        model (nn.Module): Serialize the state of this model.
        optimizer (optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training
            iterations we've completed.
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, path)


def run_load_checkpoint(
    src: str | os.PathLike[str],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    """Given a serialized checkpoint (path or file-like object), restore the serialized
    state to the given model and optimizer. Return the number of iterations that we
    previously serialized in the checkpoint.

    Args:
        src (str | os.PathLike[str]: Path or file-like object to serialized checkpoint.
        model (nn.Module): Restore the state of this model.
        optimizer (optim.Optimizer): Restore the state of this optimizer.

    Returns:
        iterations (int): the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return int(checkpoint['iteration'])


def get_tokenizer(
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
    special_tokens: list[str] | None = None,
) -> dltk.Tokenizer:
    """Given a vocabulary, a list of merges, and a list of special tokens, return
    a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[str, int]): The tokenizer vocabulary, a mapping from str (token
            in the vocabulary) to int (token id in the vocabulary).
        merges (list[tuple[str, str]): BPE merges. Each list item is a tuple of str
            (<token1>, <token2>), representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str], optinal): A list of string special tokens for the
            tokenizer. These strings will never be split into multiple tokens, and will
            always be kept as a single token.

    Returns:
        tokenizer (dltk.Tokenizer) A BPE tokenizer that uses the provided vocab, merges,
            and special tokens.
    """
    tokenizer = dltk.Tokenizer(
        dltk.BPE(vocab, merges, unk_token='unk'),
        pre_tokenizer=dltk.ByteLevelPreTokenizer(add_prefix_space=False),
        decoder=dltk.ByteLevelDecoder(),
        num_workers=1,
    )
    tokenizer.add_special_tokens(special_tokens or [])
    return tokenizer


def run_train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> dltk.Tokenizer:
    """Given the path to an input corpus, run train a BPE tokenizer and output
    its vocabulary and merges.

    Args:
        input_path (str | os.PathLike[str]): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary
            (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to
            the tokenizer vocabulary. These strings will never be split into multiple
            tokens, and will always be kept as a single token. If these special tokens
            occur in the `input_path`, they are treated as any other string.
        num_workers (int, optional): Number of workers to use for training the tokenizer.
            This is passed to the `train_from_iterator` method of the tokenizer.

    Returns:
        tokenizer (dltk.Tokenizer): A BPE tokenizer that uses the provided vocab, merges,
            and special tokens. The tokenizer has the following attributes:
            - vocab: The trained tokenizer vocabulary, a mapping from str (token in the
                vocabulary) to int (token ids).
            - merges: BPE merges. Each list item is a tuple of str (<token1>, <token2>),
                representing that <token1> was merged with <token2>. Merges are ordered
                by order of creation.
    """
    tokenizer = dltk.Tokenizer(
        dltk.BPE(),
        pre_tokenizer=dltk.ByteLevelPreTokenizer(add_prefix_space=False),
        decoder=dltk.ByteLevelDecoder(),
        num_workers=num_workers,
    )
    tokenizer.add_special_tokens(special_tokens)

    alphabet = dltk.ByteLevelPreTokenizer.alphabet()
    with open(input_path) as fp:
        tokenizer.train_from_iterator(fp, vocab_size, initial_alphabet=alphabet)

    return tokenizer
