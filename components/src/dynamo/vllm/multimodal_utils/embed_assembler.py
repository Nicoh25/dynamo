# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed token-ids/embeds assembly for the aggregated CustomEncoder path.

The encoder returns only the visual-token embeddings; this module builds the
inputs for vLLM's mixed ``EmbedsPrompt`` mode (``prompt_token_ids`` +
``prompt_is_token_ids`` + ``prompt_embeds``). vLLM embeds the ``True`` (text)
positions itself and substitutes the ``False`` (image) positions from
``prompt_embeds``, so Dynamo fills only the image rows (text rows stay zero):

    prompt_token_ids    = [ text ... <img run> ... text ... ]
    prompt_is_token_ids = [ True ...  False    ... True ... ]
    prompt_embeds       = [ zeros... img_embeds... zeros... ]   (seq_len, hidden)

The contract is **one placeholder token per image**: each placeholder in
``prompt_token_ids`` is one image (matched positionally to the encoder tensors)
and is expanded to that tensor's row count — i.e. the number of visual tokens
the encoder produced for the image. E.g. a single placeholder ``P`` with a
``(3, hidden)`` image tensor expands ``[t, P, t]`` → ``[t, P, P, P, t]``, so the
encoder dictates the span length (mirroring vLLM's own placeholder expansion).
The chat template emits exactly one placeholder per image, with no separator
between consecutive images.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def build_mixed_embeds(
    token_ids: list[int],
    img_tensors: list[torch.Tensor],
    placeholder_id: int,
) -> tuple[torch.Tensor, list[int], list[bool]]:
    """Build the mixed token-ids/embeds inputs for an aggregated request.

    Args:
        token_ids: The full prompt token IDs (text + one placeholder token per
            image).
        img_tensors: Per-image visual token tensors, each ``(n_tokens, hidden)``,
            in prompt order.
        placeholder_id: The token ID marking image positions.

    Returns:
        ``(prompt_embeds, prompt_token_ids, prompt_is_token_ids)`` where
        ``prompt_embeds`` is a CPU ``(seq_len, hidden)`` tensor (text rows zero,
        image rows from ``img_tensors``), ``prompt_token_ids`` has each
        placeholder token expanded to its tensor's row count, and
        ``prompt_is_token_ids[i]`` is ``False`` at image positions.

    Raises:
        ValueError: if ``img_tensors`` is empty, the number of placeholder
            tokens does not equal the number of image tensors, or the tensors
            are not 2D with a consistent hidden dim.
    """
    if not img_tensors:
        raise ValueError("img_tensors must not be empty")

    positions = [i for i, tid in enumerate(token_ids) if tid == placeholder_id]
    if len(positions) != len(img_tensors):
        raise ValueError(
            f"placeholder tokens ({len(positions)}) != image tensors "
            f"({len(img_tensors)}) for placeholder token {placeholder_id} "
            f"in sequence of length {len(token_ids)}"
        )

    # Check 2D before reading shape[1], so a 1D encoder output raises a clear
    # ValueError here instead of an opaque IndexError.
    if img_tensors[0].dim() != 2:
        raise ValueError(
            f"image tensor 0 has shape {tuple(img_tensors[0].shape)}; expected "
            "2D (n_tokens, hidden)"
        )
    hidden = img_tensors[0].shape[1]
    dtype = img_tensors[0].dtype
    # Validate all shapes before scattering, so a width mismatch raises a clear
    # ValueError here instead of an opaque RuntimeError in the row-copy below.
    for i, tensor in enumerate(img_tensors):
        if tensor.dim() != 2 or tensor.shape[1] != hidden:
            raise ValueError(
                f"image tensor {i} has shape {tuple(tensor.shape)}; expected "
                f"2D with hidden dim {hidden} (from image tensor 0)"
            )
        # A (0, hidden) tensor would erase the image's placeholder run entirely,
        # silently dropping the image — an encoder bug, so fail loudly.
        if tensor.shape[0] == 0:
            raise ValueError(
                f"image tensor {i} has 0 rows (shape {tuple(tensor.shape)}); the "
                "encoder returned no visual tokens for an image"
            )

    # Build the token-id / mask layout, recording where each image block lands,
    # then scatter the image rows into one pre-zeroed (seq_len, hidden) buffer.
    # Text rows stay zero (vLLM overwrites them via the embedding table).
    out_token_ids: list[int] = []
    is_token_ids: list[bool] = []
    image_slots: list[tuple[int, torch.Tensor]] = []  # (row_start, tensor)

    def _emit_text(text_ids: list[int]) -> None:
        if not text_ids:
            return
        out_token_ids.extend(text_ids)
        is_token_ids.extend([True] * len(text_ids))

    cursor = 0
    for pos, tensor in zip(positions, img_tensors):
        _emit_text(token_ids[cursor:pos])
        n = tensor.shape[0]
        image_slots.append((len(out_token_ids), tensor))
        out_token_ids.extend([placeholder_id] * n)
        is_token_ids.extend([False] * n)
        cursor = pos + 1
    _emit_text(token_ids[cursor:])

    seq_len = len(out_token_ids)
    # CPU tensor: vLLM's renderer forces prompt_embeds to CPU anyway.
    prompt_embeds = torch.zeros(seq_len, hidden, dtype=dtype)
    for row_start, tensor in image_slots:
        n = tensor.shape[0]
        prompt_embeds[row_start : row_start + n] = tensor.to(dtype=dtype, device="cpu")

    logger.debug(
        "[custom_embeds] images=%d seq_len=%d hidden=%d dtype=%s",
        len(img_tensors),
        seq_len,
        hidden,
        dtype,
    )
    return prompt_embeds, out_token_ids, is_token_ids
