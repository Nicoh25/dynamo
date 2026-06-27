# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract for a pluggable, in-process vision encoder.

The encoder runs in the same process as the vLLM aggregated worker (no separate
encode worker, no NIXL transfer): it encodes images and projects them to the LM
hidden dim, and Dynamo splices those embeds into a mixed ``EmbedsPrompt`` at the
placeholder positions (vLLM embeds the text tokens itself).

``CustomEncoder`` defines the contract only — ``load`` / async ``encode`` /
``get_image_placeholder_token_id``. For the common case subclass
``SerializedCustomEncoder``; Qwen-family models subclass
``QwenSerializedCustomEncoder``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch


def placeholder_token_id_from_tokenizer(
    tokenizer: object,
    token: str,
) -> Optional[int]:
    """Resolve a placeholder token string to its ID via a loaded tokenizer.

    Returns ``None`` if the tokenizer does not define ``token`` (it maps to the
    unknown-token id), so callers can raise a clear error.

    Args:
        tokenizer: A loaded tokenizer exposing ``convert_tokens_to_ids``.
        token: The placeholder token string, e.g. ``QWEN_IMAGE_PLACEHOLDER_TOKEN``.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    tid = convert(token)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        return None
    return int(tid)


class CustomEncoder(ABC):
    """The pluggable image-encoder contract for aggregated serving.

    Subclasses implement ``load``, async ``encode``, and
    ``get_image_placeholder_token_id``; ``encode`` returns one
    ``(n_visual_tokens, lm_hidden_dim)`` tensor per image (the encoder decides
    ``n_visual_tokens``). For the common case subclass ``SerializedCustomEncoder``
    and write only a synchronous ``_encode_blocking``; implement ``encode``
    directly only to run your own batching scheduler.
    """

    @abstractmethod
    def load(self, model_id: str, device: str) -> None:
        """Load vision encoder weights.

        Args:
            model_id: The LM checkpoint — a local dir or HF model id, passed
                verbatim from ``--model``.
            device: Target device string, e.g. ``"cuda"``.
        """
        ...

    @abstractmethod
    async def encode(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Encode images, returning one visual-token tensor per URL.

        The worker ``await``s this directly — it holds no lock and does no
        offload, so the encoder owns its execution model: ``encode`` may be
        called concurrently and must not block the event loop (offload the
        blocking forward via ``asyncio.to_thread`` or its own scheduler). Most
        authors get this for free by subclassing ``SerializedCustomEncoder``.

        Args:
            image_urls: URLs of images to encode.

        Returns:
            One ``(n_visual_tokens, lm_hidden_dim)`` tensor per URL, in order.
        """
        ...

    @abstractmethod
    def get_image_placeholder_token_id(self) -> int:
        """Return the token ID marking image positions in the prompt.

        Dynamo uses it to locate the image span and splice in the encoder
        tensors. How it is resolved is the encoder's choice (Qwen-family
        encoders inherit this from ``QwenSerializedCustomEncoder``).

        Raises:
            ValueError: if no valid id can be resolved (surfaced at startup by
                ``validate()``).
        """
        ...

    def validate(self) -> None:
        """Fail-fast checks run by the worker right after ``load()``.

        Resolves the placeholder id once so a misconfigured encoder errors at
        startup, not on the first request. Subclasses may override to add
        post-load checks (call ``super().validate()`` first).

        Raises:
            ValueError: propagated from ``get_image_placeholder_token_id``.
        """
        self.get_image_placeholder_token_id()
