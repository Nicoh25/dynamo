# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serial, off-loop default base for the pluggable CustomEncoder path.

Implement a synchronous ``_encode_blocking`` and this base provides ``encode``:
it runs the forward via ``asyncio.to_thread`` (keeping the event loop
responsive) under an internal lock (so concurrent requests are serialized — one
forward at a time, and ``_encode_blocking`` need not be re-entrant). Subclass
this unless you intend to run your own batching scheduler.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import List, Optional

import torch

from dynamo.vllm.multimodal_utils.custom_encoder import CustomEncoder


class SerializedCustomEncoder(CustomEncoder):
    """CustomEncoder base that runs a synchronous forward serially, off-loop.

    Subclasses implement ``load``, ``get_image_placeholder_token_id``, and the
    synchronous ``_encode_blocking`` (see the module docstring for the
    ``encode`` mechanism).
    """

    def _serialize_lock(self) -> asyncio.Lock:
        # Lazily created (no super().__init__() needed); the check-and-set has no
        # await between, so it is atomic on the single-threaded event loop.
        lock: Optional[asyncio.Lock] = getattr(self, "_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._lock = lock
        return lock

    async def encode(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Run ``_encode_blocking`` off the event loop, serialized."""
        async with self._serialize_lock():
            return await asyncio.to_thread(self._encode_blocking, image_urls)

    @abstractmethod
    def _encode_blocking(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Synchronous forward: encode images to per-image visual embeddings.

        Runs in a worker thread, serialized by ``encode``, so it need not be
        re-entrant. Return one ``(n_visual_tokens, lm_hidden_dim)`` tensor per
        URL, in prompt order.
        """
        ...
