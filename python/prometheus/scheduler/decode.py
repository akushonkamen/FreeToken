from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Set

import torch
from prometheus.core import Batch, Req

from .spec import NGramDraftModel, make_spec_batch, make_spec_batch_mtp

if TYPE_CHECKING:
    from .spec import MTPDraftManager


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)
    # Speculative decoding state. Exactly one draft provider is live unless the
    # scheduler enabled spec (spec_ngram / spec_mtp > 0 and the model guards pass);
    # with both None the schedule path below is byte-identical to the non-spec engine.
    spec_k: int = 0
    token_pool: torch.Tensor | None = None
    draft_model: NGramDraftModel | None = None
    mtp: "MTPDraftManager | None" = None

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)
        if self.draft_model is not None:
            self.draft_model.remove(req.uid)
        if self.mtp is not None:
            self.mtp.remove(req.uid)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                if self.draft_model is not None:
                    self.draft_model.remove(uid)
                if self.mtp is not None:
                    self.mtp.remove(uid)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        reqs = sorted(self.running_reqs, key=lambda req: req.uid)
        if self.token_pool is not None:
            if self.mtp is not None:
                batch = make_spec_batch_mtp(reqs, self.spec_k, self.token_pool, self.mtp)
                if batch is not None:
                    return batch
            elif self.draft_model is not None:
                batch = make_spec_batch(reqs, self.spec_k, self.token_pool, self.draft_model)
                if batch is not None:
                    return batch
        return Batch(reqs=reqs, phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
