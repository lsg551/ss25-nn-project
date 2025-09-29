"""Longformer-based listwise cross-encoder for passage ranking.

This model scores a list of candidate passages for a single query in one forward
pass by packing the query followed by candidates into a single Longformer input.
Candidate boundaries are marked with a special token `[CAND]`, and then the hidden
state is pooled at those marker positions to produce one score per candidate.

Longformer is generally preferred over BERT, because it can handle longer inputs,
up to 4096 tokens by default, which allows more candidates to be packed in. To
process variable-length candidate lists, we insert a dedicated candidate marker
token "[CAND]" in front of each candidate and pool at those positions.

A single relevance score is produced for each query-candidate pair. These scores
are then used to compute a probability distribution via softmax.

The same is also done for the ground-truth relevance labels during training.
The cross-entropy between these two distributions is used as the loss (ListNet loss).

Note that there are other ways (and loss functions) to implement listwise
ranking, e.g. ListMLE or Approximate nDCG (LambdaLoss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, BatchEncoding, LongformerTokenizer

tokenizer: LongformerTokenizer = LongformerTokenizer.from_pretrained(
    "allenai/longformer-base-4096"
)


# Add a dedicated marker for candidates
CAND_TOKEN = "[CAND]"
tokenizer.add_special_tokens({"additional_special_tokens": [CAND_TOKEN]})


class ListwiseRankingHead(nn.Module):
    """Listwise head that outputs one logit per candidate.

    Pools the hidden state at [CAND] positions that precede each candidate.
    """

    def __init__(
        self,
        hidden_size: int,
        cand_token_id: int,
    ):
        super().__init__()
        self.cand_token_id = cand_token_id
        # tiny MLP head for classification
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    def _candidate_marker_positions(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find [CAND] marker positions per batch.

        Returns
            tuple[torch.Tensor, torch.Tensor]:
                - positions: (B, Cmax) indices with -1 as padding for absent markers
                - mask:      (B, Cmax) bool mask for valid markers
        """
        device = input_ids.device
        B, L = input_ids.shape  # (batch size, sequence length)
        cand_mask_all = input_ids.eq(self.cand_token_id)  # (B, L)

        # collect positions per batch; pad to Cmax with -1
        idxs = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)
        counts = cand_mask_all.sum(dim=1)  # (B,)
        Cmax = int(counts.max().item()) if B > 0 else 0

        # when no candidates are found in the batch
        if Cmax == 0:
            positions = input_ids.new_full((B, 0), -1)
            mask = input_ids.new_zeros((B, 0), dtype=torch.bool)
            return positions, mask

        # gather positions per batch item, pad to Cmax with -1
        positions_list = []
        masks_list = []
        for b in range(B):
            pos_b = idxs[b][cand_mask_all[b]]  # (Cb,)
            Cb = pos_b.numel()
            pad = Cmax - Cb
            if pad > 0:
                pos_b = torch.cat([pos_b, pos_b.new_full((pad,), -1)])
            positions_list.append(pos_b)
            mask_b = torch.zeros(Cmax, dtype=torch.bool, device=device)
            if Cb > 0:
                mask_b[:Cb] = True
            masks_list.append(mask_b)

        positions = torch.stack(positions_list, dim=0)  # (B, Cmax)
        mask = torch.stack(masks_list, dim=0)  # (B, Cmax)

        return positions, mask

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-candidate logits from hidden states and input ids.

        Args:
            hidden_states: (B, L, H)
            input_ids:     (B, L)

        Returns:
            scores: (B, Cmax) logits per candidate (padded positions arbitrary)
            mask:   (B, Cmax) bool mask of valid candidate positions
        """
        B, L, H = hidden_states.shape
        pos, mask = self._candidate_marker_positions(input_ids)  # (B, Cmax)
        if pos.numel() == 0:
            # No candidates detected; return empty tensors
            return hidden_states.new_zeros((B, 0)), mask

        # Gather hidden at candidate [SEP] positions
        safe_pos = pos.clamp_min(0).unsqueeze(-1).expand(-1, -1, H)  # (B, Cmax, H)
        gathered = hidden_states.gather(dim=1, index=safe_pos)  # (B, Cmax, H)
        logits = self.mlp(gathered).squeeze(-1)  # (B, Cmax)
        return logits, mask


class CrossEncoderLongformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.longformer = AutoModel.from_pretrained("allenai/longformer-base-4096")
        # ensure embeddings cover the added [CAND] token
        self.longformer.resize_token_embeddings(len(tokenizer))
        # resolve [CAND] token id
        cand_id_list = tokenizer.convert_tokens_to_ids([CAND_TOKEN])
        if not isinstance(cand_id_list, list) or len(cand_id_list) != 1:
            raise ValueError("Failed to resolve [CAND] token id.")
        cand_token_id = int(cand_id_list[0])

        self.classifier = ListwiseRankingHead(
            hidden_size=self.longformer.config.hidden_size,
            cand_token_id=cand_token_id,
        )

    def forward(self, x: BatchEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Longformer cross-encoder.

        Example:
        >>> logits, cand_mask = model(model.tokenize(query, candidates))

        This returns the raw logits for each candidate in the input, as well as
        a mask that indicates which positions correspond to real candidates
        (as opposed to padding).

        >>> scores

        Args:
            x (BatchEncoding): _description_

        Returns:
            _type_: _description_
        """
        # B = batch size
        # L = sequence length (num tokens in input after truncation/padding)
        # H = dimensionality of the hidden representations for each token
        #     (e.g., 768 for Longformer)
        # Example: if a batch with 4 samples and max length 512 is fed into the model
        # with a hidden size of 768, this will be (4, 512, 768)
        outputs = self.longformer(**x)
        hidden = outputs.last_hidden_state  # (B, L, H)
        # x["input_ids"] is (B, L)
        logits, cand_mask = self.classifier(hidden, x["input_ids"])

        # cand_mask is the "candidates mask", i.e.

        # For loss/metrics, mask padded positions by setting -inf if desired outside
        return logits, cand_mask

    @staticmethod
    def tokenize(
        query: str,
        candidates: str | list[str],
        *,
        max_length: int = 4096,
    ) -> BatchEncoding:
        """Tokenize a query and one or more candidates for the cross-encoder.

        Uses the `LongformerTokenizer`. Query and candidates are concatenated
        into a single input sequence, separated by a separator token.

        Args:
            query (str): _description_
            candidates (str | list[str]): _description_
            max_length (int, optional): _description_. Defaults to 4096.

        Returns:
            BatchEncoding: _description_
        """
        # TODO: seems to work, but check if it can be done more elegantly / simplified
        # target embedding: <s> query </s> [CAND] cand1 </s> [CAND] cand2 </s> ...
        cands = candidates if isinstance(candidates, list) else [candidates]
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            raise ValueError("Tokenizer missing cls/sep token ids.")
        cand_id = tokenizer.convert_tokens_to_ids(CAND_TOKEN)
        if isinstance(cand_id, list):
            cand_id = cand_id[0]
        if not isinstance(cand_id, int):
            raise ValueError("Invalid [CAND] token id.")

        q_ids = tokenizer.encode(query, add_special_tokens=False)
        cand_ids_list = [tokenizer.encode(c, add_special_tokens=False) for c in cands]

        input_ids = [cls_id] + q_ids + [sep_id]
        for ids in cand_ids_list:
            input_ids += [cand_id] + ids + [sep_id]

        # truncate to max_length
        input_ids = input_ids[:max_length]
        attention_mask = [1] * len(input_ids)

        # global attention on CLS, query tokens and [CAND] markers
        global_attention_mask = [0] * len(input_ids)
        global_attention_mask[0] = 1  # CLS
        q_end = min(1 + len(q_ids), len(global_attention_mask))
        for i in range(1, q_end):
            global_attention_mask[i] = 1
        for i, tid in enumerate(input_ids):
            if tid == cand_id:
                global_attention_mask[i] = 1

        # wrap into BatchEncoding
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
        encoding = BatchEncoding(
            data={
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "global_attention_mask": torch.tensor(
                    [global_attention_mask], dtype=torch.long
                ),
            }
        )

        return encoding

    def rank(self, query: str, candidates: list[str]) -> list[float]: ...
