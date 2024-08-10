from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from args import ModelArguments


@dataclass
class ModelOutput:
    hr_vector: Optional[torch.FloatTensor] = None
    tail_vector: Optional[torch.FloatTensor] = None


class Kermit(nn.Module):
    def __init__(self, args: ModelArguments) -> None:
        super().__init__()
        self.pooling = args.pooling
        self.log_inv_t = torch.nn.Parameter(
            torch.tensor(1.0 / args.tau).log(), requires_grad=args.finetune_tau
        )
        self.add_margin = args.additive_margin
        # BERT encoders
        self.bert_hr = AutoModel.from_pretrained(args.pretrained_model)
        self.bert_entity = deepcopy(self.bert_hr)

    def forward(
        self,
        hr_token_ids: torch.LongTensor,
        hr_mask: torch.ByteTensor,
        hr_token_type_ids: torch.LongTensor,
        tail_token_ids: torch.LongTensor,
        tail_mask: torch.ByteTensor,
        tail_token_type_ids: torch.LongTensor,
        **kwargs,
    ) -> ModelOutput:
        hr_vector = self.encode_bert(self.bert_hr, hr_token_ids, hr_mask, hr_token_type_ids)
        hr_vector = F.normalize(hr_vector, dim=1)
        tail_vector = self.encode_bert(
            self.bert_entity, tail_token_ids, tail_mask, tail_token_type_ids
        )
        tail_vector = F.normalize(tail_vector, dim=1)

        return {"hr_vector": hr_vector, "tail_vector": tail_vector}

    def encode_bert(self, encoder, token_ids, mask, token_type_ids) -> torch.FloatTensor:
        outputs = encoder(input_ids=token_ids, attention_mask=mask, token_type_ids=token_type_ids)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        output = _pool_output(self.pooling, cls_output, mask, last_hidden_state)
        return output

    def compute_loss(
        self,
        hr_vector: torch.FloatTensor,
        tail_vector: torch.FloatTensor,
        positive_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        # compute logits
        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        return {
            "logits": logits,
            "loss": self.compute_contrastive_loss(logits, positive_mask),
            "inv_t": self.log_inv_t.detach().exp(),
        }

    def compute_contrastive_loss(
        self,
        logits: torch.FloatTensor,
        positive_mask: torch.LongTensor,
    ) -> torch.FloatTensor:
        neg_log_prob = -F.log_softmax(logits, dim=1)
        neg_log_prob *= positive_mask.float()
        num_pos = positive_mask.sum(dim=1)
        loss = neg_log_prob.sum(dim=1) / num_pos.float()

        return loss.mean()

    def encode_hr(self, token_ids, mask, token_type_ids) -> torch.FloatTensor:
        entity_vector = self.encode_bert(self.bert_hr, token_ids, mask, token_type_ids)
        return F.normalize(entity_vector, dim=1)

    def encode_entities(self, token_ids, mask, token_type_ids) -> torch.FloatTensor:
        entity_vector = self.encode_bert(self.bert_entity, token_ids, mask, token_type_ids)
        return F.normalize(entity_vector, dim=1)


def _pool_output(
    pooling: str, cls_output: torch.Tensor, mask: torch.Tensor, last_hidden_state: torch.Tensor
) -> torch.Tensor:
    if pooling == "cls":
        output_vector = cls_output
    elif pooling == "mean":
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, "Unknown pooling mode: {}".format(pooling)

    return output_vector
