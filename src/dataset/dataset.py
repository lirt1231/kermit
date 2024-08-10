import json
from typing import Dict, List

import torch

from .data_utils import (
    Triple,
    get_tokenizer,
    reverse_triple,
    to_indices_and_mask,
    construct_positive_mask,
)


EVALUATING_MODE = False


def set_eval_mode(mode: bool = True) -> None:
    global EVALUATING_MODE
    EVALUATING_MODE = mode


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        triple_path: str,
        max_num_tokens: int = 128,
        use_neighbor_names: bool = False,
        add_forward_triples: bool = True,
        add_backward_triples: bool = False,
    ) -> None:
        self.max_num_tokens = max_num_tokens
        self.use_neighbor_names = use_neighbor_names
        self.triples = self._parse_triple(triple_path, add_forward_triples, add_backward_triples)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.triples[idx].format(
            self.max_num_tokens,
            self.use_neighbor_names,
            EVALUATING_MODE,
        )

    def _parse_triple(
        self, path: str, add_forward_triples: bool, add_backward_triples: bool
    ) -> List[Triple]:
        triples = []
        with open(path, "r") as file:
            for triple in json.load(file):
                if add_forward_triples:
                    triples.append(
                        Triple(
                            triple["head_id"],
                            triple["relation"],
                            triple["tail_id"],
                            triple["hr_desp"],
                        )
                    )
                if add_backward_triples:
                    triples.append(Triple(**reverse_triple(triple)))
        return triples


def collate(batch_data: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(data["hr_token_ids"]) for data in batch_data],
        pad_token_id=get_tokenizer().pad_token_id,
    )
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(data["hr_token_type_ids"]) for data in batch_data], need_mask=False
    )
    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(data["tail_token_ids"]) for data in batch_data],
        pad_token_id=get_tokenizer().pad_token_id,
    )
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(data["tail_token_type_ids"]) for data in batch_data], need_mask=False
    )

    triples = [data["triple"] for data in batch_data]
    labels = torch.arange(len(batch_data))
    batch_dict = {
        "hr_token_ids": hr_token_ids,
        "hr_mask": hr_mask,
        "hr_token_type_ids": hr_token_type_ids,
        "tail_token_ids": tail_token_ids,
        "tail_mask": tail_mask,
        "tail_token_type_ids": tail_token_type_ids,
        "triples": triples,
        "labels": labels,
        "positive_mask": construct_positive_mask(row_triples=triples),
    }

    return batch_dict
