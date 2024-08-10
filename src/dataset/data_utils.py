import json
import os.path as osp
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set

import torch
from transformers import AutoTokenizer


__all__ = [
    "Triple",
    "EntityDict",
    "init",
    "get_entity_dict",
    "get_tokenizer",
    "get_triple_dict",
    "get_neighbor_dict",
    "concat_name_desc",
    "get_neighbor_desc",
    "tokenize_entity",
    "reverse_triple",
    "to_indices_and_mask",
    "construct_positive_mask",
]


entity_dict: "EntityDict" = None
inv_rel_dict: Dict[str, str] = {}
tokenizer: AutoTokenizer = None
triple_dict: "TripleDict" = None
neighbor_dict: "NeighborDict" = None


@dataclass
class Triple:
    head_id: str
    relation: str
    tail_id: str
    pred_desc: str

    def format(
        self,
        max_num_tokens: int,
        use_neighbor_names: bool = False,
        eval_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        head_name, tail_name = (
            entity_dict.get_entity_name(self.head_id),
            entity_dict.get_entity_name(self.tail_id),
        )
        head_desc, tail_desc = (
            entity_dict.get_entity_desc(self.head_id),
            entity_dict.get_entity_desc(self.tail_id),
        )
        if use_neighbor_names:
            if len(head_desc.split()) < 20:
                head_desc += " " + get_neighbor_desc(self.head_id, self.tail_id, eval_mode)
            if len(tail_desc.split()) < 20:
                tail_desc += " " + get_neighbor_desc(self.tail_id, self.head_id, eval_mode)
        head_text = concat_name_desc(head_name, head_desc)
        tail_text = concat_name_desc(tail_name, tail_desc)

        hr_encoded_inputs = tokenize_hr(head_text, self.relation, self.pred_desc, max_num_tokens)
        tail_encoded_inputs = tokenize_entity(tail_text, max_num_tokens)

        return {
            "hr_token_ids": hr_encoded_inputs["input_ids"],
            "hr_token_type_ids": hr_encoded_inputs["token_type_ids"],
            "tail_token_ids": tail_encoded_inputs["input_ids"],
            "tail_token_type_ids": tail_encoded_inputs["token_type_ids"],
            "triple": self,
        }


class EntityDict:
    def __init__(self, data_dir: str):
        with open(osp.join(data_dir, "entities.json"), "r") as f:
            self.entity_dict = json.load(f)

        self.ent2idx = {ent: i for i, ent in enumerate(self.entity_dict.keys())}
        self.idx2ent = {i: ent for ent, i in self.ent2idx.items()}

    def entity_to_idx(self, entity_id: str) -> int:
        return self.ent2idx[entity_id]

    def idx_to_entity(self, idx: int) -> str:
        return self.idx2ent[idx]

    def idx_to_name(self, idx: int) -> str:
        return self.get_entity_name(self.idx2ent[idx])

    def idx_to_desc(self, idx: int) -> str:
        return self.get_entity_desc(self.idx2ent[idx])

    def get_entity_name(self, entity_id: str) -> str:
        return self.entity_dict[entity_id]["name"]

    def get_entity_desc(self, entity_id: str) -> str:
        return self.entity_dict[entity_id]["desc"]

    def iter_entities(self) -> Generator[str, str, str]:
        # ensure the order to be the same as their indices
        for idx in range(len(self)):
            yield self.idx_to_entity(idx=idx), self.idx_to_name(idx), self.idx_to_desc(idx)

    def __len__(self) -> int:
        return len(self.entity_dict)


class TripleDict:
    def __init__(self, path_list: List[str]) -> None:
        self.hr2tails = defaultdict(set)
        for path in path_list:
            self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r") as f:
            triples = json.load(f)
        triples += [reverse_triple(trip) for trip in triples]
        for trip in triples:
            h, r, t = trip["head_id"], trip["relation"], trip["tail_id"]
            self.hr2tails[(h, r)].add(t)

    def get_relational_neighbors(self, h: str, r: str) -> Set[str]:
        return self.hr2tails.get((h, r), set())


class NeighborDict:
    def __init__(self, train_path: List[str]) -> None:
        self.ent2neighbors = defaultdict(set)
        self._load(train_path)

    def _load(self, path: str) -> None:
        with open(path, "r") as f:
            triples = json.load(f)
        for trip in triples:
            h, t = trip["head_id"], trip["tail_id"]
            self.ent2neighbors[h].add(t)
            self.ent2neighbors[t].add(h)

    def get_neighbors(self, entity_id: str, num_neighbors: int = 10) -> List[str]:
        # make sure different calls return the same results
        neighbor_ids = self.ent2neighbors.get(entity_id, set())
        return sorted(list(neighbor_ids))[:num_neighbors]

    def get_n_hop_entity_indices(
        self, entity_id: str, num_hops: int = 2, max_nodes: int = 100000
    ) -> set:
        if num_hops < 0:
            return set()

        seen_ent_ids = set()
        seen_ent_ids.add(entity_id)
        queue = deque([entity_id])
        for _ in range(num_hops):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.ent2neighbors.get(tp, set()):
                    if node not in seen_ent_ids:
                        queue.append(node)
                        seen_ent_ids.add(node)
                        if len(seen_ent_ids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_ent_ids])


def init(
    data_dir: str, pretrained_model: str, triple_path_list: List[str], train_path: str
) -> None:
    init_entity_dict(data_dir)
    init_inv_rel_dict(data_dir)
    init_tokenizer(pretrained_model)
    init_triple_dict(triple_path_list)
    init_neighbor_dict(train_path)


def init_entity_dict(data_dir: str):
    global entity_dict
    entity_dict = EntityDict(data_dir)


def get_entity_dict() -> EntityDict:
    return entity_dict


def init_inv_rel_dict(data_dir: str):
    global inv_rel_dict
    with open(osp.join(data_dir, "inverse_relations.json"), "r") as f:
        inv_rel_dict = json.load(f)


def init_tokenizer(pretrained_model: str) -> AutoTokenizer:
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)


def get_tokenizer() -> AutoTokenizer:
    return tokenizer


def init_triple_dict(path_list: List[str]) -> None:
    global triple_dict
    triple_dict = TripleDict(path_list)


def get_triple_dict() -> TripleDict:
    return triple_dict


def init_neighbor_dict(train_path: str) -> None:
    global neighbor_dict
    neighbor_dict = NeighborDict(train_path)


def get_neighbor_dict() -> NeighborDict:
    return neighbor_dict


def concat_name_desc(name: str, desc: str) -> str:
    if desc.startswith(name):
        desc = desc[len(name) :].strip()
    if desc:
        return "{}: {}".format(name, desc)
    return name


def get_neighbor_desc(head_id: str, tail_id: str = "", eval_mode: bool = False) -> str:
    neighbor_ids = neighbor_dict.get_neighbors(head_id)
    # avoid label leakage during training
    if not eval_mode:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entity_names = [entity_dict.get_entity_name(n_id) for n_id in neighbor_ids]
    return " ".join(entity_names)


def tokenize_hr(
    head_text: str,
    relation: str,
    pred_desc: str,
    max_num_tokens: int,
) -> Dict[str, List[int]]:
    hr_inputs = tokenizer(
        text=head_text, text_pair=relation, return_token_type_ids=True, return_attention_mask=False
    )
    desc_inputs = tokenizer(text=pred_desc, add_special_tokens=False, return_token_type_ids=True)

    encoded_inputs = {k: hr_inputs[k] + desc_inputs[k] for k in hr_inputs}
    if len(encoded_inputs["input_ids"]) >= max_num_tokens:
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v[:max_num_tokens]
        encoded_inputs["input_ids"][-1] = tokenizer.sep_token_id
    elif encoded_inputs["input_ids"][-1] != tokenizer.sep_token_id:
        encoded_inputs["input_ids"].append(tokenizer.sep_token_id)
        encoded_inputs["token_type_ids"].append(encoded_inputs["token_type_ids"][-1])

    return encoded_inputs


def tokenize_entity(entity_text: str, max_num_tokens: int) -> Dict[str, List[int]]:
    return tokenizer(
        text=entity_text,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=False,
        max_length=max_num_tokens,
        truncation=True,
    )


def reverse_triple(triple: Dict[str, str]) -> Dict[str, str]:
    return {
        "head_id": triple["tail_id"],
        "relation": inv_rel_dict[triple["relation"]],
        "tail_id": triple["head_id"],
        "pred_desc": triple["tr_desp"],
    }


def to_indices_and_mask(
    batch_tensor: List[torch.Tensor], pad_token_id: int = 0, need_mask: bool = True
) -> torch.LongTensor:
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, : len(t)].copy_(t)
        if need_mask:
            mask[i, : len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices


def construct_positive_mask(
    row_triples: List[Triple], col_triples: Optional[List[Triple]] = None
) -> torch.BoolTensor:
    positive_on_diagonal = col_triples is None
    num_row = len(row_triples)
    col_triples = row_triples if col_triples is None else col_triples
    num_col = len(col_triples)

    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_triples])
    col_entity_ids = (
        row_entity_ids
        if positive_on_diagonal
        else torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_triples])
    )
    # num_row x num_col
    triplet_mask = row_entity_ids.unsqueeze(1) == col_entity_ids.unsqueeze(0)

    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = row_triples[i].head_id, row_triples[i].relation
        neighbor_ids = triple_dict.get_relational_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue
            tail_id = col_triples[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = True

    return triplet_mask
