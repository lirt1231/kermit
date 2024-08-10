import json
import pathlib
import random
import tqdm
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import wandb
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from args import TrainerArguments, ModelArguments
from dataset import (
    Dataset,
    Triple,
    EntityDict,
    collate,
    get_entity_dict,
    get_triple_dict,
    get_neighbor_dict,
    init as dataset_init,
    set_eval_mode as dataset_eval,
    concat_name_desc,
    get_neighbor_desc,
    get_tokenizer,
    tokenize_entity,
    to_indices_and_mask,
)
from model import Kermit
from utils import move_to_cuda, get_model_obj


train_args: TrainerArguments = None
model_args: ModelArguments = None
model: Kermit = None
NEIGHBOR_WEIGHT: float = None
RERANK_N_HOP: int = None


def compute_metrics(
    logits: torch.LongTensor, targets: torch.LongTensor, triples: List[Triple], topk: int = 3
) -> Tuple:
    entity_dict = get_entity_dict()
    all_triple_dict = get_triple_dict()
    topk_scores, topk_indices, ranks = [], [], []
    mean_rank, mrr, hit1, hit3, hit10 = 0.0, 0.0, 0.0, 0.0, 0.0

    assert len(logits) == len(targets) and len(targets) == len(triples)

    rerank_by_graph(logits, triples)
    for scores, triple, target in zip(logits, triples, targets):
        # filter known triples
        entities_to_mask = all_triple_dict.get_relational_neighbors(triple.head_id, triple.relation)
        entities_to_mask = entities_to_mask - {triple.tail_id}
        entities_to_mask = [entity_dict.entity_to_idx(e) for e in entities_to_mask]
        # cosine similarity as score >= 0
        scores[entities_to_mask] = -1
        # sort scores
        scores, indices = torch.sort(scores, descending=True)
        # 0-based => 1-based
        rank = torch.nonzero(indices.eq(target)).item() + 1

        mean_rank += rank
        mrr += 1.0 / rank
        hit1 += 1 if rank <= 1 else 0
        hit3 += 1 if rank <= 3 else 0
        hit10 += 1 if rank <= 10 else 0
        ranks.append(rank)
        topk_scores.append(scores[:topk].tolist())
        topk_indices.append(indices[:topk].tolist())

    metrics = {
        "mean_rank": mean_rank,
        "mrr": mrr,
        "hit1": hit1,
        "hit3": hit3,
        "hit10": hit10,
    }
    return topk_scores, topk_indices, metrics, ranks


def rerank_by_graph(batch_score: torch.FloatTensor, triples: List[Triple]):
    if NEIGHBOR_WEIGHT < 1e-6:
        return
    neighbor_dict = get_neighbor_dict()

    for score, triple in zip(batch_score, triples):
        n_hop_indices = neighbor_dict.get_n_hop_entity_indices(
            triple.head_id, num_hops=RERANK_N_HOP
        )
        delta = torch.tensor([NEIGHBOR_WEIGHT for _ in n_hop_indices]).to(batch_score.device)
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)
        score.index_add_(0, n_hop_indices, delta)


def encode_hr(inputs: Dict[str, torch.Tensor], device: str) -> torch.FloatTensor:
    data = move_to_cuda(
        {
            "token_ids": inputs["hr_token_ids"],
            "mask": inputs["hr_mask"],
            "token_type_ids": inputs["hr_token_type_ids"],
        },
        device,
    )
    hr_vector = get_model_obj(model).encode_hr(**data)
    return F.normalize(hr_vector, dim=1)


def encode_entities(device: str, batch_size: int) -> torch.FloatTensor:
    entity_dict = get_entity_dict()
    encoded_inputs, encoded_entities = [], []
    for eid, name, desc in entity_dict.iter_entities():
        if model_args.use_neighbor_names and len(desc.split()) < 20:
            desc += " " + get_neighbor_desc(eid, eval_mode=True)
        entity_text = concat_name_desc(name, desc)
        encoded_inputs.append(tokenize_entity(entity_text, train_args.max_num_tokens))

    loader = DataLoader(
        encoded_inputs, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: x
    )
    for _, inputs in tqdm.tqdm(enumerate(loader), total=len(loader), desc="Encoding entities"):
        token_ids, mask = to_indices_and_mask(
            [torch.LongTensor(data["input_ids"]) for data in inputs],
            pad_token_id=get_tokenizer().pad_token_id,
        )
        token_type_ids = to_indices_and_mask(
            [torch.LongTensor(data["token_type_ids"]) for data in inputs], need_mask=False
        )
        token_ids, mask, token_type_ids = (
            token_ids.to(device),
            mask.to(device),
            token_type_ids.to(device),
        )

        encoded_entities.append(
            get_model_obj(model).encode_entities(token_ids, mask, token_type_ids)
        )

    return torch.cat(encoded_entities, dim=0)


def eval_single_direction(
    entity_vector: torch.Tensor,
    eval_forward: bool,
    batch_size: int,
) -> Tuple[Dict[str, float], List[dict]]:
    entity_dict: EntityDict = get_entity_dict()
    test_dataset = Dataset(
        train_args.test_path,
        train_args.max_num_tokens,
        model_args.use_neighbor_names,
        add_forward_triples=eval_forward,
        add_backward_triples=not eval_forward,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=4
    )
    num_samples = len(test_dataset)
    metrics = {"mrr": 0.0, "hit1": 0.0, "hit3": 0.0, "hit10": 0.0, "mean_rank": 0.0}
    pred_infos = []

    direction = "forward" if eval_forward else "backward"
    for _, inputs in tqdm.tqdm(
        enumerate(test_dataloader), desc=f"evaluating {direction}", total=len(test_dataloader)
    ):
        triples: List[Triple] = inputs["triples"]
        hr_vector = encode_hr(inputs, entity_vector.device)
        targets = [entity_dict.entity_to_idx(trip.tail_id) for trip in triples]
        targets = torch.LongTensor(targets).to(entity_vector.device)
        logits = hr_vector.mm(entity_vector.t())
        # compute metrics
        batch_topk_scores, batch_topk_indices, batch_metrics, ranks = compute_metrics(
            logits, targets, triples
        )
        # update metrics
        for k, v in batch_metrics.items():
            metrics[k] += v
        # update prediction info
        for trip, topk_scores, topk_indices, rank, target in zip(
            triples, batch_topk_scores, batch_topk_indices, ranks, targets.tolist()
        ):
            pred_idx, pred_score = topk_indices[0], topk_scores[0]
            score_info = {
                entity_dict.idx_to_name(idx): score for idx, score in zip(topk_indices, topk_scores)
            }
            pred_infos.append(
                {
                    "head": entity_dict.get_entity_name(trip.head_id),
                    "relation": trip.relation,
                    "tail": entity_dict.get_entity_name(trip.tail_id),
                    "pred_tail": entity_dict.idx_to_name(pred_idx),
                    "pred_score": round(pred_score, 4),
                    "topk_score_info": score_info,
                    "rank": rank,
                    "correct": (pred_idx == target),
                }
            )
    for k, v in metrics.items():
        metrics[k] = round(v / num_samples, 4)

    return metrics, pred_infos


@torch.inference_mode()
def evaluate(
    ckpt_path: str,
    device: str,
    batch_size: int,
    wandb_run_id: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    entity_vector = encode_entities(device, batch_size)
    forward_metrics, forward_pred_infos = eval_single_direction(entity_vector, True, batch_size)
    backward_metrics, backward_pred_infos = eval_single_direction(entity_vector, False, batch_size)

    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    print("forward metrics: {}".format(forward_metrics))
    print("backward metrics: {}".format(backward_metrics))
    print("averaged metrics: {}".format(metrics))

    ckpt_path = pathlib.Path(ckpt_path)
    experiment_dir = ckpt_path.parent.parent
    path = experiment_dir / f"test_results_{ckpt_path.stem}.txt"
    with open(path, "w") as file:
        file.write("forward metrics: {}\n".format(json.dumps(forward_metrics)))
        file.write("backward metrics: {}\n".format(json.dumps(backward_metrics)))
        file.write("average metrics: {}\n".format(json.dumps(metrics)))
    path = experiment_dir / f"forward_details_{ckpt_path.stem}.json"
    with open(path, "w") as file:
        file.write(json.dumps(forward_pred_infos))
    path = experiment_dir / f"backward_details_{ckpt_path.stem}.json"
    with open(path, "w") as file:
        file.write(json.dumps(backward_pred_infos))

    if wandb_run_id:
        wandb.init(
            id=wandb_run_id,
            entity=wandb_entity,
            project=wandb_project,
            dir=train_args.log_dir,
            resume="must",
        )
        for k, v in forward_metrics.items():
            wandb.summary[f"test/forward_{k}"] = v
        for k, v in backward_metrics.items():
            wandb.summary[f"test/backward_{k}"] = v
        for k, v in metrics.items():
            wandb.summary[f"test/avg_{k}"] = v
        wandb.finish()

    return forward_metrics, backward_metrics, metrics


def load_checkpoint(ckpt_path: str, device: str) -> None:
    global train_args, model_args, model
    # load training args
    arg_path = pathlib.Path(ckpt_path).parent.parent / "args.json"
    args = json.load(arg_path.open("r"))
    train_args = TrainerArguments(**args["train_args"])
    model_args = ModelArguments(**args["model_args"])
    # load model state dict
    checkpoint = torch.load(ckpt_path)
    model = Kermit(model_args)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    if torch.cuda.device_count() > 1 and device == "cuda":
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)
    model.eval()


def main(
    checkpoint_path: str,
    neighbor_weight: float,
    rerank_n_hop: int,
    device: str = "cuda",
    batch_size: int = 256,
    wandb_run_id: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
):
    global NEIGHBOR_WEIGHT, RERANK_N_HOP
    NEIGHBOR_WEIGHT, RERANK_N_HOP = neighbor_weight, rerank_n_hop

    load_checkpoint(checkpoint_path, device)
    if train_args.seed is not None:
        random.seed(train_args.seed)
        np.random.seed(train_args.seed)
        torch.manual_seed(train_args.seed)

    dataset_init(
        train_args.data_dir,
        model_args.pretrained_model,
        [train_args.train_path, train_args.valid_path, train_args.test_path],
        train_args.train_path,
    )
    dataset_eval()
    return evaluate(checkpoint_path, device, batch_size, wandb_run_id, wandb_entity, wandb_project)


if __name__ == "__main__":
    fire.Fire(main)
