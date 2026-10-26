"""
main_dinomaly_clad.py — Continual AD (CLAD) con Dinomaly (DINOv2-reg, 224).
Ricalcato su main_rd4ad_clad.py / main_stfpm_clad.py: stessa infrastruttura continual
(ContinualTrainer / ContinualDataset / Replay / DistilledReplay / FineTuning), cambia
solo il detector. Dinomaly allena per iterazioni (DinomalyTrainArgs.total_iters) ma il
loop per-task resta a epoche come per gli altri; encoder congelato, trainable = bottleneck+decoder.

Strategie (--model): FT_dino, MT_dino, real_replay_dino, distilled_replay_dino, replay_dino
Esempio:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src WANDB_MODE=offline ~/.local/bin/uv run python \
    experiments/main_dinomaly_clad.py --model distilled_replay_dino --mem_per_task 10 \
    --epochs 30 --batch_size 16 --seed 0
"""
from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.trainers.trainer import Trainer
from moviad.datasets.mvtec import MVTecDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.scenarios.continual.strategies.replay.replay_model import Replay
from moviad.scenarios.continual.strategies.replay.distilled_replay_model import DistilledReplay
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.fine_tuning import FineTuning
import torch
from torchvision import transforms
import wandb

import argparse
import json
import random
import numpy as np
import os
import time


CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "transistor",
    "metal_nut", "pill", "screw", "zipper", "toothbrush",
]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   choices=["FT_dino", "MT_dino", "real_replay_dino", "distilled_replay_dino", "replay_dino"])
    p.add_argument("--encoder", type=str, default="vit_small_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--device", type=int, required=False)
    p.add_argument("--dataset_path", type=str,
                   default="../../distillation/data/mvtec_anomaly_detection")
    p.add_argument("--out_dir", type=str, default="results_clad")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--eval_interval", type=int, default=10**9)
    p.add_argument("--mem_per_task", type=int, default=10)
    p.add_argument("--replay_ratio", type=float, default=0.5)
    p.add_argument("--distilled_dir", type=str,
                   default="../../distillation/outputs/distilled_dinomaly")
    p.add_argument("--distilled_template", type=str, default="{cat}/10/real/full224/images_best.pt")
    p.add_argument("--memory_size", type=int, default=100)
    p.add_argument("--max_cats", type=int, default=0, help="smoke: solo le prime N categorie (0=tutte)")
    return p.parse_args()


def _metrics():
    return [
        RocAuc(MetricLvl.IMAGE), RocAuc(MetricLvl.PIXEL),
        AvgPrec(MetricLvl.IMAGE), AvgPrec(MetricLvl.PIXEL),
        F1(MetricLvl.IMAGE), F1(MetricLvl.PIXEL),
        ProAuc(MetricLvl.PIXEL),
    ]


def make_model(encoder, device):
    return Dinomaly(encoder_name=encoder).to(device)


def _train_args(epochs, bs, eval_interval):
    return DinomalyTrainArgs(batch_size=bs, epochs=epochs, evaluation_epoch_interval=eval_interval)


def _assert_updated(model, before):
    after = [p for p in model.trainable.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "Dinomaly: nessun parametro aggiornato"


def train_FT(dataset, encoder, epochs, bs, device, ev):
    model = make_model(encoder, device)
    cm = FineTuning(model)
    trainer = ContinualTrainer(dataset, cm, device, metrics=_metrics(),
                               training_args=_train_args(epochs, bs, ev), logger=wandb)
    before = [p.clone() for p in model.trainable.parameters()]
    trainer.train()
    _assert_updated(model, before)
    return getattr(trainer, "summary", None)


def train_MT(dataset, encoder, epochs, bs, device, ev):
    train_ds, test_ds = dataset.get_all_tasks_data()
    model = make_model(encoder, device)
    ta = _train_args(epochs, bs, ev)
    ta.init_train(model)
    trainer = Trainer(ta, model, train_ds, test_ds, metrics=_metrics(),
                      device=device, logger=wandb, save_path=None, saving_criteria=None)
    before = [p.clone() for p in model.trainable.parameters()]
    trainer.train()
    _assert_updated(model, before)
    return getattr(trainer, "summary", None)


def train_replay(dataset, make_cm, encoder, epochs, bs, device, ev):
    model = make_model(encoder, device)
    cm = make_cm(model)
    trainer = ContinualTrainer(dataset, cm, device, metrics=_metrics(),
                               training_args=_train_args(epochs, bs, ev), logger=None)
    before = [p.clone() for p in model.trainable.parameters()]
    trainer.train()
    _assert_updated(model, before)
    return getattr(trainer, "summary", None)


def main():
    args = get_args()
    device = args.device if args.device is not None else 0
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wandb.init(project="moviad_test",
               name=f"{args.model}_dino_{args.epochs}ep_bs{args.batch_size}_s{args.seed}",
               mode=os.environ.get("WANDB_MODE", "offline"))

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    S = args.img_size
    mvtec_transform = [
        transforms.Resize((S, S), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
    dataset_args = {
        "dataset_path": args.dataset_path,
        "img_size": (S, S),
        "gt_mask_size": (S, S),
        "image_transform_list": mvtec_transform,
    }

    cats = CATEGORIES[:args.max_cats] if args.max_cats and args.max_cats > 0 else CATEGORIES
    continual_dataset = ContinualDataset(
        dataset_arguments=DatasetArguments(**dataset_args),
        dataset_class=MVTecDataset,
        categories=cats,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    if args.model == "FT_dino":
        summary = train_FT(continual_dataset, args.encoder, args.epochs, args.batch_size, device, args.eval_interval)
    elif args.model == "MT_dino":
        summary = train_MT(continual_dataset, args.encoder, args.epochs, args.batch_size, device, args.eval_interval)
    elif args.model in ("real_replay_dino", "distilled_replay_dino", "replay_dino"):
        if args.model == "real_replay_dino":
            make_cm = lambda m: Replay(m, replay_ratio=args.replay_ratio, mem_per_task=args.mem_per_task)
        elif args.model == "distilled_replay_dino":
            make_cm = lambda m: DistilledReplay(
                m, continual_dataset.categories, args.distilled_dir,
                mem_per_task=args.mem_per_task, replay_ratio=args.replay_ratio,
                img_size=dataset_args["img_size"], mean=imagenet_mean, std=imagenet_std,
                path_template=args.distilled_template,
            )
        else:
            make_cm = lambda m: Replay(m, memory_size=args.memory_size, replay_ratio=args.replay_ratio)
        summary = train_replay(continual_dataset, make_cm, args.encoder, args.epochs, args.batch_size, device, args.eval_interval)
    else:
        raise NotImplementedError(args.model)

    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"\nTempo totale: {elapsed/60:.1f} min | picco GPU: {peak_gb:.2f} GB")

    os.makedirs(args.out_dir, exist_ok=True)
    mem_tag = {"real_replay_dino": f"_mpt{args.mem_per_task}",
               "distilled_replay_dino": f"_mpt{args.mem_per_task}",
               "replay_dino": f"_mem{args.memory_size}"}.get(args.model, "")
    out_path = os.path.join(args.out_dir,
                            f"{args.model}{mem_tag}_dino_{args.epochs}ep_bs{args.batch_size}_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "time_s": elapsed, "peak_gpu_gb": peak_gb, **(summary or {})}, f, indent=2)
    print(f"Risultati salvati in {out_path}")


if __name__ == "__main__":
    main()
