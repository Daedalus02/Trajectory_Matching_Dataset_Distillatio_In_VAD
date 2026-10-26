"""
main_rd4ad_clad.py — Continual AD (CLAD) con RD4AD.
Ricalcato su experiments/main_stfpm_clad.py: stessa infrastruttura continual
(ContinualTrainer / ContinualDataset / Replay / DistilledReplay / FineTuning,
stesse metriche e stesso output JSON), cambia solo il detector (RD4AD).

Strategie (--model):
  FT_rd4ad               fine-tuning sequenziale (baseline, nessuna memoria)
  MT_rd4ad               multi-task / joint (riferimento, tutti i task insieme)
  real_replay_rd4ad      replay con esemplari REALI, quota fissa per task
  distilled_replay_rd4ad replay con esemplari DISTILLATI, quota fissa per task
  replay_rd4ad           riferimento "memoria reale generosa" (budget totale)

Esempio:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src WANDB_MODE=offline uv run python \
    experiments/main_rd4ad_clad.py --model distilled_replay_rd4ad \
    --mem_per_task 10 --backbone wide_resnet50_2 --epochs 30 --batch_size 16 --seed 0
"""
from moviad.models import RD4AD
from moviad.models.rd4ad.rd4ad import RD4ADTrainArgs
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


# same task sequence as main_stfpm_clad.py
CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "transistor",
    "metal_nut", "pill", "screw", "zipper", "toothbrush",
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="FT_rd4ad, MT_rd4ad, real_replay_rd4ad, distilled_replay_rd4ad, replay_rd4ad",
                        choices=["FT_rd4ad", "MT_rd4ad", "real_replay_rd4ad",
                                 "distilled_replay_rd4ad", "replay_rd4ad"])
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--device", type=int, required=False)
    parser.add_argument("--dataset_path", type=str,
                        default="../../distillation/data/mvtec_anomaly_detection")
    parser.add_argument("--out_dir", type=str, default="results_clad")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224, help="lato immagine (RD4AD nativo 224)")
    parser.add_argument("--eval_interval", type=int, default=10**9,
                        help="evaluation_epoch_interval di RD4ADTrainArgs (grande = niente eval intra-task)")
    # replay
    parser.add_argument("--mem_per_task", type=int, default=10)
    parser.add_argument("--replay_ratio", type=float, default=0.5)
    parser.add_argument("--distilled_dir", type=str,
                        default="../../distillation/outputs/distilled_rd4ad")
    parser.add_argument("--distilled_template", type=str, default="{cat}/10/real/full128/images_best.pt")
    parser.add_argument("--memory_size", type=int, default=100)
    parser.add_argument("--max_cats", type=int, default=0, help="smoke test: usa solo le prime N categorie (0=tutte)")
    return parser.parse_args()


def _metrics():
    return [
        RocAuc(MetricLvl.IMAGE), RocAuc(MetricLvl.PIXEL),
        AvgPrec(MetricLvl.IMAGE), AvgPrec(MetricLvl.PIXEL),
        F1(MetricLvl.IMAGE), F1(MetricLvl.PIXEL),
        ProAuc(MetricLvl.PIXEL),
    ]


def _trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def make_model(backbone, device):
    return RD4AD(backbone_name=backbone).to(device)


def train_rd4ad_FT(dataset, backbone, epochs, batch_size, device, eval_interval):
    model = make_model(backbone, device)
    continual_model = FineTuning(model)
    trainer = ContinualTrainer(
        dataset, continual_model, device,
        metrics=_metrics(),
        training_args=RD4ADTrainArgs(epochs=epochs, batch_size=batch_size,
                                     evaluation_epoch_interval=eval_interval),
        logger=wandb,
    )
    before = [p.clone() for p in _trainable(model)]
    trainer.train()
    after = _trainable(model)
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "RD4AD: nessun parametro aggiornato"
    return getattr(trainer, "summary", None)


def train_rd4ad_multi_task(dataset, backbone, epochs, batch_size, device, eval_interval):
    train_dataset, test_dataset = dataset.get_all_tasks_data()
    model = make_model(backbone, device)
    training_args = RD4ADTrainArgs(epochs=epochs, batch_size=batch_size,
                                   evaluation_epoch_interval=eval_interval)
    training_args.init_train(model)
    trainer = Trainer(
        training_args, model, train_dataset, test_dataset,
        metrics=_metrics(), device=device, logger=wandb,
        save_path=None, saving_criteria=None,
    )
    before = [p.clone() for p in _trainable(model)]
    trainer.train()
    after = _trainable(model)
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "RD4AD: nessun parametro aggiornato"
    return getattr(trainer, "summary", None)


def train_rd4ad_replay(dataset, make_continual_model, backbone, epochs, batch_size, device, eval_interval):
    model = make_model(backbone, device)
    continual_model = make_continual_model(model)
    trainer = ContinualTrainer(
        dataset, continual_model, device,
        metrics=_metrics(),
        training_args=RD4ADTrainArgs(epochs=epochs, batch_size=batch_size,
                                     evaluation_epoch_interval=eval_interval),
        logger=None,
    )
    before = [p.clone() for p in _trainable(model)]
    trainer.train()
    after = _trainable(model)
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "RD4AD: nessun parametro aggiornato"
    return getattr(trainer, "summary", None)


def main():
    args = get_args()
    device = args.device if args.device is not None else 0
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wandb.init(project="moviad_test",
               name=f"{args.model}_{args.backbone}_{args.epochs}ep_bs{args.batch_size}_s{args.seed}",
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

    if args.model == "FT_rd4ad":
        summary = train_rd4ad_FT(continual_dataset, args.backbone, args.epochs, args.batch_size, device, args.eval_interval)
    elif args.model == "MT_rd4ad":
        summary = train_rd4ad_multi_task(continual_dataset, args.backbone, args.epochs, args.batch_size, device, args.eval_interval)
    elif args.model in ("real_replay_rd4ad", "distilled_replay_rd4ad", "replay_rd4ad"):
        if args.model == "real_replay_rd4ad":
            make_cm = lambda m: Replay(m, replay_ratio=args.replay_ratio, mem_per_task=args.mem_per_task)
        elif args.model == "distilled_replay_rd4ad":
            make_cm = lambda m: DistilledReplay(
                m, continual_dataset.categories, args.distilled_dir,
                mem_per_task=args.mem_per_task, replay_ratio=args.replay_ratio,
                img_size=dataset_args["img_size"], mean=imagenet_mean, std=imagenet_std,
                path_template=args.distilled_template,
            )
        else:
            make_cm = lambda m: Replay(m, memory_size=args.memory_size, replay_ratio=args.replay_ratio)
        summary = train_rd4ad_replay(continual_dataset, make_cm, args.backbone, args.epochs, args.batch_size, device, args.eval_interval)
    else:
        raise NotImplementedError(args.model)

    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"\nTempo totale: {elapsed/60:.1f} min | picco GPU: {peak_gb:.2f} GB")

    os.makedirs(args.out_dir, exist_ok=True)
    mem_tag = {"real_replay_rd4ad": f"_mpt{args.mem_per_task}",
               "distilled_replay_rd4ad": f"_mpt{args.mem_per_task}",
               "replay_rd4ad": f"_mem{args.memory_size}"}.get(args.model, "")
    out_path = os.path.join(args.out_dir,
                            f"{args.model}{mem_tag}_{args.backbone}_{args.epochs}ep_bs{args.batch_size}_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "time_s": elapsed, "peak_gpu_gb": peak_gb, **(summary or {})}, f, indent=2)
    print(f"Risultati salvati in {out_path}")


if __name__ == "__main__":
    main()
