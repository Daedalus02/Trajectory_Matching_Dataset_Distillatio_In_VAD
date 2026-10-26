"""Continual STFPM on 10 MVTec AD categories: fine-tuning, joint, real / distilled / low-res replay.

Writes one JSON per run (per-task matrices, final average, forgetting) to --out_dir.
"""

from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
from moviad.trainers.trainer import Trainer
from moviad.datasets.mvtec import MVTecDataset
from torch.utils.data import Subset
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
from moviad.utilities.evaluation.evaluator import Evaluator
import os
import time


BACKBONES = [
    "mcunet-in3",
    "micronet-m0",
    "micronet-m1",
    "micronet-m2",
    "micronet-m3",
    #"phinet_2.3_0.75_5",
    "phinet_1.2_0.5_6_downsampling",
    #"phinet_0.8_0.75_8_downsampling",
    #"phinet_1.3_0.5_7_downsampling",
    #"phinet_0.9_0.5_4_downsampling_deep",
    #"phinet_0.9_0.5_4_downsampling",
    "vgg19_bn",
    "resnet18",
    "wide_resnet50_2",
    "efficientnet_b5",
    "mobilenet_v2",
]


def get_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Es. FT_stfpm, MT_stfpm, real_replay_stfpm, distilled_replay_stfpm",
                        choices=["FT_stfpm", "MT_stfpm", "real_replay_stfpm", "distilled_replay_stfpm", "replay_stfpm", "lowres_replay_stfpm"])
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", help="Es. wide_resnet50_2", choices=BACKBONES)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--device", type=int, required=False)
    parser.add_argument("--dataset_path", type=str, default="../../distillation/data/mvtec_anomaly_detection")
    parser.add_argument("--out_dir", type=str, default="results_clad")
    parser.add_argument("--seed", type=int, default=0)
    # replay strategies keep a fixed number of exemplars per task
    parser.add_argument("--mem_per_task", type=int, default=10)
    parser.add_argument("--replay_ratio", type=float, default=0.5)
    parser.add_argument("--distilled_dir", type=str, default="../../distillation/outputs/distilled")
    parser.add_argument("--distilled_template", type=str, default="{cat}/10/real/full128/images_best.pt")
    parser.add_argument("--store_res", type=int, default=128, help="lowres_replay: risoluzione di memorizzazione")
    # replay_stfpm = riferimento opzionale "memoria reale generosa" (budget totale condiviso, quota = memory_size // n_task)
    parser.add_argument("--memory_size", type=int, default=100)
    args = parser.parse_args()

    return args



def train_stfpm_FT(dataset, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], epochs=20, batch_size=16, device="cpu"):

    teacher = CustomFeatureExtractor(backbone, layers, device, frozen=True)
    student = CustomFeatureExtractor(backbone, layers, device, frozen=False)

    model = STFPM(teacher, student)
    model.to(device)
    continual_model = FineTuning(model)

    trainer = ContinualTrainer(
        dataset,
        continual_model,
        device,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        training_args=STFPMTrainArgs(epochs=epochs, batch_size=batch_size),
        logger=wandb
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
    return getattr(trainer, "summary", None)



def train_stfpm_multi_task(dataset, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], epochs=50, batch_size=16, device="cpu"):

    teacher = CustomFeatureExtractor(backbone, layers, device, frozen=True)
    student = CustomFeatureExtractor(backbone, layers, device, frozen=False)

    train_dataset, test_dataset = dataset.get_all_tasks_data()

    model = STFPM(teacher, student)
    model.to(device)

    training_args = STFPMTrainArgs(epochs=epochs, batch_size=batch_size)
    training_args.init_train(model)

    trainer = Trainer(
        training_args,
        model,
        train_dataset,
        test_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=wandb,
        save_path=None,
        saving_criteria=None,
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
    # per-task evaluation, as in ContinualTrainer, so the joint model is comparable
    metrics = [RocAuc(MetricLvl.IMAGE), RocAuc(MetricLvl.PIXEL), AvgPrec(MetricLvl.IMAGE),
               AvgPrec(MetricLvl.PIXEL), F1(MetricLvl.IMAGE), F1(MetricLvl.PIXEL), ProAuc(MetricLvl.PIXEL)]
    per_task = {}
    for t in range(len(dataset.categories)):
        dl = torch.utils.data.DataLoader(dataset.get_task_data_evaluation(t), batch_size=batch_size, shuffle=False, num_workers=4)
        res = Evaluator.evaluate(model, dl, metrics, device=device)
        per_task[dataset.categories[t]] = {k: float(v) for k, v in res.items()}
        print(f"MT per-task {dataset.categories[t]}: {per_task[dataset.categories[t]]}")
    names = list(next(iter(per_task.values())).keys())
    final_average = {k: float(np.mean([per_task[c][k] for c in per_task])) for k in names}
    dl_all = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    pooled = {k: float(v) for k, v in Evaluator.evaluate(model, dl_all, metrics, device=device).items()}
    print(f"MT final_average (per-task): {final_average}")
    print(f"MT pooled: {pooled}")
    return {"categories": list(dataset.categories), "per_task": per_task, "final_average": final_average, "pooled": pooled}



def train_stfpm_replay(dataset, make_continual_model, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], epochs=10, batch_size=8, device="cpu"):

    teacher = CustomFeatureExtractor(backbone, layers, device, frozen=True)
    student = CustomFeatureExtractor(backbone, layers, device, frozen=False)
    model = STFPM(teacher, student).to(device)

    continual_model = make_continual_model(model)

    trainer = ContinualTrainer(
        dataset,
        continual_model,
        device,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        training_args=STFPMTrainArgs(epochs=epochs, batch_size=batch_size),
        logger=None
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
    return getattr(trainer, "summary", None)



def main():
    args = get_test()
    numEpochs = args.epochs
    batch_size = args.batch_size
    backbone = args.backbone
    model = args.model
    device = args.device if args.device else 0

    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wandb.init(project="moviad_test", name=f"{model}_{backbone}_{numEpochs}_epochs_{batch_size}_minibatch")
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("eval/*", step_metric="epoch")

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    mvtec_transform = [
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]

    dataset_args = {
        "dataset_path" : args.dataset_path,
        "img_size" : (256, 256),
        "gt_mask_size" : (256, 256),
        #"image_transform_list" : None
        "image_transform_list": mvtec_transform
    }

    continual_dataset = ContinualDataset(
        dataset_arguments=DatasetArguments(**dataset_args),
        dataset_class=MVTecDataset,
        categories=[
            "bottle",
            "cable",
            "capsule",
            "hazelnut",
            "transistor",
            "metal_nut",
            "pill",
            "screw",
            "zipper",
            "toothbrush"
        ],
    )

    if backbone == "mcunet-in3":
        layers = ["2", "6", "14"]
    else:
        layers = ["layer1", "layer2", "layer3"]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    if model == "FT_stfpm":
        summary = train_stfpm_FT(dataset=continual_dataset, backbone=backbone, layers=layers, epochs=numEpochs, batch_size=batch_size, device=device)
    elif model == "MT_stfpm":
        summary = train_stfpm_multi_task(dataset=continual_dataset, backbone=backbone, layers=layers, epochs=numEpochs, batch_size=batch_size, device=device)
    elif model in ("real_replay_stfpm", "distilled_replay_stfpm", "replay_stfpm", "lowres_replay_stfpm"):
        if model == "lowres_replay_stfpm":
            from moviad.scenarios.continual.strategies.replay.lowres_replay_model import LowResReplay
            make_continual_model = lambda m: LowResReplay(m, store_res=args.store_res, replay_ratio=args.replay_ratio, mem_per_task=args.mem_per_task)
        elif model == "real_replay_stfpm":
            make_continual_model = lambda m: Replay(m, replay_ratio=args.replay_ratio, mem_per_task=args.mem_per_task)
        elif model == "distilled_replay_stfpm":
            make_continual_model = lambda m: DistilledReplay(
                m, continual_dataset.categories, args.distilled_dir,
                mem_per_task=args.mem_per_task, replay_ratio=args.replay_ratio,
                img_size=dataset_args["img_size"], mean=imagenet_mean, std=imagenet_std,
                path_template=args.distilled_template,
            )
        else:
            make_continual_model = lambda m: Replay(m, memory_size=args.memory_size, replay_ratio=args.replay_ratio)
        summary = train_stfpm_replay(dataset=continual_dataset, make_continual_model=make_continual_model, backbone=backbone, layers=layers, epochs=numEpochs, batch_size=batch_size, device=device)
    else:
        raise NotImplementedError(f"Model {model} not implemented in tests.")

    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"\nTempo totale: {elapsed/60:.1f} min | picco GPU (allocated): {peak_gb:.2f} GB")

    os.makedirs(args.out_dir, exist_ok=True)
    mem_tag = {"real_replay_stfpm": f"_mpt{args.mem_per_task}", "distilled_replay_stfpm": f"_mpt{args.mem_per_task}",
               "replay_stfpm": f"_mem{args.memory_size}",
               "lowres_replay_stfpm": f"_mpt{args.mem_per_task}_res{args.store_res}"}.get(model, "")
    out_path = os.path.join(args.out_dir, f"{model}{mem_tag}_{backbone}_{numEpochs}ep_bs{batch_size}_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "time_s": elapsed, "peak_gpu_gb": peak_gb, **(summary or {})}, f, indent=2)
    print(f"Risultati salvati in {out_path}")



if __name__ == "__main__":
    main()