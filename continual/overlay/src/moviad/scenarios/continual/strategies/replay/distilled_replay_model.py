"""Replay whose memory of a past task holds its distilled images instead of real ones.
"""

import os

import torch
import torch.nn.functional as F

from moviad.models.vad_model import VADModel
from moviad.scenarios.continual.strategies.replay.replay_model import Replay


class DistilledReplay(Replay):
    """
    Replay identical to `Replay` (same train_task loop, replay_ratio and Memory),
    except that at the end of each task the memory is filled with the distilled
    images of the task category instead of the real training batches.
    """

    def __init__(
        self,
        vad_model: VADModel,
        categories: list[str],
        distilled_dir: str,
        mem_per_task: int = 10,
        replay_ratio=0.5,
        img_size=(256, 256),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        path_template: str = "{cat}/10/real/full128/images_best.pt",
    ):
        super().__init__(vad_model, replay_ratio=replay_ratio, mem_per_task=mem_per_task)
        self.categories = categories
        self.img_size = tuple(img_size)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.paths = [os.path.join(os.path.expanduser(distilled_dir), path_template.format(cat=c)) for c in categories]

        # fail fast, before hours of training
        missing = [p for p in self.paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError("Distilled images not found:\n  " + "\n  ".join(missing))

    def load_distilled(self, task_index: int) -> torch.Tensor:
        """Distilled [N,3,h,w] in [0,1] -> [N,3,256,256] ImageNet-normalized, same space as the real batches."""
        images = torch.load(self.paths[task_index], map_location="cpu").float()
        print(f"Distilled {self.categories[task_index]}: {tuple(images.shape)} "
              f"range [{images.min():.3f}, {images.max():.3f}] <- {self.paths[task_index]}")
        images = F.interpolate(images, size=self.img_size, mode="bilinear", align_corners=False, antialias=True)
        return (images - self.mean) / self.std

    def populate_memory(self, task_index: int, train_dataloader):
        samples = self.load_distilled(task_index)
        if len(samples) < self.memory.mem_per_task:
            print(f"WARNING: {len(samples)} distilled images < mem_per_task={self.memory.mem_per_task}")
        self.memory.add_samples(task_id=task_index, samples=samples)
