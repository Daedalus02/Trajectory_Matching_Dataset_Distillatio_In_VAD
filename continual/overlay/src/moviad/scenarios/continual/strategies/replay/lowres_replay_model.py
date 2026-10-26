"""Replay of real images stored at low resolution (same footprint as the distilled buffer).
"""

import torch.nn.functional as F

from moviad.models.vad_model import VADModel
from moviad.scenarios.continual.strategies.replay.replay_model import Replay


class LowResReplay(Replay):
    """
    Replay identical to `Replay` (reservoir sampling of real images), except that every exemplar is
    stored at `store_res` x `store_res` and upsampled back to the input size when it is replayed.
    Control for DistilledReplay: same storage resolution as the distilled sets, real content.
    """

    def __init__(self, vad_model: VADModel, store_res: int = 128, replay_ratio=0.5, mem_per_task: int = None):
        super().__init__(vad_model, replay_ratio=replay_ratio, mem_per_task=mem_per_task)
        self.store_res = store_res

    def populate_memory(self, task_index: int, train_dataloader):
        for batch in train_dataloader:
            size = batch.shape[-2:]
            low = F.interpolate(batch, size=(self.store_res, self.store_res), mode="bilinear", align_corners=False, antialias=True)
            back = F.interpolate(low, size=size, mode="bilinear", align_corners=False, antialias=True)
            self.memory.add_samples(task_id=task_index, samples=back)
