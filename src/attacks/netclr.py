import argparse
import os
from typing import Union

import numpy as np
import torch

from attacks.base import Attack
from attacks.modules import NetCLRNet
from attacks.modules.netclr import run_pretrain
from utils.general import parse_trace, feature_transform


class NetCLRAttack(Attack):
    """
    NetCLR attack: runs SimCLR pre-training first (if no pretrained checkpoint exists),
    then fine-tuning. Fine-tuning is closed-world or open-world based on --open-world
    (same data and metrics as base Attack).
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def run(self, one_fold_only: bool = False):
        # 1) Run pre-training when pretrained checkpoint is missing
        pretrained_path = os.path.join(self.args.checkpoints, "netclr", "NetCLR_pretrained.pth")
        if not os.path.isfile(pretrained_path):
            self.logger.info("NetCLR pretrained checkpoint not found. Running SimCLR pre-training first.")
            # Use only monitored traces for pre-training (when open-world, filter to mon_classes)
            mon_mask = self.labels < self.nmc
            flist_pretrain = self.flist[mon_mask]
            labels_pretrain = self.labels[mon_mask]
            run_pretrain(
                self.args,
                flist_pretrain,
                labels_pretrain,
                self.device,
                self.logger,
                seq_length=self.args.seq_length,
            )
        else:
            self.logger.info("NetCLR using existing pretrained checkpoint: {}".format(pretrained_path))

        # 2) Fine-tuning: closed-world or open-world based on --open-world (handled by base)
        super().run(one_fold_only=one_fold_only)

    def _build_model(self):
        model = NetCLRNet(num_classes=self.nc, seq_length=self.args.seq_length)
        pretrained_path = os.path.join(self.args.checkpoints, "netclr", "NetCLR_pretrained.pth")
        if os.path.isfile(pretrained_path):
            self.logger.info("Loading NetCLR pretrained backbone from: {}".format(pretrained_path))
            ckpt = torch.load(pretrained_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_pretrained_backbone(state, strict=False)
        return model

    @staticmethod
    def extract(data_path: Union[str, os.PathLike], seq_length: int) -> np.ndarray:
        """
        DF (direction) feature extraction for a single trace; same as pending NetCLR.
        """
        trace = parse_trace(data_path)
        feat = feature_transform(trace, feature_type='df', seq_length=seq_length)
        return feat
