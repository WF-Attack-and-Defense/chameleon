import argparse
import os
from typing import Union

import numpy as np

from attacks.base import Attack
from attacks.modules import VarCNNNet
from utils.general import parse_trace, feature_transform


class VarCNNAttack(Attack):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def _build_model(self):
        model = VarCNNNet(length=self.args.seq_length, num_classes=self.nc, in_channels=2)
        return model

    @staticmethod
    def extract(data_path: Union[str, os.PathLike], seq_length: int) -> np.ndarray:
        """
        VarCNN feature extraction for a single trace
        """
        trace = parse_trace(data_path)
        feat = feature_transform(trace, feature_type='var_cnn', seq_length=seq_length)
        return feat

