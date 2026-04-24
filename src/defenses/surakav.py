"""
Surakav (WF-GAN) website fingerprinting defense.

Per-trace defense follows pendding/surakav/src/generate.py with gan_glue-style
generator loading (scaler.gz + generator_seqlen*_cls*_latentdim*.ckpt).
"""

import argparse
import glob
import os
import re
from typing import List, Tuple, Union

import joblib
import numpy as np
import torch
from sklearn import preprocessing

from defenses.base import Defense
from defenses.config import SurakavConfig
from utils.general import parse_trace, set_random_seed
from utils.surakav_model import Generator

GENERATOR_CKPT_RE = re.compile(
    r"generator_seqlen([0-9]+)_cls([0-9]+)_latentdim([0-9]+)\.ckpt$"
)


def extract_bursts_from_trace(trace_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Burst lengths (cell counts) alternating from first outgoing burst, with start times per burst.
    Mirrors pendding/surakav/src/generate.py extract() but keeps alignment with timestamps.
    """
    dirs = np.sign(trace_2d[:, 1]).astype(int)
    times = trace_2d[:, 0].astype(float)
    start = 0
    for i in range(len(dirs)):
        if dirs[i] > 0:
            start = i
            break
    dirs = dirs[start:]
    times = times[start:]
    if len(dirs) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    new_x: List[int] = []
    start_t: List[float] = []
    sign = int(np.sign(dirs[0]))
    cnt = 0
    burst_started_at = 0
    for idx, e in enumerate(dirs):
        if int(np.sign(e)) == sign:
            cnt += 1
        else:
            new_x.append(cnt)
            start_t.append(float(times[burst_started_at]))
            cnt = 1
            sign = int(np.sign(e))
            burst_started_at = idx
    new_x.append(cnt)
    start_t.append(float(times[burst_started_at]))
    return np.array(new_x, dtype=int), np.array(start_t, dtype=float)


def choose_morphed_class(label: int, mon_site_num: int, rng: np.random.Generator) -> int:
    if label < mon_site_num:
        return mon_site_num
    return int(rng.integers(0, mon_site_num))


class SurakavDefense(Defense):
    """WF-GAN burst padding defense (Surakav). Model is loaded lazily per worker process."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = SurakavConfig(args)
        self.config.load_config()
        self.dummy_code = int(self.config.dummy_code)
        self.mon_site_num = int(args.mon_classes)
        self.rng = np.random.default_rng(int(self.config.seed))

        model_dir = os.path.normpath(str(self.config.model_dir))
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Surakav model_dir not found or not a directory: {model_dir!r}. "
                "Train the GAN under pendding/surakav and set model_dir to the folder "
                "with scaler.gz and generator_seqlen*_cls*_latentdim*.ckpt."
            )

        scaler_paths = glob.glob(os.path.join(model_dir, "scaler.gz"))
        if not scaler_paths:
            raise FileNotFoundError(
                f"No scaler.gz in {model_dir}. Expected joblib MinMaxScaler from Surakav training."
            )
        self._scaler_path = scaler_paths[0]
        self.scaler: preprocessing.MinMaxScaler = joblib.load(self._scaler_path)

        ckpt_paths = glob.glob(os.path.join(model_dir, "generator*.ckpt"))
        if not ckpt_paths:
            raise FileNotFoundError(f"No generator*.ckpt in {model_dir}.")
        self._ckpt_path = ckpt_paths[0]
        base = os.path.basename(self._ckpt_path)
        m = GENERATOR_CKPT_RE.match(base)
        if not m:
            raise ValueError(
                f"Generator checkpoint must match "
                f"generator_seqlen<len>_cls<n>_latentdim<d>.ckpt, got {base!r}"
            )
        self._seq_len = int(m.group(1))
        self._cls_num = int(m.group(2))
        self._latent_dim = int(m.group(3))
        self._smin = float(self.scaler.data_min_[0])
        self._smax = float(self.scaler.data_max_[0])

        self._model: Union[Generator, None] = None
        self._device: Union[torch.device, None] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_device"] = None
        return state

    def ensure_model(self) -> None:
        if self._model is not None:
            return
        use_cuda = bool(
            getattr(self.args, "use_gpu", True)
            and torch.cuda.is_available()
            and int(getattr(self.args, "gpu", 0)) >= 0
        )
        self._device = torch.device(
            f"cuda:{int(self.args.gpu)}" if use_cuda else "cpu"
        )
        self._model = Generator(
            self._seq_len,
            self._cls_num,
            self._latent_dim,
            scaler_min=self._smin,
            scaler_max=self._smax,
            is_gpu=use_cuda,
        ).to(self._device)
        state = torch.load(self._ckpt_path, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.eval()

    def sample_gan_bursts(self, c_ind: int) -> np.ndarray:
        assert self._model is not None and self._device is not None
        self._model.eval()
        with torch.no_grad():
            z = np.random.randn(1, self._latent_dim).astype(np.float32)
            z_t = torch.from_numpy(z).to(self._device)
            c = torch.zeros(1, self._cls_num, device=self._device)
            c[:, int(c_ind)] = 1.0
            synthesized = self._model(z_t, c).cpu().numpy()
            synthesized = self.scaler.inverse_transform(synthesized).flatten()
            length = int(synthesized[0])
            synthesized = synthesized[1 : 1 + length].astype(int)
            synthesized = np.trim_zeros(synthesized, trim="b")
            if len(synthesized) % 2 == 1:
                synthesized = synthesized[:-1]
            synthesized[synthesized <= 0] = 1
            sign_arr = np.tile([1, -1], len(synthesized) // 2)
            synthesized = sign_arr * synthesized
        return synthesized

    def sample_until_long_enough(
        self, c_ind: int, min_bursts: int, source_label: int
    ) -> np.ndarray:
        parts: List[np.ndarray] = []
        total = 0
        c = int(c_ind)
        guard = 0
        while total < min_bursts and guard < 200:
            part = self.sample_gan_bursts(c)
            parts.append(part)
            total += len(part)
            if total < min_bursts:
                c = choose_morphed_class(source_label, self.mon_site_num, self.rng)
            guard += 1
        if not parts:
            return np.array([], dtype=int)
        out = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return out[:min_bursts]

    @set_random_seed
    def _simulate(self, data_path: Union[str, os.PathLike]) -> np.ndarray:
        self.ensure_model()

        trace_arr = parse_trace(data_path)
        if trace_arr.ndim != 2 or trace_arr.shape[1] < 2:
            raise ValueError("Trace must be Nx2 (time, direction/size)")

        x, burst_starts = extract_bursts_from_trace(trace_arr)
        if len(x) == 0 or x[0] <= 0:
            return trace_arr.copy()

        fname = os.path.basename(str(data_path)).split(self.args.suffix)[0]
        if "-" in fname:
            label = int(fname.split("-")[0])
        else:
            label = self.mon_site_num

        c_ind = choose_morphed_class(label, self.mon_site_num, self.rng)
        synthesized_x = self.sample_until_long_enough(c_ind, len(x), label)
        while len(synthesized_x) < len(x):
            c_ind = choose_morphed_class(label, self.mon_site_num, self.rng)
            extra = self.sample_gan_bursts(c_ind)
            synthesized_x = np.concatenate((synthesized_x, extra))
        synthesized_x = synthesized_x[: len(x)]

        eps = 1e-6
        t_list: List[float] = []
        d_list: List[int] = []
        for i in range(len(x)):
            sign = 1 if i % 2 == 0 else -1
            base_t = float(burst_starts[i])
            for j in range(int(x[i])):
                t_list.append(base_t + j * eps)
                d_list.append(int(sign * 1))
            syn_mag = abs(int(synthesized_x[i]))
            pad = syn_mag - int(x[i])
            for j in range(max(0, pad)):
                t_list.append(base_t + (x[i] + j) * eps)
                d_list.append(int(sign * self.dummy_code))

        out = np.column_stack(
            [np.asarray(t_list, dtype=np.float64), np.asarray(d_list, dtype=np.int64)]
        )
        if len(out) == 0:
            return trace_arr.copy()
        out = out[np.argsort(out[:, 0], kind="mergesort")]
        out[:, 0] = out[:, 0] - out[0, 0]
        return out

    def trace_normalization(self, trace: np.ndarray) -> np.ndarray:
        arr = np.array(trace, copy=True)
        real = np.abs(arr[:, 1]) != self.dummy_code
        arr[real, 1] = np.sign(arr[real, 1])
        return arr
