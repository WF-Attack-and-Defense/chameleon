"""
ALERT website fingerprinting defense (GAN perturbation in burst space).

**Inference (``AlertDefense``)** expects pretrained weights under ``model_dir``:
  - ``generator_site_<class_id>.pt`` — state dict per monitored class (0 .. mon_classes-1)
  - optionally ``df_weights.pth`` — discriminator (not required for trace simulation)

**Training (``AlertDefender``)** — supply a ``loader`` with ``num_classes``, ``config.dataset_choice``,
``load()``, and ``split()`` (see class docstring), or use ``run_defense.py --defense alert``:
by default missing ``generator_site_*.pt`` under ``model_dir`` triggers
``maybe_train_alert_generators`` first (disable with ``--no-alert-auto-train``).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

from defenses.base import Defense
from defenses.config import AlertConfig
from utils.alert import (
    Discriminator,
    Generator,
    convert_burst_row_to_trace_data,
    convert_trace_cell_to_burst,
    trace_to_cell_sequence,
)
from utils.general import parse_trace, set_random_seed

logger = logging.getLogger(__name__)

TQDM_N_COLS = 100

_WFZOO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class AlertTrainingPaths:
    """Directories for ALERT GAN training (discriminator weights, burst cache, loss plots)."""

    discriminator_dir: str
    burst_cache_dir: str
    plot_dir: str
    burst_data_filename: str = "burst_data.npz"

    @property
    def burst_data_path(self) -> str:
        return os.path.join(self.burst_cache_dir, self.burst_data_filename)

    @classmethod
    def from_wfzoo_defaults(cls) -> AlertTrainingPaths:
        base = os.path.join(_WFZOO_ROOT, "checkpoints", "alert")
        return cls(
            discriminator_dir=base,
            burst_cache_dir=os.path.join(base, "burst_cache"),
            plot_dir=os.path.join(base, "plots"),
        )


class _AbstractAlertDefender:
    """Minimal base so ``AlertDefender`` matches the old ``AbstractDefender`` constructor."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader


class NoDefDataSet(torch.utils.data.Dataset):
    """Torch dataset over numpy burst rows and integer labels."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)


class _AlertDataset:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def unwrap(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x, self.y


class _DatasetWrapper:
    def __init__(
        self,
        dataset_choice: str,
        train_data: _AlertDataset,
        test_data: _AlertDataset,
        eval_data: _AlertDataset,
    ) -> None:
        self.config = SimpleNamespace(dataset_choice=dataset_choice)
        self.train_data = train_data
        self.test_data = test_data
        self.eval_data = eval_data

    def summary(self) -> None:
        logger.info(
            "ALERT burst data train=%s test=%s eval=%s",
            self.train_data.x.shape,
            self.test_data.x.shape,
            self.eval_data.x.shape,
        )


class WfzooAlertLoader:
    """Build monitored cell-trace train/test/eval splits from ``run_defense`` file lists."""

    def __init__(
        self,
        flist: np.ndarray,
        labels: np.ndarray,
        mon_classes: int,
        seq_length: int,
        dataset_choice: str,
        seed: int = 42,
    ) -> None:
        self.num_classes = int(mon_classes)
        self.config = SimpleNamespace(dataset_choice=str(dataset_choice))
        self._flist = np.asarray(flist)
        self._labels = np.asarray(labels)
        self._seq_length = int(seq_length)
        self._seed = int(seed)

    def load(self) -> None:
        return

    def split(self) -> _DatasetWrapper:
        mask = self._labels < self.num_classes
        flist_m = self._flist[mask]
        labels_m = self._labels[mask].astype(np.int64, copy=False)
        if flist_m.size == 0:
            raise ValueError(
                "No monitored traces for ALERT training (labels must be < mon_classes)."
            )
        rows: List[np.ndarray] = []
        for p in flist_m:
            rows.append(trace_to_cell_sequence(parse_trace(str(p)), self._seq_length))
        X = np.stack(rows, axis=0).astype(np.float32, copy=False)
        y = labels_m
        rng = np.random.default_rng(self._seed)

        train_idx: List[int] = []
        test_idx: List[int] = []
        eval_idx: List[int] = []
        for c in range(self.num_classes):
            idx = np.where(y == c)[0]
            if idx.size == 0:
                raise ValueError(f"No training samples for monitored class {c}")
            rng.shuffle(idx)
            n = int(idx.size)
            if n < 3:
                raise ValueError(
                    f"ALERT training needs at least 3 traces per monitored class; class {c} has {n}."
                )
            n_train = max(1, min(int(0.7 * n), n - 2))
            remainder = n - n_train
            n_test = max(1, remainder // 2)
            n_eval = remainder - n_test
            if n_eval < 1:
                n_test = max(1, remainder - 1)
                n_eval = remainder - n_test
            train_idx.extend(idx[:n_train].tolist())
            test_idx.extend(idx[n_train : n_train + n_test].tolist())
            eval_idx.extend(idx[n_train + n_test :].tolist())

        tr = np.array(train_idx, dtype=np.int64)
        te = np.array(test_idx, dtype=np.int64)
        ev = np.array(eval_idx, dtype=np.int64)
        return _DatasetWrapper(
            self.config.dataset_choice,
            _AlertDataset(X[tr], y[tr]),
            _AlertDataset(X[te], y[te]),
            _AlertDataset(X[ev], y[ev]),
        )


def convert_trace_data_to_burst(train_x: np.ndarray, max_length: int) -> np.ndarray:
    """Batch cell-trace rows → padded burst tensors (training pipeline; see also ``utils.alert.burst``)."""
    train_burst: List[List[float]] = []
    for trace in tqdm.tqdm(train_x.tolist(), ncols=TQDM_N_COLS, desc="trace to burst"):
        burst: List[float] = []
        i = 0
        while trace[i] == 0:
            i += 1
        tmp_dir, tmp_packet = trace[i], 1
        for j in range(i + 1, len(trace)):
            cur_dir = trace[j]
            if cur_dir != tmp_dir:
                burst.append(tmp_packet * tmp_dir)
                tmp_packet = 1 if cur_dir != 0 else 0
                tmp_dir = cur_dir
            elif cur_dir == 0:
                burst.append(0)
            else:
                tmp_packet += 1
        burst.append(tmp_packet * tmp_dir)
        while len(burst) < max_length:
            burst.append(0)
        train_burst.append(burst[:max_length])

    return np.array(train_burst)


def convert_burst_to_trace_data(burst_data: np.ndarray, max_length: int) -> List[int]:
    trace: List[int] = []
    for burst in burst_data:
        if burst == 0:
            trace.append(0)
            continue
        packet = 1 if burst > 0 else -1
        packets = [packet for _ in range(abs(int(burst)))]
        trace.extend(packets)

    while len(trace) < max_length:
        trace.append(0)

    return trace[:max_length]


def get_random_value_excluding_m(n: int, m: int) -> int:
    numbers = [num for num in range(0, n) if num != m]
    return random.choice(numbers)


def add_noise(
    burst_data: torch.Tensor,
    generator: nn.Module,
    device: torch.device,
    max_length: int,
    noise_mean: float = 0.0,
    noise_std: float = 1.0,
) -> torch.Tensor:
    batch_size = burst_data.shape[0]
    noise = torch.normal(
        mean=noise_mean,
        std=noise_std,
        size=(batch_size, max_length),
        device=device,
    )
    generated_noise = generator(noise)
    sign_generated_noise = torch.where(
        burst_data > 0,
        generated_noise,
        torch.where(burst_data < 0, -generated_noise, 0 * generated_noise),
    )
    return burst_data + sign_generated_noise


def cell_trace_to_timed_array(directions: np.ndarray, duration: float) -> np.ndarray:
    """Map 1D directions to WFZoo ``(time, direction)`` rows."""
    directions = np.asarray(directions, dtype=np.int64).ravel()
    n = int(directions.size)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    d = float(max(duration, 1e-9))
    t = np.linspace(0.0, d, n, dtype=np.float64)
    return np.column_stack([t, directions])


class AlertDefender(_AbstractAlertDefender):
    """
    Train per-site generators and a DF surrogate (discriminator).

    ``loader`` must provide:

    - ``num_classes: int``
    - ``config.dataset_choice: str`` (stored in cached splits)
    - ``load()`` — no return; prepares data if needed
    - ``split()`` — returns an object with ``train_data``, ``test_data``, ``eval_data``,
      each having ``.x`` and ``.y`` numpy arrays (cell traces before burst conversion).
    """

    def __init__(
        self,
        loader: Any,
        paths: Optional[AlertTrainingPaths] = None,
        *,
        max_length: int = 5000,
        seq_length: int = 5000,
        weights_out_dir: Optional[str] = None,
        debug_limit_site_idx: Optional[int] = None,
    ) -> None:
        super().__init__(loader)
        self.paths = paths or AlertTrainingPaths.from_wfzoo_defaults()
        for d in (self.paths.discriminator_dir, self.paths.burst_cache_dir, self.paths.plot_dir):
            os.makedirs(d, exist_ok=True)

        self.batch_size = 64
        self.oh_max_threshold = 0.50
        self.oh_min_threshold = 0.10
        self.loss_alpha, self.loss_beta, self.loss_gamma = 1.0, 1.0, 1.0
        self.max_length = int(max_length)
        self.seq_length = int(seq_length)
        self.num_epochs = 60
        self.noise_param: Tuple[float, float] = (0, 1)
        self.loss_record: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.debug_limit_site_idx = debug_limit_site_idx
        self.weights_out_dir = os.path.normpath(weights_out_dir) if weights_out_dir else None

        self.dataset: Any = None
        self.discriminator: Optional[nn.Module] = None
        self.train_burst: Optional[np.ndarray] = None
        self.device: torch.device = torch.device("cpu")

    @classmethod
    def update_config(cls, config: Any) -> Any:
        config.load_data_by_self = True
        return config

    def defense(self, **kwargs: Any) -> Tuple[Any, None, Any, float]:
        self.oh_min_threshold = kwargs.get("oh_min_threshold", self.oh_min_threshold) + 0.06
        self.oh_max_threshold = kwargs.get("oh_max_threshold", self.oh_max_threshold)

        self.load_burst_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        number_classes = self.loader.num_classes

        train_x, train_y = self.dataset.train_data.unwrap()
        test_x, test_y = self.dataset.test_data.unwrap()
        eval_x, eval_y = self.dataset.eval_data.unwrap()

        self.train_burst = train_x
        logger.info(str(self.train_burst.shape))

        self.discriminator = Discriminator(self.loader.num_classes).to(self.device)
        self.train_discriminator(self.train_burst, train_y)
        logger.info("DF 训练完成，即将训练 Generator")

        dummy_length, real_length = 0, 0
        generated_train_x: List[Any] = []
        generated_train_y: List[int] = []
        generated_test_x: List[Any] = []
        generated_test_y: List[int] = []
        generated_eval_x: List[Any] = []
        generated_eval_y: List[int] = []

        for site_idx, site_o in enumerate(range(number_classes)):
            generator = self.train_generator(site_o)
            if self.weights_out_dir:
                wp = os.path.join(self.weights_out_dir, f"generator_site_{site_o}.pt")
                torch.save(generator.state_dict(), wp)
                logger.info("Saved %s", wp)
            site_burst_data = train_x[train_y == site_o]
            site_burst_test = test_x[test_y == site_o]
            site_burst_eval = eval_x[eval_y == site_o]

            for original_burst, generated_x, generated_y in zip(
                [site_burst_data, site_burst_test, site_burst_eval],
                [generated_train_x, generated_test_x, generated_eval_x],
                [generated_train_y, generated_test_y, generated_eval_y],
            ):
                original_burst_tensor = torch.from_numpy(original_burst).to(self.device).float()
                generated_data_tensor = self.add_noise_training(original_burst_tensor, generator)
                generated_data = generated_data_tensor.detach().cpu().numpy()
                for i, burst_data in enumerate(generated_data):
                    trace = convert_burst_to_trace_data(burst_data, self.seq_length)
                    dummy_length += int(np.sum(np.abs(trace)))
                    generated_x.append(trace)
                generated_y.extend([site_o for _ in range(generated_data.shape[0])])
                for trace in original_burst:
                    for item in trace:
                        real_length += abs(int(item))

            if self.debug_limit_site_idx is not None and site_idx >= self.debug_limit_site_idx:
                break

        self.draw_loss_plot()
        overhead = (dummy_length - real_length) / real_length if real_length else 0.0
        logger.info(" overhead is %s%%", round(overhead * 100, 2))
        return (
            (generated_train_x, generated_eval_x, generated_test_x),
            None,
            (generated_train_y, generated_eval_y, generated_test_y),
            overhead,
        )

    def train_and_save_weights(self) -> None:
        """Train the DF discriminator and each site generator; save weights for ``AlertDefense``."""
        if not self.weights_out_dir:
            raise ValueError("train_and_save_weights requires weights_out_dir set")
        self.load_burst_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_x, train_y = self.dataset.train_data.unwrap()
        self.train_burst = train_x
        logger.info("train_burst shape=%s", self.train_burst.shape)

        self.discriminator = Discriminator(self.loader.num_classes).to(self.device)
        self.train_discriminator(self.train_burst, train_y)
        logger.info("DF surrogate trained; training per-site generators")

        for site_o in range(self.loader.num_classes):
            generator = self.train_generator(site_o)
            path = os.path.join(self.weights_out_dir, f"generator_site_{site_o}.pt")
            torch.save(generator.state_dict(), path)
            logger.info("Saved %s", path)

        self.draw_loss_plot()
        df_src = os.path.join(self.paths.discriminator_dir, "df_weights.pth")
        df_dst = os.path.join(self.weights_out_dir, "df_weights.pth")
        if os.path.isfile(df_src) and os.path.normpath(df_src) != os.path.normpath(df_dst):
            shutil.copy2(df_src, df_dst)
            logger.info("Copied discriminator weights to %s", df_dst)

    def train_generator(self, site_o: int) -> nn.Module:
        assert self.train_burst is not None
        site_burst_data = self.train_burst[self.dataset.train_data.y == site_o]

        num_candidates = self.loader.num_classes
        if self.debug_limit_site_idx is not None:
            num_candidates = self.debug_limit_site_idx + 1
        target_site_t = get_random_value_excluding_m(num_candidates, site_o)
        logger.info("current site: %s, target_site: %s", site_o, target_site_t)
        target_burst_data = self.train_burst[self.dataset.train_data.y == target_site_t]

        train_loader = DataLoader(
            NoDefDataSet(
                site_burst_data,
                self.dataset.train_data.y[self.dataset.train_data.y == site_o],
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        generator = Generator(self.max_length, self.max_length).to(self.device)
        optimizer = optim.Adam(generator.parameters(), lr=1e-5)

        pbr = tqdm.tqdm(
            ncols=TQDM_N_COLS + 20,
            desc=f"Site: {site_o}, Epoch",
            position=0,
            total=self.num_epochs,
            ascii=True,
            leave=True,
        )
        batch_pbr = tqdm.tqdm(
            ncols=TQDM_N_COLS + 20,
            desc="Batch",
            position=1,
            total=len(train_loader),
            ascii=True,
            leave=False,
        )

        for _epoch in range(self.num_epochs):
            generator.train()
            sample = random.sample(list(np.arange(target_burst_data.shape[0])), self.batch_size)
            _ = torch.from_numpy(target_burst_data[sample]).to(self.device).float()

            loss_sim = torch.zeros((), device=self.device)
            loss_adv = torch.zeros((), device=self.device)
            loss_per = torch.zeros((), device=self.device)
            loss = torch.zeros((), device=self.device)

            for site_sample_batch, _ in train_loader:
                batch_size = site_sample_batch.shape[0]
                if batch_size != self.batch_size:
                    batch_pbr.update(1)
                    continue

                site_sample_batch = site_sample_batch.to(self.device)
                site_perturbation = self.add_noise_training(site_sample_batch, generator)

                assert self.discriminator is not None
                _, predict_site = self.discriminator(torch.unsqueeze(site_perturbation, 1))
                loss_sim = F.cross_entropy(
                    predict_site, torch.tensor([target_site_t] * batch_size).to(self.device)
                ) / 11

                _, site_batch_to_target = self.discriminator(torch.unsqueeze(site_perturbation, 1))
                site_batch_to_target_norm = F.normalize(site_batch_to_target, p=2, dim=1)
                perturbation_at_o = site_batch_to_target_norm[:, site_o : site_o + 1]
                site_batch_expect_site_o = torch.cat(
                    [
                        site_batch_to_target_norm[:, :site_o],
                        site_batch_to_target_norm[:, site_o + 1 :],
                    ],
                    dim=1,
                )
                max_except_o = torch.max(site_batch_expect_site_o, dim=1).values

                adversarial = perturbation_at_o - max_except_o
                adversarial_result = torch.max(adversarial, torch.full_like(adversarial, 0)).to(
                    self.device
                )
                loss_adv = adversarial_result.mean()

                perturbed_bandwidth = torch.norm(site_perturbation, p=1, dim=1).to(self.device)
                site_bandwidth = torch.norm(site_sample_batch, p=1, dim=1).to(self.device)
                overhead = (perturbed_bandwidth - site_bandwidth) / site_bandwidth

                overhead_greater = overhead - self.oh_max_threshold
                loss_per1 = torch.where(
                    overhead_greater < 0, torch.tensor(0.0).to(self.device), overhead_greater
                ).mean()

                overhead_less = self.oh_min_threshold - overhead
                loss_per2 = torch.where(
                    overhead_less < 0, torch.tensor(0.0).to(self.device), overhead_less
                ).mean()

                loss_per = loss_per1 + loss_per2

                loss = (
                    self.loss_alpha * loss_sim
                    + self.loss_beta * loss_adv
                    + self.loss_gamma * loss_per
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_pbr.update(1)
                batch_pbr.set_postfix(
                    dict(
                        loss_sim=round(loss_sim.item(), 3),
                        loss_adv=round(loss_adv.item(), 3),
                        loss_per=round(loss_per.item(), 3),
                    )
                )

            batch_pbr.reset()
            pbr.update(1)
            pbr.set_postfix(dict(loss=round(loss.item(), 3)))
            self.record_loss_value(
                site_o,
                [loss_sim, loss_adv, loss_per, loss],
            )

        batch_pbr.close()
        pbr.close()
        print("")

        return generator

    def train_discriminator(
        self, x_train: np.ndarray, y_train: np.ndarray, load_model: bool = True
    ) -> None:
        assert self.discriminator is not None
        lr = 0.01
        momentum = 0.5
        batch_size = 64
        epochs = 30
        weight_decay = 1e-7

        weight_path = os.path.join(self.paths.discriminator_dir, "df_weights.pth")
        if load_model and os.path.isfile(weight_path):
            self.discriminator.load_state_dict(torch.load(weight_path, map_location=self.device))
            logger.info("加载 discriminator 模型权重，跳过训练")
            return

        train_loader = DataLoader(
            NoDefDataSet(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        opt = optim.SGD(
            [{"params": self.discriminator.parameters()}],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        pbr = tqdm.tqdm(
            ncols=TQDM_N_COLS,
            desc="Epoch",
            position=0,
            total=epochs,
            ascii=True,
            leave=True,
        )
        batch_pbr = tqdm.tqdm(
            ncols=TQDM_N_COLS,
            desc="Batch",
            position=1,
            total=len(train_loader),
            ascii=True,
            leave=False,
        )

        for epoch in range(epochs):
            self.discriminator.train()
            total_loss = 0.0
            correct = 0
            predict_list: List[Any] = []
            label_list: List[Any] = []
            for batch_id, (flow, target) in enumerate(train_loader):

                flow = flow.to(self.device)
                target = target.to(self.device)
                flow, target = Variable(flow), Variable(target)
                flow = flow.unsqueeze(1)

                opt.zero_grad()
                _, prediction = self.discriminator(flow)
                if batch_id == 0:
                    logger.info("prediction shape: %s", prediction.shape)
                    logger.info("target shape: %s", target.shape)
                loss = F.cross_entropy(prediction, target.long())
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += target.eq(pred.view_as(target)).cpu().sum().item()
                if epoch == epochs - 1:
                    predict_list += prediction.cpu().data.numpy().tolist()
                    label_list += target.cpu().data.numpy().tolist()
                batch_pbr.update()

            loss_rounded = round(total_loss, 2)
            acc = round(correct * 100 / len(train_loader.dataset), 2)
            pbr.update()
            pbr.set_description(f"loss: {loss_rounded}, acc: {acc}%")
            batch_pbr.reset()

        pbr.close()
        batch_pbr.close()
        print("")
        torch.save(self.discriminator.state_dict(), weight_path)

    def add_noise_training(self, burst_data: torch.Tensor, generator: nn.Module) -> torch.Tensor:
        mean_noise, std_noise = self.noise_param
        return add_noise(burst_data, generator, self.device, self.max_length, mean_noise, std_noise)

    def load_burst_data(self) -> None:
        burst_data_path = self.paths.burst_data_path
        if os.path.isfile(burst_data_path):
            logger.info("Loading cached burst data %s", burst_data_path)
            burst_data = np.load(burst_data_path)
            train_data = _AlertDataset(burst_data["train_x"], burst_data["train_y"])
            test_data = _AlertDataset(burst_data["test_x"], burst_data["test_y"])
            eval_data = _AlertDataset(burst_data["eval_x"], burst_data["eval_y"])
            self.dataset = _DatasetWrapper(
                self.loader.config.dataset_choice,
                train_data,
                test_data,
                eval_data,
            )
            self.dataset.summary()
        else:
            self.loader.load()
            dataset = self.loader.split()
            dataset.train_data.x = convert_trace_data_to_burst(dataset.train_data.x, self.max_length)
            dataset.test_data.x = convert_trace_data_to_burst(dataset.test_data.x, self.max_length)
            dataset.eval_data.x = convert_trace_data_to_burst(dataset.eval_data.x, self.max_length)
            self.dataset = dataset
            np.savez(
                burst_data_path,
                train_x=dataset.train_data.x,
                train_y=dataset.train_data.y,
                test_x=dataset.test_data.x,
                test_y=dataset.test_data.y,
                eval_x=dataset.eval_data.x,
                eval_y=dataset.eval_data.y,
            )
            logger.info("Wrote burst cache %s", burst_data_path)

    def record_loss_value(self, site: int, loss_list: List[torch.Tensor]) -> None:
        for loss_type, loss_value in zip(
            ["loss_sim", "loss_adv", "loss_per", "loss"],
            loss_list,
        ):
            self.loss_record[site][loss_type].append(round(loss_value.item(), 3))

    def draw_loss_plot(self) -> None:
        import matplotlib.pyplot as plt

        os.makedirs(self.paths.plot_dir, exist_ok=True)
        for site_o, loss_dic in self.loss_record.items():
            _, ax = plt.subplots()

            for loss_type, loss_value in loss_dic.items():
                avg_loss: List[float] = []
                total_loss = 0.0
                for item in loss_value:
                    total_loss += item
                    avg_loss.append(total_loss / (len(avg_loss) + 1))
                ax.plot(avg_loss, label=loss_type)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()

            pic_path = os.path.join(self.paths.plot_dir, f"{site_o}.png")
            plt.savefig(pic_path)
            plt.close()


def alert_generator_weights_complete(model_dir: str, mon_classes: int) -> bool:
    """Return True if every monitored class has a ``generator_site_`` or ``generator_`` checkpoint."""
    root = os.path.normpath(model_dir)
    for i in range(int(mon_classes)):
        p1 = os.path.join(root, f"generator_site_{i}.pt")
        p2 = os.path.join(root, f"generator_{i}.pt")
        if not (os.path.isfile(p1) or os.path.isfile(p2)):
            return False
    return True


def maybe_train_alert_generators(
    args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray
) -> None:
    """
    Train ALERT burst generators when weights are missing (unless disabled).

    Uses monitored traces from ``flist``/``labels``, ``args.seq_length``, and
    ``max_length`` from ``alert.ini``. Saves ``generator_site_<k>.pt`` under
    ``model_dir`` from the same config ``AlertDefense`` uses.
    """
    cfg = AlertConfig(args)
    cfg.load_config()
    model_dir = os.path.normpath(str(cfg.model_dir))

    force = getattr(args, "alert_force_train", False)
    if not force and getattr(args, "no_alert_auto_train", False):
        return
    if not force and alert_generator_weights_complete(model_dir, args.mon_classes):
        logger.info(
            "ALERT: all generator weights present under %s; skipping training.",
            model_dir,
        )
        return

    os.makedirs(model_dir, exist_ok=True)
    burst_fn = f"burst_{args.dataset}_sl{args.seq_length}.npz"
    paths = AlertTrainingPaths(
        discriminator_dir=model_dir,
        burst_cache_dir=os.path.join(model_dir, ".alert_burst_cache", args.dataset),
        plot_dir=os.path.join(model_dir, ".alert_plots", args.dataset),
        burst_data_filename=burst_fn,
    )
    for d in (paths.discriminator_dir, paths.burst_cache_dir, paths.plot_dir):
        os.makedirs(d, exist_ok=True)

    loader = WfzooAlertLoader(
        flist,
        labels,
        mon_classes=args.mon_classes,
        seq_length=args.seq_length,
        dataset_choice=args.dataset,
    )
    defender = AlertDefender(
        loader,
        paths=paths,
        max_length=int(cfg.max_length),
        seq_length=int(args.seq_length),
        weights_out_dir=model_dir,
        debug_limit_site_idx=None,
    )
    logger.info(
        "ALERT: training burst generators under %s (long-running); then defense will run.",
        model_dir,
    )
    defender.train_and_save_weights()


class AlertDefense(Defense):
    """Per-trace burst perturbation using pretrained per-class generators."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = AlertConfig(args)
        self.config.load_config()
        self.model_dir = os.path.normpath(str(self.config.model_dir))
        self.max_length = int(self.config.max_length)
        self.seq_length = int(getattr(args, "seq_length", 5000))
        self.mon_classes = int(args.mon_classes)
        self.rng = np.random.default_rng(2024)

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            self.logger.warning(
                "Created ALERT model_dir %s (no generators yet — run with auto-train or train separately).",
                self.model_dir,
            )

        self._device: Optional[torch.device] = None
        self._generators: Dict[int, nn.Module] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_generators"] = {}
        state["_device"] = None
        return state

    def ensure_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        # ``run_defense`` uses ``multiprocessing.Pool`` (fork). If the main process used
        # CUDA (e.g. ``maybe_train_alert_generators``), forked workers must not use CUDA.
        if mp.current_process().name != "MainProcess":
            self._device = torch.device("cpu")
            return self._device
        use_cuda = bool(
            getattr(self.args, "use_gpu", True)
            and torch.cuda.is_available()
            and int(getattr(self.args, "gpu", 0)) >= 0
        )
        self._device = torch.device(
            f"cuda:{int(self.args.gpu)}" if use_cuda else "cpu"
        )
        return self._device

    def generator_path(self, site: int) -> str:
        p1 = os.path.join(self.model_dir, f"generator_site_{site}.pt")
        if os.path.isfile(p1):
            return p1
        p2 = os.path.join(self.model_dir, f"generator_{site}.pt")
        return p2

    def load_generator(self, site: int) -> Optional[nn.Module]:
        if site in self._generators:
            return self._generators[site]
        path = self.generator_path(site)
        if not os.path.isfile(path):
            self.logger.warning(
                "No generator weights for site %s (tried %s). Returning unmodified burst.",
                site,
                path,
            )
            return None
        device = self.ensure_device()
        gen = Generator(self.max_length, self.max_length).to(device)
        state = torch.load(path, map_location=device)
        gen.load_state_dict(state)
        gen.eval()
        self._generators[site] = gen
        return gen

    def site_index_from_path(self, data_path: Union[str, os.PathLike]) -> int:
        fname = os.path.basename(str(data_path)).split(self.args.suffix)[0]
        if "-" in fname:
            label = int(fname.split("-")[0])
        else:
            label = self.mon_classes
        if label >= self.mon_classes:
            return int(self.rng.integers(0, self.mon_classes))
        return int(label)

    @set_random_seed
    def _simulate(self, data_path: Union[str, os.PathLike]) -> np.ndarray:
        trace_arr = parse_trace(data_path)
        if trace_arr.ndim != 2 or trace_arr.shape[1] < 2 or len(trace_arr) == 0:
            return trace_arr.copy()

        duration = float(trace_arr[-1, 0] - trace_arr[0, 0])
        site = self.site_index_from_path(data_path)
        gen = self.load_generator(site)
        if gen is None:
            return trace_arr.copy()

        cells = trace_to_cell_sequence(trace_arr, self.seq_length)
        burst = convert_trace_cell_to_burst(cells, self.max_length)
        device = self.ensure_device()
        burst_t = torch.from_numpy(burst).unsqueeze(0).to(device).float()
        with torch.no_grad():
            perturbed = add_noise(burst_t, gen, device, self.max_length)
        burst_out = np.round(perturbed.squeeze(0).detach().cpu().numpy()).astype(np.float32)
        dirs = convert_burst_row_to_trace_data(burst_out, self.seq_length)
        nonzero = np.flatnonzero(dirs != 0)
        if nonzero.size == 0:
            return trace_arr.copy()
        dirs = dirs[nonzero[0] : nonzero[-1] + 1]
        return cell_trace_to_timed_array(dirs, duration)


__all__ = [
    "AlertDefense",
    "AlertDefender",
    "AlertTrainingPaths",
    "NoDefDataSet",
    "WfzooAlertLoader",
    "add_noise",
    "alert_generator_weights_complete",
    "convert_burst_to_trace_data",
    "convert_trace_data_to_burst",
    "get_random_value_excluding_m",
    "maybe_train_alert_generators",
]
