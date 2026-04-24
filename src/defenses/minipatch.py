import argparse
import inspect
import os
import time
from typing import Union

import numpy as np
import pandas as pd
import torch
try:
    from scipy.optimize import dual_annealing
except ImportError as _e:  # pragma: no cover
    dual_annealing = None  # type: ignore

# `disp` was added to scipy.optimize.dual_annealing in newer SciPy; omit on older builds.
DUAL_ANNEALING_SUPPORTS_DISP = (
    dual_annealing is not None and 'disp' in inspect.signature(dual_annealing).parameters
)

from defenses.base import Defense
from defenses.config import MinipatchConfig
from utils.general import feature_transform, parse_trace, set_random_seed
from utils.perturb_util import load_attack_model


def perturb_trace(traces, perturbations, highlight=False):
    """
    Perturb packet trace(s) according to the given perturbation(s).
    Not support multiple traces and multiple perturbations at the same time.

    Parameters
    ----------
    traces : array_like
        A 2-D numpy array [Length x 1] for a single trace or a 3-D numpy
        array [N x Length x 1] for N traces.
    perturbations : array_like
        A 1-D numpy array specifying a single perturbation or a 2-D numpy
        array specifying multiple perturbations.
    highlight : optional
        Highlight perturbations by setting the absolute value to 2.
    """
    if type(perturbations) == list:
        perturbations = np.array(perturbations)

    if perturbations.ndim < 2:
        perturbations = np.array([perturbations])

    if traces.ndim < 3 or traces.shape[0] == 1:
        traces = np.tile(traces, [len(perturbations), 1, 1])
    else:
        perturbations = np.tile(perturbations, [len(traces), 1])

    perturbations = perturbations.astype(int)

    for trace, perturbation in zip(traces, perturbations):
        col = trace[:, 0]
        length = int(np.count_nonzero(col))
        patches = perturbation.reshape(-1, 2)

        for patch in patches:
            x_pos, n_pkt = patch
            if x_pos > length:
                x_pos = length
            if x_pos < len(trace):
                while col[x_pos] * n_pkt < 0:
                    x_pos += 1
                    if x_pos >= len(trace):
                        break
            if x_pos < len(trace):
                while col[x_pos] * n_pkt > 0:
                    x_pos += 1
                    if x_pos >= len(trace):
                        break
            patch[0] = x_pos

        positions = []
        for patch in sorted(patches, key=lambda x: x[0], reverse=True):
            x_pos, n_pkt = patch
            direction = 1 if n_pkt > 0 else -1
            n_pkt = abs(n_pkt)
            if x_pos in positions:
                continue
            positions.append(x_pos)
            if x_pos + n_pkt >= len(trace):
                n_pkt = len(trace) - x_pos
            if n_pkt == 0:
                continue
            if x_pos < length:
                assert direction * col[x_pos - 1] > 0

            col[x_pos + n_pkt:] = col[x_pos:-n_pkt]
            if direction > 0:
                if highlight:
                    col[x_pos:x_pos + n_pkt] = 2.0
                else:
                    col[x_pos:x_pos + n_pkt] = 1.0
            else:
                if highlight:
                    col[x_pos:x_pos + n_pkt] = -2.0
                else:
                    col[x_pos:x_pos + n_pkt] = -1.0

    return traces


def patch_length(perturbation: Union[list, np.ndarray]) -> float:
    """Total inserted packet count implied by a flat (position, n_pkt, ...)* vector."""
    arr = np.array(perturbation, dtype=np.int64).ravel()
    if arr.size < 2 or arr.size % 2 != 0:
        return 0.0
    patches = np.split(arr, len(arr) // 2)
    return float(sum(abs(int(p[1])) for p in patches))


def attack_feature_type(attack: str) -> str:
    a = attack.lower()
    mapping = {
        'df': 'df',
        'awf': 'df',
        'tiktok': 'tiktok',
        'rf': 'tam',
        'var_cnn': 'var_cnn',
        'netclr': 'df',
    }
    if a not in mapping:
        raise ValueError(
            f"Minipatch is not wired for attack {attack!r}; supported: {sorted(mapping)}"
        )
    return mapping[a]


class MinipatchDefense(Defense):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = MinipatchConfig(args)
        self.config.load_config()
        # Load lazily in each worker: avoid pickling CUDA modules/tensors for
        # multiprocessing.Pool (fork + CUDA re-init in children raises).
        self._model = None
        self._attack = None
        self._device = None
        self._feature_type = attack_feature_type(args.attack)
        self._predict_cache = {}

    def ensure_attack(self) -> None:
        if self._model is not None:
            return
        model, attack = load_attack_model(self.args)
        self._model = model
        self._model.eval()
        self._attack = attack
        self._device = attack.device

    def verbose_level(self) -> int:
        return 2 if getattr(self.args, 'verbose', False) else 0

    def label_from_path(self, data_path: Union[str, os.PathLike]) -> int:
        base = os.path.basename(str(data_path))
        stem = os.path.splitext(base)[0]
        if '-' in stem:
            return int(stem.split('-')[0])
        return int(self.args.mon_classes)

    def cell_trace_to_direction_buffer(self, trace: np.ndarray) -> np.ndarray:
        """Padded [seq_length, 1] direction buffer (signs), matching DF-style traces."""
        seq_length = self.args.seq_length
        dirs = np.sign(trace[:, 1]).astype(np.float32)
        if len(dirs) > seq_length:
            dirs = dirs[:seq_length]
        buf = np.zeros((seq_length, 1), dtype=np.float32)
        buf[: len(dirs), 0] = dirs
        return buf

    def dirs_nl1_to_batch_features(self, traces_nl1: np.ndarray) -> np.ndarray:
        """Map (N, L, 1) direction arrays to model input batch (N, C, L)."""
        n, seq_length, _ = traces_nl1.shape
        batch = []
        for i in range(n):
            d = traces_nl1[i, :, 0].astype(np.float64)
            t = np.arange(seq_length, dtype=np.float64)
            trace_2d = np.column_stack([t, d])
            feat = feature_transform(
                trace_2d, feature_type=self._feature_type, seq_length=seq_length
            )
            batch.append(feat)
        return np.stack(batch, axis=0)

    def predict_raw(self, traces_nl1: np.ndarray) -> np.ndarray:
        """Model outputs (logits), same as reference minipatch Minipatch._predict."""
        self.ensure_attack()
        x = self.dirs_nl1_to_batch_features(traces_nl1)
        with torch.no_grad():
            t = torch.from_numpy(x).float().to(self._device)
            return self._model(t).cpu().numpy()

    @staticmethod
    def perturbation_key(perturbation) -> tuple:
        arr = np.asarray(perturbation, dtype=np.int64).ravel()
        return tuple(arr.tolist())

    def predict_classes_cached(
        self,
        traces: np.ndarray,
        perturbation,
        tar_class: int,
    ) -> float:
        key = (self.perturbation_key(perturbation), int(tar_class), traces.shape)
        cached = self._predict_cache.get(key)
        if cached is not None:
            return cached
        t = np.copy(traces)
        perturbed_traces = perturb_trace(t, perturbation)
        predictions = self.predict_raw(perturbed_traces)
        score = float(np.mean(predictions[:, tar_class]))
        self._predict_cache[key] = score
        return score

    def perturb_success(
        self,
        traces: np.ndarray,
        trace_ids: np.ndarray,
        perturbation,
        tar_class: int,
        threshold: float,
    ):
        t = np.copy(traces[trace_ids])
        perturbed_traces = perturb_trace(t, perturbation)
        predictions = self.predict_raw(perturbed_traces)
        pred_class = predictions.argmax(axis=-1)
        num_success = int(sum(int(pred) != tar_class for pred in pred_class))
        if num_success >= len(trace_ids) * threshold:
            return True
        return None

    def predict_classes(
        self, traces: np.ndarray, trace_ids: np.ndarray, perturbations, tar_class: int
    ) -> float:
        t = np.copy(traces[trace_ids])
        perturbed_traces = perturb_trace(t, perturbations)
        predictions = self.predict_raw(perturbed_traces)
        return float(np.mean(predictions[:, tar_class]))

    def perturb_website(
        self,
        traces: np.ndarray,
        trace_ids: np.ndarray,
        tar_class: int,
        bounds: dict,
        maxiter: int,
        maxquery: int,
        threshold: float,
        polish: bool,
    ) -> pd.DataFrame:
        traces_sel = np.copy(traces[trace_ids])
        lengths = [int(np.count_nonzero(traces_sel[i, :, 0])) for i in range(len(traces_sel))]
        length_bound = (1, int(np.percentile(lengths, 50)))
        patches = bounds['patches']
        inbound = bounds['inbound']
        outbound = bounds['outbound']
        patch_bound = (-(inbound + 1), outbound + 1)
        perturb_bounds = [length_bound, patch_bound] * patches

        start = time.perf_counter()
        v = self.verbose_level()

        self._predict_cache.clear()

        def objective_func(perturbation):
            return self.predict_classes_cached(traces_sel, perturbation, tar_class)

        def callback_func(perturbation, f, context):
            return self.perturb_success(
                traces, trace_ids, perturbation, tar_class, threshold
            )

        if dual_annealing is None:
            raise ImportError(
                "Minipatch requires SciPy (e.g. pip install scipy) for dual_annealing."
            )

        da_kw = dict(
            maxiter=maxiter,
            maxfun=maxquery,
            initial_temp=self.config.initial_temp,
            restart_temp_ratio=self.config.restart_temp_ratio,
            visit=self.config.visit,
            accept=self.config.accept,
            callback=callback_func,
            no_local_search=not polish,
        )
        if DUAL_ANNEALING_SUPPORTS_DISP and v > 1:
            da_kw['disp'] = True
        perturb_result = dual_annealing(objective_func, perturb_bounds, **da_kw)

        end = time.perf_counter()
        perturbation = perturb_result.x.astype(int).tolist()
        iteration = perturb_result.nit
        execution = perturb_result.nfev
        duration = end - start

        perturbed_traces = perturb_trace(np.copy(traces_sel), perturbation)
        predictions = self.predict_raw(perturbed_traces)

        conf_before = self.predict_raw(np.copy(traces_sel))
        true_prior = conf_before[:, tar_class]
        true_post = predictions[:, tar_class]
        true_diff = true_prior - true_post

        pred_class = predictions.argmax(axis=-1)
        pred_prior = np.array([conf_before[i, pred_class[i]] for i in range(len(trace_ids))])
        pred_post = np.array([predictions[i, pred_class[i]] for i in range(len(trace_ids))])
        pred_diff = pred_post - pred_prior

        success = [int(pred) != tar_class for pred in pred_class]
        num_valid = len(trace_ids)
        num_success = sum(success)
        successful = num_success >= num_valid * threshold

        result = {
            'website': tar_class,
            'trace_ids': trace_ids,
            'lengths': lengths,
            'num_valid': num_valid,
            'num_success': num_success,
            'successful': successful,
            'success': success,
            'patches': patches,
            'inbound': inbound,
            'outbound': outbound,
            'perturbation': perturbation,
            'iteration': iteration,
            'execution': execution,
            'duration': duration,
            'true_class': tar_class,
            'true_prior': true_prior,
            'true_post': true_post,
            'true_diff': true_diff,
            'pred_class': pred_class,
            'pred_prior': pred_prior,
            'pred_post': pred_post,
            'pred_diff': pred_diff,
        }

        if v > 0:
            rate = 100 * num_success / num_valid if num_valid else 0.0
            tag = 'Succeeded' if successful else 'Failed'
            print(
                '%s - rate: %.2f%% (%d/%d) - iter: %d (%d) - time: %.2fs'
                % (tag, rate, num_success, num_valid, iteration, execution, duration)
            )

        return pd.DataFrame([result])

    def adaptive_tuning(
        self,
        traces: np.ndarray,
        trace_ids: np.ndarray,
        tar_class: int,
        bounds: dict,
        maxiter: int,
        maxquery: int,
        threshold: float,
        polish: bool,
    ) -> pd.DataFrame:
        results = None
        trials = 0
        layer_nodes = [bounds]
        v = self.verbose_level()

        while len(layer_nodes) > 0:
            for node in layer_nodes[::-1]:
                trials += 1
                if v > 0:
                    print(
                        'Trial %d - patches: %d - bounds: %d'
                        % (trials, node['patches'], max(node['inbound'], node['outbound'])),
                        end='\n' if v > 1 else '\t',
                    )

                result = self.perturb_website(
                    traces,
                    trace_ids,
                    tar_class,
                    node,
                    maxiter,
                    maxquery,
                    threshold,
                    polish,
                )

                if not bool(result.iloc[0]['successful']):
                    layer_nodes.remove(node)

                if results is None:
                    results = result.reset_index(drop=True)
                else:
                    results = pd.concat([results, result], ignore_index=True)

            children = []
            for node in layer_nodes:
                if node['patches'] > 1:
                    left_child = {
                        'patches': node['patches'] // 2,
                        'inbound': node['inbound'],
                        'outbound': node['outbound'],
                    }
                    if left_child not in children:
                        children.append(left_child)

                if max(node['inbound'], node['outbound']) > 1:
                    right_child = {
                        'patches': node['patches'],
                        'inbound': node['inbound'] // 2,
                        'outbound': node['outbound'] // 2,
                    }
                    if right_child not in children:
                        children.append(right_child)

            layer_nodes = children

        results_df = results if results is not None else pd.DataFrame()
        success = results_df[results_df['successful'] == True]
        if len(success) > 0:
            efficiency = success.apply(
                lambda x: x['num_success'] / max(patch_length(x['perturbation']), 1e-9),
                axis=1,
            )
            best_idx = efficiency.iloc[::-1].idxmax()
        else:
            best_idx = results_df['num_success'].iloc[::-1].idxmax()

        return results_df.loc[best_idx : best_idx]

    def perturbation_to_cell(self, traces: np.ndarray, perturbation: list) -> np.ndarray:
        """Apply optimized perturbation and write .cell rows (index time, direction)."""
        perturbed = perturb_trace(np.copy(traces), perturbation)[0]
        directions = perturbed[:, 0]
        nz = np.where(directions != 0)[0]
        if len(nz) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        last = int(nz[-1]) + 1
        directions = directions[:last]
        timestamps = np.arange(len(directions), dtype=np.float64)
        out = np.column_stack([timestamps, directions.astype(np.int32)])
        return out

    @set_random_seed
    def _simulate(self, data_path: Union[str, os.PathLike]) -> np.ndarray:
        trace = parse_trace(data_path)
        tar_class = self.label_from_path(data_path)
        buf = self.cell_trace_to_direction_buffer(trace)
        traces = buf.reshape(1, self.args.seq_length, 1)
        trace_ids = np.array([0])

        bounds = {
            'patches': self.config.patches,
            'inbound': self.config.inbound,
            'outbound': self.config.outbound,
        }

        if self.config.adaptive:
            result_df = self.adaptive_tuning(
                traces,
                trace_ids,
                tar_class,
                bounds,
                self.config.maxiter,
                self.config.maxquery,
                self.config.threshold,
                self.config.polish,
            )
        else:
            result_df = self.perturb_website(
                traces,
                trace_ids,
                tar_class,
                bounds,
                self.config.maxiter,
                self.config.maxquery,
                self.config.threshold,
                self.config.polish,
            )

        perturbation = result_df.iloc[0]['perturbation']
        return self.perturbation_to_cell(traces, perturbation)
