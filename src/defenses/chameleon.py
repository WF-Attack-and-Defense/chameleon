import argparse
import atexit
import csv
import json
import multiprocessing
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from defenses.base import Defense
from defenses.config import ChameleonConfig
from utils.general import parse_trace, set_random_seed
from utils.chameleon.predataprocessing import (
    traces_selection,
    normalized_cross_correlation,
)
from utils.chameleon.radixTrie import RadixTrie
from utils.general import get_all_mon_flist_label, parse_all_mon_trace

def pack_directions(raw_dir: Union[np.ndarray, list]) -> np.ndarray:
    """
    Map a raw direction column (or 1D trace) to strictly ``1`` or ``-1`` only.
    ``np.sign(0)`` is 0; zeros are treated as ``1`` so downstream models/trie
    see a binary alphabet.
    """
    v = np.asarray(raw_dir, dtype=float).ravel()
    s = np.sign(v).astype(np.int64)
    s[s == 0] = 1
    return s.astype(int)


class ChameleonDefense(Defense):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = ChameleonConfig(args)
        self.config.load_config()
        file_list, data_labels = get_all_mon_flist_label(
            args.mon_path,
            mon_cls=args.mon_classes,
            mon_inst=args.mon_inst,
            suffix=args.suffix,
        )
        data_trace = parse_all_mon_trace(file_list)

        self.data_traces = np.asarray(data_trace, dtype=object)
        data_labels = np.asarray(data_labels, dtype=int)

        selected_traces, idx_set = traces_selection(
            self.data_traces,
            data_labels,
            k=self.config.selection_k,
            select_ratio=self.config.selection_ratio,
            min_select=self.config.selection_min,
            alpha=self.config.selection_alpha,
            beta=self.config.selection_beta,
            gamma=self.config.selection_gamma,
            seq_len=self.config.selection_seq_len,
        )

        # Save to a npz file for evaluation

        _root = Path(__file__).resolve().parents[2]
        _chameleon_dir = _root / "defense_results" / "chameleon"
        _chameleon_dir.mkdir(parents=True, exist_ok=True)
        # _npz_path = _chameleon_dir / f"selected_traces_and_idx_set_{args.dataset}.npz"
        # np.savez_compressed(_npz_path, selected_traces=selected_traces, idx_set=idx_set)
        # self.logger.info("Saved selected traces and idx_set to %s", _npz_path)


        # save selected_traces and idx_set to a json file
        # _json_path = _chameleon_dir / f"selected_traces_and_idx_set_{args.dataset}.json"
        # payload = {
        #     "selected_traces": [
        #         np.asarray(tr, dtype=float).tolist() for tr in selected_traces
        #     ],
        #     "idx_set": np.asarray(idx_set, dtype=int).tolist(),
        # }
        # with open(_json_path, "w", encoding="utf-8") as f:
        #     json.dump(payload, f)
        # self.logger.info("Saved selected traces and idx_set to %s", _json_path)

        exit()

        idx_arr = np.asarray(idx_set, dtype=int)
        self.traces_idx: Dict[int, np.ndarray] = {
            int(idx): selected_traces[i] for i, idx in enumerate(idx_arr)
        }

        # # Each trace as 1D directions in {1, -1} only
        self.direction_traces = np.array(
            [
                pack_directions(np.asarray(tr)[:, 1])
                if np.asarray(tr).ndim == 2
                else pack_directions(tr)
                for tr in selected_traces
            ],
            dtype=object,
        )

        self.grouped_traces, self.grouped_indices = normalized_cross_correlation(
            direction_traces=self.direction_traces,
            idx_set=idx_set,
            mon_inst=args.mon_inst,
            trace_threshold=self.config.trace_threshold,
            selection_k=self.config.selection_k,
            corr_threshold=0.80,   # tune if needed
            vec_len=self.config.selection_seq_len,
        )

        #

        self.grouped_index_sets = [set(map(int, g)) for g in self.grouped_indices]
        self.selected_data_labels = data_labels[idx_set]
        self.selected_traces = np.asarray(selected_traces, dtype=object)
        self.selected_idx_set = np.asarray(idx_set, dtype=int)
        self.radix_trie: RadixTrie | None = None

        # self.plot_radix_trie_performance_3d()

    # def __getstate__(self) -> Dict[str, object]:
    #     # Multiprocessing pickles the defense; a built RadixTrie is a deep TrieNode
    #     # tree and can exceed pickle's recursion limit. Workers rebuild via
    #     # ensure_radix_trie().
    #     state = self.__dict__.copy()
    #     state["radix_trie"] = None
    #     return state

    # def __setstate__(self, state: Dict[str, object]) -> None:
    #     self.__dict__.update(state)

    def ensure_radix_trie(self) -> None:
        """
        Build radix trie from direction-only traces, truncated to
        ``radix_trie_build_length`` for prefix matching.
        """
        if self.radix_trie is not None:
            return
        cap = int(self.config.radix_trie_build_length)
        n = len(self.direction_traces)
        truncated = np.empty(n, dtype=object)
        for i in range(n):
            truncated[i] = np.asarray(self.direction_traces[i], dtype=int)[:cap]
        self.radix_trie = RadixTrie(truncated, self.selected_data_labels)

    # def plot_radix_trie_performance_3d(
    #     self,
    #     data_traces: Optional[np.ndarray] = None,
    #     output_path: Optional[Union[str, os.PathLike]] = None,
    #     match_threshold: int = 5,
    #     max_data_traces: Optional[int] = 1024,
    # ) -> None:
    #     """
    #     Visualize radix-trie prefix ambiguity on original monitored traces.

    #     For each index ``i`` (prefix length ``i + 1``, capped by
    #     ``radix_trie_build_length``), query the trie built from
    #     ``self.direction_traces`` and count how many distinct pool traces extend
    #     the prefix. If that count is below ``match_threshold``, the prefix is
    #     "sparse". The z coordinate is ``m``: over all data traces, how many
    #     traces are sparse at index ``i``. Each sparse cell contributes one 3D
    #     point ``(i, T[i, 0], m)``.

    #     Parameters
    #     ----------
    #     data_traces
    #         Traces to analyze; defaults to ``self.data_traces`` (all monitored
    #         originals). Pass a subset to speed up experiments.
    #     output_path
    #         Where to save the PNG. Default:
    #         ``defense_results/chameleon/radix_trie_performance_3d.png`` under
    #         the project ``src`` parent.
    #     match_threshold
    #         Sparse if the number of matched direction traces is strictly less
    #         than this value (default 5).
    #     max_data_traces
    #         If set, only the first this many rows of ``data_traces`` are used
    #         (default 1024). Use ``None`` for all traces (can be slow).
    #     """
    #     # PNG plotting needs matplotlib; CSV export does not.
    #     plt = None
    #     try:
    #         import matplotlib

    #         matplotlib.use("Agg")
    #         import matplotlib.pyplot as plt
    #     except ImportError:
    #         pass

    #     self.ensure_radix_trie()
    #     assert self.radix_trie is not None

    #     pool = self.data_traces if data_traces is None else data_traces
    #     pool = np.asarray(pool, dtype=object)
    #     if max_data_traces is not None and int(max_data_traces) > 0:
    #         pool = pool[: int(max_data_traces)]

    #     cap = int(self.config.radix_trie_build_length)

    #     def _uniq_match_count(directions_prefix: np.ndarray) -> int:
    #         pairs = self.radix_trie.trace_match(directions_prefix)
    #         return len({int(tid) for _, tid in pairs if int(tid) >= 0})

    #     sparse_traces_at_i: Dict[int, int] = defaultdict(int)
    #     for tr in pool:
    #         T = np.asarray(tr, dtype=float)
    #         if T.ndim != 2 or T.shape[0] < 1 or T.shape[1] < 2:
    #             continue
    #         dirs = pack_directions(T[:, 1])
    #         n = min(int(T.shape[0]), int(dirs.size), cap)
    #         for i in range(n):
    #             pref = dirs[: i + 1]
    #             if _uniq_match_count(pref) < match_threshold:
    #                 sparse_traces_at_i[i] += 1

    #     xs: List[float] = []
    #     ys: List[float] = []
    #     zs: List[float] = []
    #     for tr in pool:
    #         T = np.asarray(tr, dtype=float)
    #         if T.ndim != 2 or T.shape[0] < 1 or T.shape[1] < 2:
    #             continue
    #         dirs = pack_directions(T[:, 1])
    #         n = min(int(T.shape[0]), int(dirs.size), cap)
    #         for i in range(n):
    #             pref = dirs[: i + 1]
    #             if _uniq_match_count(pref) < match_threshold:
    #                 m_i = sparse_traces_at_i[i]
    #                 xs.append(float(i))
    #                 ys.append(float(T[i, 0])/100)
    #                 zs.append(float(m_i))


    #     root = Path(__file__).resolve().parents[2]
    #     save_path = (
    #         root
    #         / "defense_results"
    #         / "chameleon"
    #         / f"radix_trie_performance_3d_{self.args.dataset}_CW.csv"
    #     )
    #     save_path.parent.mkdir(parents=True, exist_ok=True)
    #     with open(save_path, "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["x", "y", "z"])
    #         for x, y, z in zip(xs, ys, zs):
    #             writer.writerow([x, y, z])
    #     exit()

        # if not xs:
        #     return

        # if output_path is None:
        #     root = Path(__file__).resolve().parents[2]
        #     out = root / "defense_results" / "chameleon" / "radix_trie_performance_3d.png"
        # else:
        #     out = Path(output_path)
        # out.parent.mkdir(parents=True, exist_ok=True)

        # fig = plt.figure(figsize=(9, 6))
        # ax = fig.add_subplot(projection="3d")
        # sc = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", alpha=0.55, s=8)
        # label_kw = {"fontsize": 18, "labelpad": 14}
        # ax.set_xlabel("Prefix Matched Location", **label_kw)
        # ax.set_ylabel("Timestamp (s)", **label_kw)
        # ax.set_zlabel("Number of traces", **label_kw)
        # tick_kw = {"labelsize": 16, "pad": 8}
        # ax.tick_params(axis="x", **tick_kw)
        # ax.tick_params(axis="y", **tick_kw)
        # ax.tick_params(axis="z", **tick_kw)
        # fig.colorbar(sc, ax=ax, shrink=0.55)
        # fig.tight_layout()
        # fig.savefig(out, dpi=150)
        # plt.close(fig)
        # exit()

  

    @set_random_seed
    def _simulate(self, data_path: Union[str, os.PathLike]) -> np.ndarray:
        in_trace = parse_trace(data_path)

        while np.sign(in_trace[0, 1]) == -1:
            in_trace = in_trace[1:]
        in_trace = np.asarray(in_trace, dtype=float)

        if in_trace[0, 0] != 0.0:
            for i in range(len(in_trace)):
                in_trace[i, 0] = in_trace[i, 0] - in_trace[0, 0]

        morphed_trace = self.morphing(in_trace)

        morphed_trace[:, 1] = pack_directions(morphed_trace[:, 1])

        return morphed_trace

        # return morphed_trace

    def closed_morphing_trace(self, trace_arr: np.ndarray, L: int) -> List[int]:
        """
        Fallback: pick a reference direction sequence from the training pool with
        minimum Hamming distance to the input prefix (length L).
        """
        arr = np.asarray(trace_arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return []
        in_dirs = pack_directions(arr[:, 1])
        prefix_len = int(max(1, min(L, in_dirs.size)))
        in_prefix = in_dirs[:prefix_len]
        best: Optional[np.ndarray] = None
        best_dist = np.inf
        best_len = np.inf
        for ref in np.asarray(self.direction_traces, dtype=object):
            cand = np.asarray(ref, dtype=int).ravel()
            if cand.size == 0:
                continue
            common = int(min(prefix_len, cand.size))
            if common <= 0:
                continue
            dist = int(np.sum(in_prefix[:common] != cand[:common]))
            if dist < best_dist or (dist == best_dist and cand.size < best_len):
                best_dist = float(dist)
                best_len = float(cand.size)
                best = cand
        if best is None:
            return []
        return best.astype(np.int64).tolist()

    def finalize_morphing_trace(self, morphed_trace: np.ndarray) -> np.ndarray:
        arr = np.asarray(morphed_trace, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return arr
        return arr.copy()

    def morphing(self, trace: Union[np.ndarray, list]) -> np.ndarray:

        self.ensure_radix_trie()

        trace_arr = np.asarray(trace, dtype=float)
        if trace_arr.ndim == 1:
            raise ValueError("Trace must be a 2D array with time and direction")

        directions = pack_directions(trace_arr[:, 1])

        n = len(directions)

        if n < 2:
            return self.finalize_morphing_trace(trace_arr.copy())

        mixed: List[List[float, int]] = trace_arr[:1, :].astype(float).tolist()

        
        buffer_trace: deque[int] = deque()


        # Cache trie matches for each prefix length (all matched trace indices).
        match_cache: Dict[int, List[int]] = {}

        def trace_match_cached(prefix_len: int) -> List[int]:
            if prefix_len in match_cache:
                return match_cache[prefix_len]
            assert self.radix_trie is not None
            pairs = self.radix_trie.trace_match(directions[:prefix_len])
            idxs = [int(trace_idx) for _, trace_idx in pairs if int(trace_idx) >= 0]
            match_cache[prefix_len] = idxs
            return idxs

        def mutation_check(target_ref: np.ndarray, direction: int, L: int) -> int:
            end = min(int(target_ref.shape[0]), L + int(self.config.mutation_length))
            for i in range(L + 1, end, 1):
                if target_ref[i, 1] == direction:
                    return i
            return -1

        def optimize_mutation_location(target_ref: np.ndarray, in_dirs: np.ndarray, loc: int, L: int) -> int:
            """
            Pick the best index in [loc, loc + mutation_length] to flip in target_ref.
            Objective:
            - reduce local mismatch with input directions (lower overhead),
            - prefer earlier flips (smaller shift from loc),
            - avoid creating abrupt local boundaries.
            """
            nref  = int(target_ref.shape[0])
            in_arr = np.asarray(in_dirs, dtype=float)
            if in_arr.ndim == 2 and in_arr.shape[1] >= 2:
                in_dir_1d = pack_directions(in_arr[:, 1])
            else:
                in_dir_1d = pack_directions(in_arr)
            n_in = int(in_dir_1d.size)
            if nref <= 0:
                return max(0, loc)

            left = max(0, int(loc))
            right = min(nref - 1, int(loc) + int(self.config.mutation_length))
            if left > right:
                return max(0, min(nref - 1, left))

            w_right = min(right, n_in - 1)
            if w_right < left:
                return left

            ref_dirs = pack_directions(target_ref[:, 1])
            in_window = in_dir_1d[left : w_right + 1]
            ref_window = ref_dirs[left : w_right + 1]

            best_idx = left
            best_score = float("inf")

            for cand in range(left, w_right + 1):
                local = ref_window.copy()
                local[cand - left] *= -1

                mismatch = int(np.sum(local != in_window))
                # Favor lower added latency/overhead by mutating earlier.
                distance_penalty = abs(cand - int(loc))

                # Preserve local smoothness with neighbors.
                continuity_penalty = 0
                if cand - 1 >= 0:
                    continuity_penalty += int(local[cand - 1 - left] != local[cand - left]) if cand - 1 >= left else int(ref_dirs[cand - 1] != local[cand - left])
                if cand + 1 < nref:
                    continuity_penalty += int(local[cand + 1 - left] != local[cand - left]) if cand + 1 <= w_right else int(ref_dirs[cand + 1] != local[cand - left])

                score = mismatch + 0.25 * distance_penalty + 0.5 * continuity_penalty
                if score < best_score:
                    best_score = score
                    best_idx = cand

            return int(best_idx)

        def separability_score(candidate_dirs: np.ndarray) -> float:
            """
            Lower score means lower separability (more similar to selected pool).
            """
            cand = pack_directions(candidate_dirs)
            if cand.size == 0:
                return float("inf")
            dists: List[float] = []
            for ref in np.asarray(self.direction_traces, dtype=object):
                ref_dirs = np.asarray(ref, dtype=int).ravel()
                common = min(int(cand.size), int(ref_dirs.size))
                if common <= 0:
                    continue
                dist = float(np.mean(cand[:common] != ref_dirs[:common]))
                dists.append(dist)
            if not dists:
                return float("inf")
            # Focus on nearest neighbors to preserve low-separability behavior.
            k = min(5, len(dists))
            dists.sort()
            return float(np.mean(dists[:k]))

        def trace_morphing(trace_arr: np.ndarray, target_ref: np.ndarray, L) -> np.ndarray:
            """
            Morph the trace_arr to the target_ref.
            """

            if target_ref.shape[0] < trace_arr.shape[0]:
                trace_len = target_ref.shape[0]
            else:
                trace_len = trace_arr.shape[0]
            # if trace_arr[L, 0] < target_ref[L, 0]:
            #     delay = 0
            # else:
            #     delay = 1.0

            delay = 0.0
            
            target_ref_loc = 2
            target_ref[0, 0] = float(trace_arr[0, 0])
            target_ref[1, 0] = target_ref[1, 0] + delay
            
            for L in range(2,trace_len):
                if target_ref_loc+1 >= target_ref.shape[0]:
                    break
                while target_ref_loc < target_ref.shape[0] and (target_ref[target_ref_loc-1, 0]) < float(trace_arr[L, 0]):
                    # if (
                    #     target_ref_loc + 1 < target_ref.shape[0]
                    #     and (target_ref[target_ref_loc+1, 0] - target_ref[target_ref_loc, 0]) > 1
                    # ):
                #         delay = delay - 1.0;
                    target_ref[target_ref_loc, 0] = target_ref[target_ref_loc, 0] + delay
                    target_ref_loc += 1
                # The scan can advance target_ref_loc to len(target_ref); only then compare.
                if target_ref_loc >= target_ref.shape[0]:
                    break
                if np.sign(target_ref[target_ref_loc, 1]) == np.sign(trace_arr[L, 1]):
                    if (
                        target_ref_loc + 1 < target_ref.shape[0]
                        and (target_ref[target_ref_loc+1, 0] - target_ref[target_ref_loc, 0]) > 1
                    ):
                        delay = delay - 1.0;
                    target_ref[target_ref_loc, 0] = target_ref[target_ref_loc, 0] + delay
                    target_ref_loc += 1
                else:
                    while (
                        target_ref_loc < target_ref.shape[0]
                        and np.sign(target_ref[target_ref_loc, 1]) != np.sign(trace_arr[L, 1])
                    ):
                        if (
                            target_ref_loc + 1 < target_ref.shape[0]
                            and (target_ref[target_ref_loc+1, 0] - target_ref[target_ref_loc, 0]) > 1
                        ):
                            delay = delay - 1.0;
                        target_ref[target_ref_loc, 0] = target_ref[target_ref_loc, 0] + delay
                        target_ref_loc += 1
                    if target_ref_loc >= target_ref.shape[0]:
                        break
            # while target_ref_loc < target_ref.shape[0] and (target_ref[target_ref_loc, 0] + delay) < float(trace_arr[-1, 0]):
                #     if (
                #         target_ref_loc + 1 < target_ref.shape[0]
                #         and (target_ref[target_ref_loc+1, 0] - target_ref[target_ref_loc, 0]) > 1
                #     ):
                #         delay = delay - 1.0;
            return target_ref[:target_ref_loc]

        def mutation_morphing(trace_arr: np.ndarray, target_ref: np.ndarray, L: int) -> np.ndarray:
            """
            Morph the trace_arr to the target_ref.
            """
            if target_ref.shape[0] < trace_arr.shape[0]:
                trace_len = target_ref.shape[0]
            else:
                trace_len = trace_arr.shape[0]

            if trace_arr[L, 0] < target_ref[L, 0]:
                delay = 0
            else:
                delay = 1.0
            
            target_ref_loc = 2
            target_ref[0, 0] = float(trace_arr[0, 0])
            target_ref[1, 0] = target_ref[1, 0] + delay
            
            for L in range(2,trace_len):
                if target_ref_loc+1 >= target_ref.shape[0]:
                    break
                while target_ref_loc < target_ref.shape[0] and (target_ref[target_ref_loc-1, 0]) < float(trace_arr[L, 0]):
                    # if (
                    #     target_ref_loc + 1 < target_ref.shape[0]
                    #     and (target_ref[target_ref_loc+1, 0] - target_ref[target_ref_loc, 0]) > 1
                    # ):
                    #     delay = delay - 1.0;
                    target_ref[target_ref_loc, 0] = target_ref[target_ref_loc, 0] + delay
                    target_ref_loc += 1
                # The scan can advance target_ref_loc to len(target_ref); only then compare.
                if target_ref_loc >= target_ref.shape[0]:
                    break
                if np.sign(target_ref[target_ref_loc, 1]) == np.sign(trace_arr[L, 1]):
                    if (
                        target_ref_loc + 1 < target_ref.shape[0]
                        and (target_ref[target_ref_loc+1, 0] - target_ref[target_ref_loc, 0]) > 1
                    ):
                        delay = delay - 1.0;
                    target_ref[target_ref_loc, 0] = target_ref[target_ref_loc, 0] + delay
                    target_ref_loc += 1
                else:
                    while (
                        target_ref_loc < target_ref.shape[0]
                        and np.sign(target_ref[target_ref_loc, 1]) != np.sign(trace_arr[L, 1])
                    ):
                        if (
                            target_ref_loc + 1 < target_ref.shape[0]
                            and (target_ref[target_ref_loc+1, 0] - target_ref[target_ref_loc, 0]) > 1
                        ):
                            delay = delay - 1.0;
                        target_ref[target_ref_loc, 0] = target_ref[target_ref_loc, 0] + delay
                        target_ref_loc += 1
                    if target_ref_loc >= target_ref.shape[0]:
                        break
            return target_ref[:target_ref_loc]

        # def finalize_mutated_target(target_ref: np.ndarray, loc: int, min_len: int) -> np.ndarray:
        #     """
        #     Build a finalized mutated reference under length constraints:
        #     - length >= min_len
        #     - length <= floor(1.5 * min_len)
        #     Return the candidate with the lowest inter-class separability.
        #     """
        #     nref = int(target_ref.shape[0])
        #     if nref <= 0:
        #         return target_ref
        #     loc = max(0, min(int(loc), nref - 1))

        #     min_len = max(1, int(min_len))
        #     lower_bound = max(int(loc + 1), min_len)
        #     upper_bound = min(nref, int(np.floor(1.5 * min_len)))

        #     if upper_bound < lower_bound:
        #         # Fall back to the longest allowed prefix from target_ref.
        #         fallback_len = min(nref, max(int(loc + 1), min_len))
        #         return np.asarray(target_ref[:fallback_len], dtype=float).copy()

        #     best: Optional[np.ndarray] = None
        #     best_score = float("inf")

        #     # Evaluate every feasible candidate length and keep the lowest
        #     # inter-class separability trace.
        #     for final_len in range(lower_bound, upper_bound + 1):
        #         cand = np.asarray(target_ref[:final_len], dtype=float).copy()
        #         add_start = loc + 1

        #         if add_start < final_len:
        #             # Greedy direction tuning on the appended tail.
        #             for idx in range(add_start, final_len):
        #                 cand[idx, 1] = 1.0
        #                 score_pos = separability_score(cand[:, 1])
        #                 cand[idx, 1] = -1.0
        #                 score_neg = separability_score(cand[:, 1])
        #                 cand[idx, 1] = 1.0 if score_pos <= score_neg else -1.0

        #         score = separability_score(cand[:, 1])
        #         if score < best_score:
        #             best_score = score
        #             best = cand

        #     if best is None:
        #         return np.asarray(target_ref[:lower_bound], dtype=float).copy()
        #     return best


        # Full 2D reference trace for the current morphing segment.
        target_ref: Optional[np.ndarray] = None
        mutation_location = -1 # the location of the mutation
        last_loc = 1
        L_start = -1


        for L in range(2, n):
            if L >= int(self.args.seq_length):
                return self.finalize_morphing_trace(np.asarray(mixed, dtype=float))

            picked_ref = False
            # 1) Choose a reference trace when none is active for this segment.
            if target_ref is None:
                
                if L > int(self.config.radix_trie_build_length):
                    matched_idxs = trace_match_cached(L-1)
                    matched_set = set(matched_idxs)
                    ms_list = list(matched_set)
                    if len(matched_set) > 1:
                        pick = int(ms_list[np.random.randint(0, len(ms_list))])
                        target_ref = np.asarray(self.selected_traces[pick], dtype=float)
                        picked_ref = True
                    else: 
                        sel = int(np.random.randint(0, len(self.selected_idx_set)))
                        orig = int(self.selected_idx_set[sel])
                        # target_ref = np.asarray(self.data_traces[orig], dtype=float)
                        target_ref = np.asarray(self.traces_idx[orig], dtype=float)
                        picked_ref = True
                else:
                    matched_idxs = trace_match_cached(L)
                    matched_set = set(matched_idxs)
                    ms_list = list(matched_set)

                    if 1 < len(matched_set) < 5:
                        pick = int(ms_list[np.random.randint(0, len(ms_list))])
                        target_ref = np.asarray(self.selected_traces[pick], dtype=float)
                        picked_ref = True
                    elif len(matched_set) == 1:
                        matched_orig = {int(self.selected_idx_set[i]) for i in matched_set}
                        for group_set in self.grouped_index_sets:
                            if matched_orig.issubset(group_set):
                                gs = sorted(group_set)
                                random_idx = int(np.random.randint(0, len(gs)))
                                orig_idx = int(gs[random_idx])
                                # target_ref = np.asarray(self.data_traces[orig_idx], dtype=float)
                                target_ref = np.asarray(self.traces_idx[orig_idx], dtype=float)
                                picked_ref = True
                                break
                        
                        if target_ref is None:
                            mi = int(next(iter(matched_set)))
                            target_ref = np.asarray(self.selected_traces[mi], dtype=float)
                            picked_ref = True
                    elif len(matched_set) == 0:
                        matched_idxs = trace_match_cached(L-1)
                        matched_set = set(matched_idxs)
                        ms_list = list(matched_set)
                        if len(matched_set) > 1:
                            pick = int(ms_list[np.random.randint(0, len(ms_list))])
                            target_ref = np.asarray(self.selected_traces[pick], dtype=float)
                            picked_ref = True

            if picked_ref and target_ref is not None:
                if self.config.mutation == 1:
                    if mutation_location == -1:
                        loc = L
                    else: 
                        loc = mutation_location
                    if L_start == -1:
                        L_start = L
                    if loc >= target_ref.shape[0]:
                        loc = int(target_ref.shape[0] - 1)
                    last_loc = int(loc)
                    if target_ref[loc, 1] != trace_arr[L, 1]:
                        if mutation_check(target_ref, trace_arr[L, 1], loc) != -1:
                            # If a matching direction already exists in the window,
                            # move mutation_location there to minimize extra edits.
                            mutation_location = mutation_check(target_ref, trace_arr[L, 1], loc)
                        else:
                            # Otherwise mutate one direction inside the window and
                            # continue comparing from that flipped location.
                            best_loc = optimize_mutation_location(target_ref, trace_arr, loc, L)
                            target_ref[best_loc, 1] = -float(target_ref[best_loc, 1])
                            mutation_location = int(best_loc)
                            last_loc = int(best_loc)
                            # print(f"mutation_location: {mutation_location}, best_loc: {best_loc}")
                    else:
                        mutation_location += 1

                elif self.config.mutation == 0: 

                    finalized_ref = trace_morphing(trace_arr, target_ref, L)
                    mixed.extend(finalized_ref[2:].tolist())
                    break
                else:
                    mixed.extend(finalized_ref[2:].tolist())
                    break




                    # loc = 0
                    # nref = int(target_ref.shape[0])
                    # tL = float(trace_arr[L, 0])
                    # while loc < nref and float(target_ref[loc, 0]) < tL:
                    #     loc += 1
                    # if loc < nref:
                    #     mixed.extend(target_ref[loc:].tolist())
                    # else:
                    #     mixed.append([tL, float(trace_arr[L, 1])])
                    # break

        if self.config.mutation == 1 and target_ref is not None:
            # Input has terminated: finalize the timing of the mutated reference trace.
            finalized_ref = mutation_morphing(trace_arr, target_ref, L_start)
            
            # final_loc = int(mutation_location) if mutation_location != -1 else int(last_loc)
            # finalized_ref = finalize_mutated_target(np.asarray(target_ref, dtype=float), final_loc, trace_arr.shape[0])

            mixed.extend(finalized_ref[1:].tolist())


        return self.finalize_morphing_trace(np.asarray(mixed, dtype=float))

   