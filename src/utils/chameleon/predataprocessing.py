import gc
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors


def zscore_and_l2(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Row-wise normalize vectors for NCC:
      - subtract mean
      - divide std
      - l2 normalize
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    Xc = Xc / (Xc.std(axis=1, keepdims=True) + eps)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / (norms + eps)


def normalized_cross_correlation(
    direction_traces: np.ndarray,
    idx_set: np.ndarray,
    mon_inst: int,          # kept for API compatibility
    trace_threshold: int,
    selection_k: int,       # kept for API compatibility
    corr_threshold: float = 0.80,
    vec_len: int = 1000,
    corr_batch_size: int = 256,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Group traces by high normalized cross-correlation of direction sequences.

    IMPORTANT:
    - Do NOT remove any trace.
    - Return grouped traces (and grouped indices) containing all input traces exactly once.
    - If number of groups > trace_threshold, merge extra groups into kept groups.

    Memory: avoids materializing a full n×n correlation matrix (which OOMs for
    large selected pools). Uses chunked matmuls + union-find instead.
    """
    del mon_inst, selection_k  # API compatibility only
    seqs = np.asarray(direction_traces, dtype=object)
    idx_set = np.asarray(idx_set, dtype=int)

    n = len(seqs)
    if n == 0:
        return [], []

    # 1) sequence -> fixed vectors (pad short sequences)
    vl = int(vec_len)
    X = np.zeros((n, vl), dtype=np.float32)
    for i, s in enumerate(seqs):
        arr = np.asarray(s, dtype=np.float32).ravel()
        take = min(arr.size, vl)
        if take > 0:
            X[i, :take] = arr[:take]

    # 2) normalize (no full corr matrix)
    Xn = zscore_and_l2(X)

    # 3) connected components via union-find + chunked row correlations
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    batch = max(1, int(corr_batch_size))
    for start in range(0, n, batch):
        end = min(start + batch, n)
        # (batch, n) — peak ~ batch * n * 4 bytes instead of n * n * 4
        corr_batch = Xn[start:end] @ Xn.T
        for bi, i in enumerate(range(start, end)):
            nbrs = np.where(corr_batch[bi] >= corr_threshold)[0]
            for v in nbrs:
                if int(v) > i:
                    union(i, int(v))
        del corr_batch

    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    groups_map: Dict[int, List[int]] = {}
    for i, r in enumerate(roots):
        groups_map.setdefault(int(r), []).append(i)
    groups_local: List[List[int]] = list(groups_map.values())

    def mean_pairwise_corr(g: List[int]) -> float:
        if len(g) <= 1:
            return 0.0
        sub = Xn[np.asarray(g, dtype=int)]
        # mean of upper-triangle of sub @ sub.T without full materialization when small
        c = sub @ sub.T
        iu = np.triu_indices(len(g), k=1)
        return float(c[iu].mean()) if iu[0].size > 0 else 0.0

    def group_score(g: List[int]) -> Tuple[int, float]:
        return (len(g), mean_pairwise_corr(g))

    groups_local = sorted(groups_local, key=group_score, reverse=True)

    # 4) Keep at most trace_threshold groups, but DO NOT DROP any trace:
    #    merge overflow groups into the most similar kept group.
    k_groups = max(1, int(trace_threshold))
    if len(groups_local) > k_groups:
        kept = [list(g) for g in groups_local[:k_groups]]
        overflow = groups_local[k_groups:]
        kept_mat = [Xn[np.asarray(kg, dtype=int)].mean(axis=0) for kg in kept]
        for g in overflow:
            g_vec = Xn[np.asarray(g, dtype=int)].mean(axis=0)
            best_j = 0
            best_sim = -np.inf
            for j, km in enumerate(kept_mat):
                sim = float(g_vec @ km)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            kept[best_j].extend(g)
            kept_mat[best_j] = Xn[np.asarray(kept[best_j], dtype=int)].mean(axis=0)
        groups_local = kept

    # 5) Build outputs (all traces preserved)
    grouped_traces: List[np.ndarray] = []
    grouped_indices: List[np.ndarray] = []

    for g in groups_local:
        g_arr = np.array(sorted(set(g)), dtype=int)
        grouped_traces.append(seqs[g_arr])
        grouped_indices.append(idx_set[g_arr])

    return grouped_traces, grouped_indices


def trace_to_feature(trace: np.ndarray, max_dir_len: int = 64) -> np.ndarray:
    """
    Convert one variable-length trace (N,2) -> fixed-length feature vector.
    trace[:,0] = timestamp, trace[:,1] = packet size/direction sign source.
    """
    arr = np.asarray(trace, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return np.zeros(32 + max_dir_len, dtype=np.float32)

    t = arr[:, 0]
    s = arr[:, 1]
    d = np.sign(s)
    d[d == 0] = 1.0

    n = float(len(arr))
    out_cnt = float(np.sum(s > 0))
    in_cnt = float(np.sum(s < 0))
    out_bytes = float(np.sum(np.clip(s, 0, None)))
    in_bytes = float(-np.sum(np.clip(s, None, 0)))
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    ipt = np.diff(t) if len(t) > 1 else np.array([0.0], dtype=float)
    ipt_mean = float(np.mean(ipt))
    ipt_std = float(np.std(ipt))
    ipt_q25, ipt_q50, ipt_q75 = np.quantile(ipt, [0.25, 0.5, 0.75])

    # Burst lengths (run-lengths of same sign)
    if len(d) > 1:
        change_pos = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
        seg_ends = np.concatenate([change_pos, [len(d)]])
        burst = np.diff(np.concatenate([[0], seg_ends])).astype(float)
    else:
        burst = np.array([1.0], dtype=float)

    b_mean = float(np.mean(burst))
    b_std = float(np.std(burst))
    b_q50, b_q90 = np.quantile(burst, [0.5, 0.9])

    # Direction prefix sketch
    d_prefix = d[:max_dir_len]
    if len(d_prefix) < max_dir_len:
        d_prefix = np.pad(d_prefix, (0, max_dir_len - len(d_prefix)), mode="constant")

    core = np.array(
        [
            n, out_cnt, in_cnt, out_bytes, in_bytes, duration,
            ipt_mean, ipt_std, ipt_q25, ipt_q50, ipt_q75,
            b_mean, b_std, b_q50, b_q90,
            (out_cnt + 1.0) / (in_cnt + 1.0),
            (out_bytes + 1.0) / (in_bytes + 1.0),
            float(np.mean(d > 0)),
            float(np.mean(d < 0)),
            float(np.mean(np.abs(d))),
            float(np.max(np.abs(s))) if len(s) else 0.0,
            float(np.mean(np.abs(s))) if len(s) else 0.0,
            float(np.std(np.abs(s))) if len(s) else 0.0,
            float(np.quantile(np.abs(s), 0.5)) if len(s) else 0.0,
            float(np.quantile(np.abs(s), 0.9)) if len(s) else 0.0,
            float(np.min(ipt)) if len(ipt) else 0.0,
            float(np.max(ipt)) if len(ipt) else 0.0,
            float(np.mean(burst <= 2.0)),
            float(np.mean(burst >= 8.0)),
            float(len(burst)),
            float(np.mean(d_prefix[: min(16, max_dir_len)] > 0)),
            float(np.mean(d_prefix[: min(16, max_dir_len)] < 0)),
        ],
        dtype=np.float32,
    )

    return np.concatenate([core, d_prefix.astype(np.float32)], axis=0)


def traces_selection(
    data_traces: np.ndarray,
    data_labels: np.ndarray,
    k: int = 15,
    select_ratio: float = 0.2,
    min_select: int = 1,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 2.0,
    seq_len: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select traces with high intra-class diversity and low/misleading inter-class separation.

    Score_i = alpha * intra_i - beta * inter_i + gamma * mislead_i
      intra_i   : avg distance to same-class neighbors (higher -> more diverse in class)
      inter_i   : avg distance to different-class neighbors (lower -> less separated)
      mislead_i : fraction of different-class neighbors in top-k (higher -> misleading)

    Additional rule:
      - Ignore traces whose length is < 1000.
    """
    traces_all = np.asarray(data_traces, dtype=object)
    labels_all = np.asarray(data_labels, dtype=int)

    n_total = len(traces_all)
    if n_total == 0:
        return np.asarray([], dtype=object), np.asarray([], dtype=int)

    # Ignore short traces (< 1000 packets)
    valid_mask = np.array([len(np.asarray(tr)) >= seq_len for tr in traces_all], dtype=bool)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return np.asarray([], dtype=object), np.asarray([], dtype=int)

    traces = traces_all[valid_indices]
    labels = labels_all[valid_indices]
    n = len(traces)

    if n == 1:
        return traces.copy(), valid_indices.astype(int)

    # 1) Feature extraction
    X = np.stack([trace_to_feature(tr) for tr in traces], axis=0).astype(np.float32)

    # 2) Feature normalization
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sigma

    # 3) kNN (exclude self neighbor later)
    k_eff = min(max(3, int(k)), n - 1)
    knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean", algorithm="auto")
    knn.fit(Xn)
    dists, nbrs = knn.kneighbors(Xn, return_distance=True)

    dists = dists[:, 1:]  # remove self
    nbrs = nbrs[:, 1:]

    # 4) Hard-sample score
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nn_idx = nbrs[i]
        nn_dist = dists[i]
        nn_lab = labels[nn_idx]

        same_mask = nn_lab == labels[i]
        diff_mask = ~same_mask

        intra = float(np.mean(nn_dist[same_mask])) if np.any(same_mask) else float(np.max(nn_dist) + 1.0)
        inter = float(np.mean(nn_dist[diff_mask])) if np.any(diff_mask) else float(np.max(nn_dist) + 1.0)
        mislead = float(np.mean(diff_mask))

        scores[i] = alpha * intra - beta * inter + gamma * mislead

    # 5) Select top hard samples among valid traces
    m = max(int(np.ceil(select_ratio * n)), int(min_select))
    m = min(m, n)

    selected_indices_local = np.argsort(-scores)[:m]
    selected_indices_local = np.sort(selected_indices_local)

    selected_traces = traces[selected_indices_local]
    # Map local indices back to original dataset indices
    selected_indices = valid_indices[selected_indices_local].astype(int)

    return selected_traces, selected_indices


def _parse_trace_light(fdir: str) -> np.ndarray:
    """Lightweight (time, direction) loader for selection; avoids pandas overhead."""
    rows: List[Tuple[float, float]] = []
    with open(fdir, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def traces_selection_from_files(
    file_list: Sequence[str],
    data_labels: np.ndarray,
    k: int = 15,
    select_ratio: float = 0.2,
    min_select: int = 1,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 2.0,
    seq_len: int = 1000,
    log_every: int = 10000,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same scoring as ``traces_selection``, but never holds the full dataset in RAM.

    Pass 1: parse each file, keep only features for traces with length >= seq_len.
    Pass 2: reload only the selected traces.
    """
    from utils.general import parse_trace

    labels_all = np.asarray(data_labels, dtype=int)
    n_total = len(file_list)
    if n_total == 0:
        return np.asarray([], dtype=object), np.asarray([], dtype=int)

    features: List[np.ndarray] = []
    valid_indices: List[int] = []
    valid_labels: List[int] = []

    for i, path in enumerate(file_list):
        if logger is not None and log_every > 0 and (i + 1) % log_every == 0:
            logger.info(
                "Chameleon selection scan %d/%d (valid so far: %d)",
                i + 1,
                n_total,
                len(valid_indices),
            )
        tr = _parse_trace_light(path)
        if len(tr) < int(seq_len):
            continue
        features.append(trace_to_feature(tr))
        valid_indices.append(i)
        valid_labels.append(int(labels_all[i]))
        del tr

    if not valid_indices:
        return np.asarray([], dtype=object), np.asarray([], dtype=int)

    labels = np.asarray(valid_labels, dtype=int)
    n = len(valid_indices)
    if n == 1:
        only = parse_trace(file_list[valid_indices[0]])
        return np.asarray([only], dtype=object), np.asarray(valid_indices, dtype=int)

    X = np.stack(features, axis=0).astype(np.float32)
    del features
    gc.collect()

    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sigma
    del X

    k_eff = min(max(3, int(k)), n - 1)
    knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean", algorithm="auto")
    knn.fit(Xn)
    dists, nbrs = knn.kneighbors(Xn, return_distance=True)
    dists = dists[:, 1:]
    nbrs = nbrs[:, 1:]

    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nn_idx = nbrs[i]
        nn_dist = dists[i]
        nn_lab = labels[nn_idx]
        same_mask = nn_lab == labels[i]
        diff_mask = ~same_mask
        intra = float(np.mean(nn_dist[same_mask])) if np.any(same_mask) else float(np.max(nn_dist) + 1.0)
        inter = float(np.mean(nn_dist[diff_mask])) if np.any(diff_mask) else float(np.max(nn_dist) + 1.0)
        mislead = float(np.mean(diff_mask))
        scores[i] = alpha * intra - beta * inter + gamma * mislead

    m = max(int(np.ceil(select_ratio * n)), int(min_select))
    m = min(m, n)
    selected_local = np.sort(np.argsort(-scores)[:m])
    selected_indices = np.asarray(valid_indices, dtype=int)[selected_local]

    if logger is not None:
        logger.info(
            "Chameleon selection: %d valid / %d total, keeping %d; reloading selected traces",
            n,
            n_total,
            len(selected_indices),
        )

    selected_traces = np.empty(len(selected_indices), dtype=object)
    for j, idx in enumerate(selected_indices):
        selected_traces[j] = parse_trace(file_list[int(idx)])

    gc.collect()
    return selected_traces, selected_indices.astype(int)


def predataprocessing(
    data_traces: np.ndarray,
    data_labels: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper for Chameleon:
      - runs traces_selection
      - returns selected traces, selected labels, selected original indices
    """
    selected_traces, selected_indices = traces_selection(
        data_traces=data_traces,
        data_labels=data_labels,
        **kwargs
    )
    selected_labels = np.asarray(data_labels, dtype=int)[selected_indices]
    return selected_traces, selected_labels, selected_indices
