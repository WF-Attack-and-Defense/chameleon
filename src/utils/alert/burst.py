"""Burst ↔ cell-sequence conversions aligned with ALERT (see ``defenses.alert.convert_trace_data_to_burst``)."""

from typing import List

import numpy as np


def trace_to_cell_sequence(trace_2d: np.ndarray, max_len: int) -> np.ndarray:
    """
    Map a tab-separated trace (time, direction) to a padded 1D cell sequence
    in {-1, 0, 1} style used by the original ALERT pipeline.
    """
    if trace_2d.ndim != 2 or trace_2d.shape[1] < 2:
        return np.zeros(max_len, dtype=np.float32)
    signs = np.sign(trace_2d[:, 1]).astype(np.float32)
    start = 0
    for i in range(len(signs)):
        if signs[i] > 0:
            start = i
            break
    seq = signs[start:]
    out = np.zeros(max_len, dtype=np.float32)
    n = min(len(seq), max_len)
    out[:n] = seq[:n]
    return out


def convert_trace_cell_to_burst(trace_row: np.ndarray, max_length: int) -> np.ndarray:
    """Single-trace version of ``defenses.alert.convert_trace_data_to_burst`` (batch API)."""
    trace = trace_row.tolist()
    burst: List[float] = []
    i = 0
    while i < len(trace) and trace[i] == 0:
        i += 1
    if i >= len(trace):
        return np.zeros(max_length, dtype=np.float32)
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
    return np.asarray(burst[:max_length], dtype=np.float32)


def convert_burst_row_to_trace_data(burst_row: np.ndarray, max_length: int) -> np.ndarray:
    """Single-row version of pendding ``convert_burst_to_trace_data``."""
    trace: List[int] = []
    for burst in burst_row:
        if burst == 0:
            trace.append(0)
            continue
        packet = 1 if burst > 0 else -1
        packets = [packet for _ in range(abs(int(burst)))]
        trace.extend(packets)
    while len(trace) < max_length:
        trace.append(0)
    return np.asarray(trace[:max_length], dtype=np.int64)
