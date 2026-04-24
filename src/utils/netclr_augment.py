"""
NetCLR augmentation: burst-based augmentor and SimCLR train dataset.
Used by run_netclr_pretrain.py for contrastive pre-training.
"""
import bisect
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def find_bursts(x):
    if len(x) == 0:
        return []
    direction = x[0]
    bursts = []
    start = 0
    temp_burst = x[0]
    for i in range(1, len(x)):
        if x[i] == 0.0:
            break
        elif x[i] == direction:
            temp_burst += x[i]
        else:
            bursts.append((start, i, temp_burst))
            start = i
            temp_burst = x[i]
            direction *= -1
    return bursts


def compute_outgoing_burst_cdf(x_train, max_samples=1000, seq_length=5000):
    """Compute CDF of outgoing burst sizes from a subset of traces (for augmentor)."""
    n = min(max_samples, len(x_train))
    indices = np.random.choice(len(x_train), size=n, replace=False)
    outgoing_burst_sizes = []
    for idx in indices:
        x = x_train[idx]
        if x.ndim > 1:
            x = x.flatten()
        x = x[:seq_length]
        bursts = find_bursts(x)
        outgoing_burst_sizes.extend([b[2] for b in bursts if b[2] > 0])
    if not outgoing_burst_sizes:
        return 1, np.array([0.0, 1.0])
    max_outgoing = int(np.ceil(max(outgoing_burst_sizes)))
    num_bins = max(1, max_outgoing - 1)
    count, bins = np.histogram(outgoing_burst_sizes, bins=num_bins)
    pdf = count / np.sum(count)
    cdf = np.zeros_like(bins)
    cdf[1:] = np.cumsum(pdf)
    return max_outgoing, cdf


class Augmentor:
    def __init__(self, max_outgoing_burst_size: int, outgoing_burst_size_cdf: np.ndarray, seq_length: int = 5000):
        self.seq_length = seq_length
        self.large_burst_threshold = 10
        self.upsample_rate = 1.0
        self.downsample_rate = 0.5
        self.num_bursts_to_merge = 5
        self.merge_burst_rate = 0.1
        self.add_outgoing_burst_rate = 0.3
        self.outgoing_burst_sizes = list(range(max(1, max_outgoing_burst_size)))
        self.outgoing_burst_size_cdf = outgoing_burst_size_cdf
        self.shift_param = 10

    def find_bursts(self, x):
        if len(x) == 0:
            return []
        direction = x[0]
        bursts = []
        start = 0
        temp_burst = x[0]
        for i in range(1, len(x)):
            if x[i] == 0.0:
                break
            elif x[i] == direction:
                temp_burst += x[i]
            else:
                bursts.append((start, i, temp_burst))
                start = i
                temp_burst = x[i]
                direction *= -1
        return bursts

    def increase_incoming_bursts(self, burst_sizes):
        out = []
        for size in burst_sizes:
            if size <= -self.large_burst_threshold:
                up_sample_rate = random.random() * self.upsample_rate
                new_size = int(size * (1 + up_sample_rate))
                out.append(new_size)
            else:
                out.append(size)
        return out

    def decrease_incoming_bursts(self, burst_sizes):
        out = []
        for size in burst_sizes:
            if size <= -self.large_burst_threshold:
                up_sample_rate = random.random() * self.downsample_rate
                new_size = int(size * (1 - up_sample_rate))
                out.append(new_size)
            else:
                out.append(size)
        return out

    def change_content(self, trace):
        bursts = self.find_bursts(trace)
        burst_sizes = [b[2] for b in bursts]
        if len(trace) < 1000:
            new_burst_sizes = self.increase_incoming_bursts(burst_sizes)
        elif len(trace) > 4000:
            new_burst_sizes = self.decrease_incoming_bursts(burst_sizes)
        else:
            if random.random() >= 0.5:
                new_burst_sizes = self.increase_incoming_bursts(burst_sizes)
            else:
                new_burst_sizes = self.decrease_incoming_bursts(burst_sizes)
        return new_burst_sizes

    def merge_incoming_bursts(self, burst_sizes):
        out = []
        i = 0
        num_cells = 0
        while i < len(burst_sizes) and num_cells < 20:
            num_cells += abs(burst_sizes[i])
            out.append(burst_sizes[i])
            i += 1
        while i < len(burst_sizes) - self.num_bursts_to_merge:
            if burst_sizes[i] > 0:
                out.append(burst_sizes[i])
                i += 1
                continue
            if random.random() < self.merge_burst_rate:
                num_merges = random.randint(2, self.num_bursts_to_merge)
                merged_size = 0
                while i < len(burst_sizes) and num_merges > 0:
                    if burst_sizes[i] < 0:
                        merged_size += burst_sizes[i]
                        num_merges -= 1
                    i += 1
                out.append(merged_size)
            else:
                out.append(burst_sizes[i])
                i += 1
        return out

    def add_outgoing_burst(self, burst_sizes):
        out = []
        i = 0
        num_cells = 0
        while i < len(burst_sizes) and num_cells < 20:
            num_cells += abs(burst_sizes[i])
            out.append(burst_sizes[i])
            i += 1
        for size in burst_sizes[i:]:
            if size > -10:
                out.append(size)
                continue
            if random.random() < self.add_outgoing_burst_rate:
                index = bisect.bisect_left(self.outgoing_burst_size_cdf, random.random())
                index = min(index, len(self.outgoing_burst_sizes) - 1) if self.outgoing_burst_sizes else 0
                outgoing_burst_size = self.outgoing_burst_sizes[index] if self.outgoing_burst_sizes else 1
                max_divide = max(3, abs(size) - 3)
                divide_place = random.randint(3, max_divide) if max_divide > 3 else 3
                out += [-divide_place, outgoing_burst_size, -(abs(size) - divide_place)]
            else:
                out.append(size)
        return out

    def create_trace_from_burst_sizes(self, burst_sizes):
        out = []
        for size in burst_sizes:
            val = 1 if size > 0 else -1
            out += [val] * int(abs(size))
        if len(out) < self.seq_length:
            out += [0] * (self.seq_length - len(out))
        return np.array(out[:self.seq_length], dtype=np.float32)

    def shift(self, x):
        pad = np.random.randint(0, 2, size=(self.shift_param,))
        pad = 2 * pad - 1
        zpad = np.zeros_like(pad)
        shift_val = np.random.randint(-self.shift_param, self.shift_param + 1, 1)[0]
        shifted = np.concatenate((x, zpad, pad), axis=-1)
        shifted = np.roll(shifted, shift_val, axis=-1)
        return shifted[:self.seq_length].astype(np.float32)

    def augment(self, trace):
        if trace.ndim > 1:
            trace = trace.flatten()
        trace = trace[:self.seq_length]
        if len(trace) == 0:
            return np.zeros(self.seq_length, dtype=np.float32)
        bursts = self.find_bursts(trace)
        burst_sizes = [b[2] for b in bursts]
        mapping = [self.change_content, self.merge_incoming_bursts, self.add_outgoing_burst]
        aug_method = mapping[random.randint(0, len(mapping) - 1)]
        augmented_sizes = aug_method(burst_sizes)
        augmented_trace = self.create_trace_from_burst_sizes(augmented_sizes)
        return self.shift(augmented_trace)


class NetCLRTrainDataset(Dataset):
    """Dataset that returns n_views augmented versions of each trace (for SimCLR)."""

    def __init__(self, x_train, y_train, augmentor, n_views=2):
        self.x = x_train
        self.y = y_train
        self.augmentor = augmentor
        self.n_views = n_views

    def __getitem__(self, index):
        x = self.x[index]
        if isinstance(x, np.ndarray) and x.ndim > 1:
            x = x.squeeze()
        views = [self.augmentor.augment(x) for _ in range(self.n_views)]
        views = [torch.from_numpy(v).float() for v in views]
        return views, self.y[index]

    def __len__(self):
        return len(self.x)
