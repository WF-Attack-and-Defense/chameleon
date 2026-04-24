import os
import random
from datetime import datetime
from functools import wraps
from time import strftime
from time import time
from typing import Tuple, Union, Callable, Optional, Any

import numpy as np
import pandas as pd
try:
    import torch  # optional: only needed for seed_everything
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def seed_everything(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_random_seed(f: Callable):
    @wraps(f)
    def wrap(*args: Optional[Any], **kw: Optional[Any]) -> Any:
        np.random.seed(datetime.now().microsecond)
        result = f(*args, **kw)
        return result

    return wrap


def timeit(f: Callable):
    @wraps(f)
    def wrap(*args: Optional[Any], **kw: Optional[Any]) -> Any:
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap


def parse_trace(fdir: str, sanity_check: bool = False) -> np.ndarray:
    """
    Parse a trace file based on our predefined format
    """
    trace = pd.read_csv(fdir, delimiter="\t", header=None)
    # Keep exactly time + direction; coerce bad cells so numpy never builds ragged/object rows
    if trace.shape[1] > 2:
        trace = trace.iloc[:, :2]
    trace = trace.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if trace.empty:
        return np.zeros((0, 2), dtype=np.float64)
    trace = np.asarray(trace, dtype=np.float64)

    if sanity_check:
        # it is possible the trace has a long tail
        # if there is a time gap between two bursts larger than CUT_OFF_THRESHOULD
        # We cut off the trace here sicne it could be a long timeout or
        # maybe the loading is already finished
        # Set a very conservative value
        CUT_OFF_THRESHOLD = 15
        start, end = 0, len(trace)
        ipt_burst = np.diff(trace[:, 0])
        ipt_outlier_inds = np.where(ipt_burst > CUT_OFF_THRESHOLD)[0]

        if len(ipt_outlier_inds) > 0:
            outlier_ind_first = ipt_outlier_inds[0]
            if outlier_ind_first < 50:
                start = outlier_ind_first + 1
            outlier_ind_last = ipt_outlier_inds[-1]
            if outlier_ind_last > 50:
                end = outlier_ind_last + 1
        trace = trace[start:end].copy()

        # remove the first few lines that are incoming packets
        start = -1
        for time, size in trace:
            start += 1
            if size > 0:
                break

        trace = trace[start:].copy()
        trace[:, 0] -= trace[0, 0]
        assert trace[0, 0] == 0
    return trace


def feature_transform(sample: np.ndarray, feature_type: str, seq_length: int) -> np.ndarray:
    """
    Transform a raw sample to the specific feature space.
    :return a numpy array of shape (1 or 2, seq_length)
    """
    if feature_type == 'df':
        feat = np.sign(sample[:, 1])

    elif feature_type == 'tiktok':
        feat = sample[:, 0] * np.sign(sample[:, 1])

    elif feature_type == 'tam':
        max_load_time = 80.0  # s
        time_window = 0.044  # s

        # cut_off_time = min(max_load_time, float(sample[-1, 0]))
        # num_bins = int(cut_off_time / time_window) + 1
        def _tam_times(t: np.ndarray) -> np.ndarray:
            """Map timestamps into [0, inf); np.histogram drops NaNs and values < bins[0]."""
            t = np.asarray(t, dtype=np.float64)
            t = np.nan_to_num(t, nan=0.0, posinf=max_load_time, neginf=0.0)
            return np.clip(t, 0.0, np.inf)

        times = _tam_times(sample[:, 0])
        t_max = float(np.max(times)) if times.size > 0 else 0.0
        cut_off_time = min(max_load_time, t_max)
        num_bins = max(1, int(cut_off_time / time_window) + 1)
        bins = np.linspace(0, num_bins * time_window, num_bins).tolist() + [np.inf]

        outgoing = sample[np.sign(sample[:, 1]) > 0]
        incoming = sample[np.sign(sample[:, 1]) < 0]

        # cnt_outgoing, _ = np.histogram(outgoing[:, 0], bins=bins)
        # cnt_incoming, _ = np.histogram(incoming[:, 0], bins=bins)
        cnt_outgoing, _ = np.histogram(_tam_times(outgoing[:, 0]), bins=bins)
        cnt_incoming, _ = np.histogram(_tam_times(incoming[:, 0]), bins=bins)

        # merge to 2d feature
        feat = np.stack((cnt_outgoing, cnt_incoming), axis=1)
        # assert feat.flatten().sum() == len(sample), \
        #     "Sum of feature ({}) is not equal to the length of the trace ({}). BUG?".format(
        #         feat.flatten().sum(), len(sample))
        n_binned = len(outgoing) + len(incoming)
        assert feat.flatten().sum() == n_binned, (
            "Sum of TAM bins ({}) != outgoing+incoming ({}). BUG?"
        ).format(int(feat.flatten().sum()), n_binned)

    elif feature_type == 'burst':
        sample = sample[:, 1]
        # Create a mask for consecutive elements that are the same
        mask = np.where(np.sign(sample[:-1]) != np.sign(sample[1:]))[0] + 1
        mask = np.concatenate((mask, [len(sample)]))  # add the last index
        # Count the number of elements between sign changes
        feat = np.diff(mask, prepend=0)
        assert sum(feat) == len(sample), \
            "Sum of burst lengths ({}) is not equal to the length of the trace ({}). BUG?".format(sum(feat), len(sample))
    elif feature_type == 'var_cnn':
        # VarCNN uses interval time (inter-packet time differences) and direction
        # Extract timestamps and packet sizes
        time = sample[:, 0]  # absolute timestamps
        direction = np.sign(sample[:, 1])  # direction: +1 (outgoing), -1 (incoming), 0 (zero-size)
        
        # Convert from absolute times to inter-packet times
        # Each spot holds time diff between curr packet and prev packet
        inter_time = np.zeros_like(time)
        inter_time[1:] = time[1:] - time[:-1]  # First packet has inter-time = 0
        
        # Stack inter_time and direction as 2D feature
        # Shape: (seq_length, 2) -> will be transposed to (2, seq_length) at the end
        feat = np.stack((inter_time, direction), axis=1)
    else:
        raise NotImplementedError("Feature type {} is not implemented.".format(feature_type))

    # make sure 2d
    if len(feat.shape) == 1:
        feat = feat[:, np.newaxis]
    # pad to seq_length
    if len(feat) < seq_length:
        pad = np.zeros((seq_length - len(feat), feat.shape[1]))
        feat = np.concatenate((feat, pad))
    feat = feat[:seq_length, :]
    return np.transpose(feat, (1, 0))


def get_flist_label(mon_path: Union[str, os.PathLike], unmon_path: Union[str, os.PathLike, None] = None,
                    mon_cls: int = 0, mon_inst: int = 0, unmon_inst: int = 0,
                    suffix: str = '.cell') \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a list of file paths and corresponding labels.
    
    :param mon_path: the path to the monitored data directory
    :param unmon_path: the path to the unmonitored data directory (optional, defaults to mon_path if None)
    :param mon_cls: number of monitored classes
    :param mon_inst: number of monitored instances per class
    :param unmon_inst: number of unmonitored instances
    :param suffix: file suffix
    :return: a list of file paths and a list of corresponding labels
    """

    flist = []
    labels = []


    # Load monitored data from mon_path
    for cls in range(mon_cls):
        for inst in range(mon_inst):
            pth = os.path.join(mon_path, '{}-{}{}'.format(cls, inst, suffix))
            if os.path.exists(pth):
                flist.append(pth)
                labels.append(cls)
        
    # Load unmonitored data from unmon_path
    if unmon_inst > 0 and unmon_path is not None:
        for inst in range(unmon_inst):
            pth = os.path.join(unmon_path, '{}{}'.format(inst, suffix))
            if os.path.exists(pth):
                flist.append(pth)
                labels.append(mon_cls)

    assert len(flist) > 0, "No files found!"
    return np.array(flist), np.array(labels)


def get_all_mon_flist_label(mon_path: Union[str, os.PathLike], mon_cls: int = 0, mon_inst: int = 0,
                    suffix: str = '.cell') \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a list of file paths and corresponding labels.
    
    :param mon_path: the path to the monitored data directory
    :param mon_cls: number of monitored classes
    :param mon_inst: number of monitored instances per class
    :param suffix: file suffix
    :return: a list of file paths and a list of corresponding labels
    """

    flist = []
    labels = []


    # Load monitored data from mon_path
    for cls in range(mon_cls):
        for inst in range(mon_inst):
            pth = os.path.join(mon_path, '{}-{}{}'.format(cls, inst, suffix))
            if os.path.exists(pth):
                flist.append(pth)
                labels.append(cls)

    assert len(flist) > 0, "No files found!"
    return np.array(flist), np.array(labels)

def parse_all_mon_trace(fdirs: list[str], sanity_check: bool = False) -> np.ndarray:
    """
    Parse a regression trace file based on our predefined format
    """
    all_trace = []
    for fdir in fdirs:
        all_trace.append(parse_trace(fdir, sanity_check=sanity_check))
    return np.array(all_trace, dtype=object)


def init_directories(output_parent_dir: Union[str, os.PathLike], defense_name: str) -> str:
    # Create a results dir if it doesn't exist yet
    if not os.path.exists(output_parent_dir):
        os.makedirs(output_parent_dir)

    # Define output directory
    timestamp = strftime('%m%d_%H%M%S')
    output_dir = os.path.join(output_parent_dir, defense_name + '_' + timestamp)
    os.makedirs(output_dir)
    return output_dir
