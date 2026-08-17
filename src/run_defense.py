import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union

import numpy as np

from defenses import ChameleonDefense
from utils.general import get_flist_label, timeit

defense_funcs = {
    'chameleon': ChameleonDefense,
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='WF transfer project')
    parser.add_argument('--defense', choices=['chameleon'], default='chameleon', help='choose the defense')
    # paths and file config
    parser.add_argument('--dataset', choices=['DF', 'ds-19', 'GTT23', 'test'], default='DF', help='choose the dataset')
    parser.add_argument('--checkpoints', type=str, default='../checkpoints/',
                        help='location of model checkpoints')
    # config-path
    parser.add_argument('--config-path', type=str, default=None, help="config path")
    parser.add_argument('--config-section', '-c', type=str, default='default', help="config section")
    parser.add_argument('--output-dir', type=str, default='../defense_results/',
                        help='location of defense dataset')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multiple gpus')
    parser.add_argument('--amp', action='store_true', default=False, help='use mixed precision training')


    parser.add_argument('--suffix', type=str, default='.cell', help='suffix of the output file')
    parser.add_argument('--open-world', default=False, action="store_true", help='Open world or not')
    parser.add_argument('--seq-length', default=5000, type=int, help='The input trace length')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')

    # nworkers
    parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                        help='number of data loading workers (default: 20)')

    # LOG
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed performance')

    _args = parser.parse_args()
    return _args

# Module-level handle so Pool workers inherit the defense via fork instead of
# pickling a multi-GB bound method for every task chunk (OOM on GTT23-scale).
_DEFENSE_WORKER = None


def _simulate_one(data_path: str) -> None:
    # Dump happens inside simulate(); never return the ndarray — Pool.map would
    # accumulate ~200k traces in the parent and OOM on GTT23-scale datasets.
    _DEFENSE_WORKER.simulate(data_path)


@timeit
def parallel_simulate(flist: Union[List[str], np.ndarray], defense, workers: int):
    """
    Simulate traces in parallel.
    
    Args:
        flist: List of file paths
        defense: Defense instance
        workers: Number of worker processes
    """
    global _DEFENSE_WORKER
    _DEFENSE_WORKER = defense
    if hasattr(defense, "ensure_radix_trie"):
        defense.ensure_radix_trie()
    n = len(flist)
    workers = max(1, int(workers))
    # Moderate chunks: fewer IPC round-trips, without buffering huge result batches.
    chunksize = max(1, min(64, n // (workers * 8) if n else 1))
    with Pool(workers) as p:
        # imap_unordered avoids holding the full result list; workers return None.
        for _ in p.imap_unordered(_simulate_one, flist, chunksize=chunksize):
            pass

if __name__ == '__main__':
    args = parse_arguments()

    # Check if config_path is provided
    if args.config_path is None:
        print("ERROR: --config-path is required but not provided.")
        print("Please specify the path to the defense configuration file using --config-path.")
        exit(1)

    data_path = '../datasets/'
    mon_path = ''
    unmon_path = ''

    if args.dataset == 'DF':
        args.mon_path = data_path + 'DF/CW/'
        args.unmon_path = data_path + 'DF/OW/'
        args.mon_classes = 95
        args.mon_inst = 1000
        args.unmon_inst = 40716
    elif args.dataset == 'ds-19':
        args.mon_path = data_path + 'ds-19/CW/'
        args.unmon_path = data_path + 'ds-19/OW/'
        args.mon_classes = 100
        args.mon_inst = 100
        args.unmon_inst = 10000
    elif args.dataset == 'GTT23':
        args.mon_path = data_path + 'GTT23/CW/'
        args.unmon_path = data_path + 'GTT23/OW/'
        args.mon_classes = 100
        args.mon_inst = 1000
        args.unmon_inst = 10000
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    

    # Validate arguments based on open-world setting
    if args.open_world:
        args.output_dir = str(Path(args.output_dir) / args.defense / 'OW')
    else:
        args.unmon_path = None
        args.output_dir = str(Path(args.output_dir) / args.defense / 'CW')
    
    if args.defense not in defense_funcs:
        raise NotImplementedError(f"Defense {args.defense} not implemented")

    # Get file list and labels
    flist, labels = get_flist_label(args.mon_path, args.unmon_path, mon_cls=args.mon_classes, mon_inst=args.mon_inst, unmon_inst=args.unmon_inst, suffix=args.suffix)

    defense = defense_funcs[args.defense](args)
    parallel_simulate(flist, defense, args.workers)
