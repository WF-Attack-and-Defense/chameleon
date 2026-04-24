import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union

import numpy as np

from defenses import FrontDefense, RegulatorDefense, WtfpadDefense, TrafficSliverDefense, MinipatchDefense, DynaflowDefense, PaletteDefense, MockingbirdDefense, GapdisDefense, ChameleonDefense, SurakavDefense, AlertDefense
from utils.general import get_flist_label, get_all_mon_flist_label, parse_all_mon_trace,timeit, init_directories
from utils.perturb_util import verify_defense_with_attack

defense_funcs = {
    'wtfpad': WtfpadDefense,
    'front': FrontDefense,
    'regulator': RegulatorDefense,
    'trafficsliver': TrafficSliverDefense,
    'minipatch': MinipatchDefense,
    'dynaflow': DynaflowDefense,
    'palette': PaletteDefense,
    'mockingbird': MockingbirdDefense,
    'chameleon': ChameleonDefense,
    'gapdis': GapdisDefense,
    'surakav': SurakavDefense,
    'alert': AlertDefense,
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='WF transfer project')
    parser.add_argument('--defense', type=str, help='choose the defense')
    parser.add_argument('--attack', choices=['df', 'tiktok', 'rf', 'var_cnn', 'awf', 'netclr'], default=None, help='choose the attack')
    # paths and file config
    parser.add_argument('--dataset', choices=['DF', 'ds-19', 'test'], default='DF', help='choose the dataset')
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
    parser.add_argument(
        '--no-alert-auto-train',
        action='store_true',
        default=False,
        help='ALERT only: skip GAN training when generator weights are missing',
    )
    parser.add_argument(
        '--alert-force-train',
        action='store_true',
        default=False,
        help='ALERT only: retrain generators even if all generator_site_*.pt exist',
    )
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')

    # nworkers
    parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                        help='number of data loading workers (default: 20)')

    # LOG
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed performance')

    _args = parser.parse_args()
    return _args

@timeit
def parallel_simulate(flist: Union[List[str], np.ndarray], defense, workers: int):
    """
    Simulate traces in parallel.
    
    Args:
        flist: List of file paths
        defense: Defense instance
        workers: Number of worker processes
    """
    with Pool(workers) as p:
        p.map(defense.simulate, flist)

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
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    

    # Validate arguments based on open-world setting
    if args.open_world:
        args.output_dir = str(Path(args.output_dir) / args.defense / 'OW')
    else:
        args.unmon_path = None
        args.output_dir = str(Path(args.output_dir) / args.defense / 'CW')
    
    if args.defense in defense_funcs:
        if args.defense == 'gapdis' or args.defense == 'minipatch':
            if args.attack not in ['df', 'tiktok', 'rf', 'var_cnn', 'awf', 'netclr']:
                raise ValueError(f"Attack {args.attack} not supported for gapdis defense")
            if args.attack == None:
                raise ValueError("Attack is required for gapdis defense")
    else:
        raise NotImplementedError(f"Defense {args.defense} not implemented")

    # Get file list and labels
    flist, labels = get_flist_label(args.mon_path, args.unmon_path, mon_cls=args.mon_classes, mon_inst=args.mon_inst, unmon_inst=args.unmon_inst, suffix=args.suffix)
 
    if args.defense == 'gapdis':
        GapdisDefense(args, flist, labels)
    else:
        if args.defense == 'alert':
            from defenses.alert import maybe_train_alert_generators

            maybe_train_alert_generators(args, flist, labels)
        defense = defense_funcs[args.defense](args)
        parallel_simulate(flist, defense, args.workers)