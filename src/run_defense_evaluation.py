import argparse

import torch

from defenses import ChameleonDefense
from utils.perturb_util import verify_defense_with_attack
from utils.logger import init_logger

defense_funcs = {
    'chameleon': ChameleonDefense,
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='WF transfer project')
    parser.add_argument('--attack', choices=['df', 'rf', 'var_cnn', 'netclr'], default='df', help='choose the attack')
    parser.add_argument('--defense', choices=sorted(defense_funcs.keys()), default='chameleon', help='choose the defense')
    parser.add_argument('--dataset', choices=['DF', 'wang14', 'ds-19', 'Walkie-Talkie'], default='DF', help='choose the dataset')
    parser.add_argument('--defense-dataset', default='', help='choose the defense dataset')
    parser.add_argument('--checkpoints', type=str, default='../checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--suffix', type=str, default='.cell', help='suffix of the output file')
    parser.add_argument('--one-fold', default=False, action="store_true", help='Run one fold or ten folds')
    parser.add_argument('--open-world', default=False, action="store_true", help='Open world or not')
    parser.add_argument('--seq-length', default=5000, type=int, help='The input trace length')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('-j', '--workers', default=20, type=int, metavar='N', help='number of data loading workers (default: 20)')

    # LOG
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed performance')

    _args = parser.parse_args()
    return _args

if __name__ == '__main__':
    args = parse_arguments()
    logger = init_logger("DefenseEvaluation")

    data_path = '../defense_results/' + args.defense + ('/OW/' if args.open_world else '/CW/')

    if args.dataset == 'DF':
        args.defense_mon_path = data_path + 'DF/'
        args.defense_unmon_path = data_path + 'DF/'
        args.defense_mon_classes = 95
        args.defense_mon_inst = 1000
        args.defense_unmon_inst = 40716 if args.open_world else 0
    elif args.dataset == 'wang14':
        args.defense_mon_path = data_path + 'wang14/'
        args.defense_unmon_path = data_path + 'wang14/'
        args.defense_mon_classes = 100
        args.defense_mon_inst = 90
        args.defense_unmon_inst = 9000 if args.open_world else 0
    elif args.dataset == 'ds-19':
        args.defense_mon_path = data_path + 'ds-19/'
        args.defense_unmon_path = data_path + 'ds-19/'
        args.defense_mon_classes = 100
        args.defense_mon_inst = 100
        args.defense_unmon_inst = 10000 if args.open_world else 0
    elif args.dataset == 'Walkie-Talkie':
        args.defense_mon_path = data_path + 'Walkie-Talkie/'
        args.defense_unmon_path = data_path + 'Walkie-Talkie/'
        args.defense_mon_classes = 100
        args.defense_mon_inst = 100
        args.defense_unmon_inst = 9900 if args.open_world else 0
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    if args.attack:
        # Attack.__init__ (VarCNNAttack, etc.) needs the same fields as run_attack.py
        data_path_ds = '../datasets/'
        if args.dataset == 'DF':
            args.mon_path = data_path_ds + 'DF/CW/'
            args.unmon_path = data_path_ds + 'DF/OW/' if args.open_world else None
            args.mon_classes = 95
            args.mon_inst = 1000
            args.unmon_inst = 40716 if args.open_world else 0
        elif args.dataset == 'wang14':
            args.mon_path = data_path_ds + 'wang14/CW/'
            args.unmon_path = data_path_ds + 'wang14/OW/' if args.open_world else None
            args.mon_classes = 100
            args.mon_inst = 90
            args.unmon_inst = 9000 if args.open_world else 0
        elif args.dataset == 'ds-19':
            args.mon_path = data_path_ds + 'ds-19/CW/'
            args.unmon_path = data_path_ds + 'ds-19/OW/' if args.open_world else None
            args.mon_classes = 100
            args.mon_inst = 100
            args.unmon_inst = 10000 if args.open_world else 0
        elif args.dataset == 'Walkie-Talkie':
            args.mon_path = data_path_ds + 'Walkie-Talkie/CW/'
            args.unmon_path = data_path_ds + 'Walkie-Talkie/OW/' if args.open_world else None
            args.mon_classes = 100
            args.mon_inst = 100
            args.unmon_inst = 9900 if args.open_world else 0

        # Add required arguments for attack verification
        args.use_gpu = bool(torch.cuda.is_available())
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = '0,1,2,3,4,5,6,7'
        args.amp = False

        logger.info("=" * 50)
        logger.info(f"Verifying defense with {args.attack.upper()} attack model...")
        logger.info("=" * 50)

        args.attack_model = args.attack + '_' + args.dataset + ('_OW' if args.open_world else '_CW') + '.h5'

        verify_defense_with_attack(args=args, logger=logger)