import argparse
import os

import torch

from attacks import DFAttack, TiktokAttack, RFAttack, VarCNNAttack, AWFAttack, NetCLRAttack
from utils.general import seed_everything


def parse_arguments():
    parser = argparse.ArgumentParser(description='WF transfer project')
    parser.add_argument('--attack', choices=['df', 'tiktok', 'rf', 'var_cnn', 'awf', 'netclr'], default='df', help='choose the attack')
    parser.add_argument('--dataset', choices=['DF', 'ds-19', 'defense', 'test'], default='DF', help='choose the dataset')

    parser.add_argument('--checkpoints', type=str, default='../checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--suffix', type=str, default='.cell', help='suffix of the output file')
    parser.add_argument('--one-fold', default=False, action="store_true", help='Run one fold or ten folds')
    parser.add_argument('--open-world', default=False, action="store_true", help='Open world or not')
    parser.add_argument('--seq-length', default=5000, type=int, help='The input trace length')

    # optimization
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr0', type=float, default=0.002, help='initial optimizer learning rate')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam',
                        help='optimizer')

    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multiple gpus')
    parser.add_argument('--amp', action='store_true', default=False, help='use mixed precision training')

    # LOG
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed performance')
    parser.add_argument('--log_itr_interval', type=int, default=100, help='log iteration interval')

    _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    args = parse_arguments()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    data_path = '../datasets/'

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
    elif args.dataset == 'defense':
        args.mon_path = '../defense_results/chameleon/OW/ds-19_2/'
        args.unmon_path = '../defense_results/chameleon/OW/ds-19_2/'
        args.mon_classes = 100
        args.mon_inst = 100
        args.unmon_inst = 10000
    elif args.dataset == 'test':
        args.mon_path = '../defense_results/minipatch/CW/DF_netclr/'
        args.unmon_path = '../defense_results/minipatch/OW/DF_netclr/'
        args.mon_classes = 95
        args.mon_inst = 1000
        args.unmon_inst = 40716
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")


    # Normalize checkpoints directory path
    # Normalize the path and ensure it ends with a separator
    args.checkpoints = os.path.normpath(args.checkpoints)
    if not args.checkpoints.endswith(os.sep):
        args.checkpoints = args.checkpoints + os.sep
    
    seed_everything(2024)
    attack = None
    if args.attack == 'df':
        attack = DFAttack(args)
    elif args.attack == 'tiktok':
        attack = TiktokAttack(args)
    elif args.attack == 'rf':
        attack = RFAttack(args)
    elif args.attack == 'var_cnn':
        attack = VarCNNAttack(args)
    elif args.attack == 'awf':
        attack = AWFAttack(args)
    elif args.attack == 'netclr':
        attack = NetCLRAttack(args)
    else:
        raise NotImplementedError("Attack not implemented")

    attack.run(args.one_fold)
