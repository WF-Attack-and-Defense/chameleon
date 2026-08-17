# calc data overhead based on original and new dataset
import argparse
import logging
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

global original_dir, new_dir

logger = logging.getLogger('ovhd')


def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != 'stdout':
        log_file = open(args.log, 'w')
    ch = logging.StreamHandler(log_file)

    # Set logging format
    LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate overhead for a trace folder.')

    parser.add_argument('-o',
                        metavar='<traces path>',
                        required=True,
                        help='original trace.')

    parser.add_argument('-p',
                        metavar='<new trace path>',
                        required=True,
                        help='new trace.')
    parser.add_argument('-format',
                        metavar='<file suffix>',
                        default=".cell",
                        help='file format suffix, default is .cell')
    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')
    parser.add_argument('--num_monitored',
                        type=int,
                        default=100,
                        help='number of monitored classes, default is 100')
    parser.add_argument('--num_inst',
                        type=int,
                        default=100,
                        help='number of instances per monitored class, default is 100')
    parser.add_argument('--num_unmonitored',
                        type=int,
                        default=0,
                        help='number of non-monitored classes, default is 0')
    parser.add_argument('--n_jobs',
                        type=int,
                        default=40,
                        help='number of jobs for multiprocessing, default is 40')

    args = parser.parse_args()
    config_logger(args)

    return args


def load_trace(fdir):
    with open(fdir, 'r') as f:
        trace = f.readlines()
    trace = pd.Series(trace).str.slice(0, -1).str.split('\t', expand=True).astype("float")
    return np.array(trace)


def calc_single_ovhd(ff):
    global original_dir, new_dir
    original, new = os.path.join(original_dir, ff), os.path.join(new_dir, ff)
    nt = load_trace(new)
    ot = load_trace(original)

    if len(nt) < 50 or len(ot) < 50:
        return None, None, None, None

    # |P| and |P'|: total number of cells in original and defended traces
    n_original = len(ot)
    n_defended = len(nt)
    # t_|P| and t_k: transmission time of original and defended traces
    t_original = ot[-1, 0]
    t_defended = nt[-1, 0]
    return n_original, n_defended, t_original, t_defended


def parallel(flist, n_jobs):
    pool = mp.Pool(n_jobs)
    ovhds = pool.map(calc_single_ovhd, flist)
    return ovhds


if __name__ == '__main__':
    args = parse_arguments()
    global original_dir, new_dir
    original_dir, new_dir = args.o, args.p

    flist = []
    for i in range(args.num_monitored):
        for j in range(args.num_inst):
            if os.path.exists(os.path.join(original_dir, str(i) + "-" + str(j) + args.format)) and os.path.exists(
                    os.path.join(new_dir, str(i) + "-" + str(j) + args.format)):
                flist.append(str(i) + "-" + str(j) + args.format)
    for i in range(args.num_unmonitored):
        if os.path.exists(os.path.join(original_dir, str(i) + args.format)) and os.path.exists(
                os.path.join(new_dir, str(i) + args.format)):
            flist.append(str(i) + args.format)

    ovhds = parallel(flist, args.n_jobs)
    ovhds = list(zip(*ovhds))

    n_original = sum(list(filter(None, ovhds[0])))
    n_defended = sum(list(filter(None, ovhds[1])))
    t_original = sum(list(filter(None, ovhds[2])))
    t_defended = sum(list(filter(None, ovhds[3])))

    # B(D) = (|P'| - |P|) / |P|,  L(D) = (t_k - t_|P|) / t_|P|
    bandwidth_ovhd = (n_defended - n_original) / n_original
    latency_ovhd = (t_defended - t_original) / t_original
    print("{} {:.4f} {:.4f}".format(len(flist), bandwidth_ovhd, latency_ovhd))