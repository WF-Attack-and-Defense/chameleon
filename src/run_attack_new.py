"""
Run WF attacks with a single stratified 8:1:1 train / validation / test split.

After each run, writes 15 (recall, precision) pairs suitable for plotting
precision–recall curves (tab-separated: recall, precision). Output file:
``{attack_model_name}_{name}.txt`` where ``--name`` sets the suffix.
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from attacks import DFAttack, RFAttack, VarCNNAttack, NetCLRAttack
from utils.checkpoint import load_checkpoint, save_final_model_and_cleanup
from utils.general import seed_everything
from utils.logger import init_logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="WF attack with 8:1:1 split and PR sampling")
    parser.add_argument("--attack", choices=["df", "rf", "var_cnn", "netclr"], default="df",
                        help="choose the attack")
    parser.add_argument("--dataset", choices=["DF", "ds-19", "defense", "test"], default="DF",
                        help="choose the dataset")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Suffix for the output file: {attack_model_name}_{name}.txt",
    )

    parser.add_argument("--checkpoints", type=str, default="../checkpoints/",
                        help="location of model checkpoints")
    parser.add_argument("--suffix", type=str, default=".cell", help="suffix of the output file")
    parser.add_argument("--open-world", default=False, action="store_true", help="Open world or not")
    parser.add_argument("--seq-length", default=5000, type=int, help="The input trace length")

    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N",
                        help="mini-batch size (default: 128)")
    parser.add_argument("--lr0", type=float, default=0.002, help="initial optimizer learning rate")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="Adam",
                        help="optimizer")

    parser.add_argument("-j", "--workers", default=10, type=int, metavar="N",
                        help="number of data loading workers (default: 10)")

    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3,4,5,6,7", help="device ids of multiple gpus")
    parser.add_argument("--amp", action="store_true", default=False, help="use mixed precision training")

    parser.add_argument("--verbose", action="store_true", default=False, help="print detailed performance")
    parser.add_argument("--log_itr_interval", type=int, default=100, help="log iteration interval")

    parser.add_argument("--pr-points", type=int, default=15,
                        help="number of (recall, precision) pairs to write for PR curves")
    parser.add_argument("--split-seed", type=int, default=2024,
                        help="random seed for the 8:1:1 stratified split")

    _args = parser.parse_args()
    return _args


def stratified_train_val_test_indices(labels: np.ndarray, train_frac=0.8, val_frac=0.1, test_frac=0.1,
                                      random_state: int = 2024):
    """Return index arrays for stratified 8:1:1 split."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1")
    n = len(labels)
    all_idx = np.arange(n)
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=(val_frac + test_frac),
        stratify=labels,
        random_state=random_state,
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test,
        stratify=labels[temp_idx],
        random_state=random_state,
    )
    return train_idx, val_idx, test_idx


def sample_pr_pairs(precision: np.ndarray, recall: np.ndarray, n_points: int):
    """
    Take sklearn precision_recall_curve outputs and return n_points (recall, precision) pairs
    by evenly subsampling along the curve order.
    """
    m = len(recall)
    if m == 0:
        return np.zeros(n_points), np.zeros(n_points)
    if m <= n_points:
        idx = np.arange(m)
    else:
        idx = np.linspace(0, m - 1, n_points, dtype=int)
    return recall[idx], precision[idx]


def collect_test_scores_open_world(model, test_loader, device, nmc: int):
    """Binary labels: 1 = monitored, 0 = unmonitored. Score = monitored likelihood (same as base Attack)."""
    score_parts = []
    label_parts = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            monitored_score = 1.0 - probs[:, nmc]
            score_parts.append(monitored_score.detach().cpu().numpy())
            label_parts.append((y_batch < nmc).long().detach().cpu().numpy())
    if not score_parts:
        return np.array([]), np.array([])
    y_score = np.concatenate(score_parts, axis=0)
    y_true = np.concatenate(label_parts, axis=0)
    return y_true, y_score


def collect_test_scores_closed_world(model, test_loader, device):
    """
    Binary PR: positive = top-1 prediction is correct; score = max softmax probability (confidence).
    """
    y_correct = []
    y_conf = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            correct = (pred == y_batch).long()
            conf = torch.max(probs, dim=1).values
            y_correct.append(correct.detach().cpu().numpy())
            y_conf.append(conf.detach().cpu().numpy())
    if not y_correct:
        return np.array([]), np.array([])
    return np.concatenate(y_correct, axis=0), np.concatenate(y_conf, axis=0)


def output_pr_filename(attack_type_name: str, name: str) -> str:
    """``{attack_model_name}_{name}.txt`` — sanitize ``name`` so it stays a single filename."""
    safe = name.replace(os.sep, "_").replace("/", "_").strip() or "run"
    return f"{attack_type_name}_{safe}.txt"


def write_pr_file(output_path: str, recall: np.ndarray, precision: np.ndarray, logger):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("recall\tprecision\n")
        for r, p in zip(recall, precision):
            f.write(f"{r:.10f}\t{p:.10f}\n")
    logger.info(f"Saved PR curve points ({len(recall)} pairs) to: {output_path}")


def run_attack_811(attack, args):
    logger = init_logger("run_attack_new")
    flist = attack.flist
    labels = attack.labels

    train_idx, val_idx, test_idx = stratified_train_val_test_indices(
        labels, train_frac=0.8, val_frac=0.1, test_frac=0.1, random_state=args.split_seed
    )

    train_list, train_labels = flist[train_idx], labels[train_idx]
    val_list, val_labels = flist[val_idx], labels[val_idx]
    test_list, test_labels = flist[test_idx], labels[test_idx]

    logger.info(
        f"8:1:1 split — train: {len(train_list)}, val: {len(val_list)}, test: {len(test_list)}"
    )

    fold = 1
    res_one_fold, _, _ = attack.train(
        fold, train_list, train_labels, val_list, val_labels
    )

    attack_type_name = attack.__class__.__name__.lower().replace("attack", "")
    model_checkpoint_dir = os.path.join(args.checkpoints, attack_type_name)
    checkpoint_filename = f"fold{fold}.pth"
    checkpoint_path = os.path.join(model_checkpoint_dir, checkpoint_filename)

    model = attack._build_model().to(attack.device)
    load_checkpoint(checkpoint_path, model=model, device=attack.device)

    _, test_loader = attack._get_data(test_list, test_labels, attack.extract, is_train=False)

    n_points = args.pr_points

    if args.open_world:
        y_true, y_score = collect_test_scores_open_world(
            model, test_loader, attack.device, attack.nmc
        )
        if y_true.size == 0:
            logger.warning("No test samples for PR; skipping PR file.")
        else:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            rec_s, prec_s = sample_pr_pairs(prec, rec, n_points)
            pr_path = output_pr_filename(attack_type_name, args.name)
            write_pr_file(pr_path, rec_s, prec_s, logger)

        tp, fp, p, n = res_one_fold[0], res_one_fold[1], res_one_fold[2], res_one_fold[3]
        tn = n - fp
        fn = p - tp
        tpr = tp / p if p > 0 else 0.0
        fpr = fp / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / p if p > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fn + fp) if (2 * tp + fn + fp) > 0 else 0.0
        logger.info("=" * 50)
        logger.info("Validation-set aggregate after training (last epoch, open-world):")
        logger.info("TP: {:.0f}, FP: {:.0f}, P: {:.0f}, N: {:.0f}".format(tp, fp, p, n))
        logger.info("TPR: {:.4f}, FPR: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
            tpr, fpr, precision, recall, f1))
        logger.info("=" * 50)
    else:
        y_true, y_score = collect_test_scores_closed_world(
            model, test_loader, attack.device
        )
        if y_true.size == 0:
            logger.warning("No test samples for PR; skipping PR file.")
        else:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            rec_s, prec_s = sample_pr_pairs(prec, rec, n_points)
            pr_path = output_pr_filename(attack_type_name, args.name)
            write_pr_file(pr_path, rec_s, prec_s, logger)

        tp, fp, p, n = res_one_fold[0], res_one_fold[1], res_one_fold[2], res_one_fold[3]
        accuracy = tp / p if p > 0 else 0.0
        logger.info("=" * 50)
        logger.info("Validation-set aggregate after training (last epoch, closed-world):")
        logger.info("TP: {:.0f}, P: {:.0f}, Accuracy: {:.4f}".format(tp, p, accuracy))
        logger.info("=" * 50)

    save_final_model_and_cleanup(
        attack_type_name=attack_type_name,
        checkpoints_dir=args.checkpoints,
        device=attack.device,
        model_builder=attack._build_model,
        dataset=args.dataset,
        open_world=args.open_world,
        seq_length=args.seq_length,
    )


if __name__ == "__main__":
    args = parse_arguments()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    data_path = "../datasets/"

    if args.dataset == "DF":
        args.mon_path = data_path + "DF/CW/"
        args.unmon_path = data_path + "DF/OW/"
        args.mon_classes = 95
        args.mon_inst = 1000
        args.unmon_inst = 40716
    elif args.dataset == "ds-19":
        args.mon_path = data_path + "ds-19/CW/"
        args.unmon_path = data_path + "ds-19/OW/"
        args.mon_classes = 100
        args.mon_inst = 100
        args.unmon_inst = 10000
    elif args.dataset == "defense":
        args.mon_path = "../defense_results/chameleon/CW/ds-19/"
        args.unmon_path = "../defense_results/tamaraw/OW/ds-19/"
        args.mon_classes = 100
        args.mon_inst = 100
        args.unmon_inst = 10000
    elif args.dataset == "test":
        args.mon_path = "../defense_results/gapdis/OW/DF_rf/"
        args.unmon_path = "../defense_results/gapdis/OW/DF_rf/"
        args.mon_classes = 95
        args.mon_inst = 1000
        args.unmon_inst = 40716
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    args.checkpoints = os.path.normpath(args.checkpoints)
    if not args.checkpoints.endswith(os.sep):
        args.checkpoints = args.checkpoints + os.sep

    seed_everything(2024)

    attack = None
    if args.attack == "df":
        attack = DFAttack(args)
    elif args.attack == "rf":
        attack = RFAttack(args)
    elif args.attack == "var_cnn":
        attack = VarCNNAttack(args)
    elif args.attack == "netclr":
        attack = NetCLRAttack(args)
    else:
        raise NotImplementedError("Attack not implemented")

    run_attack_811(attack, args)
