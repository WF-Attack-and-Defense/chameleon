import argparse
import os
import importlib.util
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Loss

from attacks import DFAttack, RFAttack, VarCNNAttack, AWFAttack, TiktokAttack, NetCLRAttack
from utils.general import get_flist_label
from utils.metric import WFMetric


def load_attack_model(args: argparse.Namespace):
    """
    Load attack model from model file.
    
    Args:
        args: Arguments namespace with model configuration
    
    Returns:
        Loaded model and attack instance
    """
    # Normalize attack model directory path
    model_file = os.path.normpath(args.checkpoints)
    if not model_file.endswith(os.sep):
        model_file = model_file + os.sep
    
    # Determine model filename based on open-world setting
    attack_model_name = args.attack.lower()
    if '_' in attack_model_name:
        attack_model_name = attack_model_name.replace('_', '')
    suffix = "OW" if args.open_world else "CW"
    model_filename = f"{attack_model_name}_{args.dataset}_{suffix}.h5"
    model_path = os.path.join(model_file, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Attack model not found: {model_path}. Please train the attack model first using run_attack.py")
    
    # Create attack instance to get model class and extract function
    attack_classes = {
        'df': DFAttack,
        'rf': RFAttack,
        'var_cnn': VarCNNAttack,
        'awf': AWFAttack,
        'tiktok': TiktokAttack,
        'netclr': NetCLRAttack,
    }
    
    if args.attack.lower() not in attack_classes:
        raise ValueError(f"Unknown attack type: {args.attack}")
    
    AttackClass = attack_classes[args.attack.lower()]
    
    # Create attack instance
    attack = AttackClass(args)
    
    # Load trained model
    model = attack._build_model()
    attack_model = torch.load(model_path, map_location=attack.device)
    # Handle both state dict and full checkpoint formats
    if isinstance(attack_model, dict) and 'model_state_dict' in attack_model:
        model.load_state_dict(attack_model['model_state_dict'])
    else:
        # Assume it's just the state dict
        model.load_state_dict(attack_model)
    model = model.to(attack.device)
    model.eval()
    
    return model, attack


def verify_defense_with_attack(args: argparse.Namespace, logger):
    """
    Verify defense effectiveness using attack models.
    
    Args:
        defense_mon_dir: Path to defense-generated monitored traces
        defense_unmon_dir: Path to defense-generated unmonitored traces
        attack_type: Type of attack to use for verification
        model_file: Path to the attack model file
        args: Arguments namespace
        logger: Logger instance
    """
    # Load attack model
    try:
        model, attack = load_attack_model(args)
        logger.info(f"Attack model loaded successfully: {args.attack.upper()}")
    except Exception as e:
        logger.error(f"Failed to load attack model: {e}")
        return
    
    # Get defended trace file list (from defense output directory)
    if not os.path.exists(args.defense_mon_path):
        logger.error(f"Defended traces directory not found: {args.defense_mon_path}")
        logger.error("Please run defense simulation first")
        return
    
    # Get file list and labels for defended traces
    defended_flist, defended_labels = get_flist_label(
        args.defense_mon_path,
        args.defense_unmon_path if args.open_world else None,
        mon_cls=args.defense_mon_classes,
        mon_inst=args.defense_mon_inst,
        unmon_inst=args.defense_unmon_inst,
        suffix=args.suffix
    )
    
    logger.info(f"Found {len(defended_flist)} defended traces")
    
    # Use same 90/10 split as attacks
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=2024)
    train_index, test_index = next(sss.split(defended_flist, defended_labels))
    
    train_list = defended_flist[train_index]
    train_labels = defended_labels[train_index]
    test_list = defended_flist[test_index]
    test_labels = defended_labels[test_index]
    
    logger.info(f"Train: {len(train_list)}, Test: {len(test_list)}")
    
    # Evaluate on test set
    _, test_loader = attack._get_data(test_list, test_labels, attack.extract, is_train=False)
    
    # Evaluate model
    criterion = nn.CrossEntropyLoss()
    val_metrics = {
        "accuracy": WFMetric(attack.nmc),
        "loss": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=attack.device)
    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    
    # Extract and log results
    tp, fp, p, n = metrics['accuracy'][0], metrics['accuracy'][1], metrics['accuracy'][2], metrics['accuracy'][3]
    
    if args.open_world:
        tn = n - fp
        fn = p - tp
        tpr = tp / p if p > 0 else 0.0
        fpr = fp / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / p if p > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fn + fp) if (2 * tp + fn + fp) > 0 else 0.0
        
        logger.info("=" * 50)
        logger.info(f"Defense Verification Results (Open-World) - {args.attack.upper()} Attack:")
        logger.info("TP: {:.0f}, FP: {:.0f}, P: {:.0f}, N: {:.0f}".format(tp, fp, p, n))
        logger.info("TPR: {:.4f} ({:.2f}%)".format(tpr, tpr * 100))
        logger.info("FPR: {:.4f} ({:.2f}%)".format(fpr, fpr * 100))
        logger.info("Precision: {:.4f} ({:.2f}%)".format(precision, precision * 100))
        logger.info("Recall: {:.4f} ({:.2f}%)".format(recall, recall * 100))
        logger.info("F1: {:.4f} ({:.2f}%)".format(f1, f1 * 100))
        logger.info("=" * 50)
    else:
        accuracy = tp / p if p > 0 else 0.0
        logger.info("=" * 50)
        logger.info(f"Defense Verification Results (Closed-World) - {args.attack.upper()} Attack:")
        logger.info("TP: {:.0f}, P: {:.0f}".format(tp, p))
        logger.info("Accuracy: {:.4f} ({:.2f}%)".format(accuracy, accuracy * 100))
        logger.info("=" * 50)