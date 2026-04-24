import argparse
import os
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from utils.data import MyDataset
from utils.general import get_flist_label
from utils.general import timeit
from utils.logger import init_logger
from utils.metric import WFMetric
from utils.checkpoint import save_checkpoint, load_checkpoint, del_checkpoint, save_final_model_and_cleanup


class Attack(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = init_logger(str(self.__class__.__name__))

        self.device = self._acquire_device()

        self.nmc = args.mon_classes  # number of monitored classes
        self.nc = args.mon_classes + 1 if args.open_world else args.mon_classes  # number of total classes
        self.unmon_inst = args.unmon_inst if args.open_world else 0
        
        # Load data - get_flist_label handles both open-world and closed-world cases
        self.logger.info("Loading dataset as type {} ({} world)...".format(
            self.__class__.__name__.lower().replace('attack', ''),
            'open' if args.open_world else 'closed'
        ))
        self.flist, self.labels = get_flist_label(
            args.mon_path,
            args.unmon_path,
            mon_cls=self.nmc,
            mon_inst=args.mon_inst,
            unmon_inst=self.unmon_inst,
            suffix=args.suffix
        )
        self.logger.info("Total number of data: {}".format(len(self.flist)))

    def _build_model(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def extract(data_path: Union[str, os.PathLike], seq_length: int) -> np.ndarray:
        """
        Input format from the raw traces
        """
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            self.logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            self.logger.info('Use CPU')
        return device

    def _get_data(self, flist: np.ndarray, labels: np.ndarray, feat_extract_func: Callable, is_train=True) -> (
            MyDataset, DataLoader):
        dataset = MyDataset(self.args, flist, labels, feat_extract_func)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=is_train,
                            num_workers=self.args.workers)
        return dataset, loader

    @timeit
    def run(self, one_fold_only: bool = False):
        res = np.zeros(4)  # tp, fp, p, n
        all_open_world_scores = []
        all_open_world_labels = []
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
        for fold, (train_index, test_index) in enumerate(sss.split(self.flist, self.labels)):
            if one_fold_only and fold > 0:
                break
            train_list, train_labels = self.flist[train_index], self.labels[train_index]
            test_list, test_labels = self.flist[test_index], self.labels[test_index]
            res_one_fold, fold_scores, fold_labels = self.train(
                fold + 1, train_list, train_labels, test_list, test_labels
            )
            res += res_one_fold
            if self.args.open_world and fold_scores.size > 0 and fold_labels.size > 0:
                all_open_world_scores.append(fold_scores)
                all_open_world_labels.append(fold_labels)
            self.logger.info("-" * 10)
        
        # Extract values
        tp, fp, p, n = res[0], res[1], res[2], res[3]
        
        # Calculate and output metrics based on open-world setting
        if self.args.open_world:
            # Open-world metrics: TPR, FPR, F1, Precision, Recall
            # TN = correctly classified unmonitored = N - FP
            # FN = monitored misclassified as unmonitored = P - TP
            tn = n - fp
            fn = p - tp
            
            # TPR = TP / (TP + FN) = TP / P
            tpr = tp / p if p > 0 else 0.0
            
            # FPR = FP / (FP + TN) = FP / N
            fpr = fp / n if n > 0 else 0.0
            
            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall = TP / (TP + FN) = TP / P (same as TPR)
            recall = tp / p if p > 0 else 0.0
            
            # F1 = 2TP / (2TP + FN + FP)
            f1 = (2 * tp) / (2 * tp + fn + fp) if (2 * tp + fn + fp) > 0 else 0.0
            
            self.logger.info("=" * 50)
            self.logger.info("Final Results (Open-World):")
            self.logger.info("TP: {:.0f}, FP: {:.0f}, P: {:.0f}, N: {:.0f}".format(tp, fp, p, n))
            self.logger.info("TPR: {:.4f} ({:.2f}%)".format(tpr, tpr * 100))
            self.logger.info("FPR: {:.4f} ({:.2f}%)".format(fpr, fpr * 100))
            self.logger.info("Precision: {:.4f} ({:.2f}%)".format(precision, precision * 100))
            self.logger.info("Recall: {:.4f} ({:.2f}%)".format(recall, recall * 100))
            self.logger.info("F1: {:.4f} ({:.2f}%)".format(f1, f1 * 100))
            self.logger.info("=" * 50)
            if all_open_world_scores and all_open_world_labels:
                y_score = np.concatenate(all_open_world_scores, axis=0)
                y_true = np.concatenate(all_open_world_labels, axis=0)
                roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true, y_score)
                roc_output_path = f"{self.args.attack}_{self.args.dataset}.txt"
                with open(roc_output_path, "w", encoding="utf-8") as f:
                    f.write("fpr\ttpr\tthreshold\n")
                    for fpr_i, tpr_i, thr_i in zip(roc_fpr, roc_tpr, roc_thresholds):
                        f.write(f"{fpr_i:.10f}\t{tpr_i:.10f}\t{thr_i:.10f}\n")
                self.logger.info(f"Saved ROC points to: {roc_output_path}")
        else:
            # Closed-world metric: Accuracy
            # Accuracy = TP / P (since all samples are monitored, N=0)
            accuracy = tp / p if p > 0 else 0.0
            
            self.logger.info("=" * 50)
            self.logger.info("Final Results (Closed-World):")
            self.logger.info("TP: {:.0f}, P: {:.0f}".format(tp, p))
            self.logger.info("Accuracy: {:.4f} ({:.2f}%)".format(accuracy, accuracy * 100))
            self.logger.info("=" * 50)
        
        # After successful training completion, clean up checkpoints and save final model
        attack_type_name = self.__class__.__name__.lower().replace('attack', '')
        save_final_model_and_cleanup(
            attack_type_name=attack_type_name,
            checkpoints_dir=self.args.checkpoints,
            device=self.device,
            model_builder=self._build_model,
            dataset=self.args.dataset,
            open_world=self.args.open_world,
            seq_length=self.args.seq_length,
        )

    def train(self, fold: int, train_list: np.ndarray, train_labels: np.ndarray, val_list: np.ndarray,
              val_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, train_loader = self._get_data(train_list, train_labels, self.extract, is_train=True)
        _, val_loader = self._get_data(val_list, val_labels, self.extract, is_train=False)

        model = self._build_model().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adamax(model.parameters(), lr=self.args.lr0)
        
        # Automatically check for checkpoint to resume training
        start_epoch = 0
        attack_type_name = self.__class__.__name__.lower().replace('attack', '')
        model_checkpoint_dir = os.path.join(self.args.checkpoints, attack_type_name)
        checkpoint_filename = f"fold{fold}.pth"
        checkpoint_path = os.path.join(model_checkpoint_dir, checkpoint_filename)
        
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Found checkpoint, resuming training from: {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer, device=self.device)
            start_epoch = checkpoint.get('epoch', 0)
            self.logger.info(f"Resumed from epoch {start_epoch}")
        else:
            self.logger.info(f"No checkpoint found at {checkpoint_path}, starting from scratch")

        amp_mode = 'amp' if self.args.amp and self.device != torch.device("cpu") else None

        trainer = create_supervised_trainer(model, optimizer, criterion, self.device, amp_mode=amp_mode)
        val_metrics = {
            "accuracy": WFMetric(self.nmc),
            "loss": Loss(criterion)
        }
        val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self.device, amp_mode=amp_mode)
        
        # Note: PyTorch Ignite doesn't directly support resuming from a specific epoch
        # The checkpoint is loaded, but training will start from epoch 1
        # The model and optimizer states are restored, which is the most important part

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_loss(engine: Engine):
            self.logger.info(f"Fold[{fold}] | Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] |"
                             f" Loss: {engine.state.output:.2f}")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine: Engine):
            val_evaluator.run(val_loader)
            _metrics = val_evaluator.state.metrics
            self.logger.info(
                f"Validation Results - Fold[{fold}] Epoch[{engine.state.epoch}] | "
                f"Avg loss: {_metrics['loss']:.2f} | "
                f"tp: {_metrics['accuracy'][0]:4.0f} fp: {_metrics['accuracy'][1]:4.0f} "
                f"p: {_metrics['accuracy'][2]:4.0f} n: {_metrics['accuracy'][3]:4.0f}"
            )
            
            # Save checkpoint after each epoch for recovery from crashes/interruptions
            # Always save checkpoints to enable recovery
            attack_type_name = self.__class__.__name__.lower().replace('attack', '')
            model_checkpoint_dir = os.path.join(self.args.checkpoints, attack_type_name)
            os.makedirs(model_checkpoint_dir, exist_ok=True)
            checkpoint_filename = f"fold{fold}.pth"
            checkpoint_path = os.path.join(model_checkpoint_dir, checkpoint_filename)
                
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=engine.state.epoch,
                metrics={
                    'tp': _metrics['accuracy'][0],
                    'fp': _metrics['accuracy'][1],
                    'p': _metrics['accuracy'][2],
                    'n': _metrics['accuracy'][3],
                    'loss': _metrics['loss']
                },
                filepath=checkpoint_path,
                attack_type=self.__class__.__name__,
                fold=fold,
                num_classes=self.nc,
                seq_length=self.args.seq_length
            )

        # Run training (model and optimizer states are restored if checkpoint was loaded)
        trainer.run(train_loader, max_epochs=self.args.epochs)
        metrics = val_evaluator.state.metrics
        fold_scores = np.array([], dtype=float)
        fold_labels = np.array([], dtype=int)
        if self.args.open_world:
            model.eval()
            score_parts = []
            label_parts = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = model(x_batch)
                    probs = torch.softmax(logits, dim=1)
                    # Binary score for monitored-vs-unmonitored ROC:
                    # higher score => more likely monitored.
                    monitored_score = 1.0 - probs[:, self.nmc]
                    score_parts.append(monitored_score.detach().cpu().numpy())
                    label_parts.append((y_batch < self.nmc).long().detach().cpu().numpy())
            if score_parts:
                fold_scores = np.concatenate(score_parts, axis=0)
                fold_labels = np.concatenate(label_parts, axis=0)

        torch.cuda.empty_cache()
        return np.array(metrics['accuracy']), fold_scores, fold_labels

    def test(self):
        pass

    def vali(self, **kwargs):
        pass
