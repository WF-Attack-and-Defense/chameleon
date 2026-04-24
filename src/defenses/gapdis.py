import argparse
import os
import random
import copy
from pathlib import Path
from typing import Union, List
from collections import defaultdict, namedtuple, Counter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from itertools import chain

import time

import numpy as np

from defenses.base import Defense
from defenses.config import GapdisConfig
from utils.general import parse_trace, set_random_seed
from utils.perturb_util import load_attack_model
from attacks.modules import DFNet, RFNet, VarCNNNet, AWFNet, NetCLRNet


# Helper classes and functions from tools.py
class TabuTable:
    def __init__(self, max_length):
        self.max_length = max_length
        self.tabu_dict = {}  # main table：{hash: {"frequency": freq, "list_data": lst}}
        self.frequency_dict = defaultdict(set)  # record item frequency：{freq: {hash1, hash2, ...}}
        self.min_frequency = 0

    def compute_hash(self, lst: list):
        return hash(frozenset(lst))

    def _remove_least_frequent(self):
        if self.min_frequency in self.frequency_dict and self.frequency_dict[self.min_frequency]:
            # find lowest frequency item
            hash_to_remove = self.frequency_dict[self.min_frequency].pop()
            if not self.frequency_dict[self.min_frequency]:  # if set() is empty
                del self.frequency_dict[self.min_frequency]
            del self.tabu_dict[hash_to_remove]  # del main table item

    def insert(self, lst: list):
        hash_value = self.compute_hash(lst)
        if hash_value in self.tabu_dict:
            return False
        self.tabu_dict[hash_value] = {"frequency": 0, "list_data": lst}
        self.frequency_dict[0].add(hash_value)
        self.min_frequency = 0

        if len(self.tabu_dict) > self.max_length:
            self._remove_least_frequent()

    def query(self, lst: list):
        hash_value = self.compute_hash(lst)

        if hash_value in self.tabu_dict:  # update frequency
            current_freq = self.tabu_dict[hash_value]["frequency"]
            self.tabu_dict[hash_value]["frequency"] += 1

            self.frequency_dict[current_freq].remove(hash_value)  # move curr freq to +1 freq set
            if not self.frequency_dict[current_freq]:
                del self.frequency_dict[current_freq]
            new_freq = current_freq + 1
            self.frequency_dict[new_freq].add(hash_value)

            if not self.frequency_dict[self.min_frequency]:  # update min_frequency
                self.min_frequency += 1
            return True
        return False

    def delete(self, lst: list):
        hash_value = self.compute_hash(lst)
        if hash_value in self.tabu_dict:
            current_freq = self.tabu_dict[hash_value]["frequency"]
            del self.tabu_dict[hash_value]

            self.frequency_dict[current_freq].remove(hash_value)
            if not self.frequency_dict[current_freq]:
                del self.frequency_dict[current_freq]

            if not self.frequency_dict[self.min_frequency]:
                self.min_frequency += 1

    def __len__(self):
        return len(self.tabu_dict)

    def __str__(self):
        return f"TabuTable: {self.tabu_dict}"


Solution = namedtuple("Solution", ["reward", "m", "position", "mode", "perturbations", "acc"])


class SolutionList:
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.solutions = []

    def add_solution(self, sol: Solution) -> list:
        remove_pert = None
        if len(self.solutions) >= self.max_length:  # find largest acc solutions
            max_acc = max(self.solutions, key=lambda sol: sol.acc).acc
            max_acc_items = [sol for sol in self.solutions if sol.acc == max_acc]
            to_remove = random.choice(max_acc_items)
            remove_pert = get_perturbation_of_solution(to_remove)
            self.solutions.remove(to_remove)
        self.solutions.append(sol)
        return remove_pert

    def pop(self):
        if not self.solutions:
            return None
        min_acc = min(self.solutions, key=lambda sol: sol.acc).acc
        min_acc_items = [sol for sol in self.solutions if sol.acc == min_acc]

        best_item = max(min_acc_items, key=lambda sol: sol.reward)
        self.solutions.remove(best_item)
        return best_item

    def prob_pop(self):
        if not self.solutions:
            return None
        # weight is 1 - acc
        weights = [1 - sol.acc for sol in self.solutions]
        total_weight = sum(weights)
        if total_weight == 0:
            selected_item = random.choice(self.solutions)
        else:
            probabilities = [w / total_weight for w in weights]
            selected_item = random.choices(self.solutions, weights=probabilities, k=1)[0]
        self.solutions.remove(selected_item)
        return selected_item

    def __len__(self):
        return len(self.solutions)

    def __str__(self):
        return f"SolutionList({[sol for sol in self.solutions]})"


CriticalPosition = namedtuple("CriticalPosition", ["m", "position", "acc_drop"])


class CriticalPositionManager:
    def __init__(self, capacity):
        self.capacity = capacity
        self.position_map = {}  # position as key，save CriticalPosition

    def update(self, critical_position: CriticalPosition):
        pos = critical_position.position
        acc_drop = critical_position.acc_drop
        if acc_drop <= 0:
            return
        if len(self.position_map) == self.capacity and \
                acc_drop <= min(self.position_map.items(), key=lambda item: item[1].acc_drop)[1].acc_drop:
            return

        if pos in self.position_map:  # if pos exist
            existing = self.position_map[pos]
            if acc_drop > existing.acc_drop:
                self.position_map[pos] = critical_position
        else:  # insert
            self.position_map[pos] = critical_position

        if len(self.position_map) > self.capacity:
            min_pos, min_cp = min(self.position_map.items(), key=lambda item: item[1].acc_drop)
            del self.position_map[min_pos]

    def roulette_sample(self):
        if not self.position_map:
            return None
        positions, critical_positions = zip(*self.position_map.items())
        chosen = random.choices(critical_positions, weights=[cp.acc_drop for cp in critical_positions], k=1)
        return chosen[0]

    def sample(self, curr_pert, length_limit, return_CriticalPosition=False):
        curr_count = Counter(curr_pert)
        valid_positions = [cp for cp in self.position_map.values() if curr_count[cp.position] < cp.m]
        if not valid_positions:
            if return_CriticalPosition:
                return curr_pert, None
            return curr_pert
        # roulette_wheel_selection
        chosen = random.choices(valid_positions, k=1)[0]  # , weights=[cp.acc_drop for cp in valid_positions]
        insert_count = min(chosen.m - curr_count[chosen.position], length_limit - len(curr_pert))
        curr_pert.extend([chosen.position] * insert_count)

        if return_CriticalPosition:
            return curr_pert, CriticalPosition(m=insert_count, position=chosen.position, acc_drop=None)
        return curr_pert[:length_limit]

    def __str__(self):
        return str(self.position_map)


def get_perturbation_of_solution(sol: Solution) -> List[Union[int, float]]:
    position = int(sol.position)
    if sol.mode == 'insert':
        return list(chain(sol.perturbations, [position] * sol.m))
    elif sol.mode == 'delete':
        m = sol.m
        counter = Counter(sol.perturbations)
        counter[position] -= m
        return [x for x in sol.perturbations if counter[x] > 0 or (x != position)]
    else:
        raise ValueError(f"Invalid mode {sol.mode}, and sol object:", sol)
    return []


class BestSolutionTracker:
    def __init__(self, track_by_length=False):
        self.global_best = None
        self.track_by_length = track_by_length  # whether record best solution by len(perturbations)
        self.best_by_length = {}  # best solution of len(perturbations)

    def update(self, acc, perturbations):
        is_global_best = False
        # check global_best
        if self.global_best is None or acc < self.global_best["acc"]:
            self.global_best = {"acc": acc, "perturbations": perturbations}
            is_global_best = True
        # check len(perturbations) best
        if self.track_by_length:
            length = len(perturbations)
            if length not in self.best_by_length or acc < self.best_by_length[length]["acc"]:
                self.best_by_length[length] = {"acc": acc, "perturbations": perturbations}  # update
        return is_global_best

    def get_global_best(self):
        return self.global_best

    def get_best_by_length(self, length):
        return self.best_by_length.get(length, None)

    def __str__(self):
        # Format global best solution
        global_best_str = (
            f"Global Best - acc: {self.global_best['acc']}, "
            f"perturbations: {self.global_best['perturbations']}"
            if self.global_best
            else "Global Best - None"
        )

        # Format best_by_length solutions
        best_by_length_str = "Best by Length:\n"
        if self.track_by_length and self.best_by_length:
            for length, solution in sorted(self.best_by_length.items()):
                best_by_length_str += (
                    f"  Length {length}: acc: {solution['acc']}, "
                    f"perturbations: {solution['perturbations']}\n"
                )
        else:
            best_by_length_str += "  None"

        # Combine and return
        return f"{global_best_str}\n{best_by_length_str}"


def _attack_is_varcnn(attack: Union[str, None]) -> bool:
    if not attack:
        return False
    return str(attack).lower().replace("_", "") == "varcnn"


def _pack_directions_for_varcnn(data: torch.Tensor) -> torch.Tensor:
    """VarCNN expects (B, 2, L) inter-time + direction; gapdis keeps a 1D direction track."""
    if data.dim() == 3:
        dirs = data.squeeze(1)
    else:
        dirs = data
    b, l = dirs.shape
    inter = torch.zeros(b, l, device=data.device, dtype=data.dtype)
    if l > 1:
        inter[:, 1:] = 1.0
    return torch.stack([inter, dirs], dim=1)


def perturbation_replace_by_best(curr: list, best: list, ex_length: int, lst_len_limit: int,
                                 weighted_choice: bool = False, return_CriticalPosition=False):
    # counter frequency
    freq = Counter(best)
    # get element that freq ≤ ex_length
    candidates = [(elem, count) for elem, count in freq.items() if count <= ex_length]
    if not candidates:
        if return_CriticalPosition:
            return curr, None
        return curr
    if weighted_choice:  # weighted select
        elements, weights = zip(*candidates)
        a = random.choices(elements, weights=weights, k=1)[0]
        X = dict(candidates)[a]
    else:  # uniform select
        a, X = random.choice(candidates)
    b = random.choice(curr)
    b_indices = [i for i, val in enumerate(curr) if val == b]  # find b indices
    # replace b to a
    if len(b_indices) >= X:  # if b num > a num
        for i in b_indices[:X]:
            curr[i] = a
        curr = [val for i, val in enumerate(curr) if i not in b_indices[X:]]  # del others b
    else:  # b num <= a num
        for i in b_indices:
            curr[i] = a
        curr.extend([a] * (X - len(b_indices)))
    if return_CriticalPosition:
        return curr[:lst_len_limit], CriticalPosition(m=X, position=a, acc_drop=None)
    return curr[:lst_len_limit]


def perturbation_gene_mutation(curr: list, max_num, ex_length: int, return_CriticalPosition=False):
    # random select sublist from best
    selected_elements = random.randint(0, max_num)
    replace_start = random.randint(0, len(curr) - ex_length)
    curr[replace_start:replace_start + ex_length] = [selected_elements] * ex_length
    if return_CriticalPosition:
        return curr, CriticalPosition(m=ex_length, position=selected_elements, acc_drop=None)
    return curr


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=False, delta=0, logging=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.perturbations = []
        self.early_stop = False
        self.delta = delta
        self.logging = logging

    def __call__(self, val_loss, perturbations: list):
        if self.best_score is None or val_loss > self.best_score + self.delta:
            self.best_score = val_loss
            self.perturbations = perturbations.copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.logging:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def read_best_perturbations(self):
        self.counter = 0
        self.early_stop = False
        return self.perturbations.copy()

    def __str__(self):
        return (f'EarlyStopping counter: {self.counter}, Best loss: {self.best_score}, '
                f'Best perturbation: {self.perturbations}')


def label_accuracy(pred, target):
    """Computes the accuracy of model predictions matching the target labels"""
    if len(pred) == 0 or len(target) == 0:
        return 100.0
    batch_size = len(target)
    correct = np.sum(pred == target)
    accuracy = correct / batch_size * 100.0
    return accuracy


class DynamicDataset(Dataset):
    """PyTorch Dataset equivalent of TF DynamicDataset."""
    def __init__(self, x, y, batch_size=512, return_idx=True, device=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        self.x = x
        self.y = y
        self.indices = None
        self.return_idx = return_idx
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if self.return_idx:
            return self.x[idx], self.y[idx], idx
        return self.x[idx], self.y[idx]

    def get_dataset(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def setX(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B,F]-->[B,C,F]
        self.x = x

    def setXY(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B,F]-->[B,C,F]
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

class GAPDiS:
    """Lightweight GAPDiS helper used by `GapdisDefense`.

    In this project we do not rely on the original training/codebook
    interface from `GAPDiS_torch.py`. The heavy search logic is
    implemented in `GapdisDefense.generate_perturbations_simplified`,
    and this class provides only perturbation bookkeeping utilities
    and data transformation helpers.
    """

    def __init__(self, model, x, y, perturbations=None, device=None, attack: str = None):
        self.model = model
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
             p.requires_grad = False
        self.perturbations = perturbations or []
        self._pack_varcnn_input = _attack_is_varcnn(attack)
        self.Feat_dim = x.shape[-1]
        self.criterion = F.cross_entropy

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        self.dyn_dataset = DynamicDataset(x, y, device=self.device)

        self.origin_x = self.dyn_dataset.getX().clone().to(self.device)
        if len(self.perturbations) != 0:
            x = self.get_perturbed_data(self.origin_x, self.perturbations)
            self.dyn_dataset.setX(x)


    def delete_generated_perturbations(self):
        self.perturbations = []

    def generate_adversarial_examples_tabu_search(self, max_len_perturbations, max_iter=0, target_acc=0, topk_num=50,
                                                  m_max=8, tabu_len_multi=5, sol_len_multi=2, cpm_len_den=2, init_rd_num=20,
                                                  init_m_num=8, exch_len=16, repl_rate=0.1, muta_rate=0.2, smp_cpm_rate=0.2,
                                                  toler=30, max_iter_multi=8, args=None):

        # Create a tracker to store the best (lowest‑accuracy) solution found so far
        bst_sol = BestSolutionTracker(track_by_length=True)
        # If max_iter is not specified, scale it with the allowed perturbation length
        if max_iter == 0:
            max_iter = max_iter_multi * max_len_perturbations

        # Tabu table to avoid revisiting recently explored perturbation patterns
        tabu_table = TabuTable(max_length=topk_num * tabu_len_multi)
        # Solution list that maintains a pool of promising candidate solutions
        sol_lst = SolutionList(max_length=topk_num * sol_len_multi)
        # Manager that tracks positions that are likely to be critical for the attack
        cpm = CriticalPositionManager(capacity=max_len_perturbations//cpm_len_den)
        # Iteration counter for the tabu search loop
        epoch = 0
        # Early stopping controller to abort when no further improvements occur
        estp = EarlyStopping(patience=150)

        # Initialize the solution list with random insertion‑based solutions
        for i in range(init_rd_num):
            sol_lst.add_solution(Solution(m=init_m_num, position=random.randint(0, self.Feat_dim - max_len_perturbations),
                                          reward=0, mode='insert', acc=90, perturbations=[]))
        # Main search loop runs until reaching the maximum number of iterations
        while epoch < max_iter:
            # Increment the iteration counter
            epoch += 1
            # Initialize current perturbation, previous solution and critical position
            curr_pert, pre_sol, curr_crit_pos = [], None, None
            # If there are candidate solutions, sample one according to its probability
            if sol_lst.__len__() > 0:
                pre_sol = sol_lst.prob_pop()
                curr_pert = pre_sol.perturbations

                # With probability repl_rate, try to partially replace the perturbation using the global best
                is_replace = random.random() < repl_rate
                if is_replace and bst_sol.get_global_best() is not None and \
                        min(len(curr_pert), len(bst_sol.get_global_best()['perturbations'])) > exch_len:
                    # Replace a segment of the current perturbation with that from the best solution
                    curr_pert, _ = perturbation_replace_by_best(curr_pert, bst_sol.get_global_best()['perturbations'],
                                                                ex_length=exch_len, lst_len_limit=max_len_perturbations,
                                                                weighted_choice=False, return_CriticalPosition=True)
                    # Record the new critical position if returned
                    curr_crit_pos = _ or curr_crit_pos
                else:
                    # Otherwise, with probability depending on previous accuracy, mutate the perturbation
                    is_mutation = random.random() < pre_sol.acc/100*muta_rate
                    if is_mutation and len(curr_pert) > exch_len:
                        # Apply genetic‑style mutation on a random segment of the perturbation
                        curr_pert, _ = perturbation_gene_mutation(curr_pert, self.Feat_dim - max_len_perturbations
                                                                  , exch_len, return_CriticalPosition=True)
                        # Record the new critical position if returned
                        curr_crit_pos = _ or curr_crit_pos
                    else:
                        # Otherwise, possibly sample a perturbation guided by the critical position manager
                        is_sample_from_cpm = random.random() < smp_cpm_rate*(1-pre_sol.acc/100)
                        if is_sample_from_cpm:
                            # Extend or adjust current perturbation using positions considered critical
                            curr_pert, _ = cpm.sample(curr_pert, max_len_perturbations, return_CriticalPosition=True)
                            # Record the new critical position if returned
                            curr_crit_pos = _ or curr_crit_pos
                        else:
                            # Fall back to using perturbation reconstructed from the previous solution
                            curr_pert = get_perturbation_of_solution(pre_sol)
                            # If the previous action was insertion, construct an initial critical position object
                            if pre_sol.mode == 'insert':
                                curr_crit_pos = CriticalPosition(m=pre_sol.m, position=int(pre_sol.position), acc_drop=None)

            # Fetch original labels for the dynamic dataset
            y = self.dyn_dataset.getY()
            # Start from a clean copy of the original input features
            x = self.origin_x.clone()
            # Apply the current perturbation pattern to obtain adversarial inputs
            x = self.get_perturbed_data(x, curr_pert)
            # Replace dataset contents with the perturbed inputs and original labels
            self.dyn_dataset.setXY(x, y)

            # Upper bound on number of inserted 1s for local search at this step
            m_insert_limit = min(m_max + 1, max_len_perturbations - len(curr_pert) + 1)
            # Upper bound on deletions based on the longest continuous insertion segment
            m_delete_limit = min(m_max + 1, Counter(curr_pert).most_common(1)[0][1] + 1 if len(curr_pert) > 0 else 0)
            # Evaluate the current perturbation and compute gradient‑based rewards for possible insert/delete moves
            sum_loss, sum_grads, sum_rewards_dict, del_rewards_dict, acc_eqY = \
                self.__evaluate_and_get_rewards__(curr_pert, m_insert_limit, m_delete_limit, self.dyn_dataset.get_dataset(), m_max)
            # Log current iteration statistics including loss and accuracy
            print(f'Epoch[{epoch}]\tloss={sum_loss} Acc(eq_Y)={acc_eqY} preAcc={pre_sol.acc if pre_sol else None}',
                  end='\n')
            # Accuracy change relative to the previous solution (negative is good for the attacker)
            acc_drop = acc_eqY - pre_sol.acc if pre_sol is not None else 0
            # If we have a critical position, update its observed impact and feed it back to CPM
            if curr_crit_pos is not None:
                curr_crit_pos = curr_crit_pos._replace(acc_drop=acc_drop)
                cpm.update(curr_crit_pos)
            # Update global best solution if the new perturbation yields lower accuracy
            is_best = bst_sol.update(acc_eqY, curr_pert)
            if is_best:
                # Report whenever a new best (stronger) attack is found
                print(f'Found new lowest acc:{acc_eqY}', curr_pert)
            # Stop early if the target accuracy threshold has already been reached
            if bst_sol.global_best["acc"] <= target_acc:
                break
            # Update early stopping state based on the negative accuracy (treat low accuracy as improvement)
            estp(-acc_eqY, curr_pert)
            # If early stopping decides the process has converged, exit the loop
            if estp.early_stop:
                print('Early stopping with patience', estp.patience)
                break

            # Threshold for tolerable degradation relative to the previous solution
            tolerance = toler
            # If current accuracy is much worse than previous (by more than tolerance), discard this branch
            if pre_sol is not None and acc_eqY > pre_sol.acc + tolerance:
                continue

            # List that will store all candidate neighbor solutions (insert/delete moves)
            candidate_lst = []
            # Explore all possible insertion moves with m from 1 to m_insert_limit-1
            for m in range(1, m_insert_limit):
                # Mask out positions already occupied by insertions in the final sequence
                repeat_mask = self.get_repeat_mask(self.perturbations, self.Feat_dim)
                # Mask to avoid inserting too close to the end and violating maximum perturbation length
                truncation_mask = [False] + [True]*(self.Feat_dim-max_len_perturbations-1) + [False]*max_len_perturbations
                # Convert truncation mask to tensor on the correct device
                truncation_mask = torch.tensor(truncation_mask, dtype=torch.bool, device=self.device)
                # Valid positions are those with finite rewards, not repeated and within truncation bounds
                sum_mask = (~torch.isnan(sum_rewards_dict[m]) & (repeat_mask == 1) & truncation_mask)
                # Extract indices of valid locations
                valid_pos = torch.arange(self.Feat_dim, dtype=torch.long, device=self.device)[sum_mask]

                # Gather rewards corresponding to all valid locations
                selected_rewards = sum_rewards_dict[m][valid_pos]
                if len(valid_pos) > 0:
                    # Limit the number of top positions we consider to topk_num_actual
                    topk_num_actual = min(len(valid_pos), topk_num)
                    # Top‑k positions that most increase the loss (good for attacker)
                    topk_values, topk_pos = torch.topk(selected_rewards, topk_num_actual)
                    # Top‑k positions with lowest rewards (for diversity / exploration)
                    topk_values_ver, topk_pos_ver = torch.topk(selected_rewards, topk_num_actual, largest=False)

                    # Generate insertion candidate solutions at each high‑reward position
                    for i in range(len(topk_pos)):
                        sol = Solution(m=m, position=int(valid_pos[topk_pos[i]].item()), reward=topk_values[i].item(), mode='insert',
                                       perturbations=curr_pert, acc=acc_eqY)
                        # Keep only non‑tabu solutions
                        if not tabu_table.query(get_perturbation_of_solution(sol)):
                            candidate_lst.append(sol)
                        # Also add candidates from low‑reward positions to enrich search diversity
                        sol = Solution(m=m, position=int(valid_pos[topk_pos_ver[i]].item()), reward=topk_values_ver[i].item(),
                                       mode='insert', perturbations=curr_pert, acc=acc_eqY)
                        if not tabu_table.query(get_perturbation_of_solution(sol)):
                            candidate_lst.append(sol)
            # Explore all possible deletion moves with m from 1 to m_delete_limit-1
            for m in range(1, m_delete_limit):
                # Positions where deletion reward is defined (not NaN)
                sum_mask = ~torch.isnan(del_rewards_dict[m])
                # Indices of deletable positions in the current sequence
                valid_pos = torch.arange(self.Feat_dim, dtype=torch.long, device=self.device)[sum_mask]
                if len(valid_pos) > 0:
                    # Rewards for deleting m consecutive insertions starting at each valid position
                    selected_rewards = del_rewards_dict[m][valid_pos]
                    # Limit the number of candidates for efficiency
                    topk_num_ = min(len(selected_rewards), topk_num)
                    # Top‑k deletions that are most beneficial for the attacker
                    topk_values, topk_pos = torch.topk(selected_rewards, topk_num_)
                    # Top‑k deletions with smallest rewards (again for diversity)
                    topk_values_ver, topk_pos_ver = torch.topk(selected_rewards, topk_num_, largest=False)

                    # Generate deletion candidate solutions at each promising position
                    for i in range(len(topk_pos)):
                        sol = Solution(m=m, position=int(valid_pos[topk_pos[i]].item()), reward=topk_values[i].item(), mode='delete',
                                       perturbations=curr_pert, acc=acc_eqY)
                        # Keep candidate only if its resulting perturbation is not tabu
                        if not tabu_table.query(get_perturbation_of_solution(sol)):
                            candidate_lst.append(sol)
                        # Add candidate from low‑reward position as an exploratory neighbor
                        sol = Solution(m=m, position=int(valid_pos[topk_pos_ver[i]].item()), reward=topk_values_ver[i].item(),
                                       mode='delete', perturbations=curr_pert, acc=acc_eqY)
                        if not tabu_table.query(get_perturbation_of_solution(sol)):
                            candidate_lst.append(sol)

            # Rank all candidates by their reward in descending order
            candidate_lst = sorted(candidate_lst, key=lambda x: x.reward, reverse=True)
            # Counter for how many candidates we successfully insert into the solution list
            insert_count = 0
            # Iterate over ranked candidates, adding up to topk_num of them
            for i in range(len(candidate_lst)):
                # Stop if we already inserted enough candidates
                if insert_count == topk_num:
                    break
                # If reward becomes non‑positive, reverse remaining tail to focus on worst cases
                if candidate_lst[i].reward <= 0:
                    candidate_lst[i:] = candidate_lst[i:][::-1]
                # Ensure each candidate carries the current perturbation as its base
                candidate_lst[i] = candidate_lst[i]._replace(perturbations=curr_pert)
                # Decode perturbation from solution representation
                pert = get_perturbation_of_solution(candidate_lst[i])
                # Skip already tabu perturbations
                if tabu_table.query(pert):
                    continue
                # Insert candidate into solution list, possibly removing an old one
                remove_pert = sol_lst.add_solution(candidate_lst[i])
                # If a removed perturbation was also in the tabu table, remove it from tabu memory
                if remove_pert is not None and tabu_table.query(remove_pert):
                    tabu_table.delete(remove_pert)
                # Add new perturbation pattern to tabu memory
                tabu_table.insert(pert)
                # Increase number of successfully inserted candidates
                insert_count += 1
        # with open('dyn_dataset1.pkl', 'wb') as f:
        #     pickle.dump(self.dyn_dataset.get_dataset(), f)

        for x, y, index in self.dyn_dataset.get_dataset():
            # `run_defense.py` already sets `args.output_dir` to include:
            #   .../<defense>/<CW|OW>
            # So here we only need to add the dataset subdirectory and ensure it exists.
            output_dir = str(args.output_dir)
            mode_dir = 'OW' if args.open_world else 'CW'
            if Path(output_dir).name != mode_dir:
                output_dir = os.path.join(output_dir, mode_dir)
            output_dir = os.path.join(output_dir, str(args.dataset)+'_'+str(args.attack))

            for i in range(x.shape[0]):
                idx = index[i].detach().cpu().numpy()
                data = x[i].detach().cpu().numpy()[0]
                data = data[data != 0]
                data = np.stack([np.zeros(len(data), dtype=np.float32), data], axis=1)
                if args.open_world:
                    if idx // args.mon_inst < args.mon_classes:
                        fname = f'{idx // args.mon_inst}-{idx % args.mon_inst}.cell'
                    else:
                        fname = f'{idx % args.unmon_inst}.cell'
                else:
                    fname = f'{idx // args.mon_inst}-{idx % args.mon_inst}.cell'

                dump_dir = os.path.join(output_dir, fname)
                os.makedirs(os.path.dirname(dump_dir), exist_ok=True)
                # Ensure we can serialize torch tensors even if the search ran on GPU.
                np.savetxt(dump_dir, data, fmt='%.2f\t%d')
        # Report whether the attack achieved the required target accuracy or not
        if bst_sol.global_best["acc"] <= target_acc:
            print(f"Attack succeed: acc={bst_sol.global_best['acc']}, pert={bst_sol.global_best['perturbations']}")
        else:
            print(f"Attack failed: acc={bst_sol.global_best['acc']}, pert={bst_sol.global_best['perturbations']}")
        # Save the best perturbation pattern found during the search
        self.perturbations = bst_sol.global_best['perturbations']
        # Return the best solution tracker holding the search results
        return bst_sol

    def __evaluate_and_get_rewards__(self, curr_pert, m_insert_limit, m_delete_limit, dataset, m_max):
        sum_rewards_dict = {m: 0 for m in range(1, m_insert_limit + 1)}
        del_rewards_dict = {}
        sum_grads = None
        sum_loss = 0.0
        counter = 0
        all_outputs, all_targets = [], []
        for data, target, indices in dataset:
            counter += 1
            data = data.to(self.device).float()
            target = target.to(self.device).long()
            indices = indices.to(self.device)

            data = data.requires_grad_(True)
            model_in = _pack_directions_for_varcnn(data) if self._pack_varcnn_input else data
            output = self.model(model_in)
            correct_indices = output.argmax(dim=1) == target
            correct_output = output[correct_indices]
            correct_target = target[correct_indices]
            if correct_output.numel() == 0:
                loss = self.criterion(output, target)
            else:
                loss = self.criterion(correct_output, correct_target)

            self.model.zero_grad()
            loss.backward()
            grads = data.grad

            all_outputs.append(output.argmax(dim=1).detach())
            all_targets.append(target.detach())

            data_flat = data.squeeze(1)
            grads_flat = grads.squeeze(1)

            for m in range(1, m_insert_limit):
                rewards_k = 1 * self.get_cos_similarity_when_insert_m_1(data_flat, grads_flat, m, using_cumulative_amount=False)
                if isinstance(sum_rewards_dict[m], torch.Tensor):
                    sum_rewards_dict[m] = sum_rewards_dict[m] + rewards_k.detach()
                else:
                    sum_rewards_dict[m] = rewards_k.detach()

            for m in range(1, m_delete_limit):
                origin_data = self.origin_x[indices]
                if len(origin_data.shape) > 2:
                    origin_data = origin_data.squeeze(1)
                del_rewards_k = 1 * self.get_cos_similarity_when_delete_m_1(data_flat, grads_flat, curr_pert, origin_data, m)
                if m in del_rewards_dict:
                    del_rewards_dict[m] = del_rewards_dict[m] + del_rewards_k.detach()
                else:
                    del_rewards_dict[m] = del_rewards_k.detach()

            if sum_grads is None:
                sum_grads = grads_flat.sum(dim=0)
            else:
                sum_grads = sum_grads + grads_flat.sum(dim=0)
            sum_loss += loss.item()

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        acc_eqY = label_accuracy(all_outputs, all_targets)

        for m in range(1, m_max + 1):
            if m in sum_rewards_dict and not (isinstance(sum_rewards_dict[m], int) and sum_rewards_dict[m] == 0):
                sum_rewards_dict[m] = sum_rewards_dict[m] / counter
            if m in del_rewards_dict:
                del_rewards_dict[m] = del_rewards_dict[m] / counter
        return sum_loss, sum_grads, sum_rewards_dict, del_rewards_dict, acc_eqY

    def get_cos_similarity_when_insert_m_1(self, data, grads, m, using_cumulative_amount=False, keep_n=True):
        grads = grads * 1e10
        batch_size, n = data.shape[0], data.shape[1]
        assert len(data.shape) == 2, "Data must match shape: [batch_size, n]"
        assert 0 < m < n, f"The number of inserted 1 (m) should be [0, {n}], but {m}"
        device = data.device

        difference = data[:, :-m] - data[:, m:]
        right_weight = difference * grads[:, m:]
        right_cumsum = torch.cumsum(torch.flip(right_weight, dims=[1]), dim=1)
        right_cumsum = torch.flip(right_cumsum, dims=[1])
        right_rewards = torch.cat([right_cumsum, torch.zeros((batch_size, m), device=device, dtype=data.dtype)], dim=1)

        difference_1 = 1 - data
        rewards_1 = grads * difference_1
        kernel = torch.ones((1, 1, m), dtype=torch.float32, device=device)
        ipt = rewards_1.unsqueeze(1)
        rewards_1_k_sum = F.conv1d(ipt, kernel, stride=1, padding=0).squeeze(1)

        rewards = right_rewards.clone()
        if m > 1:
            rewards[:, :-m + 1] = rewards[:, :-m + 1] + rewards_1_k_sum
        else:
            rewards = rewards + rewards_1_k_sum

        ver_difference = torch.flip(difference ** 2, dims=[1])
        ver_difference_cumsum = torch.cumsum(ver_difference, dim=1)
        difference_norm = torch.sqrt(torch.flip(ver_difference_cumsum, dims=[1]))
        data_norm_diff = torch.cat([difference_norm, torch.zeros((batch_size, 1), device=device, dtype=data.dtype)], dim=1)
        ipt_norm = (difference_1 ** 2).unsqueeze(1)
        data_norm_1 = torch.sqrt(F.conv1d(ipt_norm, kernel, stride=1, padding=0).squeeze(1))
        data_norm = torch.sqrt(data_norm_diff ** 2 + data_norm_1 ** 2)
        data_norm = torch.cat([data_norm, torch.zeros((batch_size, m - 1), device=device, dtype=data.dtype)], dim=1)

        grads_norm = torch.norm(grads, dim=-1, keepdim=True)
        denomi = grads_norm * data_norm
        cos_sim = rewards / denomi

        cos_sim = torch.where(torch.isnan(cos_sim), torch.zeros_like(cos_sim), cos_sim)
        cos_sim = torch.where(torch.isinf(cos_sim), torch.zeros_like(cos_sim), cos_sim)

        if not keep_n:
            cos_sim = cos_sim[:, :-m + 1 if m > 1 else None]
        if using_cumulative_amount:
            max_values = cos_sim.max(dim=-1, keepdim=True)[0]
            one_hot = (cos_sim == max_values).to(data.dtype)
            return one_hot.sum(dim=0)
        else:
            return cos_sim.mean(dim=0)

    def get_cos_similarity_when_delete_m_1(self, data, grads, perturbations, original_data, m):
        grads = grads * 1e10
        batch_size, n = data.shape[0], data.shape[1]
        assert len(data.shape) == 2, "Data must match shape: [batch_size, n]"
        counter = Counter(perturbations)
        assert len(perturbations) > 0 and counter.most_common(1)[0][1] >= m, f"The maximum continuous insertion of 1 is less than m({m})"
        device = data.device
        start_idx = -len(perturbations)
        end_idx = None if start_idx + m == 0 else start_idx + m
        data_before = torch.cat([data[:, m:], original_data[:, start_idx:end_idx]], dim=-1)
        difference = data_before - data
        right_weight = difference * grads

        ver_right_weight = torch.flip(right_weight, dims=[1])
        ver_right_cumsum = torch.cumsum(ver_right_weight, dim=1)
        right_rewards = torch.flip(ver_right_cumsum, dims=[1])

        rewards = right_rewards
        ver_difference = torch.flip(difference ** 2, dims=[1])
        ver_difference_cumsum = torch.cumsum(ver_difference, dim=1)
        difference_norm = torch.sqrt(torch.flip(ver_difference_cumsum, dims=[1]))

        grads_norm = torch.norm(grads, dim=-1, keepdim=True)
        denomi = grads_norm * difference_norm
        cos_sim = rewards / denomi
        cos_sim = torch.where(torch.isnan(cos_sim), torch.zeros_like(cos_sim), cos_sim)
        cos_sim = torch.where(torch.isinf(cos_sim), torch.zeros_like(cos_sim), cos_sim)

        nan_indices = torch.ones(n, dtype=torch.bool, device=device)
        for element, count in counter.items():
            if count >= m:
                nan_indices[element] = False
        nan_mask = nan_indices.unsqueeze(0).expand(batch_size, -1)
        cos_sim = torch.where(nan_mask, torch.full_like(cos_sim, float('nan')), cos_sim)
        return cos_sim.mean(dim=0)

    def get_perturbed_data(self, data, perturbations):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if len(data.shape) == 3:
            data = np.squeeze(data, axis=1)
        perturbations = sorted(perturbations)
        batch_size, n = data.shape
        result = np.ones((batch_size, n), dtype=np.float32)

        original_index = 0
        insert_count = 0
        for pos in perturbations:
            if pos + insert_count >= n:
                break
            num_elements_to_copy = max(0, pos - original_index)
            result[:, original_index + insert_count:original_index + insert_count + num_elements_to_copy] = \
                data[:, original_index:original_index + num_elements_to_copy]
            insert_count += 1
            original_index = pos
        if original_index < n:
            result[:, original_index + insert_count:] = data[:, original_index: n - insert_count]
        result = np.expand_dims(result, axis=1)
        result = torch.from_numpy(result).float().to(self.device)
        return result

    def get_actual_position(self, perturbations, selected_insert_position):
        if isinstance(perturbations, list):
            perturbations = torch.tensor(perturbations, dtype=torch.long, device=self.device)
        sort_perturbations, _ = torch.sort(perturbations)
        cumulative_offsets = torch.arange(len(perturbations), device=self.device, dtype=torch.long)
        inserted_position = sort_perturbations + cumulative_offsets
        sp = selected_insert_position if torch.is_tensor(selected_insert_position) else torch.tensor(selected_insert_position, device=self.device)
        insertions_before = (inserted_position < sp).sum().item()
        out = sp.item() - insertions_before if torch.is_tensor(selected_insert_position) else selected_insert_position - insertions_before
        return out

    def get_repeat_mask(self, perturbations: list, Feat_dim=5000):
        sorted_perturbations = sorted(perturbations)
        repeat_mask = np.ones(Feat_dim, dtype=np.float32)
        for idx, pos in enumerate(sorted_perturbations):
            insert_pos = pos + idx
            if insert_pos < Feat_dim:
                repeat_mask[insert_pos] = 0
        return torch.from_numpy(repeat_mask).float().to(self.device)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def label_accuracy(pred, target):
    """Computes the accuracy of model predictions matching the target labels"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu()
    if isinstance(target, torch.Tensor):
        target = target.cpu()
    batch_size = target.shape[0]
    correct = (pred == target).sum().item()
    accuracy = correct / batch_size * 100.0
    return accuracy

def pad_samples(all_samples, pad_value=0, max_len=None):
    """Pad variable-length samples to a uniform length (right-padding)."""
    if max_len is None:
        max_len = max(len(s) for s in all_samples)
    padded = np.full((len(all_samples), max_len), pad_value, dtype=np.float32)
    for i, s in enumerate(all_samples):
        s = np.asarray(s, dtype=np.float32)
        length = min(len(s), max_len)
        padded[i, :length] = s[:length]
    return padded

class GapdisDefense:
    def __init__(self, args: argparse.Namespace, flist: Union[List[str], np.ndarray], labels: Union[List[int], np.ndarray]):
        self.config = GapdisConfig(args)
        self.config.load_config()


        # Default: use GPU (cuda:0) when available; use --no_cuda to force CPU
        if args.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cpu")
            print("Warning: CUDA not available, using CPU.")

        traces = []
        self.wf_model, _ = load_attack_model(args)

        for data_path in flist:
            data = parse_trace(data_path)
            data = np.where(data[:, 1] > 0, 1, np.where(data[:, 1] < 0, -1, 0))
            traces.append(data)
        traces = pad_samples(traces, max_len=args.seq_length)
        start_time = time.time()
        gapdis = GAPDiS(self.wf_model, traces, labels, perturbations=None, device=device, attack=args.attack)
        best_sol = gapdis.generate_adversarial_examples_tabu_search(max_len_perturbations=self.config.max_perturb,
                        max_iter=self.config.max_iterations, target_acc=self.config.target_acc, topk_num=self.config.topk_num, m_max=self.config.max_dummy,
                        tabu_len_multi=self.config.tabu_len_multi, sol_len_multi=self.config.sol_len_multi,
                        cpm_len_den=self.config.cpm_len_den, init_rd_num=self.config.init_rd_num, init_m_num=self.config.init_m_num,
                        exch_len=self.config.exch_len, repl_rate=self.config.repl_rate, muta_rate=self.config.muta_rate,
                        smp_cpm_rate=self.config.smp_cpm_rate, toler=self.config.toler, max_iter_multi=self.config.max_iter_multi, args=args)
        print(f"Total running time: {time.time() - start_time:.2f} seconds")
