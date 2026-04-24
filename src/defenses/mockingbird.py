import argparse
import os
from typing import Union

import numpy as np

from defenses.base import Defense
from defenses.config import MockingbirdConfig
from utils.general import parse_trace, set_random_seed, get_flist_label


class MockingbirdDefense(Defense):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = MockingbirdConfig(args)
        self.config.load_config()
        
        # Initialize target traces pool for adversarial generation
        # We'll use a subset of traces from other classes as targets
        self.logger.info("Initializing Mockingbird defense...")
        self.initialize_target_pool(args)
        self.logger.info("Mockingbird defense initialized")
    
    def initialize_target_pool(self, args: argparse.Namespace):
        """
        Initialize a pool of target traces from other classes.
        In the original Mockingbird, these are used as targets for adversarial generation.
        """
        # Same layout as run_defense.py / PaletteDefense: monitored + optional OW traces
        flist, labels = get_flist_label(
            args.mon_path,
            args.unmon_path,
            mon_cls=args.mon_classes,
            mon_inst=args.mon_inst,
            unmon_inst=args.unmon_inst if args.open_world else 0,
            suffix=args.suffix
        )
        
        # Sample a subset for target pool (to avoid loading all traces)
        if len(flist) > self.config.target_pool_size:
            indices = np.random.choice(len(flist), self.config.target_pool_size, replace=False)
            self.target_flist = flist[indices]
            self.target_labels = labels[indices]
        else:
            self.target_flist = flist
            self.target_labels = labels
        
        self.logger.info(f"Initialized target pool with {len(self.target_flist)} traces")
    
    @set_random_seed
    def _simulate(self, data_path: Union[str, os.PathLike]) -> np.ndarray:
        """
        Apply Mockingbird adversarial defense to the trace.
        """
        trace = parse_trace(data_path)
        
        # Step 1: Convert trace to burst representation
        burst_sequence = self.trace_to_bursts(trace)
        
        # Step 2: Select target trace(s) from the pool
        target_bursts = self.select_target_bursts(burst_sequence)
        
        # Step 3: Apply adversarial modifications
        defended_bursts = self.apply_adversarial_modification(burst_sequence, target_bursts)

        # Step 4–5: Match pendding/mockingbird.py post-processing (see comments there):
        # (1) Rescale float burst values to integer counts — original uses rescale_X with xmax
        #     from training (part1_data.dill); here bursts are already counts, so we round.
        # (2) Expand integer bursts to one packet per row (+/-1), like expand_me — attacks use
        #     packet sequences, not burst tensors.
        bursts_int = self.rescale_bursts_to_int(defended_bursts)
        defended_trace = self.expand_signed_bursts_to_packet_trace(bursts_int)
        if len(defended_trace) == 0:
            return trace

        return defended_trace
    
    def trace_to_bursts(self, trace: np.ndarray) -> np.ndarray:
        """
        Convert packet trace to burst representation.
        A burst is a sequence of consecutive packets in the same direction.
        Returns: array of burst sizes (positive for outgoing, negative for incoming)
        """
        if len(trace) == 0:
            return np.array([])
        
        # Extract direction from second column
        directions = []
        for val in trace[:, 1]:
            direction = int(np.sign(val))
            directions.append(direction)
        directions = np.array(directions)
        
        # Find where direction changes
        change_indices = np.where(np.diff(directions) != 0)[0] + 1
        change_indices = np.concatenate(([0], change_indices, [len(directions)]))
        
        # Create bursts
        bursts = []
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            burst_size = end_idx - start_idx
            direction = directions[start_idx]
            bursts.append(burst_size * direction)
        
        bursts = np.array(bursts)
        
        # Limit to max_bursts
        if len(bursts) > self.config.max_bursts:
            bursts = bursts[:self.config.max_bursts]
        elif len(bursts) < self.config.max_bursts:
            # Pad with zeros
            pad_size = self.config.max_bursts - len(bursts)
            bursts = np.pad(bursts, (0, pad_size), mode='constant')
        
        return bursts
    
    def select_target_bursts(self, source_bursts: np.ndarray) -> np.ndarray:
        """
        Select target burst sequence from the target pool.
        """
        # Randomly select a target trace from the pool
        if len(self.target_flist) == 0:
            # Fallback: use a random target based on source
            target_bursts = self.generate_random_target(source_bursts)
            return target_bursts
        
        target_idx = np.random.randint(0, len(self.target_flist))
        target_path = self.target_flist[target_idx]
        
        try:
            target_trace = parse_trace(target_path)
            target_bursts = self.trace_to_bursts(target_trace)
        except Exception as e:
            self.logger.debug(f"Error loading target trace {target_path}: {e}")
            # Fallback: use a random target
            target_bursts = self.generate_random_target(source_bursts)
        
        return target_bursts
    
    def generate_random_target(self, source_bursts: np.ndarray) -> np.ndarray:
        """
        Generate a random target burst sequence as fallback.
        """
        # Create a target with similar structure but different values
        target_bursts = source_bursts.copy()
        # Randomly modify some bursts
        num_modifications = max(1, len(target_bursts) // 4)
        indices = np.random.choice(len(target_bursts), num_modifications, replace=False)
        for idx in indices:
            # Modify burst size while keeping direction
            if target_bursts[idx] != 0:
                direction = np.sign(target_bursts[idx])
                new_size = max(1, int(abs(target_bursts[idx]) * np.random.uniform(0.5, 2.0)))
                target_bursts[idx] = new_size * direction
        
        return target_bursts
    
    def apply_adversarial_modification(self, source_bursts: np.ndarray, target_bursts: np.ndarray) -> np.ndarray:
        """
        Apply adversarial modifications to move source bursts closer to target bursts.
        This is a simplified version without gradient computation.
        """
        defended_bursts = source_bursts.copy()
        
        # Compute distance between source and target
        distance = self.compute_distance(defended_bursts, target_bursts)
        
        # Apply iterative modifications
        for iteration in range(self.config.num_iterations):
            # Find indices where we can modify (non-zero bursts)
            non_zero_indices = np.where(defended_bursts != 0)[0]
            if len(non_zero_indices) == 0:
                break
            
            # Select a random burst to modify
            idx = np.random.choice(non_zero_indices)
            
            # Compute modification direction
            source_val = defended_bursts[idx]
            target_val = target_bursts[idx] if idx < len(target_bursts) else 0
            
            # Move source towards target
            if target_val != 0 and source_val != 0:
                # Same direction: move closer
                if np.sign(source_val) == np.sign(target_val):
                    diff = target_val - source_val
                    modification = int(diff * self.config.alpha)
                    defended_bursts[idx] = source_val + modification
                    # Ensure we don't change direction
                    if np.sign(defended_bursts[idx]) != np.sign(source_val):
                        defended_bursts[idx] = np.sign(source_val) * max(1, abs(defended_bursts[idx]))
                else:
                    # Different direction: increase magnitude to move away
                    modification = int(abs(source_val) * self.config.alpha)
                    defended_bursts[idx] = source_val + np.sign(source_val) * modification
            else:
                # Increase magnitude
                modification = int(abs(source_val) * self.config.alpha * 0.1)
                defended_bursts[idx] = source_val + np.sign(source_val) * modification
            
            # Ensure minimum burst size
            if defended_bursts[idx] != 0:
                defended_bursts[idx] = np.sign(defended_bursts[idx]) * max(1, abs(defended_bursts[idx]))
            
            # Check if we've made sufficient progress
            new_distance = self.compute_distance(defended_bursts, target_bursts)
            if new_distance < distance * 0.5:  # Significant improvement
                break
        
        return defended_bursts
    
    def compute_distance(self, source: np.ndarray, target: np.ndarray) -> float:
        """
        Compute Euclidean distance between source and target burst sequences.
        """
        min_len = min(len(source), len(target))
        source_trimmed = source[:min_len]
        target_trimmed = target[:min_len]
        diff = source_trimmed - target_trimmed
        distance = np.sqrt(np.sum(diff ** 2))
        return distance

    def rescale_bursts_to_int(self, bursts: np.ndarray) -> np.ndarray:
        """
        Analogue of mockingbird_utility.rescale_X: turn defended burst values into integer
        burst lengths. Original multiplies normalized [0,1] values by per-position xmax from
        part1_data.dill; WFZoo operates on real counts, so we round and clamp magnitude.
        """
        bursts = np.asarray(bursts, dtype=np.float64)
        out = np.zeros_like(bursts, dtype=np.int64)
        nz = bursts != 0
        signs = np.sign(bursts[nz]).astype(np.int64)
        mags = np.maximum(1, np.round(np.abs(bursts[nz])).astype(np.int64))
        out[nz] = signs * mags
        return out

    def expand_signed_bursts_to_packet_trace(self, bursts_int: np.ndarray) -> np.ndarray:
        """
        Analogue of mockingbird_utility.expand_me (inner expand): expand signed run-length
        bursts to one row per packet, e.g. [+2, -3] -> [+1,+1,-1,-1,-1].
        Pads with direction 0 to seq_length when shorter (see expand_me padding).
        """
        directions = []
        for b in np.asarray(bursts_int).flatten():
            if b == 0:
                continue
            if b > 0:
                directions.extend([1] * int(b))
            else:
                directions.extend([-1] * int(-b))

        n = len(directions)
        if n == 0:
            return np.zeros((0, 2))

        seq_length = int(getattr(self.args, "seq_length", 5000))
        if seq_length > 0:
            if n < seq_length:
                directions.extend([0] * (seq_length - n))
            elif n > seq_length:
                directions = directions[:seq_length]

        times = np.arange(len(directions), dtype=np.float64)
        return np.column_stack((times, np.asarray(directions, dtype=np.int64)))

