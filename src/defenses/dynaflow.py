import argparse
import os
from typing import Union, List

import numpy as np

from defenses.base import Defense
from defenses.config import DynaflowConfig
from utils.general import parse_trace, set_random_seed


class DynaflowDefense(Defense):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = DynaflowConfig(args)
        self.config.load_config()

    @set_random_seed
    def _simulate(self, data_path: Union[str, os.PathLike]) -> np.ndarray:
        """
        Apply DynaFlow defense to the trace.
        """
        
        trace = parse_trace(data_path)
        
        # Convert trace format: [time, direction] -> [time, direction, length]
        # The second column can be:
        # - If abs(value) == 1: direction only (-1 or 1), ignore packet size
        # - If abs(value) > 1: value includes both direction (sign) and packet size (abs value)
        old_packets = []
        for i in range(len(trace)):
            time_val = trace[i, 0]
            dir_val = trace[i, 1]
            
            # Determine direction and packet size
            if abs(int(dir_val)) == 1:
                # Only direction, no packet size - use default size
                direction = int(np.sign(dir_val))
                packet_size = 0  # Default packet size when not specified
            else:
                # Packet size is included in the value
                direction = int(np.sign(dir_val))
                packet_size = int(abs(dir_val))
            
            old_packets.append([time_val, direction, packet_size])
        
        # Create end sizes based on config
        end_sizes = self.create_end_sizes(self.config.m)
        
        # Parse switch sizes from config (comma-separated string or list)
        if isinstance(self.config.switch_sizes, str):
            switch_sizes = [int(x.strip()) for x in self.config.switch_sizes.split(',')]
        else:
            switch_sizes = self.config.switch_sizes
        
        # Parse possible time gaps from config
        if isinstance(self.config.poss_time_gaps, str):
            poss_time_gaps = [float(x.strip()) for x in self.config.poss_time_gaps.split(',')]
        else:
            poss_time_gaps = self.config.poss_time_gaps
        # Apply defense
        new_packets = self.defend(
            old_packets,
            switch_sizes,
            end_sizes,
            self.config.first_time_gap,
            poss_time_gaps,
            self.config.subseq_length,
            self.config.memory
        )
        
        # Convert back to [time, direction] format
        # new_packets contains 2-element lists [time, direction] - no packet_size info
        # Handle empty case
        if len(new_packets) == 0:
            return np.array([]).reshape(0, 2)
        
        # Convert to numpy array - all packets are [time, direction] format
        # Note: packet_size information is lost in the defense transformation
        defended_trace = np.array([[pkt[0], pkt[1]] for pkt in new_packets])
        
        return defended_trace
    
    def defend(self, old_packets: List, switch_sizes: List[int], end_sizes: List[int],
                first_time_gap: float, poss_time_gaps: List[float], subseq_length: int, memory: int) -> List:
        """
        Creates defended sequence from old packets.
        """
        
        # Make a copy of old packet sequence
        packets = old_packets[:]
        # Initialize new sequence
        new_packets = []
        past_times = []
        past_packets = []
        index = 0
        time_gap = first_time_gap
        # First packet at time zero
        curr_time = -1 * time_gap


        min_index = 99999999
        max_end_size = max(end_sizes) if end_sizes else 10000000
        # Add reasonable safety limit to prevent infinite loops (cap at 100000)
        max_index = min(max_end_size + 1000, 100000)
        iteration_count = 0
        max_iterations = 100000
        
        while (len(packets) != 0 or index not in end_sizes) and index < max_index and iteration_count < max_iterations:
            iteration_count += 1
            
            # Get time and direction of next packet
            curr_time = curr_time + time_gap
            if index % subseq_length == 0:
                curr_dir = 1
            else:
                curr_dir = -1
                
            # Add needed packet
            # If possible, packet combination
            packet_size = 0
            num_used_packets = 0
            for i in range(0, len(packets)):
                if packets[i][0] <= curr_time and packets[i][1] == curr_dir and packets[i][2] + packet_size <= 498:
                    num_used_packets += 1
                    packet_size += packets[i][2]
                    if i == 0:
                        past_times.append(packets[i][0])
                    past_packets.append(packets[i])
                else:
                    break
                
            del packets[0:num_used_packets]
                
            new_packets.append([curr_time, curr_dir])
                
            # Find new time gap if time to switch
            # Const for weighted average
            const = 400
            if index in switch_sizes:
                time_gap_info = self.find_new_time_gap(
                    past_times, curr_time, time_gap, poss_time_gaps, memory, const
                )
                time_gap = time_gap_info[0]
                
            # Move on to next packet
            index += 1

            # Get length of defended sequence before any extra padding at end
            if len(packets) == 0 and min_index > index:
                min_index = index
            
            # Safety check: break if we've exceeded max iterations
            if iteration_count >= max_iterations:
                break
        

        # Ensure we have at least some packets
        if len(new_packets) == 0:
            return []
        
        return new_packets
    
    def find_new_time_gap(self, past_times: List[float], curr_time: float, time_gap: float,
                           poss_time_gaps: List[float], memory: int, block_size: int) -> List:
        """
        Finds new time gap for defended sequence.
        """
        # Safety check: if past_times is empty, return current time_gap
        # If no past times available, return current time gap
        if len(past_times) == 0:
            # Find the closest time gap in poss_time_gaps to the current time_gap
            min_diff = 99999
            closest_idx = 0
            for i in range(0, len(poss_time_gaps)):
                diff = abs(time_gap - poss_time_gaps[i])
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            return [poss_time_gaps[closest_idx], closest_idx]
        
        # Find average time gap
        if len(past_times) >= memory:
            average_time_gap = float(past_times[-1] - past_times[-memory]) / (memory - 1)
        elif len(past_times) > 10:
            average_time_gap = float(past_times[-1] - past_times[0]) / (len(past_times) - 1)
        else:
            average_time_gap = time_gap
        
        # Safety check: avoid division by zero
        if average_time_gap <= 0:
            average_time_gap = time_gap
        
        # Find expected time gap
        exp_packet_num = block_size + 1 * float(curr_time - past_times[-1]) / average_time_gap
        exp_time_gap = block_size / exp_packet_num * average_time_gap
        
        # Choose next time gap
        min_diff = 99999
        for i in range(0, len(poss_time_gaps)):
            if min_diff > abs(exp_time_gap - poss_time_gaps[i]):
                min_diff = abs(exp_time_gap - poss_time_gaps[i])
            else:
                return [poss_time_gaps[i - 1], (i - 1)]
        return [poss_time_gaps[-1], len(poss_time_gaps) - 1]
    
    def create_end_sizes(self, k: float) -> List[int]:
        """
        Creates list of possible sizes for defended sequence.
        """
        end_sizes = []
        for i in range(0, 9999):
            if k ** i > 10000000:
                break
            end_sizes.append(round(k ** i))
        return end_sizes
