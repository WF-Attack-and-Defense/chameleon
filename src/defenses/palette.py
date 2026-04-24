import argparse
import os
import random
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.utils.data as Data

from defenses.base import Defense
from defenses.config import PaletteConfig
from utils.general import parse_trace, set_random_seed, get_flist_label


class PaletteDefense(Defense):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = PaletteConfig(args)
        self.config.load_config()
        
        # Initialize palette model (super-matrices, anonymity sets, PMF)
        # This needs to be done once using training data
        self.logger.info("Initializing Palette defense model...")
        self.initialize_palette_model(args)
        self.logger.info("Palette defense model initialized")
    
    def initialize_palette_model(self, args: argparse.Namespace):
        """
        Initialize palette model by processing training traces.
        Steps: 1) Extract features, 2) Cluster, 3) Refine, 4) Compute PMF
        """

        flist, labels = get_flist_label(
            args.mon_path,
            args.unmon_path,
            mon_cls=args.mon_classes,
            mon_inst=args.mon_inst,
            unmon_inst=args.unmon_inst if args.open_world else 0,
            suffix=args.suffix
        )

        
        # For palette, we use all traces to build the model
        self.train_flist = flist
        self.train_labels = labels
        
        self.logger.info(f"Processing {len(self.train_flist)} traces for Palette model...")
        
        # Step 1: Extract features from training traces
        features, feature_labels = self.extract_features(self.train_flist, self.train_labels)
        
        # Step 2: Build super-matrices and cluster
        super_matrices_websites = self.build_super_matrices(features, feature_labels)
        total_sets, super_matrices, website_to_set = self.cluster_websites(
            super_matrices_websites, feature_labels
        )
        
        # Step 3: Compute PMF
        PMF_upload, PMF_download = self.get_PMF(features, feature_labels, website_to_set, len(total_sets))

        # Step 4: Refine super-matrices (simplified - skip optimization for now)
        # For now, we'll use the super-matrices directly, but apply a simple refinement
        shrunk_super_matrices = self.refine_super_matrices(total_sets, super_matrices, website_to_set, PMF_upload, PMF_download)
        
        # Store for use in _simulate
        self.total_sets = total_sets
        self.shrunk_super_matrices = shrunk_super_matrices
        self.website_to_set = website_to_set
        self.PMF_upload = PMF_upload
        self.PMF_download = PMF_download
    
    def extract_features(self, flist: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract packets_per_slot features from traces (similar to TAM feature).
        Returns features of shape (N, 2, TAM_LENGTH) and labels.
        """
        features = []
        feature_labels = []
        
        for fpath, label in zip(flist, labels):
            try:
                trace = parse_trace(fpath)

                times = np.array(trace[:, 0])
                length_seq = np.array(trace[:, 1]).astype("int")

                # Extract packets_per_slot feature (similar to TAM)
                feature = self.packets_per_slot(times, length_seq)
                features.append(feature)
                feature_labels.append(label)
            except Exception as e:
                self.logger.debug(f"Error extracting feature from {fpath}: {e}")
                continue
        
        return np.array(features), np.array(feature_labels)
    
    def packets_per_slot(self, times: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        feature = [[0 for _ in range(self.config.tam_length)], [0 for _ in range(self.config.tam_length)]]
        for i in range(0, len(sizes)):
            if sizes[i] > 0:
                if times[i] >= self.config.cutoff_time:
                    feature[0][-1] += 1
                else:
                    idx = int(times[i] * (self.config.tam_length - 1) / self.config.cutoff_time)
                    feature[0][idx] += 1
            if sizes[i] < 0:
                if times[i] >= self.config.cutoff_time:
                    feature[1][-1] += 1
                else:
                    idx = int(times[i] * (self.config.tam_length - 1) / self.config.cutoff_time)
                    feature[1][idx] += 1

        return feature
    
    def build_super_matrices(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Build super-matrices for each website by taking max over all traces.
        Returns array of shape (num_websites, 2, TAM_LENGTH).
        """
        num_websites = self.args.mon_classes
        super_matrices = np.zeros((num_websites, 2, self.config.tam_length))
        
        # Group features by website
        for website_id in range(num_websites):
            website_features = features[labels == website_id]
            if len(website_features) > 0:
                # Take max over all traces for this website
                super_matrix = np.max(website_features, axis=0)
                super_matrices[website_id] = super_matrix
        
        return super_matrices
    
    def cluster_websites(self, super_matrices: np.ndarray, labels: np.ndarray) -> Tuple[List, List, Dict]:
        """
        Cluster websites into anonymity sets.
        Returns: total_sets, super_matrices, website_to_set mapping
        """
        total_sets = []
        super_matrices_clustered = []
        website_to_set = {}
        
        website_indices = np.array([i for i in range(self.args.mon_classes)])
        np.random.shuffle(website_indices)
        
        partition_1 = website_indices
        partition_2 = []
        
        for i in range(self.config.round):
            anonymity_sets_fir, super_matrices_fir = self.website_clustering(
                partition_1, super_matrices, self.config.set_size
            )
            anonymity_sets_sec, super_matrices_sec = self.website_clustering(
                partition_2, super_matrices, self.config.set_size
            )
            
            partition_1 = []
            partition_2 = []
            
            for anonymity_set, super_matrix in zip(anonymity_sets_fir, super_matrices_fir):
                super_matrix = np.where(super_matrix == 0, 1, super_matrix)
                for website in anonymity_set:
                    if len(partition_1) < len(partition_2):
                        partition_1.append(website)
                    else:
                        partition_2.append(website)
                if anonymity_set not in total_sets:
                    total_sets.append(anonymity_set)
                    super_matrices_clustered.append(super_matrix)
            
            for anonymity_set, super_matrix in zip(anonymity_sets_sec, super_matrices_sec):
                super_matrix = np.where(super_matrix == 0, 1, super_matrix)
                for website in anonymity_set:
                    if len(partition_1) < len(partition_2):
                        partition_1.append(website)
                    else:
                        partition_2.append(website)
                if anonymity_set not in total_sets:
                    total_sets.append(anonymity_set)
                    super_matrices_clustered.append(super_matrix)
            
            partition_1 = np.array(partition_1)
            partition_2 = np.array(partition_2)
        
        # Build website_to_set mapping
        for idx, anonymity_set in enumerate(total_sets):
            for website in anonymity_set:
                if website not in website_to_set:
                    website_to_set[website] = []
                website_to_set[website].append(idx)
        
        return total_sets, super_matrices_clustered, website_to_set
    
    def website_clustering(self, partition: np.ndarray, super_matrices: np.ndarray, k: int) -> Tuple[List, List]:
        """
        Cluster websites in partition into anonymity sets of size k.
        """
        if len(partition) == 0:
            return [], []
        
        partition_set = []
        centers = np.empty((0, 2, self.config.tam_length), float)
        partition = np.sort(partition)
        tar_label = np.random.choice(partition, 1)[0]
        visited = np.zeros(self.args.mon_classes, dtype=bool)
        visited[tar_label] = True
        partition_set.append([tar_label])
        centers = np.append(centers, super_matrices[tar_label:tar_label + 1], axis=0)
        node = 0
        cnt = 1
        
        while cnt < len(partition):
            if len(partition_set[node]) == k:
                if len(partition) - cnt < k:
                    break
                node += 1
                max_dis = -1
                max_idx = -1
                for i in partition:
                    if visited[i]:
                        continue
                    for ct in centers:
                        dis = np.linalg.norm(super_matrices[i] - ct)
                        if dis > max_dis:
                            max_dis = dis
                            max_idx = i
                
                if max_idx == -1:
                    break
                else:
                    partition_set.append([max_idx])
                    centers = np.append(centers, super_matrices[max_idx][np.newaxis, :], axis=0)
                    visited[max_idx] = True
                    cnt += 1
            
            min_dis = 1e9
            min_idx = -1
            
            for i in partition:
                if visited[i]:
                    continue
                if min_dis > np.linalg.norm(super_matrices[i] - centers[node]):
                    min_dis = np.linalg.norm(super_matrices[i] - centers[node])
                    min_idx = i
            
            if min_idx == -1:
                break
            else:
                visited[min_idx] = True
                partition_set[node].append(min_idx)
                centers[node] = np.maximum(centers[node], super_matrices[min_idx])
                cnt += 1
        
        # Assign remaining websites to nearest cluster
        for i in partition:
            if not visited[i]:
                min_dis = 1e9
                min_idx = -1
                for idx, ct in enumerate(centers):
                    dis = np.linalg.norm(super_matrices[i] - ct)
                    if dis < min_dis:
                        min_dis = dis
                        min_idx = idx
                
                if min_idx >= 0:
                    partition_set[min_idx].append(i)
                    centers[min_idx] = np.maximum(centers[min_idx], super_matrices[i])
        
        return partition_set, centers
    
    def refine_super_matrices(self, total_sets: List, super_matrices: List, website_to_set: Dict, PMF_upload: List, PMF_download: List) -> List:
        """
        Refine super-matrices (simplified version - just apply ceiling).
        In full implementation, this would use optimization.
        """
        seed = self.config.seed
        lr = self.config.lr
        batch_size = self.config.batch_size
        k = self.config.k
        num_epochs = self.config.num_epochs

        random.seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tam = self.config.tam_length

        train_feats: List[np.ndarray] = []
        train_y_list: List = []
        for fpath, label in zip(self.train_flist, self.train_labels):
            try:
                trace = parse_trace(fpath)
                times = np.array(trace[:, 0])
                length_seq = np.array(trace[:, 1]).astype("int")
                feat = self.packets_per_slot(times, length_seq)
                train_feats.append(np.asarray(feat, dtype=np.float32))
                train_y_list.append(label)
            except Exception as e:
                self.logger.debug(f"Error extracting trace from {fpath}: {e}")
                continue

        if not train_feats:
            self.logger.warning("No training traces for refine_super_matrices; returning ceil(super_matrices)")
            return [np.ceil(np.asarray(m)).astype(np.int16) for m in super_matrices]

        train_X = np.stack(train_feats, axis=0)
        train_y = np.asarray(train_y_list)
        train_X = torch.from_numpy(train_X).unsqueeze(1)
        train_y = torch.from_numpy(train_y).long()

        shrunk_super_matrices = []

        for idx, st in enumerate(total_sets):
            ct = torch.from_numpy(super_matrices[idx][np.newaxis, np.newaxis, :]).float().to(device)
            st_set = set(st)
            mask = np.isin(train_y.numpy(), list(st_set))
            new_train_X = train_X[mask]
            new_train_y = train_y[mask]

            if new_train_X.size(0) == 0:
                pruned_ct = torch.ceil(torch.clamp(ct, min=0))
                shrunk_super_matrices.append(pruned_ct.detach().cpu().numpy().astype(np.int16))
                continue

            train_dataset = Data.TensorDataset(new_train_X, new_train_y)
            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            weight = torch.randn(2, tam, device=device)
            bias = torch.zeros(2, tam, device=device)

            adv_weight = weight.clone().detach()
            adv_bias = bias.clone().detach()

            adv_weight.requires_grad = True
            adv_bias.requires_grad = True

            th = torch.tensor([[10.], [10.]]).to(device)
            th = torch.where(ct > th, th, ct)

            optimizer = torch.optim.Adam([adv_weight, adv_bias], lr=lr)

            for epoch in range(num_epochs):
                for batch_idx, (tr_x, _) in enumerate(train_loader):
                    tr_x = tr_x.to(device)
                    optimizer.zero_grad()

                    adv_weight.requires_grad = True
                    adv_bias.requires_grad = True

                    trans_weight = torch.sigmoid(adv_weight)
                    trans_bias = th * torch.sigmoid(adv_bias)

                    pruned = torch.clamp(torch.clamp(ct * trans_weight, min=th) - trans_bias, min=0)

                    loss = - 0.5 * torch.mean(torch.clamp(pruned * torch.sign(tr_x) - tr_x, max=0)) \
                        + torch.mean(torch.clamp(pruned * torch.sign(tr_x) - tr_x, min=10.))

                    loss.backward()
                    optimizer.step()

                # test_loss = 0
                # for batch_idx, (ts_x, _) in enumerate(test_loader):
                #     ts_x = ts_x.to(device)
                #     adv_weight.requires_grad = False
                #     adv_bias.requires_grad = False
                
                #     trans_weight = torch.sigmoid(adv_weight)
                #     trans_bias = th * torch.sigmoid(adv_bias)
                
                #     pruned = torch.clamp(torch.clamp(ct * trans_weight, min=th) - trans_bias, min=0)
                
                #     loss = - 0.5 * torch.mean(torch.clamp(pruned * torch.sign(ts_x) - ts_x, max=0)) \
                #            + torch.mean(torch.clamp(pruned * torch.sign(ts_x) - ts_x, min=10.))
                #     test_loss += loss.item()

                # print('Epoch: {} Test Loss: {:.6f}'.format(epoch, test_loss / len(test_loader)))

            trans_weight = torch.sigmoid(adv_weight)
            trans_bias = th * torch.sigmoid(adv_bias)

            pruned_ct = torch.ceil(torch.clamp(torch.clamp(ct * trans_weight, min=th) - trans_bias, min=0))
            shrunk_super_matrices.append(pruned_ct.detach().cpu().numpy().astype(np.int16))

        return shrunk_super_matrices
    
    def get_PMF(self, features: np.ndarray, labels: np.ndarray, 
                     website_to_set: Dict, set_num: int) -> Tuple[List, List]:
        """
        Compute PMF (Probability Mass Function) for upload and download.
        """
        arr = [[] for i in range(set_num)]
        for i in range(len(labels)):
            index = int(labels[i])
            if index in website_to_set:
                for j in range(len(website_to_set[index])):
                    arr[website_to_set[index][j]].append(i)
        
        trace_cum_upload = [[0] * self.config.tam_length for _ in range(set_num)]
        trace_cum_download = [[0] * self.config.tam_length for _ in range(set_num)]
        
        for i in range(len(arr)):
            trace_sum_res_upload = [0] * self.config.tam_length
            trace_sum_res_download = [0] * self.config.tam_length
            
            for j in range(len(arr[i])):
                trace = features[arr[i][j]]
                # Handle different feature shapes
                if len(trace.shape) == 2:
                    if trace.shape[0] == 1:
                        trace_upload = trace[0]
                        trace_download = trace[0]
                    else:
                        trace_upload = trace[0]
                        trace_download = trace[1]
                elif len(trace.shape) == 1:
                    # If 1D, assume it's upload only, duplicate for download
                    trace_upload = trace
                    trace_download = trace
                else:
                    # Fallback
                    trace_upload = np.zeros(self.config.tam_length)
                    trace_download = np.zeros(self.config.tam_length)
                
                new_trace_upload = [1 if x != 0 else 0 for x in trace_upload]
                new_trace_download = [1 if x != 0 else 0 for x in trace_download]
                trace_sum_res_upload = [x + y for x, y in zip(new_trace_upload, trace_sum_res_upload)]
                trace_sum_res_download = [x + y for x, y in zip(new_trace_download, trace_sum_res_download)]
            
            trace_cum_upload[i] = trace_sum_res_upload
            trace_cum_download[i] = trace_sum_res_download
        
        # Normalize to get PMF
        trace_cum_res_upload = []
        trace_cum_res_download = []
        for i in range(len(trace_cum_upload)):
            sum_upload = sum(trace_cum_upload[i])
            sum_download = sum(trace_cum_download[i])
            if sum_upload > 0:
                normalized_upload = [x / sum_upload for x in trace_cum_upload[i]]
            else:
                normalized_upload = [1.0 / self.config.tam_length] * self.config.tam_length
            if sum_download > 0:
                normalized_download = [x / sum_download for x in trace_cum_download[i]]
            else:
                normalized_download = [1.0 / self.config.tam_length] * self.config.tam_length
            
            trace_cum_res_upload.append(normalized_upload)
            trace_cum_res_download.append(normalized_download)
        
        return trace_cum_res_upload, trace_cum_res_download
    
    @set_random_seed
    def _simulate(self, data_path: Union[str, os.PathLike]) -> np.ndarray:
        """
        Apply Palette regularization to generate defended trace.
        """
        trace = parse_trace(data_path)
        
        # Get file name to determine website ID
        fname = os.path.basename(data_path)
        try:
            if '-' in fname:
                website_id = int(fname.split('-')[0])
                instance_id = int(fname.split('-')[1].split('.')[0])
            else:
                # Unmonitored site
                website_id = self.args.mon_classes
                instance_id = 0
        except (ValueError, IndexError):
            # Fallback: use random assignment
            website_id = random.randint(0, self.args.mon_classes - 1)
            instance_id = 0
        
        # Select anonymity set for this website/instance
        if website_id < self.args.mon_classes and website_id in self.website_to_set:
            anonymity_sets = self.website_to_set[website_id]
            if len(anonymity_sets) > 0:
                anonymity_set_idx = anonymity_sets[instance_id % len(anonymity_sets)]
            else:
                anonymity_set_idx = 0
        else:
            # Unmonitored or website not in mapping - randomly select an anonymity set
            if len(self.total_sets) > 0:
                anonymity_set_idx = random.randint(0, len(self.total_sets) - 1)
            else:
                # Fallback: create a default super-matrix
                self.logger.warning("No anonymity sets available, using default")
                return trace
        
        # Get shrunk super matrix for this anonymity set
        if anonymity_set_idx < len(self.shrunk_super_matrices):
            cur_shrunk_super_matrix = np.asarray(
                np.ceil(self.shrunk_super_matrices[anonymity_set_idx]), dtype=np.float64
            )
            # Refinement stores (1, 1, 2, T) or (1, 2, T); peel leading 1-dims to (2, T)
            while cur_shrunk_super_matrix.ndim > 2:
                cur_shrunk_super_matrix = cur_shrunk_super_matrix[0]
            if cur_shrunk_super_matrix.ndim == 1:
                cur_shrunk_super_matrix = np.stack(
                    [cur_shrunk_super_matrix, cur_shrunk_super_matrix], axis=0
                )
            if cur_shrunk_super_matrix.shape[0] != 2:
                self.logger.warning(
                    "Unexpected shrunk super-matrix shape %s; using ones",
                    cur_shrunk_super_matrix.shape,
                )
                cur_shrunk_super_matrix = np.ones((2, self.config.tam_length))
        else:
            self.logger.warning(f"Anonymity set index {anonymity_set_idx} out of range, using default")
            cur_shrunk_super_matrix = np.ones((2, self.config.tam_length))
        
        # Sample slots based on PMF
        if anonymity_set_idx < len(self.PMF_upload) and anonymity_set_idx < len(self.PMF_download):
            sampled_slots_upload = self.sample_slots(
                self.PMF_upload[anonymity_set_idx], 
                self.config.alpha_upload
            )
            sampled_slots_download = self.sample_slots(
                self.PMF_download[anonymity_set_idx],
                self.config.alpha_download
            )
        else:
            # Fallback: sample all slots
            sampled_slots_upload = list(range(self.config.tam_length))
            sampled_slots_download = list(range(self.config.tam_length))
        
        # Generate defended trace using regularization
        defended_trace = self.generate_defense_trace(
            trace, cur_shrunk_super_matrix, sampled_slots_upload, sampled_slots_download
        )
        
        return defended_trace
    
    def sample_slots(self, trace_prob: List[float], threshold: float) -> List[int]:
        """
        Sample slots based on PMF probability distribution.
        """
        slot_idx = list(range(0, self.config.tam_length))
        random.shuffle(slot_idx)
        sampled_slots = []
        cum_prob = 0
        for i in range(self.config.tam_length):
            if cum_prob >= threshold:
                return sampled_slots
            cum_prob = cum_prob + trace_prob[slot_idx[i]]
            sampled_slots.append(slot_idx[i])
        return sampled_slots
    
    def generate_defense_trace(self, trace: np.ndarray, cur_shrunk_super_matrix: np.ndarray,
                                 sampled_slots_upload: List[int], sampled_slots_download: List[int]) -> np.ndarray:
        """
        Generate defended trace using Palette regularization.
        """
        if len(trace) == 0:
            return np.array([])
        
        times = trace[:, 0] - trace[0, 0]  # Normalize to start at 0
        # Extract direction from second column
        # If abs(value) == 1: direction only
        # If abs(value) > 1: value includes packet size, extract direction
        length_seq = []
        for val in trace[:, 1]:
            # Extract direction (sign)
            direction = int(np.sign(val))
            length_seq.append(direction)
        length_seq = np.array(length_seq)
        
        packets = np.empty((0, 2), dtype=np.float32)
        
        time_slot = self.config.cutoff_time / self.config.tam_length
        now_timestamp = time_slot
        now_slot_idx = 0
        
        cum_upload = 0.
        cum_download = 0.
        total_pkt_upload = 0.
        total_pkt_download = 0.
        now_sequence_idx = 0
        
        u_upload = random.randint(0, self.config.u_upload - 1)
        u_download = random.randint(0, self.config.u_download - 1)
        
        flag_end_upload = False
        flag_end_download = False
        flag_first_upload = False
        flag_first_download = False
        
        while now_timestamp <= self.config.cutoff_time:
            budget_upload = np.sum(cur_shrunk_super_matrix[0, now_slot_idx: now_slot_idx + u_upload])
            budget_download = np.sum(cur_shrunk_super_matrix[1, now_slot_idx: now_slot_idx + u_download])
            
            sm_upload = cur_shrunk_super_matrix[0, now_slot_idx]
            sm_download = cur_shrunk_super_matrix[1, now_slot_idx]
            
            target_upload = sm_upload
            target_download = sm_download
            
            # Super-matrix sampling
            if now_slot_idx not in sampled_slots_upload:
                target_upload = 0.
            if now_slot_idx not in sampled_slots_download:
                target_download = 0.
            
            # Get real packets in current slot
            while now_sequence_idx < len(times) and times[now_sequence_idx] < now_timestamp:
                if length_seq[now_sequence_idx] > 0:
                    cum_upload += 1
                    total_pkt_upload += 1
                elif length_seq[now_sequence_idx] < 0:
                    cum_download += 1
                    total_pkt_download += 1
                now_sequence_idx += 1

            # Match Palette regularization.py: resume dummy traffic after new real packets arrive
            if cum_upload != 0.0:
                flag_end_upload = False
            if cum_download != 0.0:
                flag_end_download = False

            # Regulate upload packets
            send_upload = target_upload
            if cum_upload == 0:
                if flag_end_upload:
                    send_upload = 0
                elif now_slot_idx != 0 and now_slot_idx % self.config.b == 0:
                    flag_end_upload = True
            
            if total_pkt_upload >= 1 and flag_first_upload:
                if cum_upload != 0 and cum_upload >= budget_upload:
                    send_upload = max(min(10, cum_upload), sm_upload)
            else:
                if cum_upload != 0:
                    send_upload = max(cum_upload, send_upload)
                else:
                    send_upload = 0
            
            # Regulate download packets
            send_download = target_download
            if cum_download == 0:
                if flag_end_download:
                    send_download = 0
                elif now_slot_idx != 0 and now_slot_idx % self.config.b == 0:
                    flag_end_download = True
            
            if total_pkt_download >= 1 and flag_first_download:
                if cum_download != 0 and cum_download >= budget_download:
                    send_download = max(min(30, cum_download), sm_download)
            else:
                if cum_download != 0:
                    send_download = max(cum_download, send_download)
                else:
                    send_download = 0
            
            if total_pkt_upload >= 1:
                flag_first_upload = True
            if total_pkt_download >= 1:
                flag_first_download = True
            
            cum_upload = max(0., cum_upload - send_upload)
            cum_download = max(0., cum_download - send_download)
            
            # Sample timestamps (match Simulation/palette/regularization.py clipping)
            eps = 1e-6
            tam_upper = float(self.config.tam_length)
            upload_timestamps = np.clip(
                now_timestamp + np.random.rayleigh(0.03, int(send_upload)),
                a_min=now_timestamp,
                a_max=now_timestamp + tam_upper - eps,
            )
            download_timestamps = np.clip(
                now_timestamp + np.random.rayleigh(0.03, int(send_download)),
                a_min=now_timestamp,
                a_max=now_timestamp + tam_upper - eps,
            )
            
            for j in range(len(upload_timestamps)):
                packets = np.append(packets, np.array([[upload_timestamps[j], 1]]), axis=0)
            for j in range(len(download_timestamps)):
                packets = np.append(packets, np.array([[download_timestamps[j], -1]]), axis=0)
            
            now_slot_idx += 1
            now_timestamp = time_slot * (now_slot_idx + 1)
        
        # Send remaining buffered packets
        if cum_upload > 0.:
            upload_timestamps = np.clip(
                now_timestamp + np.random.rayleigh(0.1, int(cum_upload)),
                a_min=now_timestamp,
                a_max=now_timestamp + 5.
            )
            for p in range(len(upload_timestamps)):
                packets = np.append(packets, np.array([[upload_timestamps[p], 1]]), axis=0)
        
        if cum_download > 0.:
            # Original Palette uses raw Rayleigh then clips to [now, now+5] (no now_timestamp offset)
            download_timestamps = np.clip(
                np.random.rayleigh(0.1, int(cum_download)),
                a_min=now_timestamp,
                a_max=now_timestamp + 5.0,
            )
            for p in range(len(download_timestamps)):
                packets = np.append(packets, np.array([[download_timestamps[p], -1]]), axis=0)
        
        # Sort by timestamp and normalize
        packets_idx = np.argsort(packets[:, 0])
        packets = packets[packets_idx]
        packets[:, 0] = packets[:, 0] - packets[0, 0]
        
        return packets

