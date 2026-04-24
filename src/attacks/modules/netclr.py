"""
NetCLR module: DFNet backbone and DFsimCLR for SimCLR pre-training.
NetCLRNet is the classifier used for attack (fine-tuning); it can load
pretrained backbone weights from a DFsimCLR checkpoint.

Pre-training is run via run_pretrain(); the attack then runs fine-tuning
(closed-world or open-world based on --open-world).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


class DFNet(nn.Module):
    """
    Direction-based 1D CNN backbone. Input shape: (B, 1, seq_len), typically seq_len=5000.
    """
    def __init__(self, out_dim: int):
        super(DFNet, self).__init__()
        kernel_size = 8
        conv_stride = 1
        pool_stride = 4
        pool_size = 8

        self.conv1 = nn.Conv1d(1, 32, kernel_size, stride=conv_stride)
        self.conv1_1 = nn.Conv1d(32, 32, kernel_size, stride=conv_stride)

        self.conv2 = nn.Conv1d(32, 64, kernel_size, stride=conv_stride)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size, stride=conv_stride)

        self.conv3 = nn.Conv1d(64, 128, kernel_size, stride=conv_stride)
        self.conv3_3 = nn.Conv1d(128, 128, kernel_size, stride=conv_stride)

        self.conv4 = nn.Conv1d(128, 256, kernel_size, stride=conv_stride)
        self.conv4_4 = nn.Conv1d(256, 256, kernel_size, stride=conv_stride)

        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(256)

        self.max_pool_1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)

        self.fc = nn.Linear(5120, out_dim)

    def weight_init(self):
        for _, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inp):
        x = inp
        x = F.pad(x, (3, 4))
        x = F.elu(self.conv1(x))
        x = F.pad(x, (3, 4))
        x = F.elu(self.batch_norm1(self.conv1_1(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_1(x)
        x = self.dropout1(x)

        x = F.pad(x, (3, 4))
        x = F.relu(self.conv2(x))
        x = F.pad(x, (3, 4))
        x = F.relu(self.batch_norm2(self.conv2_2(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_2(x)
        x = self.dropout2(x)

        x = F.pad(x, (3, 4))
        x = F.relu(self.conv3(x))
        x = F.pad(x, (3, 4))
        x = F.relu(self.batch_norm3(self.conv3_3(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_3(x)
        x = self.dropout3(x)

        x = F.pad(x, (3, 4))
        x = F.relu(self.conv4(x))
        x = F.pad(x, (3, 4))
        x = F.relu(self.batch_norm4(self.conv4_4(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DFsimCLR(nn.Module):
    """
    SimCLR wrapper: backbone + projection head for contrastive pre-training.
    """
    def __init__(self, df: DFNet, out_dim: int = 128):
        super(DFsimCLR, self).__init__()
        self.backbone = df
        self.backbone.weight_init()
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, inp):
        return self.backbone(inp)


class NetCLRNet(nn.Module):
    """
    Classifier for attack (closed-world or open-world fine-tuning).
    Same architecture as DFNet(num_classes). Supports loading pretrained
    backbone from a DFsimCLR checkpoint (strips 'backbone.' prefix and skips 'backbone.fc').
    """
    def __init__(self, num_classes: int, seq_length: int = 5000):
        super(NetCLRNet, self).__init__()
        self.num_classes = num_classes
        self.seq_length = seq_length
        self._net = DFNet(out_dim=num_classes)
        self._net.weight_init()

    def load_pretrained_backbone(self, state_dict: dict, strict: bool = False):
        """
        Load backbone weights from a DFsimCLR checkpoint.
        Keys expected as 'backbone.xxx'; 'backbone.fc' is skipped (classifier head kept).
        """
        new_state = {}
        for k, v in state_dict.items():
            if not k.startswith("backbone."):
                continue
            sub = k[len("backbone."):]
            if sub.startswith("fc."):
                continue
            new_state["_net." + sub] = v
        self.load_state_dict(new_state, strict=strict)

    def forward(self, x):
        return self._net(x)


# ---------------------------------------------------------------------------
# SimCLR pre-training (used when running NetCLR attack; invoked before fine-tuning)
# ---------------------------------------------------------------------------


def simclr_accuracy(output, target, topk=(1, 5)):
    """Accuracy over top-k for SimCLR contrastive logits (for logging)."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def info_nce_loss(features, batch_size, n_views, temperature, device):
    """InfoNCE loss for SimCLR (normalized features, same batch = positive)."""
    labels = torch.cat([torch.arange(batch_size, device=features.device) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / temperature
    return logits, labels


def run_pretrain(args, flist, labels, device, logger, seq_length=None, extract_fn=None):
    """
    Run SimCLR pre-training on direction features and save the model to checkpoints.

    Called automatically from NetCLRAttack.run() when the pretrained checkpoint
    does not exist. Uses flist (and labels for shuffling only) to build
    augmented views and trains DFsimCLR with InfoNCE loss.

    :param args: argparse namespace with checkpoints, batch_size, seq_length, etc.
    :param flist: array of paths to .cell trace files (monitored set)
    :param labels: array of labels (same length as flist; used for shuffle only)
    :param device: torch device
    :param logger: logger instance for info messages
    :param seq_length: trace length (default from args.seq_length or 5000)
    :param extract_fn: optional (path, seq_len) -> feat; default uses parse_trace + feature_transform df
    :return: path to saved checkpoint (checkpoints/netclr/NetCLR_pretrained.pth)
    """
    from attacks.config.const import PRETRAIN_SAVE_EVERY, PRETRAIN_TEMPERATURE
    from utils.general import parse_trace, feature_transform
    from utils.netclr_augment import compute_outgoing_burst_cdf, Augmentor, NetCLRTrainDataset

    try:
        import tqdm
    except ImportError:
        tqdm = None

    seq_length = seq_length or getattr(args, "seq_length", 5000)
    batch_size = getattr(args, "batch_size", 128)
    pretrain_epochs = getattr(args, "epochs", 100)
    pretrain_lr = getattr(args, "lr0", 0.002)
    temperature = PRETRAIN_TEMPERATURE
    save_every = PRETRAIN_SAVE_EVERY
    fp16 = getattr(args, "amp", False) and device.type == "cuda"
    workers = getattr(args, "workers", 4)
    seed = getattr(args, "pretrain_seed", 42)

    checkpoints_dir = getattr(args, "checkpoints", "./checkpoints")
    out_dir = os.path.join(checkpoints_dir, "netclr")
    os.makedirs(out_dir, exist_ok=True)

    if extract_fn is None:
        def extract_fn(path, slen):
            trace = parse_trace(path)
            feat = feature_transform(trace, feature_type="df", seq_length=slen)
            return feat

    np.random.seed(seed)
    x_list = []
    for path in flist:
        feat = extract_fn(path, seq_length)
        if isinstance(feat, np.ndarray) and feat.ndim == 2:
            feat = feat[0]
        x_list.append(np.asarray(feat, dtype=np.float32))
    x_train = np.stack(x_list, axis=0)
    y_train = np.asarray(labels, dtype=np.int64)
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    logger.info("NetCLR pre-training: {} traces, seq_length={}, epochs={}".format(
        len(x_train), seq_length, pretrain_epochs))

    max_outgoing, cdf = compute_outgoing_burst_cdf(
        x_train, max_samples=min(1000, len(x_train)), seq_length=seq_length
    )
    augmentor = Augmentor(max_outgoing, cdf, seq_length=seq_length)
    dataset = NetCLRTrainDataset(x_train, y_train, augmentor, n_views=2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    df = DFNet(out_dim=512)
    model = DFsimCLR(df, out_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader), eta_min=0
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler(enabled=fp16)
    n_views = 2

    for epoch in range(pretrain_epochs + 1):
        model.train()
        n_iter = 0
        it = loader
        if tqdm is not None:
            it = tqdm.tqdm(it, unit="batch")
        for data, _ in it:
            data = torch.cat(data, dim=0)
            data = data.view(data.size(0), 1, data.size(1)).float().to(device)
            with autocast(enabled=fp16):
                features = model(data)
                logits, targets = info_nce_loss(
                    features, batch_size, n_views, temperature, device
                )
                loss = criterion(logits, targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if tqdm is not None and n_iter % 100 == 0:
                top1, _ = simclr_accuracy(logits, targets, topk=(1, 5))
                it.set_postfix(loss=loss.item(), acc=top1.item())
            n_iter += 1

        if epoch >= 10:
            scheduler.step()
        if epoch % save_every == 0:
            path = os.path.join(out_dir, "NetCLR_pretrained.pth")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, path)
            logger.info("NetCLR pre-train checkpoint saved (epoch {}): {}".format(epoch, path))

    path = os.path.join(out_dir, "NetCLR_pretrained.pth")
    torch.save({"model_state_dict": model.state_dict(), "epoch": pretrain_epochs}, path)
    logger.info("NetCLR pre-training done. Model saved to: {}".format(path))
    return path
