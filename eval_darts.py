#!/usr/bin/env python3
"""
DARTS Evaluation Phase - Evaluación de Arquitecturas con Múltiples Seeds

Uso:
    python eval_darts.py                            # Evalúa todas las seeds con genotipos
    python eval_darts.py --seeds 42 43 44          # Evalúa solo seeds específicas
    python eval_darts.py --fast                    # Modo rápido (menos épocas)

Detecta automáticamente seeds ya evaluadas (busca best_model.pt en results/seed_X/eval_full/)
y las salta para continuar desde donde se quedó.
"""

import os
import sys
import json
import time
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch.amp import autocast, GradScaler
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

print(f"Device: {DEVICE} | AMP: {'Enabled' if USE_AMP else 'Disabled'}")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURACIONES
# ============================================================================

EVAL_CFG_NORMAL = {
    "batch_size": 128,
    "learning_rate": 0.025,
    "momentum": 0.9,
    "weight_decay": 3e-4,
    "epochs": 200,
    "init_channels": 36,
    "layers": 20,
    "auxiliary": True,
    "cutout": True,
    "cutout_length": 16,
    "drop_path_prob": 0.2,
    "num_workers": 6,
    "seed": 42,
    "data": "./data",
    "stem_multiplier": 3,
    "report_freq": 50,
    "grad_clip": 5,
    "learning_rate_min": 0.0,
    "early_stopping": True,
    "early_stopping_patience": 15,
    "overfitting_threshold": 10.0,
    "target_batch_size": 256,
    "use_amp": True,
    "use_preprocessed": True,
    "val_freq": 10,
}

EVAL_CFG_FAST = {
    "batch_size": 64,
    "learning_rate": 0.025,
    "momentum": 0.9,
    "weight_decay": 3e-4,
    "epochs": 1,
    "init_channels": 16,
    "layers": 4,
    "auxiliary": True,
    "cutout": False,
    "cutout_length": 16,
    "drop_path_prob": 0.1,
    "num_workers": 4,
    "seed": 42,
    "data": "./data",
    "stem_multiplier": 3,
    "report_freq": 50,
    "grad_clip": 5,
    "learning_rate_min": 0.0,
    "target_batch_size": 128,
    "use_amp": True,
    "use_preprocessed": True,
    "val_freq": 1,
}

EVAL_CFG_SPEED = {
    "batch_size": 96,
    "learning_rate": 0.025,
    "momentum": 0.9,
    "weight_decay": 3e-4,
    "epochs": 150,
    "init_channels": 36,
    "layers": 20,
    "auxiliary": True,
    "cutout": True,
    "cutout_length": 16,
    "drop_path_prob": 0.2,
    "num_workers": 4,
    "seed": 42,
    "data": "./data",
    "stem_multiplier": 3,
    "report_freq": 25,
    "grad_clip": 5,
    "learning_rate_min": 0.0,
    "early_stopping": True,
    "early_stopping_patience": 30,
    "overfitting_threshold": 10.0,
    "target_batch_size": 192,
    "use_amp": True,
    "use_preprocessed": True,
    "val_freq": 5,
}

# ============================================================================
# OPERACIONES PRIMITIVAS
# ============================================================================

PRIMITIVES = [
    'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
    'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'
]

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}

# ============================================================================
# CELL DISCRETA PARA EVALUACIÓN
# ============================================================================

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.size(0), 1, 1, 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    x = x.div(keep_prob) * random_tensor
    return x

class CellOp(nn.Module):
    def __init__(self, op_name, C, stride):
        super().__init__()
        self.op_name = op_name
        self.op = OPS[op_name](C, stride, True)
        if 'pool' in op_name:
            self.op = nn.Sequential(self.op, nn.BatchNorm2d(C, affine=True))
    def forward(self, x):
        return self.op(x)

class CellDiscrete(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, drop_path_prob=0.0):
        super().__init__()
        self.reduction = reduction
        self.drop_path_prob = drop_path_prob
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._indices = indices
        self._op_names = op_names
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = CellOp(name, C, stride)
            self._ops.append(op)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for _ in range(self._steps):
            h1 = states[self._indices[offset]]
            h2 = states[self._indices[offset + 1]]
            op1 = self._ops[offset]
            op2 = self._ops[offset + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and self.drop_path_prob > 0.0:
                if op1.op_name != 'skip_connect':
                    h1 = drop_path(h1, self.drop_path_prob, self.training)
                if op2.op_name != 'skip_connect':
                    h2 = drop_path(h2, self.drop_path_prob, self.training)
            s = h1 + h2
            states.append(s)
            offset += 2
        return torch.cat([states[i] for i in self._concat], dim=1)

class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class NetworkCIFARFinal(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, drop_path_prob=0.0, stem_multiplier=3):
        super().__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = drop_path_prob
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        self.auxiliary_head_index = 2 * layers // 3 if auxiliary else -1
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = CellDiscrete(genotype=genotype, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr,
                             reduction=reduction, reduction_prev=reduction_prev, drop_path_prob=self.drop_path_prob)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == self.auxiliary_head_index:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        logits_aux = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if self._auxiliary and self.training and i == self.auxiliary_head_index:
                logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self._auxiliary and self.training:
            return logits, logits_aux
        return logits

# ============================================================================
# UTILIDADES
# ============================================================================

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

def analyze_early_stopping_post_hoc(eval_history, patience=50):
    val_accs = eval_history["val_acc"]
    best_val_acc = 0.0
    best_epoch = 0
    counter = 0
    
    for epoch, val_acc in enumerate(val_accs):
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    if best_epoch == 0:
        best_epoch = len(val_accs)
        best_val_acc = val_accs[-1]
    
    return best_epoch, best_val_acc

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return tuple(res)

def count_params_in_MB(model):
    return sum(v.numel() for name, v in model.named_parameters() if 'auxiliary' not in name) / 1e6

def compute_flops(model, input_size=(1, 3, 32, 32)):
    total_flops = 0
    hooks = []
    def conv_flops_hook(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0)
        out_h, out_w = output.size(2), output.size(3)
        k_h, k_w = module.kernel_size
        in_ch = module.in_channels
        out_ch = module.out_channels
        groups = module.groups
        flops = batch_size * out_h * out_w * k_h * k_w * (in_ch // groups) * out_ch
        total_flops += flops
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_flops_hook))
    with torch.no_grad():
        was_training = model.training
        model.eval()
        dummy = torch.randn(input_size).to(DEVICE)
        model(dummy)
        if was_training:
            model.train()
    for hook in hooks:
        hook.remove()
    return total_flops

def set_reproducibility(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================================
# DATA LOADERS
# ============================================================================

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

class PreprocessedCIFAR10(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class Cutout:
    def __init__(self, length):
        self.length = length
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=torch.float32)
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        mask[y1:y2, x1:x2] = 0.
        img *= mask.unsqueeze(0).expand_as(img)
        return img

def preprocess_and_save_cifar10(data_dir='./data'):
    import torchvision
    import torchvision.transforms as transforms
    
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, 'cifar10_preprocessed.pt')
    
    if os.path.exists(cache_path):
        return cache_path
    
    print("  Preprocessing CIFAR-10 (one-time operation)...")
    
    preprocess_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    
    train_dataset_full = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=preprocess_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=preprocess_train
    )
    
    train_images = torch.stack([img for img, _ in train_dataset_full])
    train_labels = torch.tensor([label for _, label in train_dataset_full])
    test_images = torch.stack([img for img, _ in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])
    
    torch.save({
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }, cache_path)
    
    print(f"  ✓ Preprocessing complete. Saved to {cache_path}")
    return cache_path

import torchvision.transforms as transforms

def build_eval_dataloaders(cfg):
    import torchvision.datasets as dset
    
    use_preprocessed = cfg.get("use_preprocessed", True)
    target_batch = cfg.get("target_batch_size", cfg["batch_size"])
    accumulation_steps = target_batch // cfg["batch_size"] if target_batch > cfg["batch_size"] else 1
    
    if use_preprocessed:
        preprocess_and_save_cifar10(cfg["data"])
        data = torch.load(os.path.join(cfg["data"], 'cifar10_preprocessed.pt'))
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        if cfg["cutout"]:
            train_transform.transforms.append(Cutout(cfg["cutout_length"]))
        
        train_dataset = PreprocessedCIFAR10(data['train_images'], data['train_labels'], transform=train_transform)
        test_dataset = PreprocessedCIFAR10(data['test_images'], data['test_labels'])
        
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(0.8 * num_train)
        rng = np.random.RandomState(cfg["seed"])
        rng.shuffle(indices)
        
        train_queue = DataLoader(
            train_dataset, batch_size=cfg["batch_size"],
            sampler=SubsetRandomSampler(indices[:split]),
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=True if cfg["num_workers"] > 0 else False,
            prefetch_factor=4 if cfg["num_workers"] > 0 else None
        )
        val_queue = DataLoader(
            train_dataset, batch_size=cfg["batch_size"],
            sampler=SubsetRandomSampler(indices[split:]),
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=True if cfg["num_workers"] > 0 else False,
            prefetch_factor=4 if cfg["num_workers"] > 0 else None
        )
        test_queue = DataLoader(
            test_dataset, batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=True if cfg["num_workers"] > 0 else False,
            prefetch_factor=4 if cfg["num_workers"] > 0 else None
        )
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        eval_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        train_data = dset.CIFAR10(root=cfg["data"], train=True, download=True, transform=train_tf)
        test_data = dset.CIFAR10(root=cfg["data"], train=False, download=True, transform=eval_tf)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(0.8 * num_train)
        rng = np.random.RandomState(cfg["seed"])
        rng.shuffle(indices)
        train_queue = DataLoader(train_data, batch_size=cfg["batch_size"], sampler=SubsetRandomSampler(indices[:split]),
                                num_workers=cfg["num_workers"], pin_memory=True)
        val_queue = DataLoader(train_data, batch_size=cfg["batch_size"], sampler=SubsetRandomSampler(indices[split:]),
                              num_workers=cfg["num_workers"], pin_memory=True)
        test_queue = DataLoader(test_data, batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=cfg["num_workers"], pin_memory=True)
    
    return train_queue, val_queue, test_queue, accumulation_steps

# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_confusion_matrix(predictions, targets, output_dir, num_classes=10):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    with open(os.path.join(output_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump(cm.tolist(), f)
    return cm

def save_classification_report(targets, predictions, output_dir):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    report = classification_report(targets, predictions, target_names=class_names, digits=4)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    return report

def plot_training_curves(history, output_dir, phase="evaluation"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{phase.capitalize()} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc@1', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc@1', linewidth=2)
    if 'test_acc' in history:
        axes[1].plot(epochs, history['test_acc'], 'g-', label='Test Acc@1', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{phase.capitalize()} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()

def plot_overfitting_gap(gap_history, output_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(gap_history) + 1)
    plt.plot(epochs, gap_history, 'r-', linewidth=2, label='Train-Val Accuracy Gap')
    plt.axhline(y=0, color='g', linestyle='-', alpha=0.3, label='No Overfitting')
    plt.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='High Overfitting Threshold (10%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.title('Overfitting Gap During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overfitting_gap.png'), dpi=150)
    plt.close()

# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def train_eval_epoch(train_q, model, criterion, optimizer, cfg, epoch, scaler, accumulation_steps, report_freq=50):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    if cfg["epochs"] > 1:
        current_drop_path = cfg["drop_path_prob"] * epoch / (cfg["epochs"] - 1)
    else:
        current_drop_path = cfg["drop_path_prob"]
    model.drop_path_prob = current_drop_path
    for cell in model.cells:
        cell.drop_path_prob = current_drop_path
    
    for step, (inp, tgt) in enumerate(train_q):
        n = inp.size(0)
        inp = inp.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(
        device_type=DEVICE.type,
        enabled=USE_AMP and cfg.get("use_amp", True)):
            logits = model(inp)
            if isinstance(logits, tuple):
                logits, logits_aux = logits
                loss = criterion(logits, tgt) + 0.4 * criterion(logits_aux, tgt)
            else:
                loss = criterion(logits, tgt)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        
        prec1, prec5 = accuracy(logits if not isinstance(logits, tuple) else logits[0], tgt, topk=(1, 5))
        objs.update(loss.item() * accumulation_steps, n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        if step % report_freq == 0 and step > 0:
            print(f"    Step {step:3d} | loss={objs.avg:.4f} | acc@1={top1.avg:.2f}%")
    return {"loss": objs.avg, "top1": top1.avg, "top5": top5.avg}

def evaluate_final(data_q, model, criterion, subset_batches=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.inference_mode():
        for step, (inp, tgt) in enumerate(data_q):
            if subset_batches is not None and step >= subset_batches:
                break
            inp = inp.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)
            
            with autocast('cuda', enabled=USE_AMP):
                logits = model(inp)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, tgt)
            
            prec1, prec5 = accuracy(logits, tgt, topk=(1, 5))
            n = inp.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(tgt.cpu().numpy())
    return {"loss": objs.avg, "top1": top1.avg, "top5": top5.avg, "predictions": all_preds, "targets": all_targets}

# ============================================================================
# RUN EVALUATION
# ============================================================================

def genotype_to_dict(genotype):
    return {
        "normal": [(op, int(idx)) for op, idx in genotype.normal],
        "normal_concat": [int(x) for x in genotype.normal_concat],
        "reduce": [(op, int(idx)) for op, idx in genotype.reduce],
        "reduce_concat": [int(x) for x in genotype.reduce_concat],
    }

def dict_to_genotype(d):
    return Genotype(
        normal=d["normal"],
        normal_concat=d["normal_concat"],
        reduce=d["reduce"],
        reduce_concat=d["reduce_concat"]
    )

def run_evaluation(cfg, genotype, output_dir):
    set_reproducibility(cfg["seed"])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config_eval.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(output_dir, "genotype.json"), "w") as f:
        json.dump(genotype_to_dict(genotype), f, indent=2)
    
    train_queue, val_queue, test_queue, accumulation_steps = build_eval_dataloaders(cfg)
    
    criterion = nn.CrossEntropyLoss()
    model = NetworkCIFARFinal(C=cfg["init_channels"], num_classes=10, layers=cfg["layers"],
                            auxiliary=cfg["auxiliary"], genotype=genotype,
                            drop_path_prob=cfg["drop_path_prob"],
                            stem_multiplier=cfg.get("stem_multiplier", 3)).to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"],
                               momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"],
                                                           eta_min=cfg.get("learning_rate_min", 0.0))
    
    scaler = GradScaler(enabled=USE_AMP and cfg.get("use_amp", True))
    
    use_early_stopping = cfg.get("early_stopping", False)
    early_stopper = None
    if use_early_stopping:
        patience = cfg.get("early_stopping_patience", 50)
        early_stopper = EarlyStopping(patience=patience, min_delta=0.001, mode='max')
    
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    stopped_early = False
    
    history = {"train_loss": [], "train_acc": [], "train_acc5": [], "val_loss": [],
                "val_acc": [], "val_acc5": [], "overfitting_gap": []}
    total_epochs = cfg["epochs"]
    t0 = time.time()
    flops = compute_flops(model)
    params = count_params_in_MB(model)
    mode_name = 'FAST' if total_epochs < 50 else ('SPEED' if cfg.get("early_stopping", False) else 'NORMAL')
    
    val_freq = cfg.get("val_freq", 5)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION PHASE - Seed {cfg['seed']} - Mode: {mode_name}")
    print(f"{'='*70}")
    print(f"Config: epochs={total_epochs}, channels={cfg['init_channels']}, layers={cfg['layers']}")
    print(f"Model: {params:.2f}M params, {flops/1e9:.2f} GFLOPs")
    print(f"Batch: {cfg['batch_size']} -> effective {cfg.get('target_batch_size', cfg['batch_size'])} (accumulation={accumulation_steps}x)")
    print(f"Auxiliary: {cfg['auxiliary']}, Cutout: {cfg['cutout']}, DropPath: {cfg['drop_path_prob']}")
    print(f"Val frequency: every {val_freq} epoch(s)")
    if use_early_stopping:
        print(f"EARLY STOPPING: ENABLED (patience={patience})")
    print(f"{'='*70}")
    
    for epoch in range(total_epochs):
        remaining_epochs = total_epochs - epoch - 1
        start_epoch_time = time.time()
        progress = (epoch + 1) / total_epochs * 100
        
        print(f"\n[Epoch {epoch+1:4d}/{total_epochs}] ({progress:6.2f}%) | Remaining: {remaining_epochs} epochs")
        
        report_freq = cfg.get("report_freq", 50)
        train_metrics = train_eval_epoch(train_queue, model, criterion, optimizer, cfg, epoch, scaler, accumulation_steps, report_freq)
        
        if (epoch + 1) % val_freq == 0 or epoch == total_epochs - 1:
            val_metrics = evaluate_final(val_queue, model, criterion, subset_batches=50)
            
            if val_metrics['top1'] > best_val_acc:
                best_val_acc = val_metrics['top1']
                best_epoch = epoch + 1
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            if use_early_stopping and early_stopper(val_metrics['top1'], epoch):
                stopped_early = True
                print(f"\n⚠️ EARLY STOPPING TRIGGERED at epoch {epoch+1}!")
                break
        else:
            val_metrics = {"top1": 0.0, "top5": 0.0, "loss": 0.0}
        
        scheduler.step()
        
        epoch_time = time.time() - start_epoch_time
        elapsed = time.time() - t0
        eta_remaining = epoch_time * remaining_epochs
        
        overfitting_gap = train_metrics['top1'] - val_metrics['top1']
        history["overfitting_gap"].append(float(overfitting_gap))
        
        print(f"  Time: {epoch_time:.1f}s | Elapsed: {elapsed:.1f}s | ETA: {eta_remaining:.1f}s")
        print(f"  Train | loss={train_metrics['loss']:.4f} | acc@1={train_metrics['top1']:.2f}%")
        if val_metrics['top1'] > 0:
            print(f"  Val   | loss={val_metrics['loss']:.4f} | acc@1={val_metrics['top1']:.2f}% | Gap: {overfitting_gap:.2f}%")
        
        history["train_loss"].append(float(train_metrics['loss']))
        history["train_acc"].append(float(train_metrics['top1']))
        history["train_acc5"].append(float(train_metrics['top5']))
        history["val_loss"].append(float(val_metrics['loss']))
        history["val_acc"].append(float(val_metrics['top1']))
        history["val_acc5"].append(float(val_metrics['top5']))
    
    eval_time = time.time() - t0
    actual_epochs = len(history["train_acc"])
    
    history["best_val_acc"] = float(best_val_acc)
    history["best_epoch"] = best_epoch
    history["actual_epochs"] = actual_epochs
    history["stopped_early"] = stopped_early
    history["max_overfitting_gap"] = float(max(history["overfitting_gap"])) if history["overfitting_gap"] else 0.0
    history["avg_overfitting_gap"] = float(np.mean(history["overfitting_gap"])) if history["overfitting_gap"] else 0.0
    
    es_info = {
        "early_stopping_triggered": stopped_early,
        "best_val_acc": float(best_val_acc),
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "patience_used": cfg.get("early_stopping_patience", 50) if use_early_stopping else None
    }
    with open(os.path.join(output_dir, "es_trigger_info.json"), "w") as f:
        json.dump(es_info, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(output_dir, "best_model.pt"))
        model.load_state_dict(best_model_state)
        
        print(f"\nRunning final evaluation on TEST set...")
        final_test = evaluate_final(test_queue, model, criterion)
        
        plot_confusion_matrix(final_test['predictions'], final_test['targets'], output_dir)
        save_classification_report(final_test['targets'], final_test['predictions'], output_dir)
        
        history["final_test_acc"] = float(final_test['top1'])
        history["final_test_acc5"] = float(final_test['top5'])
        history["final_test_loss"] = float(final_test['loss'])
    else:
        final_test = {"top1": 0.0, "top5": 0.0, "loss": 0.0}
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    plot_training_curves(history, output_dir, "evaluation")
    plot_overfitting_gap(history["overfitting_gap"], output_dir)
    
    test_error = 100 - final_test['top1']
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS:")
    print(f"  Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"  TEST Acc@1: {final_test['top1']:.2f}% | TEST Error: {test_error:.2f}%")
    print(f"  TEST Acc@5: {final_test['top5']:.2f}%")
    print(f"  Params: {params:.2f}M | FLOPs: {flops/1e9:.2f}G")
    print(f"  Epochs trained: {actual_epochs}/{total_epochs}")
    if stopped_early:
        print(f"  ⏹ Stopped early!")
    print(f"  Max Gap: {history['max_overfitting_gap']:.2f}% | Avg Gap: {history['avg_overfitting_gap']:.2f}%")
    print(f"  Eval Time: {eval_time:.2f}s")
    print(f"{'='*70}")
    
    return final_test, history, params, flops, eval_time, best_val_acc, best_epoch, stopped_early

# ============================================================================
# HELPERS PARA DETECTAR EVALUACIONES COMPLETADAS
# ============================================================================

def is_eval_completed(seed_dir):
    """Check if evaluation is already completed for this seed"""
    metrics_file = os.path.join(seed_dir, "eval_full", "metrics.json")
    best_model_file = os.path.join(seed_dir, "eval_full", "best_model.pt")
    return os.path.exists(metrics_file) and os.path.exists(best_model_file)

def has_genotype(seed_dir):
    """Check if genotype exists for this seed"""
    genotype_file = os.path.join(seed_dir, "search", "genotype.json")
    return os.path.exists(genotype_file)

def load_genotype(seed_dir):
    """Load genotype from seed directory"""
    genotype_file = os.path.join(seed_dir, "search", "genotype.json")
    with open(genotype_file, 'r') as f:
        d = json.load(f)
    return dict_to_genotype(d)

def get_seeds_to_evaluate(requested_seeds, output_dir):
    """Get list of seeds that need evaluation (skip if already evaluated or no genotype)"""
    seeds_to_eval = []
    seeds_skipped = []
    seeds_no_genotype = []
    
    for seed in requested_seeds:
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        
        if not has_genotype(seed_dir):
            seeds_no_genotype.append(seed)
            print(f"  SKIP seed {seed}: no genotype found (run search first)")
            continue
            
        if is_eval_completed(seed_dir):
            seeds_skipped.append(seed)
            print(f"  SKIP seed {seed}: evaluation already completed (best_model.pt exists)")
            continue
        
        seeds_to_eval.append(seed)
    
    return seeds_to_eval, seeds_skipped, seeds_no_genotype

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DARTS Evaluation Phase - Multiple Seeds')
    parser.add_argument('--fast', action='store_true', help='Use fast mode (1 epoch)')
    parser.add_argument('--speed', action='store_true', help='Use speed mode (150 epochs, early stopping)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='Specific seeds to evaluate')
    parser.add_argument('--start-seed', type=int, default=42, help='Starting seed (default: 42)')
    parser.add_argument('--num-seeds', type=int, default=5, help='Number of seeds to evaluate')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    if args.seeds:
        base_seeds = args.seeds
    else:
        base_seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    
    if args.speed:
        eval_cfg = copy.deepcopy(EVAL_CFG_SPEED)
        mode = "SPEED"
    elif args.fast:
        eval_cfg = copy.deepcopy(EVAL_CFG_FAST)
        mode = "FAST"
    else:
        eval_cfg = copy.deepcopy(EVAL_CFG_NORMAL)
        mode = "NORMAL"
    
    print(f"\n{'='*70}")
    print(f"DARTS EVALUATION PHASE - {mode} MODE")
    print(f"{'='*70}")
    print(f"Seeds requested: {base_seeds}")
    print(f"Output dir: {args.output}/")
    print(f"Eval Config: {eval_cfg['epochs']} epochs, {eval_cfg['init_channels']} channels, batch={eval_cfg['batch_size']}")
    if eval_cfg.get('early_stopping', False):
        print(f"EARLY STOPPING: ENABLED (patience={eval_cfg['early_stopping_patience']})")
    print(f"{'='*70}")
    
    seeds_to_eval, seeds_skipped, seeds_no_genotype = get_seeds_to_evaluate(base_seeds, args.output)
    
    print(f"\n{'='*70}")
    print(f"SEEDS TO EVALUATE: {seeds_to_eval}")
    print(f"SEEDS SKIPPED (already evaluated): {seeds_skipped}")
    print(f"SEEDS SKIPPED (no genotype): {seeds_no_genotype}")
    print(f"{'='*70}")
    
    if not seeds_to_eval:
        print("\nNo seeds to evaluate! All completed or missing genotypes.")
        return
    
    all_results = []
    total_t0 = time.time()
    
    for i, seed in enumerate(seeds_to_eval):
        print(f"\n{'#'*70}")
        print(f"# EVALUATE SEED {seed} ({i+1}/{len(seeds_to_eval)})")
        print(f"# Seeds remaining: {len(seeds_to_eval) - i - 1}")
        print(f"{'#'*70}")
        torch.cuda.empty_cache()
        
        eval_cfg['seed'] = seed
        seed_dir = os.path.join(args.output, f"seed_{seed}")
        eval_dir = os.path.join(seed_dir, "eval_full")
        
        genotype = load_genotype(seed_dir)
        
        print(f"\nGenotype for seed {seed}:")
        print(f"  Normal: {genotype.normal}")
        print(f"  Reduce: {genotype.reduce}")
        
        test_results, eval_history, params, flops, eval_time, best_val_acc, best_epoch, stopped_early = run_evaluation(eval_cfg, genotype, eval_dir)
        
        test_error = 100 - test_results['top1']
        
        result = {
            "seed": seed,
            "run": i + 1,
            "train_acc": eval_history["train_acc"][best_epoch-1] if best_epoch > 0 else eval_history["train_acc"][-1],
            "val_acc": best_val_acc,
            "test_acc": test_results['top1'],
            "test_acc5": test_results['top5'],
            "test_error": test_error,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "max_gap": eval_history.get("max_overfitting_gap", 0),
            "avg_gap": eval_history.get("avg_overfitting_gap", 0),
            "params_millions": params,
            "flops_giga": flops / 1e9,
            "eval_time_sec": eval_time,
            "genotype": genotype_to_dict(genotype),
        }
        all_results.append(result)
        
        print(f"\n[COMPLETED] Seed {seed} | Test Acc: {test_results['top1']:.2f}% | Test Error: {test_error:.2f}% | Time: {eval_time/60:.1f} min")
        
        torch.cuda.empty_cache()
    
    total_time = time.time() - total_t0
    
    if all_results:
        train_accs = [r['train_acc'] for r in all_results]
        val_accs = [r['val_acc'] for r in all_results]
        test_accs = [r['test_acc'] for r in all_results]
        test_errors = [r['test_error'] for r in all_results]
        max_gaps = [r['max_gap'] for r in all_results]
        avg_gaps = [r['avg_gap'] for r in all_results]
        
        mean_train_acc = np.mean(train_accs)
        std_train_acc = np.std(train_accs)
        mean_val_acc = np.mean(val_accs)
        std_val_acc = np.std(val_accs)
        mean_test_acc = np.mean(test_accs)
        std_test_acc = np.std(test_accs)
        mean_test_error = np.mean(test_errors)
        std_test_error = np.std(test_errors)
        mean_max_gap = np.mean(max_gaps)
        std_max_gap = np.std(max_gaps)
        mean_avg_gap = np.mean(avg_gaps)
        std_avg_gap = np.std(avg_gaps)
        
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Seeds evaluated: {seeds_to_eval}")
        print(f"Seeds skipped: {seeds_skipped}")
        print(f"Seeds no genotype: {seeds_no_genotype}")
        print(f"\n{'─'*70}")
        print("INDIVIDUAL RESULTS:")
        print(f"{'─'*70}")
        print(f"{'Seed':>6} | {'Train%':>8} | {'Val%':>8} | {'Test%':>8} | {'Err%':>8} | {'Gap%':>8} | {'Epoch':>6} | {'Stopped':>8}")
        print(f"{'─'*70}")
        for r in all_results:
            stopped_str = "YES" if r['stopped_early'] else "NO"
            print(f"{r['seed']:>6} | {r['train_acc']:>8.2f} | {r['val_acc']:>8.2f} | {r['test_acc']:>8.2f} | {r['test_error']:>8.2f} | {r['max_gap']:>8.2f} | {r['best_epoch']:>6} | {stopped_str:>8}")
        print(f"{'─'*70}")
        print(f"{'─'*70}")
        print(f"AGGREGATED (mean ± std):")
        print(f"  Train Acc:  {mean_train_acc:>7.2f}% ± {std_train_acc:>6.2f}%")
        print(f"  Val Acc:    {mean_val_acc:>7.2f}% ± {std_val_acc:>6.2f}%")
        print(f"  TEST Acc:   {mean_test_acc:>7.2f}% ± {std_test_acc:>6.2f}%")
        print(f"  TEST Err:   {mean_test_error:>7.2f}% ± {std_test_error:>6.2f}%")
        print(f"  Max Gap:   {mean_max_gap:>7.2f}% ± {std_max_gap:>6.2f}%")
        print(f"  Avg Gap:   {mean_avg_gap:>7.2f}% ± {std_avg_gap:>6.2f}%")
        print(f"{'─'*70}")
        print(f"Total time: {total_time/60:.1f} min")
        print(f"{'='*70}")
        
        import csv
        csv_path = os.path.join(args.output, "eval_results.csv")
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ["seed", "train_acc", "val_acc", "test_acc", "test_acc5", "test_error", "best_epoch", "stopped_early", "max_gap", "avg_gap", "params_millions", "flops_giga", "eval_time_sec"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                row = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items() if k in fieldnames}
                writer.writerow(row)
            writer.writerow({})
            writer.writerow({"seed": "MEAN", "train_acc": f"{mean_train_acc:.2f}", "val_acc": f"{mean_val_acc:.2f}", "test_acc": f"{mean_test_acc:.2f}", "test_error": f"{mean_test_error:.2f}", "max_gap": f"{mean_max_gap:.2f}", "avg_gap": f"{mean_avg_gap:.2f}"})
            writer.writerow({"seed": "STD", "train_acc": f"{std_train_acc:.2f}", "val_acc": f"{std_val_acc:.2f}", "test_acc": f"{std_test_acc:.2f}", "test_error": f"{std_test_error:.2f}", "max_gap": f"{std_max_gap:.2f}", "avg_gap": f"{std_avg_gap:.2f}"})
        print(f"\nResults saved to: {csv_path}")
    
    summary = {
        "mode": mode,
        "eval_config": eval_cfg,
        "seeds_requested": base_seeds,
        "seeds_evaluated": seeds_to_eval,
        "seeds_skipped": seeds_skipped,
        "seeds_no_genotype": seeds_no_genotype,
        "results": all_results,
        "aggregated": {
            "train_acc_mean": float(mean_train_acc),
            "train_acc_std": float(std_train_acc),
            "val_acc_mean": float(mean_val_acc),
            "val_acc_std": float(std_val_acc),
            "test_acc_mean": float(mean_test_acc),
            "test_acc_std": float(std_test_acc),
            "test_error_mean": float(mean_test_error),
            "test_error_std": float(std_test_error),
            "max_gap_mean": float(mean_max_gap),
            "max_gap_std": float(std_max_gap),
            "avg_gap_mean": float(mean_avg_gap),
            "avg_gap_std": float(std_avg_gap),
        } if all_results else None,
        "total_time_sec": total_time,
        "total_time_min": total_time / 60,
    }
    
    summary_path = os.path.join(args.output, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to: {summary_path}")

if __name__ == '__main__':
    main()
