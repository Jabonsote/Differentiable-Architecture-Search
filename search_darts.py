#!/usr/bin/env python3
"""
DARTS Search Phase - Búsqueda de Arquitecturas con Múltiples Seeds

Uso:
    python search_darts.py                          # Busca con seeds por defecto [42, 43, 44, 45, 46]
    python search_darts.py --seeds 42 43 44          # Busca solo seeds específicas
    python search_darts.py --start-seed 47 --num-seeds 3  # Busca 3 seeds desde 47

Detecta automáticamente seeds ya completadas (busca genotype.json en results/seed_X/search/)
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

SEARCH_CFG_NORMAL = {
    "batch_size": 64,
    "learning_rate": 0.025,
    "learning_rate_min": 0.001,
    "momentum": 0.9,
    "weight_decay": 3e-4,
    "arch_learning_rate": 3e-4,
    "arch_weight_decay": 1e-3,
    "epochs": 50,
    "init_channels": 16,
    "layers": 8,
    "steps": 4,
    "multiplier": 4,
    "stem_multiplier": 3,
    "train_portion": 0.5,
    "cutout": False,
    "cutout_length": 16,
    "grad_clip": 5,
    "unrolled": False,
    "num_workers": 8,
    "data": "./data",
    "report_freq": 50,
    "target_batch_size": 128,
    "use_amp": True,
    "use_preprocessed": True,
}

SEARCH_CFG_FAST = {
    "batch_size": 32,
    "learning_rate": 0.025,
    "learning_rate_min": 0.001,
    "momentum": 0.9,
    "weight_decay": 3e-4,
    "arch_learning_rate": 3e-4,
    "arch_weight_decay": 1e-3,
    "epochs": 1,
    "init_channels": 2,
    "layers": 1,
    "steps": 4,
    "multiplier": 4,
    "stem_multiplier": 3,
    "train_portion": 0.2,
    "cutout": False,
    "cutout_length": 16,
    "grad_clip": 5,
    "unrolled": False,
    "num_workers": 4,
    "data": "./data",
    "report_freq": 50,
    "target_batch_size": 64,
    "use_amp": True,
    "use_preprocessed": True,
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
# MIXED OP Y CELL
# ============================================================================

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for prim in PRIMITIVES:
            op = OPS[prim](C, stride, False)
            if 'pool' in prim:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        self._steps = steps
        self._multiplier = multiplier
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                self._ops.append(MixedOp(C, stride))
    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

# ============================================================================
# NETWORK SEARCH (SUPERNET)
# ============================================================================

class NetworkSearch(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alphas_normal = nn.Parameter(torch.randn(k, num_ops, device=DEVICE) * 1e-3)
        self.alphas_reduce = nn.Parameter(torch.randn(k, num_ops, device=DEVICE) * 1e-3)

    def arch_parameters(self):
        return [self.alphas_normal, self.alphas_reduce]

    def new(self):
        model_new = NetworkSearch(self._C, self._num_classes, self._layers, self._criterion).to(DEVICE)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        return self.classifier(out.view(out.size(0), -1))

    def _loss(self, x, target):
        return self._criterion(self(x), target)

    def genotype(self):
        def _parse(weights_np):
            gene = []
            n = 2
            start = 0
            none_idx = PRIMITIVES.index('none')
            for i in range(self._steps):
                end = start + n
                W = weights_np[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != none_idx)
                )[:2]
                for j in edges:
                    k_best = max(
                        range(len(W[j])),
                        key=lambda k: W[j][k] if k != none_idx else -1
                    )
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).detach().cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).detach().cpu().numpy())
        concat = list(range(2 + self._steps - self._multiplier, self._steps + 2))
        return Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )

    def get_alpha_weights(self):
        return {
            'normal': F.softmax(self.alphas_normal, dim=-1).detach().cpu().numpy(),
            'reduce': F.softmax(self.alphas_reduce, dim=-1).detach().cpu().numpy()
        }

# ============================================================================
# ARCHITECT (BI-LEVEL OPTIMIZATION)
# ============================================================================

class Architect:
    def __init__(self, model, args):
        self.network_momentum = args['momentum']
        self.network_weight_decay = args['weight_decay']
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.arch_parameters(),
            lr=args['arch_learning_rate'],
            betas=(0.5, 0.999),
            weight_decay=args['arch_weight_decay']
        )

    def step(self, inp_trg, tgt_trg, inp_val, tgt_val, eta, net_opt, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(inp_trg, tgt_trg, inp_val, tgt_val, eta, net_opt)
        else:
            self._backward_step(inp_val, tgt_val)
        self.optimizer.step()

    def _backward_step(self, inp_val, tgt_val):
        loss = self.model._loss(inp_val, tgt_val)
        loss.backward()

    def _backward_step_unrolled(self, inp_trg, tgt_trg, inp_val, tgt_val, eta, net_opt):
        unrolled = self._compute_unrolled_model(inp_trg, tgt_trg, eta, net_opt)
        unrolled_loss = unrolled._loss(inp_val, tgt_val)
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled.arch_parameters()]
        vector = [v.grad.data for v in unrolled.parameters()]
        implicit = self._hessian_vector_product(vector, inp_trg, tgt_trg)
        for g, ig in zip(dalpha, implicit):
            g.data.sub_(eta, ig.data)
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data.clone()
            else:
                v.grad.data.copy_(g.data)

    def _compute_unrolled_model(self, inp, tgt, eta, net_opt):
        loss = self.model._loss(inp, tgt)
        theta = torch.cat([p.data.flatten() for p in self.model.parameters()])
        try:
            moment = torch.cat([net_opt.state[v]['momentum_buffer'].flatten() for v in self.model.parameters()]).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
        dtheta = torch.cat([g.flatten() for g in grads]).data + self.network_weight_decay * theta
        return self._construct_model_from_theta(theta.sub(eta, moment + dtheta))

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()
        offset = 0
        for k, v in self.model.named_parameters():
            vl = v.numel()
            model_dict[k] = theta[offset:offset + vl].view(v.size())
            offset += vl
        model_new.load_state_dict(model_dict)
        return model_new

    def _hessian_vector_product(self, vector, inp, tgt, r=1e-2):
        R = r / torch.cat([v.flatten() for v in vector]).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        grads_p = torch.autograd.grad(self.model._loss(inp, tgt), self.model.arch_parameters())
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        grads_n = torch.autograd.grad(self.model._loss(inp, tgt), self.model.arch_parameters())
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

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

def build_search_dataloaders(cfg):
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
        
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(cfg["train_portion"] * num_train))
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
        valid_queue = DataLoader(
            train_dataset, batch_size=cfg["batch_size"],
            sampler=SubsetRandomSampler(indices[split:]),
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
        train_data = dset.CIFAR10(root=cfg["data"], train=True, download=True, transform=train_tf)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(cfg["train_portion"] * num_train))
        rng = np.random.RandomState(cfg["seed"])
        rng.shuffle(indices)
        train_queue = DataLoader(train_data, batch_size=cfg["batch_size"], sampler=SubsetRandomSampler(indices[:split]),
                                num_workers=cfg["num_workers"], pin_memory=True)
        valid_queue = DataLoader(train_data, batch_size=cfg["batch_size"], sampler=SubsetRandomSampler(indices[split:]),
                                num_workers=cfg["num_workers"], pin_memory=True)
    
    return train_queue, valid_queue, accumulation_steps

# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_genotype_graph(genotype_edges, filename, title):
    plt.figure(figsize=(12, 8))
    steps = len(genotype_edges) // 2
    node_x = {}
    node_y = {}
    node_x[0] = 0
    node_y[0] = 0
    node_x[1] = 0
    node_y[1] = 0.5
    for i in range(steps):
        node_x[i + 2] = 1
        node_y[i + 2] = (i + 0.5) / steps
    node_x[steps + 2] = 2
    node_y[steps + 2] = 0.5
    plt.xlim(-0.3, 2.3)
    plt.ylim(-0.1, 1.1)
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold')
    node_colors = {0: 'lightgreen', 1: 'lightblue', steps + 2: 'gold'}
    for i in range(steps):
        node_colors[i + 2] = 'lightyellow'
    for node, x in node_x.items():
        y = node_y.get(node, 0.5)
        if node == 0:
            label = f"s0"
        elif node == 1:
            label = f"s1"
        elif node == steps + 2:
            label = f"out"
        else:
            label = f"n{node-1}"
        plt.scatter(x, y, s=400, c=node_colors.get(node, 'white'), edgecolors='black', zorder=5)
        plt.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
    for i in range(steps):
        for k in range(2):
            op, j = genotype_edges[2 * i + k]
            x1, y1 = node_x[j], node_y[j]
            x2, y2 = node_x[i + 2], node_y[i + 2]
            plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            plt.text(mid_x, mid_y, op, fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_alpha_evolution(alpha_history, output_dir):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("Alpha Normal - Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Alpha Weight")
    for i in range(len(alpha_history[0]['normal'])):
        values = [h['normal'][i] for h in alpha_history]
        plt.plot(values, label=f'Edge {i}')
    plt.legend(fontsize=6, ncol=3)
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.title("Alpha Reduce - Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Alpha Weight")
    for i in range(len(alpha_history[0]['reduce'])):
        values = [h['reduce'][i] for h in alpha_history]
        plt.plot(values, label=f'Edge {i}')
    plt.legend(fontsize=6, ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_evolution.png'), dpi=150)
    plt.close()

def plot_training_curves(history, output_dir, phase="search"):
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
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{phase.capitalize()} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()

# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def train_search_epoch(train_q, valid_q, model, architect, criterion, optimizer, lr, cfg, scaler, accumulation_steps, report_freq=50):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    valid_iter = iter(valid_q)
    
    for step, (inp, tgt) in enumerate(train_q):
        n = inp.size(0)
        inp = inp.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)
        try:
            inp_search, tgt_search = next(valid_iter)
        except StopIteration:
            valid_iter = iter(valid_q)
            inp_search, tgt_search = next(valid_iter)
        inp_search = inp_search.to(DEVICE, non_blocking=True)
        tgt_search = tgt_search.to(DEVICE, non_blocking=True)
        
        architect.step(inp, tgt, inp_search, tgt_search, lr, optimizer, cfg["unrolled"])
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP and cfg.get("use_amp", True)):
            logits = model(inp)
            loss = criterion(logits, tgt)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        
        prec1, prec5 = accuracy(logits, tgt, topk=(1, 5))
        objs.update(loss.item() * accumulation_steps, n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        if step % report_freq == 0 and step > 0:
            print(f"    Step {step:3d} | loss={objs.avg:.4f} | acc@1={top1.avg:.2f}% | acc@5={top5.avg:.2f}%")
    return {"loss": objs.avg, "top1": top1.avg, "top5": top5.avg}

def validate_search(valid_q, model, criterion, subset_batches=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    
    with torch.no_grad():
        for step, (inp, tgt) in enumerate(valid_q):
            if subset_batches is not None and step >= subset_batches:
                break
            inp = inp.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)
            
            with torch.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP
            ):
                logits = model(inp)
                loss = criterion(logits, tgt)
            
            prec1, prec5 = accuracy(logits, tgt, topk=(1, 5))
            n = inp.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
    return {"loss": objs.avg, "top1": top1.avg, "top5": top5.avg}

# ============================================================================
# RUN SEARCH
# ============================================================================

def genotype_to_dict(genotype):
    return {
        "normal": [(op, int(idx)) for op, idx in genotype.normal],
        "normal_concat": [int(x) for x in genotype.normal_concat],
        "reduce": [(op, int(idx)) for op, idx in genotype.reduce],
        "reduce_concat": [int(x) for x in genotype.reduce_concat],
    }

def run_search(cfg, output_dir):
    set_reproducibility(cfg["seed"])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config_search.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    
    train_queue, valid_queue, accumulation_steps = build_search_dataloaders(cfg)
    
    criterion = nn.CrossEntropyLoss()
    model = NetworkSearch(C=cfg["init_channels"], num_classes=10, layers=cfg["layers"],
                         criterion=criterion, steps=cfg["steps"], multiplier=cfg["multiplier"],
                         stem_multiplier=cfg["stem_multiplier"]).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"],
                              momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"],
                                                         eta_min=cfg["learning_rate_min"])
    architect = Architect(model, cfg)
    
    scaler = GradScaler(enabled=USE_AMP and cfg.get("use_amp", True))
    
    history = {"train_loss": [], "train_acc": [], "train_acc5": [], "val_loss": [], "val_acc": [], "val_acc5": []}
    alpha_history = []
    total_epochs = cfg["epochs"]
    t0 = time.time()
    mode_name = 'FAST' if total_epochs < 10 else 'NORMAL'
    
    print(f"\n{'='*70}")
    print(f"SEARCH PHASE - Seed {cfg['seed']} - Mode: {mode_name}")
    print(f"{'='*70}")
    print(f"Config: epochs={total_epochs}, channels={cfg['init_channels']}, layers={cfg['layers']}")
    print(f"Batch: {cfg['batch_size']} -> effective {cfg.get('target_batch_size', cfg['batch_size'])}")
    print(f"{'='*70}")
    
    for epoch in range(total_epochs):
        lr = scheduler.get_last_lr()[0]
        remaining_epochs = total_epochs - epoch - 1
        start_epoch_time = time.time()
        
        progress = (epoch + 1) / total_epochs * 100
        
        print(f"\n[Epoch {epoch+1:3d}/{total_epochs}] ({progress:5.1f}%) | LR={lr:.6f} | ETA: {remaining_epochs} epochs")
        
        report_freq = cfg.get("report_freq", 50)
        train_metrics = train_search_epoch(train_queue, valid_queue, model, architect, criterion, optimizer, lr, cfg, scaler, accumulation_steps, report_freq)
        valid_metrics = validate_search(valid_queue, model, criterion, subset_batches=100)
        scheduler.step()
        
        epoch_time = time.time() - start_epoch_time
        elapsed = time.time() - t0
        eta_remaining = epoch_time * remaining_epochs
        
        print(f"  Train | loss={train_metrics['loss']:.4f} | acc@1={train_metrics['top1']:.2f}%")
        print(f"  Valid | loss={valid_metrics['loss']:.4f} | acc@1={valid_metrics['top1']:.2f}%")
        print(f"  Time: {epoch_time:.1f}s | Elapsed: {elapsed:.1f}s | ETA: {eta_remaining:.1f}s")
        
        history["train_loss"].append(float(train_metrics['loss']))
        history["train_acc"].append(float(train_metrics['top1']))
        history["train_acc5"].append(float(train_metrics['top5']))
        history["val_loss"].append(float(valid_metrics['loss']))
        history["val_acc"].append(float(valid_metrics['top1']))
        history["val_acc5"].append(float(valid_metrics['top5']))
        
        alpha_weights = model.get_alpha_weights()
        alpha_max_normal = float(np.max(alpha_weights['normal']))
        alpha_max_reduce = float(np.max(alpha_weights['reduce']))
        alpha_entropy_normal = float(-np.sum(alpha_weights['normal'] * np.log(alpha_weights['normal'] + 1e-10)))
        alpha_history.append({
            'normal': alpha_weights['normal'].tolist(),
            'reduce': alpha_weights['reduce'].tolist(),
            'max_normal': alpha_max_normal,
            'max_reduce': alpha_max_reduce,
            'entropy_normal': alpha_entropy_normal
        })
    
    search_time = time.time() - t0
    genotype = model.genotype()
    
    with open(os.path.join(output_dir, "genotype.json"), "w") as f:
        json.dump(genotype_to_dict(genotype), f, indent=2)
    with open(os.path.join(output_dir, "genotype.txt"), "w") as f:
        f.write(str(genotype))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, "alpha_history.json"), "w") as f:
        json.dump(alpha_history, f, indent=2)
    np.save(os.path.join(output_dir, "alpha_normal_final.npy"), model.get_alpha_weights()['normal'])
    np.save(os.path.join(output_dir, "alpha_reduce_final.npy"), model.get_alpha_weights()['reduce'])
    
    plot_alpha_evolution(alpha_history, output_dir)
    plot_training_curves(history, output_dir, "search")
    plot_genotype_graph(genotype.normal, os.path.join(output_dir, "genotype_normal.png"), "Normal Cell")
    plot_genotype_graph(genotype.reduce, os.path.join(output_dir, "genotype_reduce.png"), "Reduction Cell")
    
    print(f"\nGenotype:")
    print(f"  Normal: {genotype.normal}")
    print(f"  Reduce: {genotype.reduce}")
    print(f"Search Time: {search_time:.2f}s")
    
    return genotype, history, search_time

# ============================================================================
# HELPERS PARA DETECTAR SEMILLAS COMPLETADAS
# ============================================================================

def is_search_completed(seed_dir):
    """Check if search is already completed for this seed"""
    genotype_file = os.path.join(seed_dir, "search", "genotype.json")
    return os.path.exists(genotype_file)

def get_seeds_to_run(requested_seeds, output_dir):
    """Get list of seeds that need to be run (skip completed ones)"""
    seeds_to_run = []
    seeds_skipped = []
    
    for seed in requested_seeds:
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        if is_search_completed(seed_dir):
            seeds_skipped.append(seed)
            print(f"  SKIP seed {seed}: search already completed (genotype.json exists)")
        else:
            seeds_to_run.append(seed)
    
    return seeds_to_run, seeds_skipped

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DARTS Search Phase - Multiple Seeds')
    parser.add_argument('--fast', action='store_true', help='Use fast mode (reduced epochs)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='Specific seeds to run')
    parser.add_argument('--start-seed', type=int, default=42, help='Starting seed (default: 42)')
    parser.add_argument('--num-seeds', type=int, default=5, help='Number of seeds to run')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    if args.seeds:
        base_seeds = args.seeds
    else:
        base_seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    
    search_cfg = copy.deepcopy(SEARCH_CFG_FAST if args.fast else SEARCH_CFG_NORMAL)
    mode = "FAST" if args.fast else "NORMAL"
    
    print(f"\n{'='*70}")
    print(f"DARTS SEARCH PHASE - {mode} MODE")
    print(f"{'='*70}")
    print(f"Seeds requested: {base_seeds}")
    print(f"Output dir: {args.output}/")
    print(f"Search Config: {search_cfg['epochs']} epochs, {search_cfg['init_channels']} channels")
    print(f"{'='*70}")
    
    seeds_to_run, seeds_skipped = get_seeds_to_run(base_seeds, args.output)
    
    print(f"\n{'='*70}")
    print(f"SEEDS TO RUN: {seeds_to_run}")
    print(f"SEEDS SKIPPED (already completed): {seeds_skipped}")
    print(f"{'='*70}")
    
    if not seeds_to_run:
        print("\nAll seeds already completed! Nothing to do.")
        return
    
    all_genotypes = []
    total_t0 = time.time()
    
    for i, seed in enumerate(seeds_to_run):
        print(f"\n{'#'*70}")
        print(f"# SEARCH {seed} ({i+1}/{len(seeds_to_run)})")
        print(f"# Seeds remaining: {len(seeds_to_run) - i - 1}")
        print(f"{'#'*70}")
        torch.cuda.empty_cache()
        
        search_cfg['seed'] = seed
        seed_dir = os.path.join(args.output, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        search_dir = os.path.join(seed_dir, "search")
        
        genotype, search_history, search_time = run_search(search_cfg, search_dir)
        
        all_genotypes.append({
            "seed": seed,
            "genotype": genotype_to_dict(genotype),
            "normal": genotype.normal,
            "reduce": genotype.reduce
        })
        
        print(f"\n[COMPLETED] Seed {seed} | Time: {search_time:.1f}s | Best Val Acc: {max(search_history['val_acc']):.2f}%")
        
        torch.cuda.empty_cache()
    
    total_time = time.time() - total_t0
    
    summary = {
        "mode": mode,
        "search_config": search_cfg,
        "seeds_requested": base_seeds,
        "seeds_completed": seeds_to_run,
        "seeds_skipped": seeds_skipped,
        "genotypes": all_genotypes,
        "total_time_sec": total_time,
        "total_time_min": total_time / 60,
    }
    
    with open(os.path.join(args.output, "search_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("SEARCH SUMMARY")
    print(f"{'='*70}")
    print(f"Seeds completed: {seeds_to_run}")
    print(f"Seeds skipped: {seeds_skipped}")
    print(f"Genotypes saved to: {args.output}/seed_X/search/genotype.json")
    print(f"Summary saved to: {args.output}/search_summary.json")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
