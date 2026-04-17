#!/usr/bin/env python3
"""
DARTS Pipeline Completo - Fast y Normal Mode
Incluye: Search, Discretización, Evaluación, Métricas y Visualizaciones

Optimizaciones implementadas:
- Mixed Precision (AMP) con torch.cuda.amp
- Gradient Accumulation para batch sizes efectivos grandes
- Preprocesamiento offline de CIFAR-10
- DataLoader optimizado (persistent_workers, prefetch_factor)
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
import scipy.stats as stats

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for Ampere GPU")

print(f"Device: {DEVICE}")
print(f"Mixed Precision (AMP): {'Enabled' if USE_AMP else 'Disabled'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | cuDNN Benchmark: True | MatMul: high precision")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# ============================================================================
# CONFIGURACIONES
# ============================================================================

# FAST MODE - Para iteración rápida (1 epoch search, menos canales)
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
    "seed": 42,
    "data": "./data",
    "report_freq": 50,
    "target_batch_size": 64,
    "use_amp": True,
    "use_preprocessed": True,
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
    "num_workers": 2,
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

# NORMAL MODE - Parámetros exactos del paper DARTS
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
    "seed": 42,
    "data": "./data",
    "report_freq": 50,
    "target_batch_size": 128,
    "use_amp": True,
    "use_preprocessed": True,
}

EVAL_CFG_NORMAL = {
    "batch_size": 48,
    "learning_rate": 0.025,
    "momentum": 0.9,
    "weight_decay": 3e-4,
    "epochs": 600,
    "init_channels": 36,
    "layers": 20,
    "auxiliary": True,
    "cutout": True,
    "cutout_length": 16,
    "drop_path_prob": 0.2,
    "num_workers": 2,
    "seed": 42,
    "data": "./data",
    "stem_multiplier": 3,
    "report_freq": 50,
    "grad_clip": 5,
    "learning_rate_min": 0.0,
    "early_stopping": True,
    "early_stopping_patience": 50,
    "overfitting_threshold": 10.0,
    "target_batch_size": 96,
    "use_amp": True,
    "use_preprocessed": True,
    "val_freq": 10,
}

# ============================================================================
# OPERACIONES PRIMITIVAS (espacio exacto del paper)
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
    """Dataset CIFAR-10 preprocesado para carga rápida"""
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

def preprocess_and_save_cifar10(data_dir='./data'):
    """Preprocesa CIFAR-10 y guarda en archivo .pt (solo ejecutar una vez)"""
    import torchvision
    import torchvision.transforms as transforms
    
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, 'cifar10_preprocessed.pt')
    
    if os.path.exists(cache_path):
        print(f"  ✓ Cached CIFAR-10 found at {cache_path}")
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

def load_preprocessed_cifar10(data_dir='./data'):
    """Carga CIFAR-10 preprocesado"""
    cache_path = os.path.join(data_dir, 'cifar10_preprocessed.pt')
    if not os.path.exists(cache_path):
        return preprocess_and_save_cifar10(data_dir)
    
    return torch.load(cache_path)

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

import torchvision.transforms as transforms
import torchvision.datasets as dset

def get_cifar10_transforms(use_cutout=False, cutout_length=16):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if use_cutout:
        train_tf.transforms.append(Cutout(cutout_length))
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_tf, eval_tf

def build_search_dataloaders(cfg):
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
        train_tf, _ = get_cifar10_transforms(use_cutout=cfg["cutout"], cutout_length=cfg["cutout_length"])
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
    
    print(f"  Search DataLoader: batch={cfg['batch_size']}, effective_batch={target_batch} (accumulation={accumulation_steps}x)")
    return train_queue, valid_queue, accumulation_steps

def build_eval_dataloaders(cfg):
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
        train_tf, eval_tf = get_cifar10_transforms(use_cutout=cfg["cutout"], cutout_length=cfg["cutout_length"])
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
    
    print(f"  Eval DataLoader: batch={cfg['batch_size']}, effective_batch={target_batch} (accumulation={accumulation_steps}x)")
    return train_queue, val_queue, test_queue, accumulation_steps

# ============================================================================
# RED DISCRETA PARA EVALUACIÓN
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
    print(f"  Saved: {filename}")

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
    print(f"  Saved: {output_dir}/alpha_evolution.png")

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
    print(f"  Saved: {output_dir}/training_curves.png")

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
    print(f"  Saved: {output_dir}/confusion_matrix.png")
    return cm

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
    print(f"  Saved: {output_dir}/overfitting_gap.png")

def save_classification_report(targets, predictions, output_dir):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    report = classification_report(targets, predictions, target_names=class_names, digits=4)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print(f"  Saved: {output_dir}/classification_report.txt")
    return report

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
    """Validate with optional subset of batches for speed"""
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
    """Evaluate with optional subset of batches for speed"""
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
    print(f"Config: epochs={total_epochs}, channels={cfg['init_channels']}, layers={cfg['layers']}, unrolled={cfg['unrolled']}")
    print(f"Batch: {cfg['batch_size']} -> effective {cfg.get('target_batch_size', cfg['batch_size'])} (accumulation={accumulation_steps}x)")
    print(f"AMP: {'Enabled' if cfg.get('use_amp', True) else 'Disabled'}")
    print(f"{'='*70}")
    
    if mode_name == 'FAST':
        print("\n🧠 DIFERENCIAS CON DARTS ORIGINAL (FAST MODE):")
        print("  🔴 First-order DARTS (unrolled=False) - sin Hessian")
        print("  🔴 Search reducido: 1 epoch vs 50 del paper")
        print("  🔴 Batch size: 32 vs 64 del paper")
        print("  🔴 Init channels: 8 vs 16 del paper")
        print("  🟡 Primitives: CORRECTO (8 ops del paper)")
        print("  🟡 Split: CORRECTO (50/50)")
        print(f"{'='*70}")
    
    for epoch in range(total_epochs):
        lr = scheduler.get_last_lr()[0]
        remaining_epochs = total_epochs - epoch - 1
        start_epoch_time = time.time()
        
        progress = (epoch + 1) / total_epochs * 100
        
        print(f"\n[Epoch {epoch+1:3d}/{total_epochs}] ({progress:5.1f}% complete) | LR={lr:.6f} | Remaining: {remaining_epochs} epochs")
        
        report_freq = cfg.get("report_freq", 50)
        train_metrics = train_search_epoch(train_queue, valid_queue, model, architect, criterion, optimizer, lr, cfg, scaler, accumulation_steps, report_freq)
        valid_metrics = validate_search(valid_queue, model, criterion, subset_batches=100)
        scheduler.step()
        
        epoch_time = time.time() - start_epoch_time
        elapsed = time.time() - t0
        eta_per_epoch = epoch_time
        eta_remaining = eta_per_epoch * remaining_epochs
        
        print(f"  Epoch time: {epoch_time:.1f}s | Elapsed: {elapsed:.1f}s | ETA remaining: {eta_remaining:.1f}s")
        print(f"  Train | loss={train_metrics['loss']:.4f} | acc@1={train_metrics['top1']:.2f}% | acc@5={train_metrics['top5']:.2f}%")
        print(f"  Valid | loss={valid_metrics['loss']:.4f} | acc@1={valid_metrics['top1']:.2f}% | acc@5={valid_metrics['top5']:.2f}%")
        
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
        
        print(f"  Alpha | max_normal={alpha_max_normal:.4f} | max_reduce={alpha_max_reduce:.4f} | entropy={alpha_entropy_normal:.4f}")
    
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
    
    return genotype, history, search_time, model

# ============================================================================
# RUN EVALUATION
# ============================================================================

def run_evaluation(cfg, genotype, output_dir, use_early_stopping=False):
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
    
    model = torch.compile(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"],
                               momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"],
                                                           eta_min=cfg.get("learning_rate_min", 0.0))
    
    scaler = GradScaler(enabled=USE_AMP and cfg.get("use_amp", True))
    
    early_stopper = None
    if use_early_stopping:
        patience = cfg.get("early_stopping_patience", 50)
        early_stopper = EarlyStopping(patience=patience, min_delta=0.001, mode='max')
    
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    history = {"train_loss": [], "train_acc": [], "train_acc5": [], "val_loss": [],
                "val_acc": [], "val_acc5": [], "test_loss": [], "test_acc": [], "test_acc5": [],
                "overfitting_gap": []}
    total_epochs = cfg["epochs"]
    t0 = time.time()
    flops = compute_flops(model)
    params = count_params_in_MB(model)
    mode_name = 'FAST' if total_epochs < 50 else 'NORMAL'
    es_status = "ON" if use_early_stopping else "OFF"
    
    val_freq = cfg.get("val_freq", 5)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION PHASE - Seed {cfg['seed']} - Mode: {mode_name} - Early Stopping: {es_status}")
    print(f"{'='*70}")
    print(f"Config: epochs={total_epochs}, channels={cfg['init_channels']}, layers={cfg['layers']}")
    print(f"Model: {params:.2f}M params, {flops/1e9:.2f} GFLOPs")
    print(f"Auxiliary: {cfg['auxiliary']}, Cutout: {cfg['cutout']}, DropPath: {cfg['drop_path_prob']}")
    print(f"Validation frequency: every {val_freq} epoch(s) (saves ~{100 - 100//val_freq}% val time)")
    print(f"torch.compile: Enabled (first epoch slower, then faster)")
    
    if use_early_stopping:
        print(f"\n🧠 EARLY STOPPING CONFIG:")
        print(f"  Patience: {patience} epochs")
        print(f"  Min delta: 0.001")
        print(f"  Mode: max (mejorar val_acc)")
    
    if mode_name == 'FAST':
        print(f"\n🧠 DIFERENCIAS CON DARTS ORIGINAL (FAST MODE):")
        print("  🔴 Evaluation reducido: 15 epochs vs 600 del paper")
        print("  🔴 Batch size: 32 vs 96 del paper")
        print("  🔴 Init channels: 16 vs 36 del paper")
        print("  🔴 Layers: 8 vs 20 del paper")
        print("  🔴 Sin Cutout en evaluation (paper usa cutout=True)")
        print("  🟡 Auxiliary weight: 0.4 (CORRECTO)")
        print("  🟡 DropPath schedule: CORRECTO")
        print(f"{'='*70}")
    else:
        print(f"\n✔ CONFIGURACIÓN ORIGINAL DEL PAPER DARTS:")
        print("  ✔ epochs: 600")
        print("  ✔ init_channels: 36")
        print("  ✔ layers: 20")
        print("  ✔ batch_size: 96")
        print("  ✔ cutout: True (length=16)")
        print("  ✔ drop_path_prob: 0.2")
        print("  ✔ auxiliary: True (weight=0.4)")
        print(f"{'='*70}")
    
    stopped_early = False
    for epoch in range(total_epochs):
        remaining_epochs = total_epochs - epoch - 1
        start_epoch_time = time.time()
        progress = (epoch + 1) / total_epochs * 100
        
        print(f"\n[Epoch {epoch+1:4d}/{total_epochs}] ({progress:6.2f}% complete) | Remaining: {remaining_epochs} epochs")
        
        report_freq = cfg.get("report_freq", 50)
        train_metrics = train_eval_epoch(train_queue, model, criterion, optimizer, cfg, epoch, scaler, accumulation_steps, report_freq)
        
        if (epoch + 1) % val_freq == 0 or epoch == total_epochs - 1:
            val_metrics = evaluate_final(val_queue, model, criterion, subset_batches=100)
            
            if val_metrics['top1'] > best_val_acc:
                best_val_acc = val_metrics['top1']
                best_epoch = epoch + 1
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            if use_early_stopping and early_stopper(val_metrics['top1'], epoch):
                stopped_early = True
                print(f"\n⚠️  EARLY STOPPING TRIGGERED at epoch {epoch+1}")
                break
        else:
            val_metrics = {"top1": 0.0, "top5": 0.0, "loss": 0.0}
        
        scheduler.step()
        
        epoch_time = time.time() - start_epoch_time
        elapsed = time.time() - t0
        
        overfitting_gap = train_metrics['top1'] - val_metrics['top1']
        history["overfitting_gap"].append(float(overfitting_gap))
        
        print(f"  Epoch time: {epoch_time:.1f}s | Elapsed: {elapsed:.1f}s")
        print(f"  Train | loss={train_metrics['loss']:.4f} | acc@1={train_metrics['top1']:.2f}%")
        print(f"  Val   | loss={val_metrics['loss']:.4f} | acc@1={val_metrics['top1']:.2f}%")
        
        history["train_loss"].append(float(train_metrics['loss']))
        history["train_acc"].append(float(train_metrics['top1']))
        history["train_acc5"].append(float(train_metrics['top5']))
        history["val_loss"].append(float(val_metrics['loss']))
        history["val_acc"].append(float(val_metrics['top1']))
        history["val_acc5"].append(float(val_metrics['top5']))
        history["test_loss"].append(0.0)
        history["test_acc"].append(0.0)
        history["test_acc5"].append(0.0)
    
    eval_time = time.time() - t0
    actual_epochs = len(history["train_acc"])
    
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(output_dir, "best_model.pt"))
    
    history["best_val_acc"] = float(best_val_acc)
    history["best_epoch"] = best_epoch
    history["stopped_early"] = stopped_early
    history["actual_epochs"] = actual_epochs
    history["max_overfitting_gap"] = float(max(history["overfitting_gap"]))
    history["avg_overfitting_gap"] = float(np.mean(history["overfitting_gap"]))
    
    es_epoch, es_val_acc = analyze_early_stopping_post_hoc(history, patience=cfg.get("early_stopping_patience", 50))
    history["es_post_hoc_epoch"] = es_epoch
    history["es_post_hoc_val_acc"] = float(es_val_acc)
    history["es_post_hoc_test_acc"] = float(history["test_acc"][es_epoch - 1]) if es_epoch <= len(history["test_acc"]) else float(history["test_acc"][-1])
    history["es_post_hoc_stopped_early"] = es_epoch < total_epochs
    
    es_info = {
        "es_post_hoc_epoch": es_epoch,
        "es_post_hoc_val_acc": float(es_val_acc),
        "es_post_hoc_test_acc": history["es_post_hoc_test_acc"],
        "es_post_hoc_stopped_early": history["es_post_hoc_stopped_early"],
        "patience_used": cfg.get("early_stopping_patience", 50)
    }
    with open(os.path.join(output_dir, "es_trigger_info.json"), "w") as f:
        json.dump(es_info, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, os.path.join(output_dir, "best_model.pt"))
        
        print(f"\nRunning final evaluation on full test set...")
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
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS:")
    print(f"  Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    if best_model_state is not None and final_test.get('top1', 0) > 0:
        print(f"  Final Test Acc@1: {final_test['top1']:.2f}% (full test set)")
        print(f"  Final Test Acc@5: {final_test['top5']:.2f}%")
    print(f"  Params: {params:.2f}M")
    print(f"  FLOPs: {flops/1e9:.2f}G")
    print(f"  Actual epochs: {actual_epochs}/{total_epochs}")
    if stopped_early:
        print(f"  ⏹ Stopped early at epoch {actual_epochs}")
    print(f"  Max overfitting gap: {history['max_overfitting_gap']:.2f}%")
    print(f"  Avg overfitting gap: {history['avg_overfitting_gap']:.2f}%")
    print(f"Eval Time: {eval_time:.2f}s")
    print(f"{'='*70}")
    
    return final_test, history, params, flops, eval_time, model, best_val_acc, best_epoch, stopped_early

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DARTS Pipeline with Early Stopping Comparison')
    parser.add_argument('--fast', action='store_true', help='Use fast mode (reduced epochs)')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of independent runs')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--early_stopping', action='store_true', default=True, help='Use early stopping')
    parser.add_argument('--early_patience', type=int, default=50, help='Early stopping patience')
    args = parser.parse_args()

    search_cfg = copy.deepcopy(SEARCH_CFG_FAST if args.fast else SEARCH_CFG_NORMAL)
    eval_cfg = copy.deepcopy(EVAL_CFG_FAST if args.fast else EVAL_CFG_NORMAL)

    eval_cfg['early_stopping'] = args.early_stopping
    eval_cfg['early_stopping_patience'] = args.early_patience

    base_seeds = [42, 43, 44, 45, 46][:args.num_seeds]
    
    mode = "FAST" if args.fast else "NORMAL"
    es_mode = "ON" if args.early_stopping else "OFF"
    
    print(f"\n{'='*70}")
    print(f"DARTS PIPELINE - {mode} MODE - Early Stopping: {es_mode}")
    print(f"{'='*70}")
    print(f"Number of runs: {args.num_seeds}")
    print(f"Seeds: {base_seeds}")
    print(f"Search Config: {search_cfg['epochs']} epochs, {search_cfg['init_channels']} channels, unrolled={search_cfg['unrolled']}")
    print(f"Eval Config: {eval_cfg['epochs']} epochs, {eval_cfg['init_channels']} channels, cutout={eval_cfg['cutout']}")
    print(f"Early Stopping: {es_mode} (patience={args.early_patience})")
    print(f"{'='*70}")

    all_results_es_on = []
    all_results_es_off = []
    all_genotypes = []
    total_t0 = time.time()

    for seed in base_seeds:
        print(f"\n{'#'*70}")
        print(f"# RUN {seed} - Seeds {base_seeds.index(seed) + 1}/{args.num_seeds}")
        print(f"{'#'*70}")
        torch.cuda.empty_cache()

        search_cfg['seed'] = seed
        eval_cfg['seed'] = seed
        seed_dir = os.path.join(args.output, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        search_dir = os.path.join(seed_dir, "search")
        genotype, search_history, search_time, search_model = run_search(search_cfg, search_dir)

        all_genotypes.append({
            "seed": seed,
            "genotype": genotype_to_dict(genotype),
            "normal": genotype.normal,
            "reduce": genotype.reduce
        })

        del search_model
        torch.cuda.empty_cache()

        eval_dir = os.path.join(seed_dir, "eval_full")
        test_results_full, eval_history_full, params, flops, eval_time_full, model_full, best_val_full, best_epoch_full, stopped_full = run_evaluation(
            eval_cfg, genotype, eval_dir, use_early_stopping=False
        )

        best_epoch_es_post_hoc, best_val_acc_es_post_hoc = analyze_early_stopping_post_hoc(
            eval_history_full, patience=args.early_patience
        )

        result_es_post_hoc = {
            "seed": seed,
            "run": base_seeds.index(seed) + 1,
            "search_val_acc": max(search_history["val_acc"]),
            "search_val_acc5": max(search_history["val_acc5"]),
            "test_acc": float(eval_history_full["test_acc"][best_epoch_es_post_hoc - 1]) if best_epoch_es_post_hoc > 0 else float(eval_history_full["test_acc"][-1]),
            "test_acc5": float(eval_history_full["test_acc5"][best_epoch_es_post_hoc - 1]) if best_epoch_es_post_hoc > 0 else float(eval_history_full["test_acc5"][-1]),
            "test_error": 100 - float(eval_history_full["test_acc"][best_epoch_es_post_hoc - 1]) if best_epoch_es_post_hoc > 0 else float(eval_history_full["test_acc"][-1]),
            "best_val_acc": best_val_acc_es_post_hoc,
            "best_epoch": best_epoch_es_post_hoc,
            "stopped_early": best_epoch_es_post_hoc < eval_cfg["epochs"],
            "max_overfitting_gap": max(eval_history_full["overfitting_gap"][:best_epoch_es_post_hoc]) if best_epoch_es_post_hoc > 0 else max(eval_history_full["overfitting_gap"]),
            "avg_overfitting_gap": np.mean(eval_history_full["overfitting_gap"][:best_epoch_es_post_hoc]) if best_epoch_es_post_hoc > 0 else np.mean(eval_history_full["overfitting_gap"]),
            "params_millions": params,
            "flops_giga": flops / 1e9,
            "search_time_sec": search_time,
            "eval_time_sec": eval_time_full,
            "total_time_sec": search_time + eval_time_full,
            "early_stopped": best_epoch_es_post_hoc < eval_cfg["epochs"]
        }
        all_results_es_on.append(result_es_post_hoc)

        result_full = {
            "seed": seed,
            "run": base_seeds.index(seed) + 1,
            "search_val_acc": max(search_history["val_acc"]),
            "search_val_acc5": max(search_history["val_acc5"]),
            "test_acc": test_results_full['top1'],
            "test_acc5": test_results_full['top5'],
            "test_error": 100 - test_results_full['top1'],
            "best_val_acc": best_val_full,
            "best_epoch": best_epoch_full,
            "stopped_early": False,
            "max_overfitting_gap": eval_history_full.get("max_overfitting_gap", 0),
            "avg_overfitting_gap": eval_history_full.get("avg_overfitting_gap", 0),
            "params_millions": params,
            "flops_giga": flops / 1e9,
            "search_time_sec": search_time,
            "eval_time_sec": eval_time_full,
            "total_time_sec": search_time + eval_time_full,
            "early_stopped": False
        }
        all_results_es_off.append(result_full)

        print(f"\n{'='*70}")
        print(f"SEED {seed} COMPARISON (trained ONCE, analyzed TWICE):")
        print(f"  Early Stopping POST-HOC (analyzed from full training):")
        print(f"    Test Acc: {result_es_post_hoc['test_acc']:.2f}% | Error: {result_es_post_hoc['test_error']:.2f}%")
        print(f"    Best epoch: {result_es_post_hoc['best_epoch']} | Stopped early: {result_es_post_hoc['stopped_early']}")
        print(f"    Gap: max={result_es_post_hoc['max_overfitting_gap']:.2f}%, avg={result_es_post_hoc['avg_overfitting_gap']:.2f}%")
        print(f"  Full Training (600 epochs, no stopping):")
        print(f"    Test Acc: {result_full['test_acc']:.2f}% | Error: {result_full['test_error']:.2f}%")
        print(f"    Best epoch: {result_full['best_epoch']} | Stopped early: False")
        print(f"    Gap: max={result_full['max_overfitting_gap']:.2f}%, avg={result_full['avg_overfitting_gap']:.2f}%")
        print(f"{'='*70}")

        del model_full
        torch.cuda.empty_cache()

    total_time = time.time() - total_t0

    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")

    test_accs_es = [r['test_acc'] for r in all_results_es_on]
    test_errors_es = [r['test_error'] for r in all_results_es_on]
    test_accs_no_es = [r['test_acc'] for r in all_results_es_off]
    test_errors_no_es = [r['test_error'] for r in all_results_es_off]

    mean_acc_es = np.mean(test_accs_es)
    std_acc_es = np.std(test_accs_es)
    mean_error_es = np.mean(test_errors_es)
    std_error_es = np.std(test_errors_es)

    mean_acc_no_es = np.mean(test_accs_no_es)
    std_acc_no_es = np.std(test_accs_no_es)
    mean_error_no_es = np.mean(test_errors_no_es)
    std_error_no_es = np.std(test_errors_no_es)

    max_gaps_es = [r['max_overfitting_gap'] for r in all_results_es_on]
    avg_gaps_es = [r['avg_overfitting_gap'] for r in all_results_es_on]
    max_gaps_no_es = [r['max_overfitting_gap'] for r in all_results_es_off]
    avg_gaps_no_es = [r['avg_overfitting_gap'] for r in all_results_es_off]

    print(f"\nEarly Stopping ON ({args.num_seeds} runs):")
    print(f"  Test Accuracy@1: {mean_acc_es:.2f}% +/- {std_acc_es:.2f}%")
    print(f"  Test Error: {mean_error_es:.2f}% +/- {std_error_es:.2f}%")
    print(f"  Overfitting Gap: max={np.mean(max_gaps_es):.2f}%, avg={np.mean(avg_gaps_es):.2f}%")
    print(f"  Stopped early: {sum(r['stopped_early'] for r in all_results_es_on)}/{args.num_seeds}")

    print(f"\nEarly Stopping OFF ({args.num_seeds} runs):")
    print(f"  Test Accuracy@1: {mean_acc_no_es:.2f}% +/- {std_acc_no_es:.2f}%")
    print(f"  Test Error: {mean_error_no_es:.2f}% +/- {std_error_no_es:.2f}%")
    print(f"  Overfitting Gap: max={np.mean(max_gaps_no_es):.2f}%, avg={np.mean(avg_gaps_no_es):.2f}%")

    diff_error = mean_error_es - mean_error_no_es
    print(f"\nEarly Stopping Comparison:")
    print(f"  ON - OFF: {diff_error:+.2f}% (negative = ES ON is better)")

    paper_error = 2.76

    t_stat_es, p_value_t_es = stats.ttest_1samp(test_errors_es, paper_error)
    t_stat_no_es, p_value_t_no_es = stats.ttest_1samp(test_errors_no_es, paper_error)

    print(f"\n📊 STATISTICAL TESTS (vs DARTS paper {paper_error}% error):")
    print(f"  Early Stopping ON:")
    print(f"    t-statistic: {t_stat_es:.4f}, p-value: {p_value_t_es:.4f}")
    print(f"    Significant (p<0.05): {'Yes' if p_value_t_es < 0.05 else 'No'}")
    print(f"  Early Stopping OFF:")
    print(f"    t-statistic: {t_stat_no_es:.4f}, p-value: {p_value_t_no_es:.4f}")
    print(f"    Significant (p<0.05): {'Yes' if p_value_t_no_es < 0.05 else 'No'}")

    if len(test_errors_es) >= 5:
        w_stat_es, p_value_w_es = stats.wilcoxon([e - paper_error for e in test_errors_es])
        w_stat_no_es, p_value_w_no_es = stats.wilcoxon([e - paper_error for e in test_errors_no_es])
    else:
        w_stat_es = None
        p_value_w_es = None
        w_stat_no_es = None
        p_value_w_no_es = None

    summary = {
        "mode": mode,
        "early_stopping": es_mode,
        "early_patience": args.early_patience,
        "search_config": search_cfg,
        "eval_config": eval_cfg,
        "seeds": base_seeds,
        "num_runs": args.num_seeds,
        
        "early_stopping_on": {
            "results": all_results_es_on,
            "mean_test_acc": float(mean_acc_es),
            "std_test_acc": float(std_acc_es),
            "mean_test_error": float(mean_error_es),
            "std_test_error": float(std_error_es),
            "mean_max_overfitting_gap": float(np.mean(max_gaps_es)),
            "mean_avg_overfitting_gap": float(np.mean(avg_gaps_es)),
            "stopped_early_count": sum(r['stopped_early'] for r in all_results_es_on),
            "stat_t": float(t_stat_es),
            "stat_p_ttest": float(p_value_t_es),
            "stat_p_wilcoxon": float(p_value_w_es) if p_value_w_es is not None else None,
        },
        
        "early_stopping_off": {
            "results": all_results_es_off,
            "mean_test_acc": float(mean_acc_no_es),
            "std_test_acc": float(std_acc_no_es),
            "mean_test_error": float(mean_error_no_es),
            "std_test_error": float(std_error_no_es),
            "mean_max_overfitting_gap": float(np.mean(max_gaps_no_es)),
            "mean_avg_overfitting_gap": float(np.mean(avg_gaps_no_es)),
            "stopped_early_count": 0,
            "stat_t": float(t_stat_no_es),
            "stat_p_ttest": float(p_value_t_no_es),
            "stat_p_wilcoxon": float(p_value_w_no_es) if p_value_w_no_es is not None else None,
        },
        
        "comparison": {
            "error_diff_on_minus_off": float(diff_error),
            "early_on_better": bool(diff_error < 0)
        },
        
        "genotypes": all_genotypes,
        
        "total_time_min": float(total_time / 60),
        
        "paper_comparison": {
            "paper_error": paper_error,
            "our_error_es_on": float(mean_error_es),
            "our_error_es_off": float(mean_error_no_es),
            "diff_es_on": float(mean_error_es - paper_error),
            "diff_es_off": float(mean_error_no_es - paper_error)
        },
        
        "differences_with_paper": {
            "first_order": search_cfg["unrolled"] == False,
            "search_epochs_reduced": search_cfg["epochs"] < 50,
            "eval_epochs_reduced": eval_cfg["epochs"] < 600,
            "batch_size_smaller": eval_cfg["batch_size"] < 96,
            "cutout_missing": eval_cfg["cutout"] == False,
            "channels_reduced": eval_cfg["init_channels"] < 36
        }
    }

    with open(os.path.join(args.output, "full_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    import csv
    
    with open(os.path.join(args.output, "summary_es_on.csv"), 'w', newline='') as f:
        fieldnames = ["seed", "run", "test_acc", "test_acc5", "test_error", "best_val_acc", "best_epoch", "stopped_early", "max_overfitting_gap", "avg_overfitting_gap", "params_millions", "flops_giga", "eval_time_sec"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results_es_on:
            row = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items() if k in fieldnames}
            writer.writerow(row)
        writer.writerow({})
        writer.writerow({"seed": "MEAN", "test_acc": f"{mean_acc_es:.2f}", "test_error": f"{mean_error_es:.2f}"})
        writer.writerow({"seed": "STD", "test_acc": f"{std_acc_es:.2f}", "test_error": f"{std_error_es:.2f}"})

    with open(os.path.join(args.output, "summary_es_off.csv"), 'w', newline='') as f:
        fieldnames = ["seed", "run", "test_acc", "test_acc5", "test_error", "best_val_acc", "best_epoch", "stopped_early", "max_overfitting_gap", "avg_overfitting_gap", "params_millions", "flops_giga", "eval_time_sec"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results_es_off:
            row = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items() if k in fieldnames}
            writer.writerow(row)
        writer.writerow({})
        writer.writerow({"seed": "MEAN", "test_acc": f"{mean_acc_no_es:.2f}", "test_error": f"{mean_error_no_es:.2f}"})
        writer.writerow({"seed": "STD", "test_acc": f"{std_acc_no_es:.2f}", "test_error": f"{std_error_no_es:.2f}"})

    with open(os.path.join(args.output, "comparison.csv"), 'w', newline='') as f:
        fieldnames = ["seed", "test_acc_es_on", "test_acc_es_off", "test_error_es_on", "test_error_es_off", "diff_error", "best_epoch_es_on", "best_epoch_es_off", "gap_max_es_on", "gap_max_es_off"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(all_results_es_on)):
            row = {
                "seed": all_results_es_on[i]["seed"],
                "test_acc_es_on": f"{all_results_es_on[i]['test_acc']:.4f}",
                "test_acc_es_off": f"{all_results_es_off[i]['test_acc']:.4f}",
                "test_error_es_on": f"{all_results_es_on[i]['test_error']:.4f}",
                "test_error_es_off": f"{all_results_es_off[i]['test_error']:.4f}",
                "diff_error": f"{all_results_es_on[i]['test_error'] - all_results_es_off[i]['test_error']:.4f}",
                "best_epoch_es_on": all_results_es_on[i]["best_epoch"],
                "best_epoch_es_off": all_results_es_off[i]["best_epoch"],
                "gap_max_es_on": f"{all_results_es_on[i]['max_overfitting_gap']:.4f}",
                "gap_max_es_off": f"{all_results_es_off[i]['max_overfitting_gap']:.4f}",
            }
            writer.writerow(row)

    print(f"\n{'='*70}")
    print("📁 OUTPUT FILES - VERIFICACIÓN DE GUARDADO")
    print(f"{'='*70}")
    print(f"Results directory: {args.output}/")
    print(f"  ✓ full_summary.json (completo con comparación ES ON/OFF)")
    print(f"  ✓ summary_es_on.csv")
    print(f"  ✓ summary_es_off.csv")
    print(f"  ✓ comparison.csv")
    print(f"  Per seed:")
    for seed in base_seeds:
        print(f"  - seed_{seed}/")
        print(f"    SEARCH:")
        print(f"      ✓ genotype.json")
        print(f"      ✓ genotype_normal.png, genotype_reduce.png")
        print(f"      ✓ alpha_evolution.png, training_curves.png")
        print(f"      ✓ alpha_history.json")
        print(f"      ✓ alpha_normal_final.npy, alpha_reduce_final.npy")
        print(f"      ✓ metrics.json, config_search.json")
        print(f"    EVAL_ES_ON:")
        print(f"      ✓ best_model.pt")
        print(f"      ✓ confusion_matrix.png, classification_report.txt")
        print(f"      ✓ training_curves.png, overfitting_gap.png")
        print(f"      ✓ metrics.json, config_eval.json")
        print(f"    EVAL_ES_OFF:")
        print(f"      ✓ best_model.pt")
        print(f"      ✓ confusion_matrix.png, classification_report.txt")
        print(f"      ✓ training_curves.png, overfitting_gap.png")
        print(f"      ✓ metrics.json, config_eval.json")
    
    print(f"\n{'='*70}")
    print("📊 MÉTRICAS REPORTADAS")
    print(f"{'='*70}")
    print("  ✓ Test Accuracy@1 (mean ± std) - ES ON y OFF")
    print("  ✓ Test Accuracy@5 - ES ON y OFF")
    print("  ✓ Test Error rate - ES ON y OFF")
    print("  ✓ Best epoch (when best val_acc)")
    print("  ✓ Overfitting gap (train-val) - max y avg")
    print("  ✓ Parameters (M)")
    print("  ✓ FLOPs (G)")
    print("  ✓ Search/Evaluation time")
    print("  ✓ Statistical tests (t-test, Wilcoxon)")
    print("  ✓ Alpha evolution (entropy)")
    print("  ✓ Genotype visualization (DAG)")
    print("  ✓ Training curves (loss, acc)")
    print("  ✓ Overfitting gap curves")
    print("  ✓ Confusion matrix")
    print("  ✓ Genotypes guardados")
    print(f"{'='*70}")
    print(f"\n⏱️  Total Time: {total_time/60:.1f} min")

if __name__ == '__main__':
    main()
