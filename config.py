"""
Training Configuration & Utilities

Shared configuration classes and factory functions for model and data loading.

Data layout expected by default:
    ./data/
      cifar/       # torchvision CIFAR10/CIFAR100 root
      imagenet/    # ImageFolder-style dataset
        train/
        val/
"""
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, models, transforms
from nanogpt_model import GPT, GPTConfig   # verbatim nanoGPT model.py

# Datasets stored as nanoGPT-format token files: <data_dir>/<name>/train.bin, val.bin
TOKEN_DATASETS = ("openwebtext", "tinystories")


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Model & data
    model_name: str = "resnet50"
    dataset: str = "cifar10"
    num_classes: int = 10
    image_size: int = 32

    # Training hyperparameters
    num_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.001
    momentum: float = 0.9
    optimizer: str = "sgd"          # "sgd" | "adamw"
    nesterov: bool = True
    weight_decay: float = 5e-4
    wd_on_bn_bias: bool = False # Nesterov=False (B1) = exclude BN gamma/beta + all biases from WD
                                   #Nesterov=True  (B4) = apply WD to every param (old behaviour)
    grad_clip: Optional[float] = None

    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 0

    # DataLoader behavior
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True

    # Distributed training
    backend: str = "mpi"
    comm_algorithm: Optional[str] = None

    # ZFP compression
    zfp_rate: float = 16.0

    # Model initialization
    cifar_stem: bool = True
    pretrained: bool = False
    init_cifar_stem_from_pretrained_center: bool = False

    # Paths & misc
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42
    
    # ── nanoGPT (only read when model_name is a GPT name) ────────────────
    block_size: int = 1024          # nanoGPT train.py default
    vocab_size: int = 50304         # nanoGPT: GPT-2 50257 padded to x64
    gpt_bias: bool = False          # nanoGPT train.py default (model.py default is True!)
    dropout: float = 0.0            # nanoGPT train.py default
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95        # nanoGPT train.py default

    # ── Shared extensions (defaults = old behaviour) ─────────────────────
    grad_accum_global: Optional[int] = None  # micro-batches per optimizer step over ALL ranks; None = 1/rank
    amp: str = "none"                         # "none" | "bf16" (nanoGPT default dtype)
    allow_tf32: bool = False                  # nanoGPT sets True
    warmup_iters: int = 0                     # >0 overrides warmup_epochs
    min_lr_ratio: float = 0.0                 # cosine floor; nanoGPT min_lr/lr = 0.1
    iters_per_epoch: Optional[int] = None     # cap optimizer steps per epoch
    val_max_batches: Optional[int] = None     # cap rank-0 validation batches

    @property
    def is_lm(self) -> bool:
        return is_gpt_model(self.model_name)

    def validate(self):
        valid_datasets = ("cifar10", "cifar100", "imagenet", "imagenet_like") + TOKEN_DATASETS
        valid_schedulers = ("constant", "cosine")

        assert self.dataset in valid_datasets, f"Unknown dataset: {self.dataset}"
        assert self.backend in ("nccl", "mpi"), f"Unknown backend: {self.backend}"
        assert self.scheduler in valid_schedulers, (
            f"Unknown scheduler: {self.scheduler}"
        )
        assert self.num_epochs > 0
        assert self.batch_size > 0
        assert self.learning_rate > 0
        assert self.num_workers >= 0
        assert self.optimizer in ("sgd", "adamw"), f"Unknown optimizer: {self.optimizer}"
        assert not (self.optimizer == "sgd" and self.nesterov and self.momentum == 0), \
            "nesterov=True requires momentum > 0"
        assert self.amp in ("none", "bf16"), f"Unknown amp: {self.amp}"

        if self.is_lm or self.dataset in TOKEN_DATASETS:
            assert self.is_lm and self.dataset in TOKEN_DATASETS, (
                f"GPT models need a token dataset {TOKEN_DATASETS} (and vice "
                f"versa); got model={self.model_name}, dataset={self.dataset}"
            )
        elif self.dataset.startswith("cifar"):
            assert self.image_size >= 32, "image_size must be >= 32 for CIFAR"
            # The CIFAR stem (stride-1 conv, no maxpool) is only meaningful at
            # 32x32. At larger sizes it keeps full resolution through layer1 and
            # blows up activation memory, so require the standard stem instead.
            if self.cifar_stem:
                assert self.image_size == 32, (
                    "cifar_stem=True is only valid with image_size=32. "
                    "For upscaled CIFAR set CIFAR_STEM=false."
                )
        else:
            assert self.image_size > 0


def apply_cifar_stem(
    model: nn.Module,
    pretrained: bool = False,
    init_from_pretrained_center: bool = False,
) -> nn.Module:
    old_conv1 = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=old_conv1.in_channels,
        out_channels=old_conv1.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    if pretrained and init_from_pretrained_center:
        if old_conv1.weight.shape[-2:] != (7, 7):
            raise ValueError(
                "Cannot center-crop pretrained stem: expected old conv1 to be 7x7."
            )

        with torch.no_grad():
            model.conv1.weight.copy_(old_conv1.weight[:, :, 2:5, 2:5])

    model.maxpool = nn.Identity()
    return model


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
    else:
        raise ValueError(
            f"Do not know how to replace classifier for {type(model).__name__}"
        )
    return model


def create_model(
    model_name: str,
    num_classes: int,
    cifar_stem: bool = True,
    pretrained: bool = False,
    init_cifar_stem_from_pretrained_center: bool = False,
) -> nn.Module:
    builders = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "wide_resnet50_2": models.wide_resnet50_2,
        "resnext101_32x8d": models.resnext101_32x8d,
        "convnext_tiny": models.convnext_tiny,
        "convnext_small": models.convnext_small,
    }

    pretrained_weights = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
        "wide_resnet50_2": models.Wide_ResNet50_2_Weights.DEFAULT,
        "resnext101_32x8d": models.ResNeXt101_32X8D_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "convnext_small": models.ConvNeXt_Small_Weights.DEFAULT,
    }

    if model_name not in builders:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(builders.keys())}"
        )

    if pretrained:
        model = builders[model_name](weights=pretrained_weights[model_name])
        model = _replace_classifier(model, num_classes)
    else:
        model = builders[model_name](weights=None, num_classes=num_classes)

    if cifar_stem and model_name.startswith(("resnet", "resnext", "wide_resnet")):
        model = apply_cifar_stem(
            model,
            pretrained=pretrained,
            init_from_pretrained_center=init_cifar_stem_from_pretrained_center,
        )

    return model

_GPT_NAME = re.compile(r"^gpt2_(\d+)x(\d+)x(\d+)$")   # gpt2_<n_layer>x<n_head>x<n_embd>


def is_gpt_model(model_name: str) -> bool:
    return bool(_GPT_NAME.match(model_name))


def create_model_from_config(config: TrainingConfig) -> nn.Module:
    """Single entry point for every model family."""
    if config.is_lm:
        n_layer, n_head, n_embd = (int(v) for v in _GPT_NAME.match(config.model_name).groups())
        return GPT(GPTConfig(
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            block_size=config.block_size, vocab_size=config.vocab_size,
            bias=config.gpt_bias, dropout=config.dropout,
        ))
    return create_model(
        config.model_name,
        config.num_classes,
        cifar_stem=config.cifar_stem,
        pretrained=config.pretrained,
        init_cifar_stem_from_pretrained_center=config.init_cifar_stem_from_pretrained_center,
    )

class TokenWindowDataset(torch.utils.data.Dataset):
    """Non-overlapping windows over a nanoGPT uint16 .bin token file.

    Item i = (x, y) with x = tokens[s : s+T] and y = tokens[s+1 : s+T+1]:
    exactly nanoGPT's get_batch() pair, but at fixed offsets so that
    DistributedSampler / drop_last keep working. The memmap opens lazily so
    every DataLoader worker gets its own handle.
    """

    def __init__(self, bin_path: str, block_size: int):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"{bin_path} not found (run the data prep first)")
        self.path, self.block, self.data = bin_path, block_size, None
        self.n_tokens = os.path.getsize(bin_path) // 2   # uint16

    def __len__(self):
        return (self.n_tokens - 1) // self.block

    def __getitem__(self, i):
        if self.data is None:
            self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        s = i * self.block
        buf = torch.from_numpy(self.data[s: s + self.block + 1].astype(np.int64))
        return buf[:-1], buf[1:]
    
def _cifar_root(config: TrainingConfig) -> str:
    return f"{config.data_dir}/cifar"


def _imagenet_root(config: TrainingConfig) -> str:
    return f"{config.data_dir}/imagenet"


def get_data_loaders(
    config: TrainingConfig,
    rank: int,
    world_size: int,
) -> tuple:
    if config.dataset == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        ds_cls = datasets.CIFAR10

        train_dataset = ds_cls(
            root=_cifar_root(config),
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        )

        val_dataset = ds_cls(
            root=_cifar_root(config),
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
        )

    elif config.dataset == "cifar100":
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )
        ds_cls = datasets.CIFAR100

        train_dataset = ds_cls(
            root=_cifar_root(config),
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        )

        val_dataset = ds_cls(
            root=_cifar_root(config),
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
        )
        
    elif config.dataset in TOKEN_DATASETS:
        root = f"{config.data_dir}/{config.dataset}"
        train_dataset = TokenWindowDataset(f"{root}/train.bin", config.block_size)
        val_dataset = TokenWindowDataset(f"{root}/val.bin", config.block_size)

    elif config.dataset in ("imagenet", "imagenet_like"):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        train_dataset = datasets.ImageFolder(
            root=f"{_imagenet_root(config)}/train",
            transform=transforms.Compose([
                transforms.RandomResizedCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        )

        val_dataset = datasets.ImageFolder(
            root=f"{_imagenet_root(config)}/val",
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                normalize,
            ]),
        )

    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        drop_last=config.drop_last,  
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )

    # Validation is NOT sharded. Every rank could evaluate the full 10k test set
    # and get an identical answer (DDP keeps weights in sync), so no reduction is
    # needed — which keeps the ONLY collective in the run the gradient allreduce
    # performed by the custom comm hook. In practice only rank 0 runs it.
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,              # ← replaces sampler=val_sampler
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, train_sampler