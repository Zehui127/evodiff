import argparse
import json
import os
from datetime import datetime, timedelta
import pathlib

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from evodiff.model import ByteNetLMTime
from evodiff.utils import Tokenizer
from evodiff.dna_utils import Tokenizer as DNATokenizer
from torch.utils.data import Subset
from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.datasets import UniRefDataset
from sequence_models.constants import MSA_ALPHABET
from evodiff.collaters import OAMaskCollater, D3PMCollater
from evodiff.losses import OAMaskedCrossEntropyLoss, D3PMCELoss, D3PMLVBLoss
from sequence_models.metrics import MaskedAccuracy
from sequence_models.utils import warmup
import sys
from evodiff.dnadiff.constant import DNA_ALPHABET, DNA_GAP
from evodiff.dnadiff.dna_data import setup_epd_dataset

random_seed =0
data_dir = "/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/evodiff/dataset/sequence.csv"
ds_train, ds_valid, _ = setup_epd_dataset(data_dir, 0.8, 0.1, seed=0)
max_batch_size = 8
diffusion_timesteps = 500
tokenizer = DNATokenizer(sequences=True)
Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
collater = D3PMCollater(tokenizer=tokenizer, num_timesteps=diffusion_timesteps, Q=Q_t, Q_bar=Q_prod)


dl_train = DataLoader(dataset=ds_train,
                              batch_size=max_batch_size,
                              num_workers=16,
                              shuffle=True,
                              collate_fn=collater)

dl_train_simple = DataLoader(dataset=ds_train,
                              batch_size=max_batch_size,
                              num_workers=16,
                              shuffle=True)

dl_train_iter = iter(dl_train)
dl_train_simple_iter = iter(dl_train_simple)

sample = next(dl_train_iter)
sample_simple = next(dl_train_simple_iter)
