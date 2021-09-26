'''
=================================LICENSE====================================

MIT License

Copyright (c) 2021 Tauhid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from datetime import datetime
from time import time
from tqdm import tqdm
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.dataset import TfdataPipeline
from utils.losses import SoftDiceLoss, WBCEDICELoss, FocalTverskyLoss, VAE_loss
from model.BraTS_model import BraTSeg

import tensorflow as tf
tf.random.set_seed(4)


def train(
    dataset_dir: str,
    trained_model_dir: str,
    IMG_H: int           = 160,
    IMG_W: int           = 192,
    IMG_D: int           = 128,
    batch_size: int      = 1,
    lr: float            = 1e-4,
    dataset_split: float = 0.1,
    logdir: str          = 'logs/',
    loss_function:str    = 'softdice',
    
):
    pass