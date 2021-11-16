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

from MeSAI.utils.dataset import TfdataPipeline
from MeSAI.utils.losses import SoftDiceLoss, WBCEDICELoss, FocalTverskyLoss, VAE_loss
from MeSAI.network.vae_unet3D import VAEUnet3D
from MeSAI.network.unet3D import UNet3D

import tensorflow as tf
tf.random.set_seed(4)


def train(
    dataset_dir: str,
    trained_model_dir: str,
    IMG_H: int           = 160,
    IMG_W: int           = 192,
    IMG_D: int           = 128,
    IMG_C: int           = 3,   #hardcoded
    out_channel: int     = 3,   #hardcoded
    batch_size: int      = 1,
    epochs:int           = 100,
    lr: float            = 1e-4,
    dataset_split: float = 0.1,
    logdir: str          = 'logs/',
    model_name:str       = 'vae_unet3D',
    loss_function:str    = 'softdice',
    alpha:float          = 0.7,         
    gamma:int            = 3,

):
    assert os.path.isdir(dataset_dir), f'{dataset_dir} is not a directory'
    assert isinstance(gamma,int), f'gamma should be int, given data type {type(gamma)}'
    assert 0 <= alpha <=1, f'aplha should be between 0 - 1'

    if not os.path.exists(dataset_dir):
        tf.print(f'No such directory named as {dataset_dir} exist.')
        sys.exit()
    
    if not os.path.exists(trained_model_dir):
        os.mkdir(trained_model_dir)
    
    # instantiate tf.summary writer
    logsdir = logdir + "Model/" + "BraTS_"+loss_function+datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(logsdir + "/train/")
    val_writer = tf.summary.create_file_writer(logsdir + "/val/")

    #initialize tfdatapipeline
    tf.print('[INFO] Creating Data Pipeline...\n')
    tfdatapipeline = TfdataPipeline(
        BASE_DATASET_DIR=dataset_dir,
        IMG_H=IMG_H,
        IMG_W=IMG_W,
        IMG_D=IMG_D,
        batch_size=batch_size,
        split=dataset_split
    )
    train_data = tfdatapipeline.data_loader(dataset_type='train')
    val_data = tfdatapipeline.data_loader(dataset_type='valid')

    #instantiate optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
    )

    # instantiate seg loss and vae_loss
    _loss_function_list = ['softdice', 'w_bce_dice', 'focal_tversky']

    if loss_function not in _loss_function_list:
        tf.print(f'Loss function should be in f{_loss_function_list} list')
        sys.exit()
    if loss_function == 'softdice':
        seg_loss = SoftDiceLoss('softdice_loss')
    if loss_function == 'w_bce_dice':
        seg_loss = WBCEDICELoss(name='w_bce_dice_loss')
    if loss_function == 'focal_tversky':
        seg_loss = FocalTverskyLoss(name='focal_tversky_loss',alpha=alpha, gamma=gamma)
    
    vae_loss = VAE_loss(name='vae_loss')
    
    #instantiate Model (BraT_Seg)
    tf.print(f'[INFO] Creating Model...\n')
    seg_model = VAEUnet3D(
        name='seg_model',
        IMG_H=IMG_H,
        IMG_W=IMG_W,
        IMG_D=IMG_D,
        IMG_C=IMG_C,
        out_channel=out_channel,
    )

    #compile the model
    tf.print(f'[INFO] Compiling Model...\n')
    seg_model.compile(
        optimizer=optimizer,
        seg_loss=seg_loss,
        vae_loss=vae_loss
    )

    
    tf.print(f'[INFO] Summary of all model\n')
    tf.print(seg_model.summary())
    # tf.print(seg_model.unet3D.summary())
    # tf.print(seg_model.vae_decoder.summary(input_shape=(IMG_H//8, IMG_W//8, IMG_D//8, 256)))

    tf.print('\n')
    tf.print('*'*60)
    tf.print('\t\t\tModel Configs')
    tf.print('*'*60)
    tf.print(
        f'\n',
        f'Training and validating Model : {seg_model.name} \n',
        f'Epochs                        : {epochs} \n',
        f'learing_rate                  : {lr} \n',
        f'Input shape                   : ({IMG_H},{IMG_W},{IMG_D},{IMG_C}) \n',
        f'Batch size                    : {batch_size} \n',
        f'Loss Function                 : {loss_function} \n',
        f'Output Channel                : {out_channel} \n',
    )

    # train the model
    for e in range(epochs):
        t = time()

        for (train_img_vol, train_seg_vol) in tqdm(train_data, unit='steps', desc='training...', colour='red'):
            train_loss, train_dice, train_iou, train_precision, train_recall = seg_model.train_step(
                                                                                        x_vol=train_img_vol,
                                                                                        y_mask=train_seg_vol
                                                                                    )
        for (val_img_vol, val_seg_vol) in tqdm(val_data, unit='steps', desc='validating...', colour='green'):
            val_loss, val_dice, val_iou, val_precision, val_recall = seg_model.test_step(
                                                                                x_vol=val_img_vol,
                                                                                y_mask=val_seg_vol
                                                                            )
        tf.print(
            "ETA:{} - epoch: {} - loss: {} - dice: {} - IoU: {} - Precision: {}- Recall: {} - val_loss: {} - val_dice: {} - val_IoU: {} - val_Precision: {} - val_Recall: {}\n".format(
                round((time() - t)/60, 2), (e+1), train_loss, train_dice, train_iou, train_precision, train_recall, val_loss, val_dice, val_iou, val_precision, val_recall)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='BraTS_2020/', help='path to dataset')

    parser.add_argument('--data_split', type=float,
                        default=0.1, help='split percent of val and test data')

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--batchsize', type=int,
                        default=1, help='training batch size')

    parser.add_argument('--img_h', type=int,
                        default=160, help='input height')

    parser.add_argument('--img_w', type=int,
                        default=192, help='input width')

    parser.add_argument('--img_d', type=int,
                        default=128, help='input depth')

    parser.add_argument('--img_c', type=int,
                        default=3, help='input channel')

    parser.add_argument('--out_channel', type=int,
                        default=3, help='output channel(label)')
                        
    parser.add_argument('--trained_model_path', type=str,
                        default='trained_model/')

    parser.add_argument('--logdir', type=str, help="Tensorboard logs",
                        default='logs/')
    
    parser.add_argument('--loss_function', type=str, help="choose loss function from ['softdice', 'w_bce_dice', 'focal_tversky']",
                        default='softdice')
    
    parser.add_argument('--alpha', type=float,
                        default=0.7, help='False Negative weigth fot Focal Tversky Loss')

    parser.add_argument('--gamma', type=int,
                        default=3, help='Power factor in Focal Tversky loss (taken as 1/gamma)')

    opt = parser.parse_args()

    train(
        dataset_dir         =   opt.data_path,
        trained_model_dir   =   opt.trained_model_path,
        IMG_H               =   opt.img_h,
        IMG_W               =   opt.img_w,
        IMG_D               =   opt.img_d,
        IMG_C               =   opt.img_c,
        out_channel         =   opt.out_channel,
        batch_size          =   opt.batchsize,
        epochs              =   opt.epoch,
        lr                  =   opt.lr,
        dataset_split       =   opt.data_split,
        logdir              =   opt.logdir,
        loss_function       =   opt.loss_function,
        alpha               =   opt.alpha,
        gamma               =   opt.gamma,
    )
