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

from __future__ import annotations
from __future__ import absolute_import

import os
from typing import Tuple, Union

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from MeSAI.network.unet3D import Unet3D
from MeSAI.layers.vae_decoder import VAE_decoder
from MeSAI.utils.losses import L2loss

from MeSAI.utils.metrics import dice_coef, iou_metric, Precision, Recall


'''
TODO: Test the VAE_UNET3D
'''
class VAEUnet3D(tf.keras.Model):
    def __init__(
        self, 
        name:  str,
        IMG_H: int = 160,
        IMG_W: int = 192,
        IMG_D: int = 128,
        IMG_C: int = 3,
        number_of_class: int = 3,
        enable_deepsupervision: bool = True,
         **kwargs
    ):
        super(VAEUnet3D, self).__init__(name=name,**kwargs)
        self.IMG_H       = IMG_H
        self.IMG_W       = IMG_W
        self.IMG_D       = IMG_D
        self.IMG_C       = IMG_C
        self.number_of_class = number_of_class
        self.enable_deepsupervision = enable_deepsupervision
        self.l2_loss = L2loss(name='L2loss')

        self.unet3D      = Unet3D(name='Unet3D', number_of_class=self.number_of_class, enable_deepsupervision=self.enable_deepsupervision)
        self.vae_decoder = VAE_decoder(
                                        name='vae_decoder',
                                        feat_h=self.IMG_H//16, 
                                        feat_w= self.IMG_W//16, 
                                        feat_d=self.IMG_D//16
                                    )

    def call(self, inputs: tf.Tensor, training: bool = True):
        pred_seg_vol, x_256             = self.unet3D(inputs, training = training)
        z_mean,z_var,reconstructed_vol  = self.vae_decoder(x_256, training = training)

        kl_loss = -0.5 * tf.reduce_mean(1 + z_var - tf.square(z_mean) - tf.exp(z_var))
        self.add_loss(kl_loss)

        return pred_seg_vol, reconstructed_vol
    
    def compile(
        self, 
        optimizer:tf.keras.optimizers.Optimizer, 
        loss: Union(tf.keras.losses.Loss, Tuple[tf.keras.losses.Loss, ...]),
        **kwargs
    ):  
        super(VAEUnet3D, self).compile(**kwargs)
        self.optimizer  = optimizer
        self.loss       = loss
        
        self.unet3D.compile(
            optimizer=self.optimizer,
            loss=self.loss
        )
        self.vae_decoder.compile(
            optimizer=self.optimizer,
            loss=self.loss
        )
    
    @tf.function
    def train_step(self, x_vol:tf.Tensor, y_mask:tf.Tensor, gclip:float) -> tuple[tf.Tensor, ...]:
        '''
        Forward pass, calculates total loss, metrics, and calculate gradients with respect to loss.
        args    x_vol : Input 3D volume -> tf.Tensor
                y_mask: 3D Mask map of x_vol -> tf.Tensor
                gclip : Gradient Clipping value -> float
        
        returns total_loss, train_dice, train_iou, train_precision, train_recall
        '''
        with tf.GradientTape() as tape:
            pred_seg_vol, reconstructed_vol = self(x_vol, training=True)
            
            #Check dtype of pred_seg_vol
            if isinstance(pred_seg_vol, tuple):
                loss1 = 0
                for pred in pred_seg_vol:
                    loss1 += self.loss(y_mask, pred)
            else:
                loss1 = self.loss(y_mask, pred_seg_vol)
    
            loss2 = self.loss(y_mask, pred_seg_vol)
            loss3 = 0.1 * self.l2_loss(x_vol, reconstructed_vol)
            loss4 = 0.1 * self.losses

            train_loss = loss1 + loss2 + loss3 + loss4 
        
        gradients = tape.gradient(train_loss, self.trainable_variables)
        gradients = [tf.clip_by_value(grad, -gclip, gclip) for grad in gradients]

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        #calculate metrics
        train_dice      = dice_coef(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else dice_coef(y_mask=y_mask, y_pred=pred_seg_vol)
        train_iou       = iou_metric(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else iou_metric(y_mask=y_mask, y_pred=pred_seg_vol)
        train_precision = Precision(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else Precision(y_mask=y_mask, y_pred=pred_seg_vol)
        train_recall    = Recall(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else Recall(y_mask=y_mask, y_pred=pred_seg_vol)

        return train_loss, train_dice, train_iou, train_precision, train_recall

    @tf.function
    def test_step(self, x_vol:tf.Tensor, y_mask:tf.Tensor):
        '''
        Forward pass, Calculates loss and metric on validation set
        args    x_img: Input Image -> tf.Tensor
                y_mask: Mask map of x_img -> tf.Tensor
        
        returns total_loss, val_dice, val_iou, val_precision, val_recall
        '''
        pred_seg_vol, reconstructed_vol = self(x_vol, training=False)
        if isinstance(pred_seg_vol, tuple):
            loss1 = 0
            for pred in pred_seg_vol:
                loss1 += self.loss(y_mask, pred)
        else:
            loss1 = self.loss(y_mask, pred_seg_vol)
    
        loss2 = self.loss(y_mask, pred_seg_vol)
        loss3 = 0.1 * self.l2_loss(x_vol, reconstructed_vol)
        loss4 = 0.1 * self.losses
        val_loss = loss1 + loss2 + loss3 + loss4

        #calculate metrics
        val_dice      = dice_coef(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else dice_coef(y_mask=y_mask, y_pred=pred_seg_vol)
        val_iou       = iou_metric(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else iou_metric(y_mask=y_mask, y_pred=pred_seg_vol)
        val_precision = Precision(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else Precision(y_mask=y_mask, y_pred=pred_seg_vol)
        val_recall    = Recall(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else Recall(y_mask=y_mask, y_pred=pred_seg_vol)

        return val_loss, val_dice, val_iou, val_precision, val_recall

    def summary(self):
        x = tf.keras.Input(shape=(self.IMG_H,self.IMG_W, self.IMG_D, self.IMG_C))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='VAEUnet3D_Model')
        return model.summary()

    def get_config(self):
        config = {
            'Image_H'               : self.IMG_H,
            'Image_W'               : self.IMG_W,
            'Image_D'               : self.IMG_D,
            'Image_C'               : self.IMG_C,
            'Number of class'       : self.number_of_class,
            'enable_deepsupervision': self.enable_deepsupervision,
        }
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(config)
    
if __name__ == "__main__":

    x = tf.ones(shape=(1, 160, 192, 128, 3))
    _, h, w, d, _ = x.shape.as_list()

    vae_unet = VAEUnet3D(name='BratSeg')
    # first call to the `vae_unet` will create weights
    y = vae_unet(x)

    tf.print("weights:", len(vae_unet.weights))
    tf.print("trainable weights:", len(vae_unet.trainable_weights))
    tf.print("config:", vae_unet.get_config())
    
    for i in range(len(y)):
        tf.print(f"Out{i}: {y[i].shape}")
    tf.print(vae_unet.summary())
    tf.print(vae_unet.unet3D.summary())
    tf.print(vae_unet.vae_decoder.summary(input_shape=(h//8, w//8, d//8, 256)))