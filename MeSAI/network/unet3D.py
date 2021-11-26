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

import os
from typing import Tuple, Union
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from MeSAI.layers.encoder3d import Encoder3D
from MeSAI.layers.decoder3d import Decoder3D
from MeSAI.utils.metrics import dice_coef, iou_metric, Precision, Recall

class Unet3D(tf.keras.Model):
    def __init__(self,
        name:str,
        number_of_class:int = 3, 
        enable_deepsupervision:bool = True,
        IMG_H: int = 160,
        IMG_W: int = 192,
        IMG_D: int = 128,
        IMG_C: int = 3,
        *args, **kwargs
    ):
        super(Unet3D, self).__init__(name=name,*args, **kwargs)

        self.number_of_class = number_of_class
        self.enable_deepsupervision = enable_deepsupervision
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_D = IMG_D
        self.IMG_C = IMG_C

        self.encoder = Encoder3D(name='encoder3d')
        self.decoder = Decoder3D(name='decoder3d', number_of_class=self.number_of_class, enable_deepsupervision=self.enable_deepsupervision)


    def call(self, inputs:tf.Tensor, training:bool=None)->Tuple[Union(tf.Tensor, Tuple[tf.Tensor, ...]), tf.Tensor]:
        x_32, x_64, x_128, x_256    = self.encoder(inputs)
        output                      = self.decoder((x_32, x_64, x_128, x_256))

        # returning x_256 as input for VAE_decoder3d
        return output, x_256

    def compile(
        self, optimizer:tf.keras.optimizers.Optimizer, 
        loss: Union(tf.keras.losses.Loss, Tuple[tf.keras.losses.Loss, ...]),
        loss_weights=None, 
        **kwargs
    ):
        super(Unet3D,self).compile(**kwargs)
        self.optimizer      = optimizer
        self.loss           = loss
        self.loss_weights   = loss_weights

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
            pred_seg_vol,_ = self(x_vol, training = True)

            # Check dtype of pred_seg_vol 
            if isinstance(pred_seg_vol, tuple):
                train_loss = 0
                for pred in pred_seg_vol:
                    train_loss += self.loss(y_mask, pred)
            else:
                train_loss = self.loss(y_mask, pred_seg_vol)
        
        #Calculate gradients
        gradients = tape.gradient(train_loss, self.trainable_variables)
        gradients = [(tf.clip_by_value(grad, clip_value_min=-gclip, clip_value_max=gclip)) for grad in gradients]

        # Backpropogation
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Calculate metrics
        train_dice      = dice_coef(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else dice_coef(y_mask=y_mask, y_pred=pred_seg_vol)
        train_iou       = iou_metric(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else iou_metric(y_mask=y_mask, y_pred=pred_seg_vol)
        train_precision = Precision(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else Precision(y_mask=y_mask, y_pred=pred_seg_vol)
        train_recall    = Recall(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else Recall(y_mask=y_mask, y_pred=pred_seg_vol)

        return train_loss, train_dice, train_iou, train_precision, train_recall


    @tf.function
    def test_step(self, x_vol:tf.Tensor, y_mask:tf.Tensor) -> tuple[tf.Tensor, ...]:
        '''
        Forward pass, Calculates loss and metric on validation set
        args    x_img: Input Image -> tf.Tensor
                y_mask: Mask map of x_img -> tf.Tensor
        
        returns total_loss, val_dice, val_iou, val_precision, val_recall
        '''
        # Forword pass
        pred_seg_vol, _ = self(x_vol, training = False)

        # Check dtype of pred_seg_vol
        if isinstance(pred_seg_vol, tuple):
            val_loss = 0
            for pred in pred_seg_vol:
                val_loss += self.loss(y_mask, pred)
        else:
            val_loss = self.loss(y_mask, pred_seg_vol)

        val_dice      = dice_coef(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else dice_coef(y_mask=y_mask, y_pred=pred_seg_vol)
        val_iou       = iou_metric(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else iou_metric(y_mask=y_mask, y_pred=pred_seg_vol)
        val_precision = Precision(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else Precision(y_mask=y_mask, y_pred=pred_seg_vol)
        val_recall    = Recall(y_mask=y_mask, y_pred=pred_seg_vol[0]) if isinstance(pred_seg_vol, tuple) else Recall(y_mask=y_mask, y_pred=pred_seg_vol)

        return val_loss, val_dice, val_iou, val_precision, val_recall
        

    def summary(self):
        x = tf.keras.Input(shape=(self.IMG_H,self.IMG_W,self.IMG_D,self.IMG_C))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='Unet3D')
        return model.summary()

    def get_config(self):
        config = {
            'number_of_class': self.number_of_class,
            'enable_deepsupervision': self.enable_deepsupervision,
            'IMG_H': self.IMG_H,
            'IMG_W': self.IMG_W,
            'IMG_D': self.IMG_D,
            'IMG_C': self.IMG_C,
        }
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(config)

if __name__ == "__main__":
    unet = Unet3D(name='unet3D', number_of_class=3)
    # first call to the `unet` will create weights
    y = unet(tf.ones(shape=(1, 160, 192, 128, 3)))

    tf.print("weights:", len(unet.weights))
    tf.print("trainable weights:", len(unet.trainable_weights))
    tf.print("config:", unet.get_config())
    tf.print(y[0].shape)
    tf.print(unet.summary())
    
