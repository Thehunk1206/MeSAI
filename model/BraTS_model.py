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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from unet3D import Unet3D
from vae_decoder import VAE_decoder


class BraTSeg(tf.keras.Model):
    def __init__(
        self, 
        IMG_H: int = 160,
        IMG_W: int = 192,
        IMG_D: int = 128,
        IMG_C: int = 3,
        out_channel: int = 3,
         **kwargs
    ):
        super(BraTSeg, self).__init__(**kwargs)
        self.IMG_H       = IMG_H
        self.IMG_W       = IMG_W
        self.IMG_D       = IMG_D
        self.IMG_C       = IMG_C
        self.out_channel = out_channel,

        self.unet3D      = Unet3D(name='Unet3D', number_of_class=self.out_channel)
        self.vae_decoder = VAE_decoder(
                                        name='vae_decoder',
                                        feat_h=self.IMG_H//16, 
                                        feat_w= self.IMG_H//16, 
                                        feat_d=self.IMG_D//16
                                    )

    def call(self, inputs: tf.Tensor, training: bool = True):
        pred_seg_vol, x_256             = self.unet3D(inputs, training = training)
        z_mean,z_var,reconstructed_vol  = self.vae_decoder(x_256, training = training)
        
        return pred_seg_vol, reconstructed_vol, z_mean, z_var
    
    def compile(
        self, 
        optimizer:tf.keras.optimizers.Optimizer, 
        dice_loss:tf.keras.losses.Loss,
        vae_loss: tf.keras.losses.Loss, 
        **kwargs
    ):  
        super(BraTSeg, self).compile(**kwargs)
        self.optimizer  = optimizer
        self.dice_loss  = dice_loss
        self.vae_loss   = vae_loss

        self.unet3D.compile(
            optimizer=self.optimizer,
            loss=self.dice_loss
        )
        self.vae_decoder.compile(
            optimizer=self.optimizer,
            loss=self.vae_loss
        )
    
    def summary(self):
        x = tf.keras.Input(shape=(self.IMG_H,self.IMG_W, self.IMG_D, self.IMG_C))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='BraTSeg_Model')
        return model.summary()

    def get_config(self):
        config = {
            'Image_H':        self.IMG_H,
            'Image_W':        self.IMG_W,
            'Image_D':        self.IMG_D,
            'Image_C':        self.IMG_C,
            'Output Channel': self.out_channel
        }
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(config)
    
if __name__ == "__main__":

    x = tf.ones(shape=(1, 160, 192, 128, 3))

    brat_seg = BraTSeg(name='BratSeg')
    # first call to the `brat_seg` will create weights
    y = brat_seg(x)

    tf.print("weights:", len(brat_seg.weights))
    tf.print("trainable weights:", len(brat_seg.trainable_weights))
    tf.print("config:", brat_seg.get_config())
    
    for i in range(y):
        tf.print(f"Out{i}: {y[i].shape}")
    tf.print(brat_seg.summary())