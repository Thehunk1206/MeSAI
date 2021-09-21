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

import tensorflow as tf

from encoder3d import Encoder3D
from decoder3d import Decoder3D

class Unet3D(tf.keras.Model):
    def __init__(self,name:str,number_of_class:int = 3, *args, **kwargs):
        super(Unet3D, self).__init__(name=name,*args, **kwargs)

        self.number_of_class = number_of_class

        self.encoder = Encoder3D(name='encoder3d')
        self.decoder = Decoder3D(name='decoder3d', number_of_class=self.number_of_class)

    def call(self, inputs:tf.Tensor, training:bool=None)->tf.Tensor:
        x_32, x_64, x_128, x_256    = self.encoder(inputs)
        output                      = self.decoder((x_32, x_64, x_128, x_256))

        # returning x_256 as input for VAE_decoder3d
        return output, x_256

    def compile(
        self, optimizer:tf.keras.optimizers.Optimizer, 
        loss:tf.losses.Loss, 
        loss_weights=None, 
        **kwargs
    ):
        return super(Unet3D,self).compile(optimizer=optimizer, loss=loss,  loss_weights=loss_weights, **kwargs)

    def summary(self):
        x = tf.keras.Input(shape=(160,192,128,3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='Unet3D')
        return model.summary()
    
    def get_config(self):
        config = {
            'number_of_class':self.number_of_class
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
    
