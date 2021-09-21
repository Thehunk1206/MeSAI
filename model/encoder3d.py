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
from conv3d_module import Conv3d_module

class Encoder3D(tf.keras.layers.Layer):
    def __init__(self, name:str, **kwargs):
        super(Encoder3D, self).__init__(name=name,**kwargs)

        self._L2_reg_f = 1e-5
        self.conv_1 = tf.keras.layers.Conv3D(
            filters=32,
            kernel_size=(3,3,3),
            strides=(1,1,1),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(self._L2_reg_f)
        )

        self.spatial_dropout3d = tf.keras.layers.SpatialDropout3D(0.2)

        self.conv_module_out_1 = Conv3d_module(filters=32)

        self.conv_2_down = tf.keras.layers.Conv3D(
            filters=32,
            kernel_size=(3,3,3),
            strides=(2,2,2),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(self._L2_reg_f)
        )

        self.conv_module_2 = Conv3d_module(filters=64)

        self.conv_module_out_2 = Conv3d_module(filters=64)

        self.conv_3_down = tf.keras.layers.Conv3D(
            filters=64,
            kernel_size=(3,3,3),
            strides=(2,2,2),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(self._L2_reg_f)
        )

        self.conv_module_3 = Conv3d_module(filters=128)
        self.conv_module_out_3 = Conv3d_module(filters=128)

        self.conv_4_down = tf.keras.layers.Conv3D(
            filters=128,
            kernel_size=(3,3,3),
            strides=(2,2,2),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(self._L2_reg_f)
        )

        self.conv_module_4 = Conv3d_module(filters=256)
        self.conv_module_5 = Conv3d_module(filters=256)
        self.conv_module_out_5 = Conv3d_module(filters=256)

    def call(self, inputs: tf.Tensor, **kwargs) -> tuple:
        assert len(inputs.shape) == 5, f'Input tensor should be of 5D dim, given dim was {len(inputs.shape)}'
        x = self.conv_1(inputs)
        x = self.spatial_dropout3d(x)

        x_out1 = self.conv_module_out_1(x)
        x = self.conv_2_down(x_out1)
        
        x = self.conv_module_2(x)
        x_out2 = self.conv_module_out_2(x)
        x = self.conv_3_down(x_out2)

        x = self.conv_module_3(x)
        x_out3 = self.conv_module_out_3(x)
        x = self.conv_4_down(x_out3)

        x = self.conv_module_4(x)
        x = self.conv_module_5(x)
        x_out4 = self.conv_module_out_5(x)

        return x_out1, x_out2, x_out3, x_out4 # out1-> c,h,w,d,32  out2-> c,h,w,d,64  out3-> c,h,w,d,128  out4-> c,h,w,d,256  

    def get_config(self):
        return super(Encoder3D,self).get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

if __name__ == "__main__":
    encoder = Encoder3D(name='enc_1')
    # first call to the `encoder` will create weights
    y = encoder(tf.ones(shape=(1, 160, 192, 32, 3)))

    print("weights:", len(encoder.weights))
    print("trainable weights:", len(encoder.trainable_weights))
    print("config:", encoder.get_config())
    print(f"Y: {y[-1].shape}")