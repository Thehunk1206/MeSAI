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
    def __init__(self, **kwargs):
        super(Encoder3D, self).__init__(**kwargs)

        self.conv3d_1 = tf.keras.layers.Conv3D(
            filters=32,
            kernel_size= (3,3,3),
            strides=(1,1,1),
            padding='same'
        )

        self.spatial_dropout3d = tf.keras.layers.SpatialDropout3D(0.2)

        self.conv3d_module_out_1 = Conv3d_module(filters=32)

        self.conv3d_2_down = tf.keras.layers.Conv3D(
            filters=32,
            kernel_size=32,
            strides=(2,2,2),
            padding='same',
        )

        self.conv3d_module_2 = Conv3d_module(filters=64)

        self.conv3d_module_out_2 = Conv3d_module(filters=64)

        self.conv3d_3_down = tf.keras.layers.Conv3D(
            filters=64,
            kernel_size=32,
            strides=(2,2,2),
            padding='same',
        )

        self.conv3d_module_3 = Conv3d_module(filters=128)
        self.conv3d_module_out_3 = Conv3d_module(filters=128)

        self.conv3d_4_down = tf.keras.layers.Conv3D(
            filters=128,
            kernel_size=32,
            strides=(2,2,2),
            padding='same',
        )

        self.conv3d_module_4 = Conv3d_module(filters=256)
        self.conv3d_module_5 = Conv3d_module(filters=256)
        self.conv3d_module_out_5 = Conv3d_module(filters=256)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        assert len(inputs.shape) == 5, f'Input tensor should be of 5D dim, given dim was {inputs.shape}'
        x = self.conv3d_1(inputs)
        x = self.spatial_dropout3d(x)

        x_out1 = self.conv3d_module_out_1(x)
        x = self.conv3d_2_down(x_out1)
        
        x = self.conv3d_module_2(x)
        x_out2 = self.conv3d_module_out_2(x)
        x = self.conv3d_3_down(x_out2)

        x = self.conv3d_module_3(x)
        x_out3 = self.conv3d_module_out_3(x)
        x = self.conv3d_4_down(x_out3)

        x = self.conv3d_module_4(x)
        x = self.conv3d_module_5(x)
        x_out4 = self.conv3d_module_out_5(x)

        return x_out1, x_out2, x_out3, x_out4

    def get_config(self):
        return super(Encoder3D,self).get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

if __name__ == "__main__":
    encoder = Encoder3D()
    # first call to the `encoder` will create weights
    y = encoder(tf.ones(shape=(1, 40, 48, 32, 3)))

    print("weights:", len(encoder.weights))
    print("trainable weights:", len(encoder.trainable_weights))
    print("config:", encoder.get_config())
    print(f"Y: {y[-1].shape}")