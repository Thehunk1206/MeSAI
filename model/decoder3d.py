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

class Decoder3D(tf.keras.layers.Layer):
    def __init__(self, name:str, number_of_class:int, **kwargs):
        super(Decoder3D, self).__init__(name=name, **kwargs)
        
        self.number_of_class = number_of_class
        self._L2_reg_f = 1e-5

        self.conv_1 = tf.keras.layers.Conv3D(
            filters=128,
            kernel_size=(1,1,1),
            strides=(1,1,1),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(self._L2_reg_f)
        )
        self.upsample_1 = tf.keras.layers.UpSampling3D(size=(2,2,2))
        self.add_1 = tf.keras.layers.Add()
        self.conv_module_1 = Conv3d_module(filters=128)

        self.conv_2 = tf.keras.layers.Conv3D(
            filters=64,
            kernel_size=(1,1,1),
            strides=(1,1,1),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(self._L2_reg_f)
        )
        self.upsample_2 = tf.keras.layers.UpSampling3D(size=(2,2,2))
        self.add_2 = tf.keras.layers.Add()
        self.conv_module_2 = Conv3d_module(filters=64)

        self.conv_3 = tf.keras.layers.Conv3D(
            filters=32,
            kernel_size=(1,1,1),
            strides=(1,1,1),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(self._L2_reg_f)
        )
        self.upsample_3 = tf.keras.layers.UpSampling3D(size=(2,2,2))
        self.add_3 = tf.keras.layers.Add()
        self.conv_module_3 = Conv3d_module(filters=32)

        self.conv_4 = tf.keras.layers.Conv3D(
            filters=16,
            kernel_size=(3,3,3),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(self._L2_reg_f)
        )

        self.conv_5 = tf.keras.layers.Conv3D(
            filters=number_of_class,
            kernel_size=(1,1,1),
            strides=(1,1,1),
        )
    
    def call(self, inputs:tuple, **kwargs):
        '''
        input here will be a tuple from Encder. 
        This Tuple contains all low to high level 
        feature (out_1, out_2, out_3, out_4).
        (32channels, 64channels, 128channels, 256channels)
        '''
        x = self.conv_1(inputs[-1])
        x = self.upsample_1(x)
        x = self.add_1([x, inputs[-2]])
        x = self.conv_module_1(x)       # out 128 channels

        x = self.conv_2(x)
        x = self.upsample_2(x)
        x = self.add_2([x, inputs[1]])
        x = self.conv_module_2(x)       # out 64 channels

        x = self.conv_3(x)
        x = self.upsample_3(x)
        x = self.add_3([x, inputs[0]])
        x = self.conv_module_3(x)       # out 32 channels

        x = self.conv_4(x)              # out 16 channels

        x = self.conv_5(x)              # final output with channel as class
        x_out = tf.sigmoid(x)

        return x_out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'number of class' : self.number_of_class
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return super().from_config(config)


if __name__ == "__main__":
    decoder = Decoder3D(name='decoder_1', number_of_class=3)
    # first call to the `decoder` will create weights
    feature_1 = tf.ones(shape=(1,40,48,32,32))
    feature_2 = tf.ones(shape=(1,20,24,16,64))
    feature_3 = tf.ones(shape=(1,10,12,8,128))
    feature_4 = tf.ones(shape=(1,5,6,4,256))

    inputs = (feature_1, feature_2, feature_3, feature_4)

    y = decoder(inputs)

    print("weights:", len(decoder.weights))
    print("trainable weights:", len(decoder.trainable_weights))
    print("config:", decoder.get_config())
    print(f"Y: {y.shape}")

