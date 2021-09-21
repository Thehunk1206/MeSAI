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
from sampling import Sampling
from group_norm import GroupNormalization
from conv3d_module import Conv3d_module

class VAE_decoder(tf.keras.layers.Layer):
    def __init__(self, name:str, feat_h:int, feat_w:int, feat_d:int, feat_c:int, **kwargs):
        super(VAE_decoder, self).__init__(name=name,  **kwargs)

        self.feat_h = feat_h
        self.feat_w = feat_w
        self.feat_d = feat_d
        self.feat_c = feat_c
        self._L2_reg_f = 1e-5

        self.group_norm = GroupNormalization(groups=8)
        self.relu1 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv3D(
            filters=16,
            kernel_size=(3,3,3),
            strides=2,
            padding='same',
            kernel_regularizer= tf.keras.regularizers.L2(l2=self._L2_reg_f)
        )
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(units=256, kernel_regularizer=tf.keras.regularizers.L2(l2=self._L2_reg_f))

        self.dense_z_mean = tf.keras.layers.Dense(units=128)
        self.dense_z_var  = tf.keras.layers.Dense(units=128)

        self.sampling = Sampling(name='sampling_1')

        self.dense2 = tf.keras.layers.Dense(units=((self.feat_h//2) * (self.feat_w//2) * (self.feat_d//2) * (self.feat_c//16)))
        self.relu2 = tf.keras.layers.ReLU()
        self.reshape = tf.keras.layers.Reshape(target_shape=((self.feat_h//2), (self.feat_w//2), (self.feat_d//2), (self.feat_c//16)))

        self.conv2 = tf.keras.layers.Conv3D(
            filters=128,
            kernel_size=(1,1,1),
            strides=(1,1,1),
            padding='same',
            kernel_regularizer= tf.keras.regularizers.L2(l2=self._L2_reg_f)
        )
        self.upsample_conv2 = tf.keras.layers.UpSampling3D(size=(2,2,2))

        self.conv3 = tf.keras.layers.Conv3D(
            filters=128,
            kernel_size=(1,1,1),
            strides=(1,1,1),
            padding='same',
            kernel_regularizer= tf.keras.regularizers.L2(l2=self._L2_reg_f)
        )
        self.upsample_conv3 = tf.keras.layers.UpSampling3D(size=(2,2,2))
        self.conv_module_1 = Conv3d_module(filters=128)

        self.conv4 = tf.keras.layers.Conv3D(
            filters=64,
            kernel_size=(1,1,1),
            strides=(1,1,1),
            padding='same',
            kernel_regularizer= tf.keras.regularizers.L2(l2=self._L2_reg_f)
        )
        self.upsample_conv4 = tf.keras.layers.UpSampling3D(size=(2,2,2))
        self.conv_module_2 = Conv3d_module(filters=64)

        self.conv5 = tf.keras.layers.Conv3D(
            filters=32,
            kernel_size=(1,1,1),
            strides=(1,1,1),
            padding='same',
            kernel_regularizer= tf.keras.regularizers.L2(l2=self._L2_reg_f)
        )
        self.upsample_conv5 = tf.keras.layers.UpSampling3D(size=(2,2,2))
        self.conv_module_3 = Conv3d_module(filters=32)

        self.conv6 = tf.keras.layers.Conv3D(
            filters=16,
            kernel_size=(3,3,3),
            strides=(1,1,1),
            padding='same',
            kernel_regularizer= tf.keras.regularizers.L2(l2=self._L2_reg_f)
        )

        self.conv7_out = tf.keras.layers.Conv3D(
            filters=3,
            kernel_size=(1,1,1),
            strides=(1,1,1),
            padding='same'
        )

    def call(self, inputs:tf.Tensor, **kwargs)->tuple:
        '''
        inputs =  output from encoder3d's last layer 
        '''
        # GroupNorm + conv(16)
        x           = self.group_norm(inputs)
        x           = self.relu1(x)
        x           = self.conv1(x)
        
        # flatten + z_mean + z_var + sampling
        x_flatten   = self.flatten(x)
        x           = self.dense1(x_flatten)
        z_mean_out  = self.dense_z_mean(x)
        z_var_out   = self.dense_z_var(x)
        x           = self.sampling(z_mean=z_mean_out, z_log_var=z_var_out)

        # dense + reshape to 3d volume
        x           = self.dense2(x)
        x           = self.relu2(x)
        x           = self.reshape(x)

        #conv2(256) + upsample -> (b,h,w,d,256)
        x           = self.conv2(x)
        x           = self.upsample_conv2(x)

        #conv3(128) + upsample + Conv_module -> (b,h,w,d,128)
        x           = self.conv3(x)
        x           = self.upsample_conv3(x)
        x           = self.conv_module_1(x)

        #conv4(64) + upsample + Conv_module -> (b,h,w,d,64)
        x           = self.conv4(x)
        x           = self.upsample_conv4(x)
        x           = self.conv_module_2(x)

        #conv5(32) + upsample + Conv_module -> (b,h,w,d,32)
        x           = self.conv5(x)
        x           = self.upsample_conv5(x)
        x           = self.conv_module_3(x)

        #conv6(16) + conv7_out(3) -> reconstructed output
        x           = self.conv6(x)
        x_vae_out = self.conv7_out(x)

        return z_mean_out, z_var_out, x_vae_out

    def get_config(self):
        config = super().get_config()
        config.update({
            'feat_h': self.feat_h,
            'feat_w': self.feat_w,
            'feat_d': self.feat_d,
            'feat_c': self.feat_c
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

if __name__ == "__main__":

    x = tf.ones(shape=(1, 20, 24, 16, 256))
    _, h, w, d, c  = tf.shape(x)

    vae = VAE_decoder(name='enc_1', feat_h=h, feat_w=w, feat_d=d, feat_c=c)
    # first call to the `vae` will create weights
    y = vae(x)

    print("weights:", len(vae.weights))
    print("trainable weights:", len(vae.trainable_weights))
    print("config:", vae.get_config())
    print(f"Y: {y[-1].shape}")