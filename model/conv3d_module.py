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
from model.group_norm import GroupNormalization

class Conv3d_module(tf.keras.layers.Layer):
    def __init__(self, 
                filters:int,
                kernel_size: tuple = (3,3,3),
                stride:tuple = (1,1,1),
                padding: str = 'same',
                **kwargs
    ):
        super(Conv3d_module, self).__init__(**kwargs)
        assert len(kernel_size) == 3
        assert len(stride) == 3
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv3d_1 = tf.keras.layers.Conv3D(
            filters=self.filters,
            kernel_size=(1,1,1),
            strides=self.stride,
            padding=self.padding,
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-5)
        )

        self.group_norm_1 = GroupNormalization(groups=8, axis=-1)

        self.relu1 = tf.keras.layers.ReLU()

        self.conv3d_2 = tf.keras.layers.Conv3D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-5)
        )

        self.group_norm_2 = GroupNormalization(groups=8, axis=-1)

        self.relu2 = tf.keras.layers.ReLU()

        self.conv3d_3 = tf.keras.layers.Conv3D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-5)
        )

        self.add = tf.keras.layers.Add()

    def call(self, inputs:tf.Tensor,training:bool = False, **kwargs) -> tf.Tensor:
        x_shortcut = self.conv3d_1(inputs) 
        x = self.group_norm_1(x_shortcut, training = training)
        x = self.relu1(x)
        x = self.conv3d_2(x)
        x = self.group_norm_2(x, training= training)
        x = self.relu2(x)
        x = self.conv3d_3(x)
        output = self.add([x,x_shortcut])

        return output
    
    def get_config(self):
        config = super(Conv3d_module, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return super().from_config(config)
    

if __name__ == "__main__":
    cm = Conv3d_module(32)
    # first call to the `cm` will create weights
    y = cm(tf.ones(shape=(2, 32, 32, 32, 128)))

    print("weights:", len(cm.weights))
    print("trainable weights:", len(cm.trainable_weights))
    print("config:", cm.get_config())
    print(f"Y: {y.shape}")