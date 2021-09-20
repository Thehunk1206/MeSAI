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

class Sampling(tf.keras.layers.Layer):
    def __init__(self, name:str, **kwargs):
        super(Sampling, self).__init__(name=name,  **kwargs)

    def call(self, z_mean: tf.Tensor, z_log_var: tf.Tensor, **kwargs)->tf.Tensor:
        assert len(z_mean.shape) == len(z_log_var.shape) == 2,  f'Shape of z_mean should be (batch,dim), given shape for z_mean: {z_mean.shape}' \
                                                                f'and for z_log_var: {z_log_var.shape}'
        assert z_mean.shape == z_log_var.shape, f'shapes are incomaptible, shape of z_mean: {z_mean.shape}, shape of z_log_var: {z_log_var.shape} '

        batch = tf.shape(z_mean)[0]
        dims  = tf.shape(z_mean)[1]

        epsilon = tf.random.normal(shape=(batch,dims))

        return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        return super(Sampling, self).get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)
    
if __name__ == "__main__":
    sampling = Sampling(name='sampling_1')
    y = sampling(z_mean=tf.ones(shape=(5, 128)), z_log_var= tf.random.normal([5,128]))

    print("weights:", len(sampling.weights))
    print("trainable weights:", len(sampling.trainable_weights))
    print("config:", sampling.get_config())
    print(f"Y: {y.shape}")

