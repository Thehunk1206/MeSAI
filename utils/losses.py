'''
The following script contains loss fucntions for multiclass - 3D segmentation
1)Wieghted BCE and Dice loss
2) VAE(variational autoencoder loss)
3)KL divergence loss
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

from dataset import TfdataPipeline
import tensorflow as tf


class SoftDiceLoss(tf.keras.losses.Loss):
    def __init__(self, name: str,):
        super(SoftDiceLoss, self).__init__(name=name)

    @tf.function
    def call(self, y_mask: tf.Tensor, y_pred: tf.Tensor):
        assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
        assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

        inter = tf.reduce_sum((y_pred * y_mask), axis=(1, 2, 3))
        union = tf.reduce_sum(tf.square(y_pred) + tf.square(y_mask), axis=(1, 2, 3))

        soft_dice_loss = 1 - ((2*inter) / union+1e-15)

        return soft_dice_loss

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)



class VAE_loss(tf.keras.losses.Loss):
    def __init__(self, name: str, weight_l2: float = 0.1, weight_kl:float = 0.1):
        super(VAE_loss, self).__init__(name=name)
        self.weight_l2 = weight_l2
        self.weight_kl = weight_kl

    @tf.function
    def call(self, y_mask: tf.Tensor,y_pred: tf.Tensor, z_mean:tf.Tensor, z_var:tf.Tensor):
        assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
        assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

        B,H,W,D,C = y_mask.shape
        N = B*H*W*D*C

        l2_loss = tf.reduce_mean(tf.square(y_mask - y_pred))
        kl_loss = (1/N) * tf.reduce_sum(tf.math.exp(z_var) + tf.square(z_mean) -1.0 - z_var, axis=0)

        vae_loss = self.weight_l2 * l2_loss + self.weight_kl * kl_loss

        return vae_loss

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)


if __name__ == "__main__":

    y_pred = tf.abs(tf.random.normal([1,160,192,128,3]))
    y_mask = tf.abs(tf.random.normal([1,160,192,128,3]))

    soft_dice_loss = SoftDiceLoss(name='sotf_dice_loss') 

    total_soft_dice_loss =  soft_dice_loss(y_mask,y_pred)

    tf.print(
        f"soft_dice_loss: {total_soft_dice_loss}\n",
        )
    # tf.print(y_mask.shape)