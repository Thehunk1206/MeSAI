'''
The following script contains loss fucntions for multiclass - 3D segmentation
1)Wieghted BCE and Dice loss
2)Soft Dice Loss
3)Focal Tversky Loss
4)VAE(variational autoencoder loss)

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


class WBCEDICELoss(tf.keras.losses.Loss):
    def __init__(self, name: str,):
        super(WBCEDICELoss, self).__init__(name=name)

    @tf.function
    def call(self, y_mask: tf.Tensor, y_pred: tf.Tensor):
        assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
        assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

        bce_dice_weights = 1 + 5 * tf.abs(tf.nn.avg_pool3d(y_mask, ksize=21, strides=1, padding="SAME")-y_mask)

        # weighted BCE loss
        bce_loss = tf.keras.losses.BinaryCrossentropy()(y_mask, y_pred)
        wbce_loss = tf.reduce_sum(bce_loss*bce_dice_weights, axis=(1, 2, 3)) / tf.reduce_sum(bce_dice_weights + 1e-8, axis=(1, 2, 3))

        # weighted DICE loss
        y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32) #thresholding

        inter = tf.reduce_sum((y_pred * y_mask) * bce_dice_weights, axis=(1, 2, 3))
        union = tf.reduce_sum((y_pred + y_mask) * bce_dice_weights, axis=(1, 2, 3))
        wdice_loss = 1 - ((2*inter) / union+1e-15)

        weighted_bce_dice_loss = tf.reduce_mean(wbce_loss + wdice_loss)
        return weighted_bce_dice_loss

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)



class SoftDiceLoss(tf.keras.losses.Loss):
    def __init__(self, name: str,):
        super(SoftDiceLoss, self).__init__(name=name)

    @tf.function
    def call(self, y_mask: tf.Tensor, y_pred: tf.Tensor):
        assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
        assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

        inter = tf.reduce_sum((y_pred * y_mask), axis=(1, 2, 3, 4))
        union = tf.reduce_sum(tf.square(y_pred) + tf.square(y_mask), axis=(1, 2, 3, 4))

        soft_dice_loss = 1 - ((2*inter) / union+1e-15)

        return soft_dice_loss

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

class FocalTverskyLoss(tf.keras.losses.Loss):

    def __init__(self, name:str, alpha:float = 0.7, gamma:int = 3, smooth:float = 1e-8):
        super(FocalTverskyLoss, self).__init__(name=name)
        self.alpha = alpha # weight for false negatives and (1-aplha will be for false positives)
        self.gamma = gamma 
        self.smooth = smooth # numeric stability and to avoid divide by zero error
    
    @tf.function
    def call(self, y_mask:tf.Tensor, y_pred:tf.Tensor) -> tf.Tensor:
        assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
        assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

        inter = tf.reduce_sum((y_mask * y_pred), axis=(1,2,3))

        false_ps = tf.reduce_sum((1-y_mask) * y_pred, axis = (1,2,3))
        false_ns = tf.reduce_sum(y_mask * (1-y_pred), axis = (1,2,3))

        denom = inter +  (self.alpha * false_ns) + ((1-self.alpha) * false_ps) + self.smooth

        tversky_index = tf.reduce_mean((inter+ self.smooth) / denom)

        focal_tversky_loss = tf.pow((1-tversky_index), (1/self.gamma))

        return focal_tversky_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'smoothing': self.smooth
        })
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)


class L2loss(tf.keras.losses.Loss):
    def __init__(self, name: str):
        super(L2loss, self).__init__(name=name)

    @tf.function
    def call(self, x_vol: tf.Tensor, x_reconstructed: tf.Tensor):
        assert len(x_vol.shape) == 5, f"x_vol should be of rank 5 but got {len(x_vol.shape)} with shape as {x_vol.shape}"
        assert len(x_reconstructed.shape) == 5, f"x_reconstructed should be of rank 5 but got {len(x_reconstructed.shape)} with shape as {x_reconstructed.shape}"

        l2_loss = tf.reduce_mean(tf.square(x_vol - x_reconstructed))
        
        return l2_loss

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)


if __name__ == "__main__":
    y_pred = tf.cast(tf.greater(tf.abs(tf.random.normal([1,160,192,128,3])), 0.5), dtype=tf.float32)
    y_mask = tf.cast(tf.greater(tf.abs(tf.random.normal([1,160,192,128,3])), 0.5), dtype=tf.float32)


    soft_dice_loss  = SoftDiceLoss(name='sotf_dice_loss') 
    w_bce_dice_loss = WBCEDICELoss(name='w_bce_dice_loss')
    focal_tversky_loss = FocalTverskyLoss(name='FTL', gamma=1)
    l2_loss = L2loss(name='l2_loss')

    total_soft_dice_loss    =  soft_dice_loss(y_mask,y_pred)
    total_w_bce_dice_loss   =  w_bce_dice_loss(y_mask,y_pred)
    total_ftl               =  focal_tversky_loss(y_mask,y_pred)
    total_l2_loss          =  l2_loss(y_mask, y_pred)

    tf.print(
        f"soft_dice_loss    : {total_soft_dice_loss}\n",
        f"w_bce_dice_loss   : {total_w_bce_dice_loss}\n",
        f"focal Tversky loss: {total_ftl}\n",
        f"vae loss          : {total_l2_loss}\n",
        )