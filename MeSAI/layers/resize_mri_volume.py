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

from __future__ import annotations

import tensorflow as tf


class ResizeVolume(tf.keras.layers.Layer):
    '''
    A preprocessing layer which resizes volume.

    Args:
        target_h: int, 
            Desired height of the volume.
        target_w: int,
            Desired width of the volume. 
        target_d: int,
            Desired depth of the volume.
        interpolation: tf.image.ResizeMethod,
            Interpolation method to use for resizing.
    '''
    def __init__(
        self,
        target_h: int = 128,
        target_w: int = 128,
        target_d: int = 128,
        interpolation: tf.image.ResizeMethod = tf.image.ResizeMethod.BICUBIC,
        **kwargs
    ) -> None:
        super(ResizeVolume, self).__init__(**kwargs)
        self.target_h = target_h
        self.target_w = target_w
        self.target_d = target_d
        self.interpolation = interpolation
    
    def _resize_by_axis(self, volume: tf.Tensor, target_h: int,target_w: int, ax:int)-> tf.Tensor:
        '''
        Resize the volume along the given axis.
        args:
            volume: tf.Tensor of shape (batch_size, h, w, d)    
            target_h: int   Desired height of the volume
            target_w: int   Desired width of the volume
            ax: int         Axis along which to resize the volume
        return:
            resized_volume_along_axis: tf.Tensor
        '''
        assert ax in [1,2,3], 'Axis must be 1,2 or 3'
        assert volume.shape.rank == 4 or volume.shape.rank == 5, 'Volume must be 4D or 5D'

        resized_list = []
        unstack_img_depth_list = tf.unstack(volume, axis = ax)
        for img in unstack_img_depth_list:
            resized_list.append(tf.image.resize(img, [target_h, target_w], method = self.interpolation))
        stack_img = tf.stack(resized_list, axis=ax)

        return stack_img
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        assert inputs.shape.rank == 4 or inputs.shape.rank == 5, 'Volume must be 4D or 5D with batch_first'

        if inputs.shape.rank == 3:
            # add a batch dimension
            inputs = tf.expand_dims(inputs, axis=0)

        resized_along_depth = self._resize_by_axis(inputs, self.target_h, self.target_w, ax=3)
        resized_along_width = self._resize_by_axis(resized_along_depth, self.target_h, self.target_d, ax=2)

        return resized_along_width

    def get_config(self) -> dict:
        config = super(ResizeVolume, self).get_config()
        config.update({
            'target_h': self.target_h,
            'target_w': self.target_w,
            'target_d': self.target_d,
            'interpolation': self.interpolation
        })
        return config
    
    def from_config(cls, config: dict) -> ResizeVolume:
        return cls(**config)


if __name__ == '__main__':
    volume = tf.random.uniform([1, 40,48, 32,3])

    tf.print(volume.shape)
    tf.print("Resizing...")
    resized_volume = ResizeVolume(target_h=160, target_w=192, target_d=128)(volume)
    print(resized_volume.shape)
