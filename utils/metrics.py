'''
The following scripts contains various Metric evaluation function
to compute the performance of Multi-class 3D-segmentation.
1. Dice Coef
2. Dice Coef for Multi-Class
3. IoU measure
4. IoU measure for Multi-Class
5. Precision
6. Recall

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

def dice_coef(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Sorensen Dice coeffient.
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask
    return: Dice coeff value ranging between [0-1]
    '''
    assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
    assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

    smooth = 1e-15
    # Threshold in order to get a binary segmented map for each class
    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
    y_mask = tf.cast(tf.math.greater(y_mask, 0.5), tf.float32)

    intersection = tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=(1, 2, 3))
    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2, 3)) + smooth
    dice = tf.reduce_mean(((2*intersection+smooth) / union))

    return dice

def dice_coef_multi_class(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Sorensen Dice coeffient for each class C, where C in the segmented channels of Predicted Map and Actual Map (B,H,W,D,C)
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask
    return: dice-> tf.Tensor   Dice coeff value for each class (shape = (1,class))
    '''
    assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
    assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

    smooth = 1e-15
    # Threshold in order to get a binary segmented map for each class
    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
    y_mask = tf.cast(tf.math.greater(y_mask, 0.5), tf.float32)

    intersection = tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=(1, 2, 3))
    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2, 3)) + smooth
    dice = ((2*intersection+smooth) / union)

    return dice


def iou_metric(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Intersection over Union measure
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask
    return: iou->tf.Tensor     IoU measure value ranging between [0-1]
    '''
    assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
    assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

    smooth = 1e-15
    # Threshold in order to get a binary segmented map for each class
    y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
    y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)

    intersection = tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=(1, 2, 3))

    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2, 3)) + smooth

    iou = tf.reduce_mean((intersection)/(union-intersection))

    return iou

def iou_multi_class(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Intersection over Union measure for each class C, where C in the segmented channel of inputs (B,H,W,D,C)
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask
    return: iou->tf.Tensor     IoU measure value for each class (shape = (1,class))
    '''
    assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
    assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

    smooth = 1e-15
    # Threshold in order to get a binary segmented map for each class
    y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
    y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)

    intersection = tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=(1, 2, 3))

    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2, 3)) + smooth

    iou = intersection/(union-intersection)

    return iou

def Precision(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Calculate Precision metric given as (TP/TP+FP). 
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask
    '''
    assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
    assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

    # Threshold in order to get a binary segmented map for each class
    y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
    y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)

    TPs = tf.reduce_sum(y_mask * y_pred, axis=(1,2,3))
    FPs = tf.reduce_sum((1-y_mask) * y_pred, axis=(1,2,3))

    P = tf.reduce_mean(TPs / (TPs+FPs))

    return P

def Recall(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    assert len(y_mask.shape) == 5, f"y_mask should be of rank 5 but got {len(y_mask.shape)} with shape as {y_mask.shape}"
    assert len(y_pred.shape) == 5, f"y_pred should be of rank 5 but got {len(y_pred.shape)} with shape as {y_pred.shape}"

    y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
    y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)

    TPs = tf.reduce_sum(y_mask * y_pred, axis=(1,2,3))
    FNs = tf.reduce_sum(y_mask * (1-y_pred), axis=(1,2,3))

    R = tf.reduce_mean(TPs/(TPs+FNs))

    return R

if __name__ == "__main__":
    #creating random samples
    y_pred = tf.cast(tf.greater(tf.abs(tf.random.normal([1,160,192,128,3])), 0.5), dtype=tf.float32)
    y_mask = tf.cast(tf.greater(tf.abs(tf.random.normal([1,160,192,128,3])), 0.5), dtype=tf.float32)

    dice        =   dice_coef(y_mask=y_mask, y_pred=y_pred)
    multi_dice  =   dice_coef_multi_class(y_mask=y_mask, y_pred=y_pred)
    iou         =   iou_metric(y_mask=y_mask, y_pred=y_pred)
    iou_multi   =   iou_multi_class(y_mask=y_mask, y_pred=y_pred)
    precision   =   Precision(y_mask=y_mask, y_pred=y_pred)
    recall      =   Recall(y_mask=y_mask, y_pred=y_pred)

    tf.print(
        f'Dice coef                 : {dice}\n',
        f'Dice coef for Multi class : {multi_dice}\n',
        f'IoU                       : {iou}\n',
        f'IoU score for Multiclass  : {iou_multi}\n',
        f'Precision                 : {precision}\n',
        f'Recall                    : {recall}\n'
    )
