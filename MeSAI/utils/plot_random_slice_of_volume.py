'''
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataset import TfdataPipeline

import tensorflow as tf
import matplotlib.pyplot as plt
import random

PATH_TO_DATASET = "BraTS_2020/"

if __name__ == "__main__":
    tfdatapipeline = TfdataPipeline(BASE_DATASET_DIR=PATH_TO_DATASET, IMG_H=128, IMG_W=160   , IMG_D=128)
    train_data = tfdatapipeline.data_loader()

    data = next(train_data.as_numpy_iterator())
    seg_vol = tf.squeeze(data[1])
    img_vol = tf.squeeze(data[0])

    random_slice = random.randint(60, seg_vol.shape[2])
    plt.figure(figsize=(8,8))

    plt.subplot(221)
    plt.imshow(img_vol[:,:,random_slice,0], cmap='gray')
    plt.title('Image t2')

    plt.subplot(222)
    plt.imshow(img_vol[:,:,random_slice,1], cmap='gray')
    plt.title('Image t1ce')

    plt.subplot(223)
    plt.imshow(img_vol[:,:,random_slice,2], cmap='gray')
    plt.title('Image flair')

    plt.subplot(224)
    plt.imshow(seg_vol[:,:,random_slice,0], cmap='Accent_r')
    plt.imshow(seg_vol[:,:,random_slice,1], cmap='Greens', alpha=0.5)
    plt.imshow(seg_vol[:,:,random_slice,2], cmap='Oranges',alpha=0.7)
    plt.title('Tumor Mask')
    

    plt.show()
