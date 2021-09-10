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

This script Creates a Data Pipeline for Model to consume.

Read Nifti File -> Cast volume data to tf.Tensor -> Crop the volume -> scale the values between 0-1(for MRI scans) -> Replace the label 4 with 3 in segmented volume -> out
'''

import glob
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split



class TfdataPipeline:
    def __init__(
        self,
        BASE_DATASET_DIR: str,
        IMG_H: int = 150,
        IMG_W: int = 150,
        IMG_D: int = 128,
        batch_size: int = 1,
        split: float = 0.1
    ) -> None:
        self.BASE_DATASET_DIR = BASE_DATASET_DIR
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_D = IMG_D
        self.batch_size = batch_size
        self.split = split
        self.__datasettype = ['train', 'valid', 'test']

        if not os.path.exists(BASE_DATASET_DIR):
            print(
                f"[Error] Dataset directory {BASE_DATASET_DIR} does not exist!")
            sys.exit()

    def load_and_split_dataset_files(self, path: str, split: float):
        '''
        Loads the path name of each MRI volumnes and Segmenated Volume
        and split it in ratio 80:10:10(train:valid:test)
        '''
        assert 0.0 < split < 1.0, "The split value should be between 0 and 1"
        
        list_t2 = sorted(glob.glob(os.path.join(path,"*TrainingData/*/*t2.nii")))
        list_t1ce = sorted(glob.glob(os.path.join(path,"*TrainingData/*/*t1ce.nii")))
        list_flair = sorted(glob.glob(os.path.join(path,"*TrainingData/*/*flair.nii")))

        list_seg = sorted(glob.glob(os.path.join(path,"*TrainingData/*/*seg.nii")))
        
        assert len(list_t2) == len(list_t1ce) == len(list_seg) == len(list_flair), "Number of MRI and Segmented files are not equal, Please check the names of the files"

        total_items = len(list_t1ce)
        valid_items = int(total_items * split)
        test_items = int(total_items * split)

        train_t2, valid_t2 = train_test_split(list_t2, test_size=valid_items, random_state=12)
        train_t1ce, valid_t1ce = train_test_split(list_t1ce, test_size=valid_items, random_state=12)
        train_flair, valid_flair = train_test_split(list_flair, test_size=valid_items, random_state=12)

        train_y, valid_y = train_test_split(list_seg, test_size=valid_items, random_state=12)

        train_t2, test_t2 = train_test_split(train_t2, test_size=test_items, random_state=12)
        train_t1ce, test_t1ce = train_test_split(train_t1ce, test_size=test_items, random_state=12)
        train_flair, test_flair = train_test_split(train_flair, test_size=test_items, random_state=12)

        train_y, test_y = train_test_split(train_y, test_size=test_items, random_state=12)

        train_x = [train_t2, train_t1ce, train_flair]
        valid_x = [valid_t2, valid_t1ce, valid_flair]
        test_x  = [test_t2, test_t1ce, test_flair]

        return (train_x, train_y), (valid_x, valid_y), (test_x,test_y)


if __name__ == "__main__":
    tfdatapipeline = TfdataPipeline(BASE_DATASET_DIR="BraTS_2020/")
    (train_x, train_y), (valid_x, valid_y), (test_x,test_y) = tfdatapipeline.load_and_split_dataset_files(path="BraTS_2020", split=0.1)

    tf.print(len(train_x[0]))
    tf.print(len(valid_x[0]))
    tf.print(test_x[0])


