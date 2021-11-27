'''
This module Creates a tf Data Pipeline for Model to consume.


All BraTS multimodal scans are available as NIfTI files (.nii.gz) and described as 
a) native (T1) and b) post-contrast T1-weighted (T1ce), c) T2-weighted (T2),and 
d)Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different 
clinical protocols and various scanners from multiple (n=19) institutions, mentioned as data contributors here.

Here we stack T2, T1ce, and flair volumes on axis[-1] so we get BxHxWxDx3 dimension volume.

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
        IMG_H: int = 160,
        IMG_W: int = 192,
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

    def _load_and_split_dataset_files(self, path: str, split: float):
        '''
        Loads the path name of each MRI volumnes and Segmenated Volume
        and split it in ratio 80:10:10(train:valid:test)
        '''
        assert 0.0 < split < 1.0, "The split value should be between 0 and 1"
        
        list_t2    =    sorted(glob.glob(os.path.join(path,"*TrainingData/*/*t2.nii")))
        list_t1ce  =    sorted(glob.glob(os.path.join(path,"*TrainingData/*/*t1ce.nii")))
        list_flair =    sorted(glob.glob(os.path.join(path,"*TrainingData/*/*flair.nii")))

        list_seg   =    sorted(glob.glob(os.path.join(path,"*TrainingData/*/*seg.nii")))
        
        assert len(list_t2) == len(list_t1ce) == len(list_seg) == len(list_flair), \
        "Number of MRI and Segmented files are not equal, Please check the names of the files"

        total_items              =    len(list_t1ce)
        valid_items              =    int(total_items * split)
        test_items               =    int(total_items * split)
        
        train_t2, valid_t2       =    train_test_split(list_t2, test_size=valid_items, random_state=12)
        train_t1ce, valid_t1ce   =    train_test_split(list_t1ce, test_size=valid_items, random_state=12)
        train_flair, valid_flair =    train_test_split(list_flair, test_size=valid_items, random_state=12)

        train_y, valid_y         =    train_test_split(list_seg, test_size=valid_items, random_state=12)

        train_t2, test_t2        =    train_test_split(train_t2, test_size=test_items, random_state=12)
        train_t1ce, test_t1ce    =    train_test_split(train_t1ce, test_size=test_items, random_state=12)
        train_flair, test_flair  =    train_test_split(train_flair, test_size=test_items, random_state=12)

        train_y, test_y          =    train_test_split(train_y, test_size=test_items, random_state=12)

        train_x                  =    [zip_train for zip_train in zip(train_t2, train_t1ce, train_flair)]
        valid_x                  =    [zip_val   for zip_val   in zip(valid_t2, valid_t1ce, valid_flair)]
        test_x                   =    [zip_test  for zip_test  in zip(test_t2, test_t1ce, test_flair)]

        # train_x,valid_x and test_x is  a list of tuples containing t2,t1ce and flair(Fluid Attenuated Inversion Recovery) NIfTI files (.nii)
        return (train_x, train_y), (valid_x, valid_y), (test_x,test_y)
    
    def _read_volumes(self, path:tf.Tensor) -> tf.Tensor:
        path = path.numpy().decode('ascii')
        assert os.path.exists(path), f"file {path} does not exist"

        vol = nib.load(path)
        vol = tf.cast(vol.get_fdata(), tf.float32)

        return vol

    def _crop_volume(self, volume:tf.Tensor) -> tf.Tensor:
        if len(volume.shape) == 4:
            volume = tf.expand_dims(volume, axis=0)
        elif len(volume.shape) == 3:
            volume = tf.expand_dims(volume, axis=-1)
            volume = tf.expand_dims(volume, axis=0)
        elif len(volume.shape) < 3:
            tf.print("Volume can't have 2 or less dims")
        
        crop_top_h          = int(tf.math.ceil((volume.shape[1]-self.IMG_H)/2))
        crop_bottom_h       = int(tf.math.floor((volume.shape[1]-self.IMG_H)/2))
        crop_top_w          = int(tf.math.ceil((volume.shape[2]-self.IMG_W)/2))
        crop_bottom_w       = int(tf.math.floor((volume.shape[2]-self.IMG_W)/2))
        crop_top_d          = int(tf.math.ceil((volume.shape[3]-self.IMG_D)/2))
        crop_bottom_d       = int(tf.math.floor((volume.shape[3]-self.IMG_D)/2))

        cropped_volume = tf.keras.layers.Cropping3D(cropping=((crop_top_h,crop_bottom_h) , (crop_top_w,crop_bottom_w), (crop_top_d,crop_bottom_d)))(volume)
        cropped_volume = tf.squeeze(cropped_volume, axis=0)

        return cropped_volume

    
    def _read_and_combine_volumes(self, path_l: tf.Tensor)-> tf.Tensor:
        assert len(path_l) == 3
        t2_volume           = self._read_volumes(path_l[0])
        t1ce_volume         = self._read_volumes(path_l[1])
        flair_volume        = self._read_volumes(path_l[2])

        combined_vol        = tf.stack([t2_volume,t1ce_volume,flair_volume], axis=-1)
        combined_vol        = (combined_vol - tf.reduce_min(combined_vol)) / (tf.reduce_max(combined_vol) - tf.reduce_min(combined_vol))

        cropped_combine_vol = self._crop_volume(combined_vol)

        return cropped_combine_vol

    def _read_seg_volumes(self, path:tf.Tensor) -> tf.Tensor:

        seg_vol = self._read_volumes(path=path)
        
        # convert to multi-channel label 
        channel_label = []
        unique_label, _ = tf.unique(tf.reshape(seg_vol, [-1]))
        for label in unique_label:
            if label.numpy() != 0:
                label = seg_vol == label.numpy()
                channel_label.append(label)
        
        # Hardcoded for 3 labels
        if len(channel_label) == 2:
            channel_label.append(tf.cast(tf.zeros_like(seg_vol), tf.bool))
        
        final_seg_vol = tf.stack(channel_label, axis=-1)
        final_seg_vol = tf.cast(final_seg_vol, dtype=tf.float32)
        
        cropped_seg_vol = self._crop_volume(final_seg_vol)

        return cropped_seg_vol
    
    def _map_dataset(self, mri_path:tf.Tensor, seg_path:tf.Tensor)->tuple:
        mri_path = tf.squeeze(mri_path)
        def _map(mri_path,seg_path):
            X = self._read_and_combine_volumes(mri_path)
            Y = self._read_seg_volumes(seg_path)
            return X,Y
        X,Y = tf.py_function(func=_map,inp=[mri_path,seg_path], Tout=[tf.float32, tf.float32])
        return X,Y
    
    def _tf_dataset(self, mri_path: list, seg_path: list) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((mri_path, seg_path))

        dataset = dataset.map(
            self._map_dataset, num_parallel_calls=tf.data.AUTOTUNE)#.cache()
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
    
    def data_loader(self, dataset_type: str = 'train') -> tf.data.Dataset:
        '''
        dataset_type should be in ['train','valid','test']
        '''
        if dataset_type not in self.__datasettype:
            print(
                f"[Error] invalid option {dataset_type} option should be in {self.__datasettype}")
            sys.exit()
        (train_x, train_y), (valid_x, valid_y), (test_x,
                                                 test_y) = self._load_and_split_dataset_files(self.BASE_DATASET_DIR, split=self.split)

        if dataset_type == 'train':
            train_dataset = self._tf_dataset(train_x, train_y)
            return train_dataset
        elif dataset_type == 'valid':
            valid_dataset = self._tf_dataset(valid_x, valid_y)
            return valid_dataset
        elif dataset_type == 'test':
            test_dataset = self._tf_dataset(test_x, test_y)
            return test_dataset

#testing
if __name__ == "__main__":
    

    tfdatapipeline = TfdataPipeline(BASE_DATASET_DIR="BraTS_2020/", IMG_H=128, IMG_W=172, IMG_D=128)
    train_data = tfdatapipeline.data_loader(dataset_type='train')

    for img_vol, seg_vol in train_data.take(2):
        tf.print(
            f"Segmented VOL shape:  {seg_vol.shape} \n",
            f"MRI VOL shape:        {img_vol.shape} \n",
        )