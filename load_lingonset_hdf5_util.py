# -*- coding: utf-8 -*-

"""
Created on Tue Feb  9 10:12:39 2021
Utilit functions to load a lingon dataset stored in a HDF5-file as   

Parmeters:
---------
    file_name : HDF5-file file name (file_name) to pe open  
@author: ISOFT
"""
import numpy as np
import h5py



def load_lingonset_hdf5(hdf5_file, file_path = 'D:/OneDrive - Isoft Services AB/Bärplockning/deep_models/Create_hdf5_datasets/data_sets/' ):
    
    hdf5_file = file_path + hdf5_file
    dataset = h5py.File(hdf5_file, "r")
    dataset_x_orig = np.array(dataset["data_set_x"][:]) # your test set feature
    dataset_y = np.array(dataset["data_set_y"][:]) # your test set feature
    dataset_images = np.array(dataset["data_set_images"][:]) #List of images
    dataset_labels = np.array(dataset["correct_labels"][:])   #Correct labels
    dataset.close()
    
    return dataset_x_orig, dataset_y, dataset_images, dataset_labels
# In[1] 
#Unit test of function

dataset_x_orig, dataset_y, dataset_images, dataset_labels = load_lingonset_hdf5('First_lr_train_m210_512x512x3_T0.35.h5')

print(dataset_x_orig.shape)
print(type(dataset_x_orig))