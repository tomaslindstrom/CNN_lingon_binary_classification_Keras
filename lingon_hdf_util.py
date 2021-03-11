# -*- coding: utf-8 -*-

"""
Created on Tue Feb  9 10:12:39 2021
Utilit functions to load a lingon dataset stored in a HDF5-file as   

Parmeters:
---------
    file_name : HDF5-file file name (file_name) to be open  
@author: ISOFT
"""
import numpy as np
import h5py



def load_lingonset(hdf5_file, file_path = 'D:/OneDrive - Isoft Services AB/Bärplockning/deep_models/Create_hdf5_datasets/data_sets/' ):
    """
    Loading ligonset for logisitc regression
    
    Parameters
    ----------
    hdf5_file : str
        DESCRIPTION.Name of hdf5-file to use.
    file_path : str, optional
        DESCRIPTION. The path to hdf5 file. The default is 'D:/OneDrive - Isoft Services AB/Bärplockning/deep_models/Create_hdf5_datasets/data_sets/'.

    Returns
    -------
    dataset_x_orig : ndarray
        DESCRIPTION.
    dataset_y : ndarray
        DESCRIPTION. 
    dataset_images : ndarray
        DESCRIPTION.
    dataset_labels : TYPE
        DESCRIPTION.

    """
    hdf5_file = file_path + hdf5_file
    dataset = h5py.File(hdf5_file, "r")
    dataset_x_orig = np.array(dataset["data_set_x"][:]) # your test set feature
    dataset_y = (np.array(dataset["data_set_y"][:])).reshape(1,-1) # Test set labels
    dataset_images = np.array(dataset["data_set_images"][:]) #List of images
    dataset_labels = np.array(dataset["correct_labels"][:])   #Correct labels
    dataset.close()
    
    return dataset_x_orig, dataset_y, dataset_images, dataset_labels



def load_lingonset_one(hdf5_file, file_path = 'D:/OneDrive - Isoft Services AB/Bärplockning/deep_models/Create_hdf5_datasets/data_sets/'):
    """
    Loading NN data to model from hdf5, for Logistoc Regression and one-one shot models.
    
    Paramaters
    -----------
    hdf5_file: str
        DESCRIPTION. Name of the hdf5 - file 
    file_path: string, optional
        DESCRIPTION. The path to the hdf5-file. The default is 'D:/OneDrive - Isoft Services AB/Bärplockning/deep_models/Create_hdf5_datasets/data_sets/'
    
    Returns
    -------
    dataset_x_orig : ndarray
        DESCRIPTION.The matrix with the features.
    dataset_one_hot_lables : list
        DESCRIPTION. A list with the image labels for each image 
    dataset_one_hot_y : ndarray
        DESCRIPTION.An array with the  respective images one hot class numbers.
    dataset_images : ndarray
        DESCRIPTION.The names of the images in a vector, in order of creation. 
    onehot_classes_labels : list
        DESCRIPTION.The class labels
    one_hot_dic: dictionary
        DESCRIPTION.Mapping of the one hot class lable to the one hot class number e.g. {0:lingon}

    
    """
    hdf5_file = file_path + hdf5_file
    dataset = h5py.File(hdf5_file, "r")
    dataset_x_orig = np.array(dataset["data_set_x"][:])             # Dataset features 
    dataset_one_hot_labels = np.array(dataset["cat_label_y"][:])    # Category labels of images as binary
    dataset_one_hot_y = np.array(dataset["one_hot_y"][:])           # One hot codings of dataset  as numbers
    dataset_images = np.array(dataset["data_set_images"][:])        # List of the names of the images
    one_hot_classes = np.array(dataset['one_hot_classes'][:])       # Tuple with the one hot classes in order as coded in numbers
    dataset.close()
    
    #Decode dataset_one_hot_labels for ease of use 
    dataset_one_hot_labels = [cat_label.decode('utf-8') for cat_label in dataset_one_hot_labels]
    
    
    # Decode image names from byte to string
    dataset_images = [img_label.decode('utf-8') for img_label in dataset_images]
    
    # Decode one hot classes and create dictionary {code:label}
    one_hot_classes = [cat_dict_label.decode('utf-8') for cat_dict_label in one_hot_classes]
    one_hot_dic = {}
    for index,cat_label in enumerate(one_hot_classes):
           one_hot_dic.update({index: cat_label})
    
    return dataset_x_orig, dataset_one_hot_labels, dataset_one_hot_y, dataset_images, one_hot_classes, one_hot_dic
    

# In[2] 
#Unit test of function for load_lingonset
"""
dataset_x_orig, dataset_y, dataset_images, dataset_labels = load_lingonset('First_lr_train_m210_512x512x3_T0.35.h5')

print(dataset_x_orig.shape)
print(type(dataset_x_orig))
print(dataset_y.shape)
"""


# In[3] 
"""
#Unit test function of one_shot labels
dataset_x_orig, dataset_one_hot_labels, dataset_one_hot_y, dataset_images, one_hot_classes, one_hot_dic = load_lingonset_one('Test_one_hot_labels_m78_64x64x3_T0.14one.h5')

print(dataset_x_orig.shape)
print(type(dataset_x_orig))
print(dataset_one_hot_labels)
print(type(dataset_one_hot_labels))

print(dataset_one_hot_y)
print(dataset_images)
print(one_hot_classes)
print(type(one_hot_classes))
print(one_hot_dic)
"""