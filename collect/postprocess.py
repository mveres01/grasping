import os
import csv
import cv2
import sys
import h5py
import numpy as np
import theano
from PIL import Image

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn import preprocessing

from utils import format_htmatrix, plot_mesh


def floatX(x):
    return np.float32(x)

# This function splits a dataset into train/test splits
def split_dataset(n_samples, train_size=0.8, use_valid=False):

    # Split data into train/test/valid
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_samples_train = int(train_size*n_samples)
    train_indices = indices[:n_samples_train]
    test_indices = indices[n_samples_train:]

    # Make valid and test set be equally-sized
    if use_valid is True:
        valid_indices = test_indices[:int(0.5*len(test_indices))]
        test_indices  = test_indices[int(0.5*len(test_indices)):]

        return train_indices, test_indices, valid_indices

    return train_indices, test_indices


# Given an input directory and list of objects to load ... load them!
def load_dataset(data_dir, data_list):

    if type(data_list) != list:
        data_list = [data_list]

    grasps = []
    images = []
    labels = []
    object_props = []

    for fname in data_list:
        data = load_all(data_dir+fname)

        if data is None:
            print '%s no data returned!'%fname
            continue

        grasps.append(data[0])
        images.append(data[1])
        labels.append(data[2])
        object_props.append(data[3])
    grasps = np.vstack(grasps)
    images = np.vstack(images)
    labels = np.vstack(labels)
    object_props = np.vstack(object_props)
    return grasps, images, labels, object_props


def load_all(fname=None):

    f = h5py.File(fname, 'r')

    object_name = np.atleast_2d(f['OBJECT_NAME'][:]).T
    pregrasp = f['GRIPPER_PREGRASP']

    mask_images = floatX(f['GRIPPER_IMAGE_MASK'][:])
    depth_images = floatX(f['GRIPPER_IMAGE'][:])
    colour_images = floatX(f['GRIPPER_IMAGE_COLOUR'][:])
    images = np.concatenate((depth_images, colour_images), axis=1)

    grasps = pregrasp['grasp_wrt_cam'][:]
    cam_wrt_obj = pregrasp['cam_wrt_obj'][:]
    img_wrt_cam = pregrasp['img_wrt_cam'][:]    
    obj_wrt_world = pregrasp['obj_wrt_world'][:]
    workspace_wrt_world = pregrasp['workspace_wrt_world'][:]

    if 'cam_wrt_world' not in pregrasp:
        cam_wrt_world = np.empty(obj_wrt_world.shape)
        for i, (w2o, o2c) in enumerate(zip(obj_wrt_world, cam_wrt_obj)):
            w2o_mat = format_htmatrix(w2o.reshape(3,4))
            o2c_mat = format_htmatrix(o2c.reshape(3,4))
            cam_wrt_world[i] = np.dot(w2o_mat, o2c_mat)[:3].flatten()
    else:
        cam_wrt_world = pregrasp['cam_wrt_world'][:]
    
    labels = np.hstack([object_name, 
                        obj_wrt_world, 
                        cam_wrt_obj, 
                        cam_wrt_world,
                        workspace_wrt_world, 
                        img_wrt_cam]) 

    com_wrt_obj = pregrasp['com_wrt_cam'][:]
    mass_wrt_obj = pregrasp['mass_wrt_cam'][:]
    inertia_wrt_obj = pregrasp['inertia_wrt_cam'][:]

    assert(grasps.shape[0] == images.shape[0] and \
           grasps.shape[0] == labels.shape[0])

    obj_props = np.hstack([mass_wrt_obj, inertia_wrt_obj, com_wrt_obj])
    return (grasps, images, labels, obj_props)


def splitAndSaveDataset():

    content_file = open('train_items.txt','r')
    reader = csv.reader(content_file, delimiter=',')
    train_list = reader.next()
    content_file.close()

    print 'train_list: ',train_list    

    dataset_save_path = '/scratch/mveres/ttv_localencode_v40.hdf5'
    data_dir = '/mnt/data/datasets/grasping/scene_v40/'
    train_dir = data_dir + 'train/'
    test_dir  = data_dir + 'test/'

    # Find the names of all datafiles we're going to use
    train_objects = []
    for c_idx in train_list:
        object_list =[o for o in os.listdir(train_dir) if c_idx in o]
        train_objects += object_list

    test_objects = [o for o in os.listdir(test_dir) if '.hdf5' in o]

    print '  Loading Train data ... ' 
    train_data = load_dataset(train_dir, train_objects)
    tr_grasps, tr_images, tr_labels, tr_props = train_data

    print '  Loading Test Data ... '
    test_data = load_dataset(test_dir, test_objects)
    test_grasps, test_images, test_labels, test_props = test_data

    # Randomly shuffle the test set
    indices = np.arange(test_grasps.shape[0])
    np.random.seed(12345)
    np.random.shuffle(indices)
    test_grasps = test_grasps[indices]
    test_images = test_images[indices]
    test_labels = test_labels[indices]
    test_props = test_props[indices]
    
    assert all(x.shape[0] == tr_grasps.shape[0] for x in \
        [tr_grasps, tr_images, tr_labels, tr_props])

    assert all(x.shape[0] == test_grasps.shape[0] for x in \
        [test_grasps, test_images, test_labels, test_props])


    train_indices, val_indices = split_dataset(tr_grasps.shape[0], 0.9)

    valid_grasps = tr_grasps[val_indices]
    valid_images = tr_images[val_indices]
    valid_labels = tr_labels[val_indices]
    valid_props  = tr_props[val_indices]

    train_grasps = tr_grasps[train_indices]
    train_images = tr_images[train_indices]
    train_labels = tr_labels[train_indices]
    train_props = tr_props[train_indices]


    print 'train_labels: ',train_labels[:2]
#print 'valid_labels: ',valid_labels[:2]
#print 'test_labels: ',test_labels[:2]

    
    df = h5py.File(dataset_save_path, 'a')
    group = df.create_group('train')
    group.create_dataset('images', data=train_images)
    group.create_dataset('grasps', data=train_grasps)
    group.create_dataset('labels',  data=train_labels)
    group.create_dataset('object_props', data=train_props)
    df.close()

    df = h5py.File(dataset_save_path, 'a')
    group = df.create_group('test')
    group.create_dataset('images', data=test_images)
    group.create_dataset('grasps', data=test_grasps)
    group.create_dataset('labels',  data=test_labels)
    group.create_dataset('object_props', data=test_props)
    df.close()

    df = h5py.File(dataset_save_path, 'a')
    group = df.create_group('valid')
    group.create_dataset('images', data=valid_images)
    group.create_dataset('grasps', data=valid_grasps)
    group.create_dataset('labels',  data=valid_labels)
    group.create_dataset('object_props', data=valid_props)
    df.close()
    
    

if __name__ == '__main__':
    splitAndSaveDataset()


