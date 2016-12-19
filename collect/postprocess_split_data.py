import os
import csv
import sys
sys.path.append('..')

import h5py
import numpy as np

from lib.utils import float32, format_htmatrix
from lib.python_config import (config_dataset_path, config_train_dir,
                               config_test_dir, config_train_item_list)


def split_dataset(n_samples, train_size=0.8, use_valid=False):
    """Generate train and test indices to split a dataset."""

    # Split data into train/test/valid
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_samples_train = int(train_size*n_samples)
    train_indices = indices[:n_samples_train]
    test_indices = indices[n_samples_train:]

    # Make valid and test set be equally-sized
    if use_valid is True:
        valid_indices = test_indices[:int(0.5*len(test_indices))]
        test_indices = test_indices[int(0.5*len(test_indices)):]

        return train_indices, test_indices, valid_indices

    return train_indices, test_indices


def load_object_datasets(data_dir, data_list):
    """Given a directory, load a set of hdf5 files given as a list."""

    if isinstance(data_list, list):
        data_list = [data_list]

    grasps, images, labels, object_props = [], [], [], []

    for fname in data_list:
        data = load_grasp_data(data_dir+fname)

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


def load_grasp_data(fname=None):
    """Loads a specific hdf5 file."""

    datafile = h5py.File(fname, 'r')

    object_name = datafile['OBJECT_NAME'][:]
    object_name = np.atleast_2d(object_name).T
    depth_images = float32(datafile['GRIPPER_IMAGE'][:])
    colour_images = float32(datafile['GRIPPER_IMAGE_COLOUR'][:])
    images = np.concatenate((depth_images, colour_images), axis=1)

    # For convenience, access all of the information we'll need
    pregrasp = datafile['GRIPPER_PREGRASP']
    grasp_wrt_cam = pregrasp['grasp_wrt_cam'][:]
    frame_cam_wrt_obj = pregrasp['frame_cam_wrt_obj'][:]
    frame_img_wrt_cam = pregrasp['frame_img_wrt_cam'][:]
    frame_obj_wrt_world = pregrasp['frame_obj_wrt_world'][:]
    frame_workspace_wrt_world = pregrasp['frame_workspace_wrt_world'][:]

    # NOTE: Check if we still need this?
    if 'frame_cam_wrt_world' not in pregrasp:
        frame_cam_wrt_world = np.empty(frame_obj_wrt_world.shape)

        for i in xrange(frame_obj_wrt_world.shape[0]):
            w2o = format_htmatrix(frame_obj_wrt_world[i].reshape(3, 4))
            o2c = format_htmatrix(frame_cam_wrt_obj[i].reshape(3, 4))
            frame_cam_wrt_world[i] = np.dot(w2o, o2c)[:3].flatten()
    else:
        frame_cam_wrt_world = pregrasp['frame_cam_wrt_world'][:]

    # Labels are composed of the object name, as well as additional reference
    # frames
    labels = np.hstack([object_name, frame_obj_wrt_world, frame_cam_wrt_obj,
                        frame_cam_wrt_world, frame_workspace_wrt_world,
                        frame_img_wrt_cam])

    # Object specific properties, so we can set these in simulation when testing
    com_wrt_obj = pregrasp['com_wrt_cam'][:]
    mass_wrt_obj = pregrasp['mass_wrt_cam'][:]
    inertia_wrt_obj = pregrasp['inertia_wrt_cam'][:]

    assert all(grasp_wrt_cam.shape[0] == data.shape[0] for data in [images, labels])

    obj_props = np.hstack([mass_wrt_obj, inertia_wrt_obj, com_wrt_obj])

    return (grasp_wrt_cam, images, labels, obj_props)


def split_and_save_dataset():
    """Given a list of train items, splits a dataset into train/test split."""

    # This contains all of the objects we wish to consider in our train set
    content_file = open(config_train_item_list, 'r')
    reader = csv.reader(content_file, delimiter=',')
    train_list = reader.next()
    content_file.close()

    # For the train set, we're only going to use the object classes specified in
    # the provided 'train_list' file.
    train_objects = []
    for c_idx in train_list:
        object_list = [o for o in os.listdir(config_train_dir) if c_idx in o]
        train_objects += object_list

    # For the test set, we will eventually segment into similar/different
    # classes
    test_objects = [o for o in os.listdir(config_test_dir) if '.hdf5' in o]

    print '  Loading Train data ... '
    train_data = load_object_datasets(config_train_dir, train_objects)
    tr_grasps, tr_images, tr_labels, tr_props = train_data

    print '  Loading Test Data ... '
    test_data = load_object_datasets(config_test_dir, test_objects)
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


    # Split the training dataset into train/valid splits
    train_indices, val_indices = split_dataset(tr_grasps.shape[0], 0.9)

    valid_grasps = tr_grasps[val_indices]
    valid_images = tr_images[val_indices]
    valid_labels = tr_labels[val_indices]
    valid_props = tr_props[val_indices]

    train_grasps = tr_grasps[train_indices]
    train_images = tr_images[train_indices]
    train_labels = tr_labels[train_indices]
    train_props = tr_props[train_indices]


    # Write each of the train/test/valid splits to file
    savefile = h5py.File(config_dataset_path, 'a')
    group = savefile.create_group('train')
    group.create_dataset('images', data=train_images)
    group.create_dataset('grasps', data=train_grasps)
    group.create_dataset('labels', data=train_labels)
    group.create_dataset('object_props', data=train_props)
    savefile.close()

    savefile = h5py.File(config_dataset_path, 'a')
    group = savefile.create_group('test')
    group.create_dataset('images', data=test_images)
    group.create_dataset('grasps', data=test_grasps)
    group.create_dataset('labels', data=test_labels)
    group.create_dataset('object_props', data=test_props)
    savefile.close()

    savefile = h5py.File(config_dataset_path, 'a')
    group = savefile.create_group('valid')
    group.create_dataset('images', data=valid_images)
    group.create_dataset('grasps', data=valid_grasps)
    group.create_dataset('labels', data=valid_labels)
    group.create_dataset('object_props', data=valid_props)
    savefile.close()


if __name__ == '__main__':
    split_and_save_dataset()


