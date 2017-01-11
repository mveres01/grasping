import os
import csv
import sys
sys.path.append('..')

import h5py
import numpy as np

from lib.utils import float32, format_htmatrix
from lib.python_config import (config_dataset_path, config_train_dir,
                               config_test_dir, config_train_item_list)


def split_dataset(n_samples, train_size=0.8):
    """Generate train and test indices to split a dataset."""

    # Split data into train/test/valid
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_samples_train = int(train_size*n_samples)
    train_indices = indices[:n_samples_train]
    test_indices = indices[n_samples_train:]

    return train_indices, test_indices


def load_grasp_data(fname=None):
    """Loads a specific hdf5 file."""

    datafile = h5py.File(fname, 'r')

    image_depth_oto = float32(datafile['image_depth_oto'][:])
    image_colour_oto = float32(datafile['image_colour_oto'][:])
    images_oto = np.concatenate([image_depth_oto, image_colour_oto], axis=1)

    image_depth_otm = float32(datafile['image_depth_otm'][:])
    image_colour_otm = float32(datafile['image_colour_otm'][:])
    images_otm = np.concatenate([image_depth_otm, image_colour_otm], axis=1)

    # For convenience, access all of the information we'll need
    pregrasp = datafile['pregrasp']
    cam2grasp_oto = pregrasp['cam2grasp_oto'][:]
    cam2grasp_otm = pregrasp['cam2grasp_otm'][:]

    # Collect everything else but the grasp
    keys = [k for k in pregrasp.keys() if 'grasp' not in k]
    misc_dict = {k:pregrasp[k] for k in keys}

    return (cam2grasp_oto, cam2grasp_otm, images_oto, images_otm, misc_dict)


def load_object_datasets(data_dir, data_list):
    """Given a directory, load a set of hdf5 files given as a list."""

    if not isinstance(data_list, list):
        data_list = [data_list]

    # We'll append all data into a list before shuffle train/test/valid
    grasps_oto = []
    grasps_otm = []
    images_oto = []
    images_otm = []
    misc_props = {}

    # For each of the decoded objects in the data_dir
    for fname in data_list:

        data_path = os.path.join(data_dir, fname)
        data = load_grasp_data(data_path)

        if data is None:
            print '%s no data returned!'%fname
            continue

        grasps_oto.append(data[0])
        grasps_otm.append(data[1])
        images_oto.append(data[2])
        images_otm.append(data[3])
      
        # Misc props is stored as a dictionary, so we'll make a list out of
        # each of the keys and append to that
        if not misc_props:
            for key in data[4].keys():
                misc_props[key] = []
        for key in misc_props.keys(): 
            misc_props[key].append(np.atleast_2d(data[4][key]))

    # Merge a list of lists into a single list/numpy array
    grasps_oto = np.vstack(grasps_oto)
    grasps_otm = np.vstack(grasps_otm)
    images_oto = np.vstack(images_oto)
    images_otm = np.vstack(images_otm)

    for key in misc_props.keys():
        if key == 'object_name':
            misc_props[key] = np.hstack(misc_props[key]).T
        else:
            misc_props[key] = np.vstack(misc_props[key])

    return grasps_oto, grasps_otm, images_oto, images_otm, misc_props


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
        train_objects += [o for o in os.listdir(config_train_dir) if c_idx in o]
        
    # For the test set, we will eventually segment into similar/different
    # classes
    test_objects = [o for o in os.listdir(config_test_dir) if '.hdf5' in o]

    print '  Loading Train data ... '
    train_data = load_object_datasets(config_train_dir, train_objects)
    grasps_oto = train_data[0]
    grasps_otm = train_data[1]
    images_oto = train_data[2]
    images_otm = train_data[3]
    misc_props = train_data[4]

    # Split the training dataset into train/valid splits
    
    # -------- Train dataset
    train_indices, valid_indices = split_dataset(grasps_oto.shape[0], 0.9)
    tr_grasps_oto = grasps_oto[train_indices]
    tr_grasps_otm = grasps_otm[train_indices]
    tr_images_oto = images_oto[train_indices]
    tr_images_otm = images_otm[train_indices]
    tr_misc_props = {}
    for key in misc_props.keys():  
        if key == 'object_name': 
            data = list(misc_props[key][train_indices])
        else:
            data = misc_props[key][train_indices]
        tr_misc_props[key] = data

    # -------- Valid dataset
    va_grasps_oto = grasps_oto[valid_indices]
    va_grasps_otm = grasps_otm[valid_indices]
    va_images_oto = images_oto[valid_indices]
    va_images_otm = images_otm[valid_indices]
    va_misc_props = {}
    for key in misc_props.keys():
        if key == 'object_name': 
            data = list(misc_props[key][valid_indices])
        else:
            data = misc_props[key][valid_indices]
        va_misc_props[key] = data 


    # -------- Test dataset
    test_data = load_object_datasets(config_test_dir, test_objects)

    test_indices = np.arange(test_data[0].shape[0])
    np.random.shuffle(test_indices)

    te_grasps_oto = test_data[0][test_indices]
    te_grasps_otm = test_data[1][test_indices]
    te_images_oto = test_data[2][test_indices]
    te_images_otm = test_data[3][test_indices]
    te_misc_props = {}
    for key in test_data[4].keys():
        if key == 'object_name': 
            data = list(test_data[4][key][test_indices])
        else:
            data = test_data[4][key][test_indices]
        te_misc_props[key] = data 


    assert all(x.shape[0] == tr_grasps_oto.shape[0] for x in \
        [tr_grasps_oto, tr_grasps_otm, tr_images_oto, tr_grasps_otm])

    assert all(x.shape[0] == va_grasps_oto.shape[0] for x in \
        [va_grasps_oto, va_grasps_otm, va_images_oto, va_grasps_otm])

    assert all(x.shape[0] == te_grasps_oto.shape[0] for x in \
        [te_grasps_oto, te_grasps_otm, te_images_oto, te_grasps_otm])


    # Write each of the train/test/valid splits to file
    savefile = h5py.File(config_dataset_path, 'w')
    group = savefile.create_group('train')
    group.create_dataset('images_oto', data=tr_images_oto, compression='gzip')
    group.create_dataset('images_otm', data=tr_images_otm, compression='gzip')
    group.create_dataset('grasps_oto', data=tr_grasps_oto, compression='gzip')
    group.create_dataset('grasps_otm', data=tr_grasps_otm, compression='gzip')

    props = group.create_group('object_props')
    for key in tr_misc_props.keys():
        props.create_dataset(key, data=tr_misc_props[key], compression='gzip')
    savefile.close()

    savefile = h5py.File(config_dataset_path, 'a')
    group = savefile.create_group('test')
    group.create_dataset('images_oto', data=te_images_oto, compression='gzip')
    group.create_dataset('images_otm', data=te_images_otm, compression='gzip')
    group.create_dataset('grasps_oto', data=te_grasps_oto, compression='gzip')
    group.create_dataset('grasps_otm', data=te_grasps_otm, compression='gzip')

    props = group.create_group('object_props')
    for key in te_misc_props.keys():
        props.create_dataset(key, data=te_misc_props[key], compression='gzip')
    savefile.close()

    savefile = h5py.File(config_dataset_path, 'a')
    group = savefile.create_group('valid')
    group.create_dataset('images_oto', data=va_images_oto, compression='gzip')
    group.create_dataset('images_otm', data=va_images_otm, compression='gzip')
    group.create_dataset('grasps_oto', data=va_grasps_oto, compression='gzip')
    group.create_dataset('grasps_otm', data=va_grasps_otm, compression='gzip')

    props = group.create_group('object_props')
    for key in va_misc_props.keys():
        props.create_dataset(key, data=va_misc_props[key], compression='gzip')
    savefile.close()


if __name__ == '__main__':
    split_and_save_dataset()


