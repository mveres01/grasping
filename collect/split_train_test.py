import os
import sys
sys.path.append('..')

import csv
import random
import h5py
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


from lib.python_config import (config_processed_data_dir, config_train_dir,
                       config_test_dir)

GLOBAL_STAT_HEADER = ['idx', 'Class', 'Train Objs', 'Train Grasps',
                      'Train Objs', 'Test Grasps']

def plot_stats(dataset_path):
    """Given a table of dataset statistics, creates a nice bar plot."""

    stats_path = os.path.join(dataset_path, 'dataset_statistics.csv')

    stats = pd.read_csv(stats_path, index_col=False)

    # Combine the class and index so Seaborn doesn't try to replace one another
    stats['Class'] = stats["idx"].map(str) +'_'+ stats["Class"]

    stats_sorted = stats.sort_values(['Train Grasps'], ascending=False,
                                     inplace=False)

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(15, 20))

    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(x="Train Grasps", y="Class", data=stats_sorted,
                ci=None, label="Training Instances", color="b")

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x="Test Grasps", y="Class", data=stats_sorted,
                ci=None, label="Testing Instances", color="b")

    # Add a legend and informative axis label
    axis.legend(ncol=2, loc="lower right", frameon=True)
    axis.set(xlim=(0, np.max(stats_sorted['Train Grasps'])), ylabel="",
             xlabel="Number of Image and Grasp Instances")
    sns.despine(left=True, bottom=True)

    plt.tight_layout()

    save_path = os.path.join(dataset_path, 'dataset_statistics.pdf')
    plt.savefig(save_path, dpi=120)


def count_num_grasps(list_of_objects_in_class):
    """Given a list of objects, counts how many grasps are in each file."""

    num_grasps = np.zeros((len(list_of_objects_in_class), 1))
    for i, object_file in enumerate(list_of_objects_in_class):

        try:
            fpath = os.path.join(config_processed_data_dir, object_file)
            datafile = h5py.File(fpath, 'r')
            successful_grasps = datafile['GRIPPER_IMAGE'].shape[0]
            num_grasps[i] = successful_grasps
            datafile.close()
        except Exception as e:
            print '%s \nTRAIN SPLIT ERR: '%object_file, e
            continue

    return num_grasps


def split_train_test(files, array_of_grasps):
    """Given a list of files, splits the data into train/test splits."""

    good_grasps_idx = array_of_grasps > 0
    good_grasps = array_of_grasps[good_grasps_idx].flatten()
    good_files = files[good_grasps_idx].flatten()

    # If a class only has a single object in it
    if np.sum(good_grasps_idx) == 1:
        test_objects = good_files
        test_grasps = good_grasps
        train_objects = []
        train_grasps = [0]
    else:
        # Randomly select an item to be in test set
        indices = np.arange(len(files))
        random.shuffle(indices)

        # Split train vs Test objects
        train_objects = good_files[indices[:-1]]
        train_grasps = good_grasps[indices[:-1]]
        test_objects = good_files[indices[-1]].flatten()
        test_grasps = good_grasps[indices[-1]].flatten()

    return (train_objects, train_grasps), (test_objects, test_grasps)


def main():
    """Splits a dataset into training/testing components.

    This function takes a single element from each of the object classes, and
    places it into a 'test' folder.
    """

    if not os.path.exists(config_train_dir):
        os.makedirs(config_train_dir)
    if not os.path.exists(config_test_dir):
        os.makedirs(config_test_dir)

    # Get a list of all the decoded object files
    objects = os.listdir(config_processed_data_dir)
    objects = np.asarray([o for o in objects if '.hdf5' in o])
    class_ids = np.asarray([f.split('_')[0] for f in objects])
    class_names = np.asarray([f.split('_')[1] for f in objects])
    unique_ids = np.unique(class_ids)

    # Header
    fpath = os.path.join(config_processed_data_dir, 'dataset_statistics.csv')
    csvfile = open(fpath, 'wb')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(GLOBAL_STAT_HEADER)

    # For each of the object classes, count how many files and total grasps
    for obj_class in unique_ids:

        # Get the files belonging to each individual class, then count each #
        idx_where = np.where(class_ids[:] == obj_class)
        files = objects[idx_where]
        obj_name = class_names[idx_where][0]
        
        num_object_grasps = count_num_grasps(files)

        if len(num_object_grasps) == 0 or all(num_object_grasps == 0):
            print 'No grasps collected for object class: %s'%obj_class
            continue

        files = np.vstack(files)

        # Make sure we've recorded an equal number of objects and grasps
        assert files.shape[0] == num_object_grasps.shape[0]

        # Given our list of grasps, create a train/test split
        train, test = split_train_test(files, num_object_grasps)
        train_objects, train_grasps = train
        test_objects, test_grasps = test

        for obj in train_objects:
            old_path = os.path.join(config_processed_data_dir, obj)
            new_path = os.path.join(config_train_dir, obj)
            os.rename(old_path, new_path)
        for obj in test_objects:
            old_path = os.path.join(config_processed_data_dir, obj)
            new_path = os.path.join(config_test_dir, obj)
            os.rename(old_path, new_path)

        # Write the statistics
        if len(test_grasps) > 0 and len(train_grasps) > 0:
            writer.writerow(
                [obj_class] + # Class number
                [obj_name] + # Class name
                [len(train_grasps) if np.sum(train_grasps) > 0 else 0] +
                ['%4d'%np.sum(train_grasps)] +
                [len(test_grasps)] + # Always (at least) one file
                ['%4d'%np.sum(test_grasps)])
    csvfile.close()

if __name__ == '__main__':

    main()
    plot_stats(config_processed_data_dir)
