import os
import sys
import csv
import h5py
import random
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sns.set(style='whitegrid')


def plot_stats(dataset_path):

    plt.close('all')
    df = pd.read_csv(dataset_path+'dataset_statistics.csv', index_col = False)

    # Combine the class and index so Seaborn doesn't try to replace one another
    df['Class'] = df["idx"].map(str) +'_'+ df["Class"]

    df_sorted = df.sort_values(['Train Grasps'], ascending = False, inplace=False)

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 20))


    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(x="Train Grasps", y="Class", data=df_sorted, ci=None,
        label="Training Instances", color="b")

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x="Test Grasps", y="Class", data=df_sorted, ci=None,
        label="Testing Instances", color="b")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, np.max(df_sorted['Train Grasps'])), ylabel="",
       xlabel="Number of Image and Grasp Instances")
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(dataset_path+'dataset_statistics.pdf', dpi=120)



def split_dataset_traintest(folder):
    
    root_path = '/mnt/data/datasets/grasping/'
    data_path = os.path.join(root_path, folder)
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    csvfile = open(data_path+'/dataset_statistics.csv','wb')
    writer  = csv.writer(csvfile, delimiter=',')
 
 
    objects = os.listdir(data_path)
    objects = [o for o in objects if '.hdf5' in o]
    file_ids = [f.split('_')[0] for f in objects]

    unique_ids = np.unique(file_ids)

    # So we can index them quickly
    objects = np.asarray(objects)
    file_ids= np.asarray(file_ids)

    # Header 
    writer.writerow(['idx','Class','Train Objs','Train Grasps',
                    'Train Objs','Test Grasps'])
    for obj_class in unique_ids:

        # Get the files belonging to each individual class
        files = objects[file_ids[:] == str(obj_class)]

        # For each object in train set
        num_grasps = []
        for f in files:
          
            f = str(f) 
            try:
                fp = h5py.File(data_path+'/'+f, 'r')
                successful_grasps = fp['GRIPPER_IMAGE'].shape[0]
                num_grasps.append(successful_grasps)

                fp.close()
            except Exception as e:
                num_grasps.append(0)
                print '%s \nTRAIN SPLIT ERR: '%f,e 
                continue

        if len(num_grasps)==0: 
            continue

        print 'num_grasps: ',num_grasps

        num_grasps = np.vstack(num_grasps).flatten()
        files = np.vstack(files)       

        good_grasps = num_grasps>0
        if np.sum(good_grasps) == 0:
            print 'No good grasps!'
            continue

        assert(files.shape[0] == num_grasps.shape[0])

        grasps = num_grasps[good_grasps].flatten()
        files  = files[good_grasps].flatten()

        # If a class only has a single object in it
        if np.sum(good_grasps)==1: 

            test_objects = files
            test_grasps = grasps
            train_objects = [] 
            train_grasps = [0]

        else:

            # Randomly select an item to be in test set
            indices = np.arange(len(files))
            random.shuffle(indices)

            # Split train vs Test objects
            train_objects = files[indices[:-1]]
            train_grasps = grasps[indices[:-1]]

            test_objects = files[indices[-1]].flatten()
            test_grasps = grasps[indices[-1]].flatten()

        
        for f in train_objects:
            os.rename(data_path+'/'+f, train_path+'/'+f)
        for f in test_objects:
            os.rename(data_path+'/'+f, test_path+'/'+f)

        # Write the statistics
        if len(test_grasps) >0 and len(train_grasps)>0: 
            writer.writerow(
                [f.split('_')[0]] + 
                [f.split('_')[1]] +
                [len(train_grasps) if np.sum(train_grasps)>0 else 0] + 
                ['%4d'%np.sum(train_grasps)] + 
                [len(test_grasps)] + 
                ['%4d'%np.sum(test_grasps)]) 



if __name__ == '__main__':
   
    import argparse
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--folder',type=str, dest='folder')
    parser.set_defaults(folder='scene_v40') 
    args = parser.parse_args(sys.argv[1:])
 
    kwargs = vars(args)
    #split_dataset_traintest(kwargs['folder'])
    plot_stats('/mnt/data/datasets/grasping/scene_v40/')
