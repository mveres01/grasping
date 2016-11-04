import os
import csv
import sys
import itertools

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stl import mesh
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA

from utils import plot_mesh
from utils import format_htmatrix, invert_htmatrix, format_point


def plot_predicted_pose(finger_pos, finger_nml, axis=None):
    
    # Create a new plot
    if axis is None:
        figure = plt.figure()
        axis = mplot3d.Axes3D(figure)

    axis.scatter(*finger_pos[0], c='r', marker='o', s=75)
    axis.scatter(*finger_pos[1], c='g', marker='o', s=75)
    axis.scatter(*finger_pos[2], c='b', marker='o', s=75)
   
    axis.quiver(*np.hstack([finger_pos[0],finger_nml[0]]),color='k',length=0.1)
    axis.quiver(*np.hstack([finger_pos[1],finger_nml[1]]),color='k',length=0.1)
    axis.quiver(*np.hstack([finger_pos[2],finger_nml[2]]),color='k',length=0.1)         
    
    return axis


def import_and_rotate_mesh(obj_mesh, world2obj):

    # Find the dimensions of the mesh object
    width = np.max(obj_mesh.x) - np.min(obj_mesh.x)
    height= np.max(obj_mesh.y) - np.min(obj_mesh.y)
    depth = np.max(obj_mesh.z) - np.min(obj_mesh.z) 

    # Set the object centroid as center of bounding box
    obj_mesh.x -= abs(np.max(obj_mesh.x))-width/2.
    obj_mesh.y -= abs(np.max(obj_mesh.y))-height/2.
    obj_mesh.z -= abs(np.max(obj_mesh.z))-depth/2.

    # Translate/rotate each of the vertices
    for j, row in enumerate(obj_mesh.vectors):
 
        obj_mesh.vectors[j][0] = np.dot(world2obj, format_point(row[0]))[:3]
        obj_mesh.vectors[j][1] = np.dot(world2obj, format_point(row[1]))[:3]
        obj_mesh.vectors[j][2] = np.dot(world2obj, format_point(row[2]))[:3]

    return obj_mesh


def tile_grasps_mesh(grasp_orig, grasp_pred, labels):

    """Plots a 3d mesh object, along with predicted fingertip positions."""
    def format_data(graspData, htmatrix=None):

        n_fingers = len(graspData)/6
        positions = np.zeros((n_fingers, 3))
        normals = np.zeros((n_fingers, 3))

        for i in xrange(n_fingers):
            pos = graspData[i*n_fingers : i*n_fingers + 3]
            nml = graspData[3*n_fingers + i*n_fingers: \
                            3*n_fingers + i*n_fingers + 3]
    
            if htmatrix is not None:
                pos = np.dot(htmatrix, format_point(pos))[:3]
                nml = np.dot(htmatrix[:3, :3], nml)
            positions[i] = pos
            normals[i] = nml

        normals = - normals / np.sqrt(np.sum(normals, axis=1))
        return (positions, normals)


    # Create figure and set ranges
    n_samples, n_var = grasp_pred.shape

    if grasp_orig is not None:
        n_samples = n_samples+1
    fig = plt.figure(figsize=(3*(n_samples), 3))

    count = 0

    name = str(labels[ 0])

    # Load the object mesh
    obj = name.split('/')[-1]
  
    obj_wrt_world = labels[1:13].reshape(3,4)
    obj_wrt_world = format_htmatrix(obj_wrt_world)

    cam_wrt_obj = labels[13:25].reshape(3,4)
    cam_wrt_obj = format_htmatrix(cam_wrt_obj)
    obj_wrt_cam = invert_htmatrix(cam_wrt_obj)

    cam_wrt_world = labels[25:37].reshape(3,4)
    cam_wrt_world = format_htmatrix(cam_wrt_world)

    workspace_wrt_world = labels[37:49].reshape(3,4)
    workspace_wrt_world = format_htmatrix(workspace_wrt_world)
    world_wrt_workspace = invert_htmatrix(workspace_wrt_world)

    obj_wrt_workspace = np.dot(world_wrt_workspace, obj_wrt_world)
    cam_wrt_workspace = np.dot(world_wrt_workspace, cam_wrt_world)

    # NOTE: Not sure if this is right or now

    if grasp_orig is not None:
        count += 1
        axis = fig.add_subplot(1, n_samples, count, projection='3d')
        pos, nml = format_data(grasp_orig[0], cam_wrt_workspace) 
        axis = plot_predicted_pose(pos, nml, axis=axis)
        axis = plot_mesh(obj, obj_wrt_workspace, axis=axis)

    for j in xrange(grasp_pred.shape[0]):
       
        count += 1
        axis = fig.add_subplot(1, n_samples, count, projection='3d')
        pos, nml = format_data(grasp_pred[j], cam_wrt_workspace)
        axis = plot_predicted_pose(pos, nml, axis=axis)
        axis = plot_mesh(obj, obj_wrt_workspace, axis=axis)

    return fig
  
      
if __name__ == '__main__':
   
 
    stl_dir = '/mnt/data/datasets/grasping/stl_files'
    file_path ='/scratch/mveres/predictions/SINGLE-pred.csv'
   
    if not os.path.exists('images'):
        os.makedirs('images')
    
    data = pd.read_csv(file_path, header=None, index_col=False)
    
    for i, params in enumerate(data):
    
        plt.close('all')
        
        name = str(data.iloc[i,0])
        world2obj = format_htmatrix(data.iloc[i,1:13].astype(np.float32))
        obj2cam = data.iloc[i,13:25].astype(np.float32)
        pos0 = data.iloc[i,25:28].astype(np.float32)
        pos1 = data.iloc[i,28:31].astype(np.float32)
        pos2 = data.iloc[i,31:34].astype(np.float32)
        nml0 = data.iloc[i,34:37].astype(np.float32)
        nml1 = data.iloc[i,37:40].astype(np.float32)
        nml2 = data.iloc[i,40:43].astype(np.float32)
        
        # Load the object mesh
        obj = name.split('/')[-1]
        
        print 'Object: %s'%obj
        obj_path = os.path.join(stl_dir, obj+'.stl')
        obj_mesh = mesh.Mesh.from_file(obj_path)
        obj_mesh = import_and_rotate_mesh(obj_mesh, world2obj)
            
        pos0 = np.dot(world2obj, format_point(pos0))[:3]
        pos1 = np.dot(world2obj, format_point(pos1))[:3]
        pos2 = np.dot(world2obj, format_point(pos2))[:3]
        nml0 = np.dot(world2obj[:3,:3], nml0)
        nml1 = np.dot(world2obj[:3,:3], nml1)
        nml2 = np.dot(world2obj[:3,:3], nml2)
        
        # Flip the direction of unit vector to point inwards (towards object)
        nml0 = - nml0 / (np.sqrt(np.sum(nml0**2)))
        nml1 = - nml1 / (np.sqrt(np.sum(nml1**2)))
        nml2 = - nml2 / (np.sqrt(np.sum(nml2**2)))
        
   
        save_name =  'images/'+str(i)+'-'+obj+'.png'
        plot_predicted_pose(\
            obj_mesh, [pos0, pos1, pos2], [nml0, nml1, nml2], save_name)
        #plt.show()
   

