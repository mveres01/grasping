import os, sys, csv
import numpy as np
import pandas as pd

import trimesh
import trimesh.transformations as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import format_htmatrix, invert_htmatrix
from utils import calc_mesh_centroid, plot_equal_aspect, plot_mesh

GLOBAL_SAVE_DIR = 'candidates'
GLOBAL_MESH_DIR = '/mnt/data/datasets/grasping/stl_files'
#GLOBAL_MESH_DIR = 'C:/Users/Matt/Documents/vrep-scenes/object_files/stl_meshes'

# Controls how big the rotation angles in each direction are
GLOBAL_X = 45
GLOBAL_Y = 45
GLOBAL_Z = 45


# Function for sammpling a random angle in radians given a step size
ri = lambda x: np.float32(np.random.randint(0,x))*np.pi/180.

# Function for converting a tuple of local/global degrees to rad
to_rad = lambda x,y,z: (x*np.pi/180., y*np.pi/180., z*np.pi/180.)

# Helper function for quickly formatting a 3d (xyz) point into a 4d vector
# so we can multiply it with a 4x4 homogeneous transform matrix
cvt4d = lambda x : np.hstack([x, 1])


def get_corners_and_plances(bbox):
    """Caclucates the bounding box and 8 planes given min/max x,y,z values"""
    
    minx = bbox[0]; maxx = bbox[1]
    miny = bbox[2]; maxy = bbox[3]
    minz = bbox[4]; maxz = bbox[5]

    corners = [(minx, miny, minz), (minx, miny, maxz),
               (minx, maxy, minz), (minx, maxy, maxz),
               (maxx, miny, minz), (maxx, miny, maxz),
               (maxx, maxy, minz), (maxx, maxy, maxz)]

    # Define 3 points on each of the bounding-box planes
    xz_pos = [(maxx, maxy, maxz), (maxx, maxy, minz), (minx, maxy, minz)]
    xz_neg = [(maxx, miny, maxz), (maxx, miny, minz), (minx, miny, minz)]
    xy_pos = [(maxx, maxy, maxz), (minx, miny, maxz), (maxx, miny, maxz)]
    xy_neg = [(maxx, maxy, minz), (minx, miny, minz), (maxx, miny, minz)]
    yz_pos = [(maxx, maxy, maxz), (maxx, maxy, minz), (maxx, miny, maxz)]
    yz_neg = [(minx, maxy, maxz), (minx, maxy, minz), (minx, miny, maxz)]
    planes = [xz_pos, xz_neg, xy_pos, xy_neg, yz_pos, yz_neg]

    return corners, planes


def intersect_plane(pa, pb, p0, p1, p2):
    """Checks the intersect_plane of a vector and a plane in 3d space

    Parameters
    ----------
    pa : a point along the line
    pb : a second point along the line
    p0, p1, p2 : three different points on the plane

    Returns
    -------
    Point of intersect_plane if the line and plane intersect, otherwise False
    See: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersect_plane
    """

    mat = np.asarray(
        [[pa[0]-pb[0], p1[0]-p0[0], p2[0]-p0[0]],
         [pa[1]-pb[1], p1[1]-p0[1], p2[1]-p0[1]],
         [pa[2]-pb[2], p1[2]-p0[2], p2[2]-p0[2]]])

    vec = np.asarray(pa-p0)

    try:
        inv_mat = np.linalg.pinv(mat)
    except Exception as e:
        return False

    soln = np.dot(inv_mat, vec)

    itx = pa + (pb-pa)*soln[0]

    # Check if line is along norm (e.g. positive instead of negative)
    return itx if soln[0]>0 else False


def intersect_box(itx, minx, maxx, miny, maxy, minz, maxz):
    """Calculates whether the intersect_plane is within the bounding box"""

    if type(itx) == bool:
        return False

    if itx[0] < minx or itx[0]>maxx or \
       itx[1] < miny or itx[1]>maxy or \
       itx[2] < minz or itx[2]>maxz:
        return False
    return True


def plot_bbox(work2obj, bbox, axis=None):

    if axis is None:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.autoscale(False)

    corners, _ = get_corners_and_plances(bbox)
    for corner in corners:
        new_corner = np.dot(work2obj, cvt4d(corner))[:3]
        axis.scatter(*new_corner, color='r', marker='x', s=125)

    return axis


def plot_candidate(start, end=None, axis=None):

    if axis is None:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.autoscale(False)

    axis.scatter(*start, color='g', marker='o', s=5)

    if end is not None:
        axis.scatter(*end, color='r', marker='*', s=5)
        axis.plot([start[0], end[0]],
                  [start[1], end[1]],
                  [start[2], end[2]], 'k-')

    return axis


def generate_candidates(bbox, work2obj, obj2grip):

    # Extract the bounding box corners and planes of the object
    corners, planes = get_corners_and_plances(bbox)

    xyz_grid = [to_rad(x, y, z)
                for x in range(0,360, GLOBAL_X) \
                for y in range(0,360, GLOBAL_Y) \
                for z in range(0,360, GLOBAL_Z)]

    total_trials = len(xyz_grid)**2

    # Angles holds the information about global and local rotations
    # Matrices holds information about the final transformation.
    angles = np.empty((total_trials, 6))
    matrices = np.empty((total_trials, 12))

    # Going to extend a vector along z-direction of palm. The extra 1 o the end
    # is for multiplying with a 4x4 HT matrix
    palm_axis = np.atleast_2d([0, 0, 1, 1]).T

    curr_trial = 0
    success = 0

    # Exhaustively evaluate local and global rotations
    for gx, gy, gz in xyz_grid:
        for lx, ly, lz in xyz_grid:

            # Monitor our status, print something every 10%
            if curr_trial%int(0.1*total_trials)==0:
                print 'Trial %d/%d, successful: %d'%(curr_trial,
                                                     total_trials,
                                                     success)

            curr_trial += 1

            # Build the global rotation matrix using quaternions
            q_gx = tf.quaternion_about_axis(gx + ri(GLOBAL_X), [1, 0, 0])
            q_gy = tf.quaternion_about_axis(gy + ri(GLOBAL_Y), [0, 1, 0])
            q_gz = tf.quaternion_about_axis(gz + ri(GLOBAL_Z), [0, 0, 1])

            # Build the local rotation matrix using quaternions
            q_lx = tf.quaternion_about_axis(lx + ri(GLOBAL_X), [1, 0, 0])
            q_ly = tf.quaternion_about_axis(ly + ri(GLOBAL_Y), [0, 1, 0])
            q_lz = tf.quaternion_about_axis(lz + ri(GLOBAL_Z), [0, 0, 1])

            # Multiply global and local rotations
            global_rotation = tf.quaternion_multiply(q_gx, q_gy)
            global_rotation = tf.quaternion_multiply(global_rotation, q_gz)
            global_rotation = tf.quaternion_matrix(global_rotation)

            local_rotation = tf.quaternion_multiply(q_lx, q_ly)
            local_rotation = tf.quaternion_multiply(local_rotation, q_lz)
            local_rotation = tf.quaternion_matrix(local_rotation)

            rotation = np.dot(np.dot(global_rotation, obj2grip), local_rotation)

            # Don't want to test any candidates that are below the workspace
            workspace2grip = np.dot(work2obj, rotation)
            if workspace2grip[2,3] < 0:
                continue

            # Check if a line from the center of the grippers palm intersects
            # with the planes of the object 
            line_start = rotation[:3,3]
            line_end = np.dot(rotation, palm_axis).flatten()[:3]

            # Check the intersection of gripper palm ray and bounding box
            for p in planes:
                itx_plane = intersect_plane(line_start, line_end, *p)
                
                if intersect_box(itx_plane, *bbox) is False:
                    continue

                angles[success] = np.asarray([gx, gy, gz, lx, ly, lz])
                matrices[success] = rotation[:3].flatten()
                success = success + 1
                break
                
    # Only keep the successful transformations
    angles = np.asarray(angles)[:success]
    matrices = np.asarray(matrices)[:success]

    return angles, matrices


def main(df, to_keep=-1):

    if not os.path.exists(GLOBAL_SAVE_DIR):
        os.makedirs(GLOBAL_SAVE_DIR)
        
    # Parse the dataframe
    name = str(df[0])
    com = np.float32(df[1:4])
    inertia = np.float32(df[4:13])
    bbox = np.float32(df[13:19])
    
    # Homogeneous transform matrices (bottom row omitted)
    work2obj = np.float32(df[19:31]).reshape(3,4)
    obj2grip = np.float32(df[31:43]).reshape(3,4)

    # Reshape Homogeneous transform matrices from 3x4 into 4x4
    work2obj = format_htmatrix(work2obj)
    obj2grip = format_htmatrix(obj2grip)

    # Check that the position of the object isn't too far
    if any(work2obj[:, 3]>1):
        raise Exception('%s out of bounds'%name)

    # Generate candidates through local and global rotations
    angles, matrices = generate_candidates(bbox, work2obj, obj2grip)

    
    # ------------------------------------------------------------------------ 
    # Gather all the object information we'll need for running the sim
    success = angles.shape[0]

    if success == 0:
        raise Exception('No candidates generated. Try reducing step sizes?')
    print '%s \n# of successful TF: %d'%(name, success)

    # Choose how many candidates we want to save (-1 will save all)
    if to_keep == -1 or to_keep > success:
        to_keep = success
    random_idx = np.arange(success)
    np.random.shuffle(random_idx)
    
    # Save the data
    savefile = os.path.join(GLOBAL_SAVE_DIR, name+'.txt')
    csvfile = open(savefile, 'wb')
    writer = csv.writer(csvfile, delimiter=',')

    obj_mtx = work2obj[:3].flatten()
    for i in random_idx[:to_keep]:
        data = np.hstack([name, i, obj_mtx, angles[i], matrices[i], com, inertia])
        writer.writerow(data)
    csvfile.close()
        

    # ------------------------------------------------------------------------ 
    # To visualize the generated candidates, we need to transform the points
    # (which are the the objects reference frame) to the workspace frame
    fig = plot_mesh(name, work2obj)
    plot_bbox(work2obj, bbox, axis=fig)

    n_plot = np.minimum(500, success)
    points = matrices[:, [3,7,11]]

    np.random.shuffle(points)
    for i in xrange(n_plot):
        point = np.dot(work2obj, cvt4d(points[i]))[:3]
        plot_candidate(point, axis=fig)
    
    plt.savefig(os.path.join(GLOBAL_SAVE_DIR,name+'.png'))


if __name__ == '__main__':

    np.random.seed(np.random.randint(1, 1234567890))

    # We usually run this in parallel (using gnu parallel), so we pass in
    # a row of information at a time (i.e. from collecting initial poses)
    if len(sys.argv)==1:
        
        #path = 'C:/Users/Matt/Documents/grasping/initialize/initial_poses_v39.txt'
        #data = pd.read_csv(path, delimiter=',')
        #data = data.values
        #main(data[10])
        
        raise Exception('No input specified for generating hypothesis')    

    df = sys.argv[1]
    df = df.split(',')[:-1]
    main(df, to_keep = 10000)
