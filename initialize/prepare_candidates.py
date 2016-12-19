import os
import sys
sys.path.append('..')

import csv
import numpy as np
import pandas as pd
import trimesh.transformations as tf

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.utils import plot_mesh, format_htmatrix
from lib.python_config import (config_mesh_dir, config_candidate_dir,
                               config_pose_path)

# Controls how big the rotation angles in each direction are
GLOBAL_X = 45
GLOBAL_Y = 45
GLOBAL_Z = 45


def rand_step(max_angle):
    """Returns a random point between (0, max_angle."""
    return np.float32(np.random.randint(0, max_angle))*np.pi/180.


def to_rad(deg_x, deg_y, deg_z):
    """Converts a tuple of (x, y, z) in degrees to radians."""
    return (deg_x*np.pi/180., deg_y*np.pi/180., deg_z*np.pi/180.)


def cvt4d(point_3d):
    """Helper function to quickly format a 3d point into a 4d vector."""
    return np.hstack([point_3d, 1])


def get_mesh_properties(data_vector):
    """Parses information on objects pose collected via sim."""

    mesh_props = {}
    mesh_props['name'] = str(data_vector[0])
    mesh_props['com'] = np.float32(data_vector[1:4])
    mesh_props['inertia'] = np.float32(data_vector[4:13])
    mesh_props['bbox'] = np.float32(data_vector[13:19])

    work2obj = np.float32(data_vector[19:31]).reshape(3, 4)
    obj2grip = np.float32(data_vector[31:43]).reshape(3, 4)

    # Check that the position of the object isn't too far
    if any(work2obj[:, 3] > 1):
        raise Exception('%s out of bounds'%mesh_props['name'])

    # Reshape Homogeneous transform matrices from 3x4 into 4x4
    mesh_props['work2obj'] = format_htmatrix(work2obj)
    mesh_props['obj2grip'] = format_htmatrix(obj2grip)

    return mesh_props


def get_corners_and_plances(bbox):
    """Caclucates the bounding box and 8 planes given min/max x,y,z values"""

    minx, maxx, miny, maxy, minz, maxz = bbox

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


def intersect_plane(pt_a, pt_b, pt_0, pt_1, pt_2):
    """Checks the intersect_plane of a vector and a plane in 3d space

    Parameters
    ----------
    pt_a : a point along the line.
    pt_b : a second point along the line.
    pt_0, pt_1, pt_2 : three different points on the plane.

    Returns
    -------
    Point of intersect_plane if the line and plane intersect, otherwise False.
    See: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersect_plane
    """

    mat = np.asarray(
        [[pt_a[0] - pt_b[0], pt_1[0] - pt_0[0], pt_2[0] - pt_0[0]],
         [pt_a[1] - pt_b[1], pt_1[1] - pt_0[1], pt_2[1] - pt_0[1]],
         [pt_a[2] - pt_b[2], pt_1[2] - pt_0[2], pt_2[2] - pt_0[2]]])

    vec = np.asarray(pt_a-pt_0)

    try:
        inv_mat = np.linalg.pinv(mat)
    except Exception as e:
        return False

    soln = np.dot(inv_mat, vec)

    itx = pt_a + (pt_b-pt_a)*soln[0]

    # Check if line is along norm (e.g. positive instead of negative)
    return itx if soln[0] > 0 else False


def intersect_box(itx, bbox):
    """Calculates whether the intersect_plane is within the bounding box."""

    if isinstance(itx, bool):
        return False

    minx, maxx, miny, maxy, minz, maxz = bbox

    if not minx <= itx[0] <= maxx or \
       not miny <= itx[1] <= maxy or \
       not minz <= itx[2] <= maxz:
        return False
    return True


def plot_bbox(work2obj, bbox, axis=None):
    """Plots the objects bounding box on an (optional) given matplotlib fig."""

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
    """Plots grasp candidates the algorithm has identified."""

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


def generate_candidates(mesh_properties):
    """Generates grasp candidates via predicted 6DoF pose of gripper."""

    bbox = mesh_properties['bbox']
    work2obj = mesh_properties['work2obj']
    obj2grip = mesh_properties['obj2grip']

    # Extract the bounding box corners and planes of the object
    _, planes = get_corners_and_plances(bbox)

    xyz_grid = [to_rad(x, y, z)
                for x in range(0, 360, GLOBAL_X) \
                for y in range(0, 360, GLOBAL_Y) \
                for z in range(0, 360, GLOBAL_Z)]

    total_trials = len(xyz_grid)**2

    # Angles holds the information about global and local rotations
    # Matrices holds information about the final transformation.
    angles = np.empty((total_trials, 6))
    matrices = np.empty((total_trials, 12))

    # Going to extend a vector along z-direction of palm. The extra 1 o the end
    # is for multiplying with a 4x4 HT matrix
    palm_axis = np.atleast_2d([0, 0, 10, 1]).T

    curr_trial = 0
    success = 0

    # Exhaustively evaluate local and global rotations
    for gx, gy, gz in xyz_grid:
        for lx, ly, lz in xyz_grid:

            # Monitor our status, print something every 10%
            if curr_trial%int(0.1*total_trials) == 0:
                print 'Trial %d/%d, successful: %d'%\
                (curr_trial, total_trials, success)

            curr_trial += 1

            # Build the global rotation matrix using quaternions
            q_gx = tf.quaternion_about_axis(gx + rand_step(GLOBAL_X), [1, 0, 0])
            q_gy = tf.quaternion_about_axis(gy + rand_step(GLOBAL_Y), [0, 1, 0])
            q_gz = tf.quaternion_about_axis(gz + rand_step(GLOBAL_Z), [0, 0, 1])

            # Build the local rotation matrix using quaternions
            q_lx = tf.quaternion_about_axis(lx + rand_step(GLOBAL_X), [1, 0, 0])
            q_ly = tf.quaternion_about_axis(ly + rand_step(GLOBAL_Y), [0, 1, 0])
            q_lz = tf.quaternion_about_axis(lz + rand_step(GLOBAL_Z), [0, 0, 1])

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
            if workspace2grip[2, 3] < 0:
                continue

            # Check if a line from the center of the grippers palm intersects
            # with the planes of the object
            line_start = rotation[:3, 3]
            line_end = np.dot(rotation, palm_axis).flatten()[:3]

            for plane in planes:
                itx_plane = intersect_plane(line_start, line_end, *plane)

                if intersect_box(itx_plane, bbox) is False:
                    continue

                angles[success] = np.asarray([gx, gy, gz, lx, ly, lz])
                matrices[success] = rotation[:3].flatten()
                success = success + 1
                break

    # Only keep the successful transformations
    angles = np.asarray(angles)[:success]
    matrices = np.asarray(matrices)[:success]

    return angles, matrices


def main(to_keep=-1):

    # We usually run this in parallel (using gnu parallel), so we pass in
    # a row of information at a time (i.e. from collecting initial poses)
    # This is for test purposes only
    if len(sys.argv) == 1:
        data_vector = pd.read_csv(config_pose_path, delimiter=',')
        data_vector = (data_vector.values)[10]
    else:
        data_vector = sys.argv[1]
        data_vector = data_vector.split(',')[:-1]

    mesh_properties = get_mesh_properties(data_vector)

    # Generate candidates through local and global rotations
    angles, matrices = generate_candidates(mesh_properties)

    success = angles.shape[0]
    if success == 0:
        print 'No candidates generated. Try reducing step sizes?'
        return
    print '%s \n# of successful TF: %d'%(mesh_properties['name'], success)

    # Choose how many candidates we want to save (-1 will save all)
    if to_keep == -1 or to_keep > success:
        to_keep = success

    random_idx = np.arange(success)
    np.random.shuffle(random_idx)

    # Save the data
    savefile = os.path.join(config_candidate_dir, mesh_properties['name'] + '.txt')
    csvfile = open(savefile, 'wb')
    writer = csv.writer(csvfile, delimiter=',')

    obj_mtx = mesh_properties['work2obj'][:3].flatten()
    for i in random_idx[:to_keep]:
        writer.writerow(np.hstack(\
            [mesh_properties['name'], i, obj_mtx, angles[i], matrices[i],
             mesh_properties['com'], mesh_properties['inertia']]))
    csvfile.close()

    # ------------------------------------------------------------------------
    # To visualize the generated candidates, we need to transform the points
    # (which are the the objects reference frame) to the workspace frame
    mesh_path = os.path.join(config_mesh_dir, mesh_properties['name'] + '.stl')
    fig = plot_mesh(mesh_path, mesh_properties['work2obj'])
    plot_bbox(mesh_properties['work2obj'], mesh_properties['bbox'], axis=fig)

    points = matrices[:, [3, 7, 11]]
    np.random.shuffle(points)
    for i in xrange(np.minimum(500, success)):
        point = np.dot(mesh_properties['work2obj'], cvt4d(points[i]))[:3]
        plot_candidate(point, axis=fig)

    plt.savefig(os.path.join(config_candidate_dir, mesh_properties['name'] + '.png'))


if __name__ == '__main__':

    np.random.seed(np.random.randint(1, 1234567890))

    if not os.path.exists(config_candidate_dir):
        os.makedirs(config_candidate_dir)

    main(to_keep=10000)
