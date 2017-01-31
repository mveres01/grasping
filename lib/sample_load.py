"""
This file acts as a stand-alone module for loading the dataset and visualizing 
grasps/meshes.
"""

import os
import h5py
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def float32(x):
    return np.float32(x)


def format_htmatrix(matrix_in):
    """Formats a 3x3 rotation matrix into a 4x4 homogeneous matrix"""

    ht_matrix = np.eye(4)
    ht_matrix[:3] = matrix_in
    return ht_matrix


def format_point(point):
    """Formats a 3-element [x,y,z] vector as a 4-element vector [x,y,z,1]"""

    return np.hstack((point, 1))


def invert_htmatrix(htmatrix):
    """Inverts a homogeneous transformation matrix"""

    inv = np.eye(4)
    rot_T = htmatrix[:3, :3].T
    inv[:3, :3] = rot_T
    inv[:3, 3] = -np.dot(rot_T, htmatrix[:3, 3])
    return inv


def calc_mesh_centroid(trimesh_mesh, center_type='vrep'):
    """Calculates the center of a mesh according to three different metrics."""

    if center_type == 'centroid':
        return trimesh_mesh.centroid
    elif center_type == 'com':
        return trimesh_mesh.center_mass
    elif center_type == 'vrep': # How V-REP assigns object centroid
        maxv = np.max(trimesh_mesh.vertices, axis=0)
        minv = np.min(trimesh_mesh.vertices, axis=0)
        return 0.5*(minv+maxv)


def convert_grasp_frame(grasp_data, htmatrix):
    """Converts the reference frame of a grasp given a transform matrix."""

    n_fingers = len(grasp_data)/6
    positions = np.zeros((n_fingers, 3))
    normals = np.zeros((n_fingers, 3))

    for i in xrange(n_fingers):
        pos = grasp_data[i*3 : i*3 + 3]
        nml = grasp_data[n_fingers*3 + i*3:n_fingers*3 + i*3 + 3]

        if htmatrix is not None:
            pos = np.dot(htmatrix, format_point(pos))[:3]
            nml = np.dot(htmatrix[:3, :3], nml)
        positions[i] = pos
        normals[i] = nml

    # Contact normals point outwards, so make them point inwards
    normals = - normals / np.sqrt(np.sum(normals**2, axis=1))
    return (positions.flatten(), normals.flatten())


def plot_equal_aspect(vertices, axis):
    """Forces a matplotlib plot to maintain an equal aspect ratio.

    Notes
    -----
    See: http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    """

    max_dim = np.max(np.array(np.max(vertices, axis=0) - np.min(vertices, axis=0)))

    mid = 0.5*np.max(vertices, axis=0) + np.min(vertices, axis=0)
    axis.set_xlim(mid[0] - max_dim, mid[0] + max_dim)
    axis.set_ylim(mid[1] - max_dim, mid[1] + max_dim)
    axis.set_zlim(mid[2] - max_dim, mid[2] + max_dim)

    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')

    return axis


def plot_grasps(data, axis=None):
    """Displays/Plots grasp configurations, on an (optional) plt axis."""

    # Create figure and set ranges
    if axis is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_ylim([-0.25, 0.25])
        ax.set_xlim([-0.25, 0.25])
        ax.set_zlim([-0.25, 0.25])
    else:
        ax = axis

    # Gaussian KDE plots frequency of position using colour, but
    for sample in data:
        for i in range(0, 9, 3):
            ax.scatter(sample[i], sample[i+1], sample[i+2], marker='o', c='r')

    return ax


def plot_mesh(mesh_name, workspace2obj, mesh_dir, axis=None):
    """Visualize where we will sample grasp candidates from

    Parameters
    ----------
    mesh_name : name of the mesh (assuming it can be found in given path)
    workspace2obj : 4x4 transform matrix from the workspace to object
    axis : an (optional) matplotlib axis in which we can plot a mesh
    """

    if axis is None:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.autoscale(False)

    if not '.obj' in mesh_name:
        mesh_name = mesh_name + '.obj'

    # Load the object mesh
    path = os.path.join(mesh_dir, mesh_name)

    if not os.path.isfile(path):
        raise Exception('Mesh %s does not exist in path \'%s\''%(mesh_name,
                                                                 mesh_dir))

    mesh = trimesh.load_mesh(path)

    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    center = calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center

    # Rotate the vertices so they're in the frame of the workspace
    mesh.apply_transform(workspace2obj)

    # Construct a 3D mesh via matplotlibs 'PolyCollection'
    poly = Poly3DCollection(mesh.triangles, linewidths=0.05, alpha=0.25)
    poly.set_facecolor([0.5, 0.5, 1])
    axis.add_collection3d(poly)

    axis = plot_equal_aspect(mesh.vertices, axis)

    return axis


def plot_grasps_and_mesh(grasps, props, mesh_dir=None):
    """Takes the name of an object and plots associated grasps."""

    object_name = '94_weight_final-19-Nov-2015-08-53-47'

    print 'Visualizing grasps for: %s'%object_name

    # Find the grasps for a specific object
    object_idx, _ = np.where(props['object_name'] == object_name)
    object_props = {key: props[key][object_idx] for key in props.keys()}
    object_grasps = grasps[object_idx]

    # Homogeneous transform matrix for workspace to object mapping
    work2obj = None

    # Only want the fingertip positions
    work2grasp = np.zeros((len(object_idx), 9))

    for i in xrange(object_grasps.shape[0]):

        work2cam = object_props['frame_work2cam_otm'][i]
        work2cam = format_htmatrix(work2cam.reshape(3, 4))

        world2work = object_props['frame_world2work'][i]
        world2work = format_htmatrix(world2work.reshape(3, 4))
        work2world = invert_htmatrix(world2work)

        # Find the workspace2object matrix. This is same across all attempts
        if work2obj is None:
            world2obj = object_props['frame_world2obj'][i]
            world2obj = format_htmatrix(world2obj.reshape(3, 4))
            work2obj = np.dot(work2world, world2obj)

        work2grasp[i], _ = convert_grasp_frame(object_grasps[i], work2cam)

    ax = plot_grasps(work2grasp)
    if mesh_dir is not None:
        ax = plot_mesh(object_name, work2obj, mesh_dir, axis=ax)

    save_name = object_name+'.png'
    plt.savefig(save_name, bbox_inches='tight')


def load_dataset_hdf5(fname, subset='train', n_samples=-1):
    """Loads a dataset of images/grasps.

    Notes
    -----
    Each grasp is stored as an 27-dimensional vector of type:
    [p1, p2, p3, n1, n2, n3, f1, f2, f3], where:
    * p_i = contact position of finger i
    * n_i = contact normal of finger i
    * f_i = contact force of finger i

    In this work, we only use the positions and normals due to the closing
    strategy of the gripper, although the forces are recorded as well.
    """

    # Only a few groups available
    if subset not in ['train', 'test', 'valid']:
        raise Exception('Subset needs to be one of: train, test, valid')
    elif not isinstance(subset, str):
        raise Exception('Subset must be a string')

    # Choose to load all, or only a portion of data
    if n_samples < 0:
        idx = slice(0, None)
    else:
        idx = slice(0, n_samples)

    # Try loading the file
    try:
        f = h5py.File(fname, 'r')
    except Exception as e:
        print 'File %s not found. Enter the path to the file'%fname

    images = float32(f[subset]['images'][idx])
    grasps = float32(f[subset]['grasps'][idx, :18])
    props = {}
    for key in f[subset]['object_props'].keys():
        data = f[subset]['object_props'][key][idx]
        if key != 'object_name':
            props[key] = data.astype(np.float32)
        else:
            props[key] = data.astype(str)

    return images, grasps, props


if __name__ == '__main__':

    DATA_PATH = '/scratch/mveres/grasping/data/processed/grasping_otm.hdf5'
    train = load_dataset_hdf5(DATA_PATH, subset='train', n_samples=-1)
    test = load_dataset_hdf5(DATA_PATH, subset='test', n_samples=-1)
    valid = load_dataset_hdf5(DATA_PATH, subset='valid', n_samples=-1)

    train_images, train_grasps, train_props = train
    test_images, test_grasps, test_props = test
    valid_images, valid_grasps, valid_props = valid

    print 'train_images.shape: ', train_images.shape
    print 'test_images.shape: ', test_images.shape
    print 'valid_images.shape: ',valid_images.shape

    #MESH_OBJECT_DIR = '/scratch/mveres/grasping/data/meshes/object_files'
    MESH_OBJECT_DIR = None
    plot_grasps_and_mesh(train_grasps, train_props, mesh_dir=MESH_OBJECT_DIR)
