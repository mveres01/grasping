import os
import csv
import numpy as np
import trimesh
from trimesh import transformations as tf

from scipy import optimize
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt

from PIL import Image


def calc_mesh_centroid(trimesh_mesh,  center_type='vrep'):
    """Calculates the center of a mesh according to three different metrics."""
    
    if center_type == 'centroid':
        return trimesh_mesh.centroid
    elif center_type == 'com':
        return trimesh_mesh.center_mass
    elif center_type == 'vrep': # How V-REP assigns object centroid
        maxv = np.max(trimesh_mesh.vertices, axis=0)
        minv = np.min(trimesh_mesh.vertices, axis=0)
        return 0.5*(minv+maxv)

    
def plot_equal_aspect(vertices, axis):
    """Forces the plot to maintain an equal aspect ratio
    
    # See:
    # http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
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


def plot_mesh(mesh_name, workspace2obj, axis=None):
    """Visualize where we will sample grasp candidates from

    Parameters
    ----------
    mesh_name : name of the mesh (assuming it can be found in given path)
    workspace2obj : 4x4 transform matrix from the workspace to object
    obj2gripper : 4x4 transform matrix from the object to gripper
    corners : (n,3) list of bounding box corners, in objects frame
    points : grasp candidates where the gripper palm intersects bbox
    """

    if axis is None:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.autoscale(False)

    # Load the object mesh
    path = os.path.join(GLOBAL_MESH_DIR, mesh_name)
    mesh = trimesh.load_mesh(path+'.stl')

    # V-REP encodes the object centroid as the literal center of the object, 
    # so we need to make sure the points are centered the same way
    center = calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center

    # Rotate the vertices so they're in the frame of the workspace
    mesh.apply_transform(workspace2obj)

    # Construct a 3D mesh via matplotlibs 'PolyCollection'
    poly = Poly3DCollection(mesh.triangles, linewidths=0.05, alpha=0.25)
    poly.set_facecolor([0.5,0.5,1])
    axis.add_collection3d(poly)

    axis = plot_equal_aspect(mesh.vertices, axis)

    return axis


def format_htmatrix(matrix_in):
    """Formats a 3x3 rotation matrix into a 4x4 homogeneous matrix"""

    ht_matrix = np.eye(4)
    ht_matrix[:3] = matrix_in
    return ht_matrix


def format_point(point):
    """Formats a 3-element [x,y,z] vector as a 4-element vector [x,y,z,1]"""

    return np.hstack((point,1))


def invert_htmatrix(htmatrix):
    """Inverts a homogeneous transformation matrix"""

    inv = np.eye(4)
    rot_T = htmatrix[:3,:3].T
    inv[:3,:3] = rot_T
    inv[:3, 3] = -np.dot(rot_T, htmatrix[:3,3])
    return inv


def rot_x(theta):
    """Builds a 3x3 rotation matrix around x

    Parameters
    ----------
    theta : angle of rotation in degrees
    """

    theta = theta*math.pi/180.
    mat = np.asarray(
            [[1, 0,                0],
             [0, math.cos(theta), -math.sin(theta)],
             [0, math.sin(theta),  math.cos(theta)]])
    return mat


def rot_y(theta):
    """Builds a 3x3 rotation matrix around y

    Parameters
    ----------
    theta : angle of rotation in degrees
    """

    theta = theta*math.pi/180.
    mat = np.asarray(
            [[math.cos(theta), 0, math.sin(theta)],
             [0,               1, 0],
             [-math.sin(theta),0, math.cos(theta)]])
    return mat


def rot_z(theta):
    """Builds a 3x3 rotation matrix around z

    Parameters
    ----------
    theta : angle of rotation in degrees
    """

    theta = theta*math.pi/180.
    mat = np.asarray(
             [[math.cos(theta), -math.sin(theta), 0],
              [math.sin(theta),  math.cos(theta), 0],
              [0,                0,               1]])
    return mat


def rxyz(thetax, thetay, thetaz):
    """Calculates rotation matrices by multiplying in the order x,y,z

    Parameters
    ----------
    thetax : rotation around x in degrees
    thetay : rotation around y in degrees
    thetaz : rotation around z in degrees
    """
    
    # Convert radians to degrees
    rx = tf.rotation_matrix(thetax, [1,0,0])
    ry = tf.rotation_matrix(thetay, [0,1,0])
    rz = tf.rotation_matrix(thetaz, [0,0,1])
    rxyz = tf.concatenate_matrices(rx,ry,rz)

    return rxyz


def sample_images(hdf5_file, fname, image_dir):
    """Randomly samples a few images from the object file"""

    length = hdf5_file['GRIPPER_IMAGE'].shape[0]

    for idx in xrange(length):

        if np.random.rand() < 0.99:
            continue

        image_col = np.squeeze(hdf5_file['GRIPPER_IMAGE_COLOUR'][idx])
        image_mask = np.squeeze(hdf5_file['GRIPPER_IMAGE_MASK'][idx])
        image_depth = np.squeeze(hdf5_file['GRIPPER_IMAGE'][idx])

        channels, rows, cols = image_col.shape


        # Convert the multi-channel images to grayscale
        image_col = image_col*255.0
        image_col = image_col.transpose(1,2,0)
        image_col = image_col.astype(np.uint8)
        image_bw = Image.fromarray(image_col).convert('L')
        image_bw = np.array(image_bw, dtype=np.float32)/255.0

        # Get ride of channel column on depth image
        image_depth = np.squeeze(image_depth)

        # Combine them into a single image
        im = Image.new('L',(cols*3, rows*1))
        im.paste(Image.fromarray(np.uint8(image_bw*255.0)), \
                                (0, 0, cols, rows))
        im.paste(Image.fromarray(np.uint8(image_depth*255.0)),\
                (cols, 0, cols*2,rows))
        im.paste(Image.fromarray(np.uint8(image_mask*255.0)), \
                (cols*2, 0, cols*3, rows))
        obj_name = fname[:-4]
        im.save(os.path.join(image_dir, obj_name+str(idx)+'.png'))


def sample_poses(hdf5_file, fname, image_dir):
    """Randomly samples a few poses from the object file

    This function can be used to verify that we were able to accurately estimate
    object pose from the given image
    """

    length = hdf5_file['GRIPPER_IMAGE'].shape[0]

    csvfile = open(image_dir+'sample_poses.txt','a+')
    writer = csv.writer(csvfile, delimiter=',')

    for idx in xrange(length):

        if np.random.rand() < 0.99:
            continue

        name = hdf5_file['OBJECT_NAME'][idx]
        world2obj = hdf5_file['GRIPPER_PREGRASP']['world2obj'][idx]
        obj2cam = hdf5_file['GRIPPER_PREGRASP']['obj2cam'][idx]
        cam2img = hdf5_file['GRIPPER_PREGRASP']['cam2img'][idx]
        unproj_z = hdf5_file['GRIPPER_PREGRASP']['unproj_z'][idx]
        unproj_y = hdf5_file['GRIPPER_PREGRASP']['unproj_y'][idx]

        world2obj_mat = format_htmatrix(world2obj.reshape(3,4))
        obj2cam_mat = format_htmatrix(obj2cam.reshape(3,4))
        cam2img_mat = format_htmatrix(cam2img.reshape(3,4))

        world2cam = np.dot(world2obj_mat, obj2cam_mat)
        world2img = np.dot(world2cam, cam2img_mat)

        writer.writerow(np.hstack(\
             [name, world2obj, world2img[:3].flatten(),
             world2cam[:3].flatten(), unproj_z, unproj_y]))
    csvfile.close()



def get_maximized_prediction(sampled_mu, sampled_ls, eps=1e-6):
    """Compute y = argmax( 1/L Sum(likelihood)). """

    assert(sampled_mu.shape==sampled_ls.shape)

    c = -0.5*np.log(2.0*np.pi)

    def LogSumExp(x, axis=1):
            
        x_max = np.max(x, axis=axis, keepdims=True)
        return x_max + np.log(np.sum(np.exp(x - x_max), \
                             axis=axis, keepdims=True))

    def log_likelihood(tgt, *args):
        """Calculates log-likelihood for a given set of target variabels."""
    
        mu, ls  = args
        ll = c - ls - ((tgt-mu)**2) / (2.0*np.exp(2.0*ls)+eps)
    
        # Calc **negative** log-likelihood
        nll = np.log(tgt.shape[0]) - LogSumExp(np.sum(ll,axis=1), axis=None)
        return nll


    x0 = np.zeros((sampled_mu.shape[1], ))
    soln = optimize.minimize(log_likelihood, x0, method='cg', \
                             args=(sampled_mu, sampled_ls), tol=1e-4) 

    # Return the prediction and whether or not optimization succeeded
    return np.atleast_2d(soln.x), soln.fun


def load_dataset_hdf5(fname, sample_exp=17, seed=1234):

    f = h5py.File(fname,'r')

    train_images = f['train']['images'][:].astype(theano.config.floatX)
    train_grasps = f['train']['grasps'][:].astype(theano.config.floatX)
    train_labels = f['train']['labels'][:]
    train_props  = f['train']['object_props'][:]

    test_images = f['test']['images'][:].astype(theano.config.floatX)
    test_grasps = f['test']['grasps'][:].astype(theano.config.floatX)
    test_labels = f['test']['labels'][:]
    test_props  = f['test']['object_props'][:]

    valid_images = f['valid']['images'][:].astype(theano.config.floatX)
    valid_grasps = f['valid']['grasps'][:].astype(theano.config.floatX)
    valid_labels = f['valid']['labels'][:]
    valid_props  = f['valid']['object_props'][:]
    f.close()


    idx_len = np.minimum(int(2**sample_exp), train_images.shape[0])

    np.random.seed(seed)
    indices = np.arange(train_images.shape[0])
    np.random.shuffle(indices)

    train = [train_grasps[indices[:idx_len]],
             train_images[indices[:idx_len]],
             train_labels[indices[:idx_len]],
             train_props[indices[:idx_len]]]

    # Always use the sample test and valid sets, only change the train set
    test  = [test_grasps, test_images, test_labels, test_props]
    valid = [valid_grasps, valid_images, valid_labels, valid_props]

    return train, test, valid


def split_similar_objects(list1, list2):
    """Finds the "classes" in list2 that also appear in list1."""

    indices = np.arange(list2.shape[0])

    # Get the class number from objects in the test set
    list2_labels = [(l.split('/')[-1]).split('_')[0] for l in list2[:,0]]

    # Get the object name from items in the train set
    list1_labels = [l.split('_')[0] for l in list1]

    # Similar objects are those objects in the test set that are similar
    # to objects seen in the training set
    similar_objects = np.asarray(\
            [True if l in list1_labels else False for l in list2_labels])
    different_objects = np.asarray(\
            [False if l in list1_labels else True for l in list2_labels])

    return indices[similar_objects], indices[different_objects]


def get_network_info(network_layer, input_layer = None):
    """Given a network layer, output the name, shape, and # parameters."""

    # Get all the layers of the network
    all_layers = nn.get_all_layers(network_layer, treat_as_input=input_layer)

    # Initialize a counter to hold parameter count
    network_info = []

    prev_params = 0
    for layer in all_layers:
        params = nn.count_params(layer) - prev_params
        shape  = nn.get_output_shape(layer)

        info = '{:<20s} shape: {:<20s} params: {:<5}'\
                .format(layer.name, shape, params)

        network_info.append(info)
        prev_params += params

    # Find the total number of params
    total_params = nn.count_params(network_layer)
    info = ' Total number of parameters: {:<20}'.format(total_params)
    network_info.append(info)
    return network_info






