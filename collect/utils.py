import os
import csv
import numpy as np
import trimesh
from trimesh import transformations as tf

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt

from PIL import Image

def visualize_mesh(mesh, vis_center=False):
    """Takes a trimesh object, and visualizes it using matplotlib

    This is an alternative method for plotting, instead of using the
    trimesh.show() function
    """

    # Create a new plot
    plt.close('all')
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.autoscale(False)

    # If we want to visualize the center of the object
    if vis_center == 'centroid':
        center = mesh.centroid
    elif vis_center == 'com':
        center = mesh.center_mass
    elif vis_center == 'vrep': # How V-REP assigns object centroid
        maxv = np.max(mesh.vertices, axis=0)
        minv = np.min(mesh.vertices, axis=0)
        center = 0.5*(minv+maxv)
    if vis_center is not False:
        axes.scatter(*center, c='r', marker='o', s=75)

    # Construct a 3D mesh using the mesh faces, and matplotlibs PolyCollection 
    poly = Poly3DCollection(mesh.triangles, linewidths=0.05, alpha=0.25)
    poly.set_facecolor([0.5,0.5,1])
    axes.add_collection3d(poly)

    # Want to center the plot around the object, while maintaining an equal
    # aspect ratio
    # See: http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    max_range = np.array(
        np.max(mesh.vertices,axis=0) - np.min(mesh.vertices,axis=0)).max()
    mid = np.max(mesh.vertices,axis=0) + np.min(mesh.vertices,axis=0)*0.5
    axes.set_xlim(mid[0] - max_range, mid[0] + max_range)
    axes.set_ylim(mid[1] - max_range, mid[1] + max_range)
    axes.set_zlim(mid[2] - max_range, mid[2] + max_range)

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    #plt.show()
    return axes


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


