import os
import sys
import csv
import cv2
import h5py
import numpy as np
from PIL import Image

from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from utils import format_htmatrix, format_point, invert_htmatrix
from utils import sample_images


# Raw files are hosted on the local GPU drives
GLOBAL_PROJECT_DIR = '/home/robot/Documents/grasping'
GLOBAL_RAW_DATA_DIR = '/scratch/mveres/collected'
GLOBAL_PROCESSED_DIR = os.path.join(GLOBAL_PROJECT_DIR, 'data/processed')
GLOBAL_SAMPLE_IMAGE_DIR = os.path.join(GLOBAL_PROCESSED_DIR, 'sample_images')
GLOBAL_SAMPLE_POSE_DIR = os.path.join(GLOBAL_PROCESSED_DIR, 'sample_poses')

GLOBAL_IMAGE_WIDTH = 128
GLOBAL_IMAGE_HEIGHT = 128
GLOBAL_NEAR_CLIP = 0.01
GLOBAL_FAR_CLIP = 0.7
GLOVAL_FOV = 50.*np.pi/180.


# Helpers
def float32(data):
    return np.float32(data)


def reshape(data, shape=(3, 3)):
    return data.reshape(shape)


def get_unique_idx(data_in, n_nbrs=2, thresh=1e-4):
    """Finds the unique elements of a dataset using NearestNeighbors algorithm

    Parameters
    ----------
    data_in : array of datapoints
    n : number of nearest neighbours to find
    thresh : float specifying how close two items must be to be considered
        duplicates

    Notes
    -----
    The nearest neighbour algorithm will usually flag the query datapoint
    as being a neighbour. So we generally need n>1
    """

    if data_in.shape[0] == 1:
        return np.asarray([True], dtype=bool)

    # Scale the data so points are weighted equally/dont get misrepresented
    scaler = preprocessing.StandardScaler().fit(data_in)
    data_in = scaler.transform(data_in)
    nbrs = NearestNeighbors(n_neighbors=n_nbrs, algorithm='brute').fit(data_in)


    # This vector will contain a list of all indices that may be duplicated.
    # We're going to use each datapoint as a query.
    exclude_vector = np.zeros((data_in.shape[0],), dtype=bool)
    for i in xrange(data_in.shape[0]):

        # If we've already classified the datapoint at this index as being a
        # duplicate, there's no reason to process it again
        if exclude_vector[i:i+1] is True:
            continue

        # Find how close each point is to the query. If we find a point that
        # is less then some threshold, we add it to our exlude list
        distances, indices = nbrs.kneighbors(data_in[i:i+1])
        if distances[0, 0] <= thresh and indices[0, 0] != i:
            exclude_vector[indices[0, 0]] = True
        if distances[0, 1] <= thresh and indices[0, 1] != i:
            exclude_vector[indices[0, 1]] = True

    # Return a list of indices that represent unique elements of the dataset
    return np.bitwise_not(exclude_vector)


def get_outlier_mask(data_in, m=3):
    """Find dataset outliers by whether or not it falls within a given number
    of standard deviations from a given population

    Parameters
    ----------
    data_in : array of (n_samples, n_variables) datapoints
    m : number of standard deviations to evaluate
    """

    if data_in.ndim == 1:
        data_in = np.atleast_2d(data_in).T

    # Collect mean and std
    mean = np.mean(data_in, axis=0)
    std = np.std(data_in, axis=0)

    # Create a boolean mask of data within *m* std
    mask = abs(data_in - mean) < m*std
    mask = np.sum(mask, axis=1)

    # Want samples where all variables are within a 'good' region
    return mask == data_in.shape[1]


# Assumes field of view (GLOVAL_FOVy) is given in radians
def unprojectPoint(px, py, depth, fov_y, image_shape):

    # Make sure the values are encoded as floating point numbers
    px = np.float32(px)
    py = np.float32(py)
    scene_y = -(2.*(py-0.)/image_shape[0] - 1.0)*depth*np.tan(fov_y/2.)
    scene_x = -(2.*(px-0.)/image_shape[1] - 1.0)*depth*np.tan(fov_y/2.)

    return np.asarray([scene_x, scene_y, depth])


def get_image_matrix(y_axis, z_axis, center):

    # compute the rotational elements of the homogeneous transform matrix
    z_axis = z_axis/(np.sqrt(np.sum(z_axis**2)))
    y_axis = y_axis/(np.sqrt(np.sum(y_axis**2)))

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis/(np.sqrt(np.sum(x_axis**2)))

    # Camera to center of image is just a translation to estimated CoM
    cam2img = np.eye(4)
    cam2img[:3, 3] = center

    rmat = np.eye(4)
    rmat[0, 0:3] = x_axis
    rmat[1, 0:3] = y_axis
    rmat[2, 0:3] = z_axis

    cam2img = np.dot(cam2img, rmat.T)

    return cam2img


def get_image_centroid(mask):
    """Given a binary mask image, returns the centroid."""

    # Flip the binary mask, so the image is in white, then find coords
    img = np.uint8((1.0 - mask)*255.0).copy()
    coordsx, coordsy = np.where(img > 0)
    coords = np.vstack([coordsy, coordsx]).T

    # Find object centroid using average of pixels in x and y direction
    cx = np.int8(np.mean(coords[:, 0]), dtype=np.int8)
    cy = np.int8(np.mean(coords[:, 1]), dtype=np.int8)

    # If the object doesn't exist at that location, find nearest pixel
    if int(mask[cy, cx]) != 0:
        distances = coords - np.atleast_2d([cx, cy])
        distances = np.sum(distances**2, axis=1)
        cx, cy = coords[np.argmin(distances)]

    return cx, cy, coords


def estimate_object_pose(mask, depth_image, fov_y=50*np.pi/180):

    estimated_poses = []
    unprojected_y = []
    unprojected_z = []

    n_samples, _, n_rows, n_cols = mask.shape

    for i in xrange(n_samples):

        if i % np.ceil(int(0.1*n_samples)) == 0:
            print '  Converting %d/%d'%(i, n_samples)

        cx, cy, coords = get_image_centroid(mask[i, 0])

        # Estimate the major axis via principle components of the binary image
        pca = PCA(n_components=2).fit(coords)
        pc1 = np.asarray(pca.components_[0])
        pc2 = np.asarray(pca.components_[1])

        # Use PCA estimates to compute major axes
        c = 10.0 # arbitrary
        dir_z = (np.floor(cx + 0.5 + pc1[0]*c), np.floor(cy + 0.5 + pc1[1]*c))
        dir_y = (np.floor(cx + 0.5 + pc2[0]*c), np.floor(cy + 0.5 + pc2[1]*c))

        # Get the depth information at estimated object center of mass
        # Then convert the image pixel to coordinateed realtive to camera
        depth = depth_image[i, 0, int(cy), int(cx)]
        com = unprojectPoint(cx, cy, depth, GLOVAL_FOV, [n_rows, n_cols])
        unproj_z = unprojectPoint(dir_z[0], dir_z[1], depth, fov_y, [n_rows, n_cols])
        unproj_y = unprojectPoint(dir_y[0], dir_y[1], depth, fov_y, [n_rows, n_cols])

        # Make sure the major vectors have direction only
        z_axis = unproj_z - com
        y_axis = unproj_y - com

        # Now we can build the homogeneous transform matrix
        cam2img = get_image_matrix(y_axis, z_axis, com)

        # Make sure x-axis points towards the object and not the camera. If it
        # points towards the camera, rotation 180 degrees around z-direction
        img2cam = invert_htmatrix(cam2img)
        if img2cam[0, 3] > 0:
            rmat = np.eye(4)
            rmat[0, :2] = np.asarray([np.cos(np.pi), -np.sin(np.pi)])
            rmat[1, :2] = np.asarray([np.sin(np.pi), np.cos(np.pi)])
            img2cam = np.dot(rmat, img2cam)
            cam2img = invert_htmatrix(img2cam)

        # Make sure the generated transform matrix is valid
        det = int(0.5 + np.linalg.det(cam2img[:3, :3]))
        if det != 1:
            raise Exception('Index %d \nComputed determinant (%2.4f) is '\
                            'not equal to 1.'%(i, det))

        unprojected_z.append(np.atleast_2d(unproj_z))
        unprojected_y.append(np.atleast_2d(unproj_y))
        estimated_poses.append(cam2img[:3].flatten())

        # Check whether the selected pixel is on object using ground truth
        if np.random.randn() > 0.99:
            img = np.uint8((1.0 - mask[i, 0])*255.0).copy()
            p1 = (cx + int(pc1[0]*15), cy + int(pc1[1]*15))
            p2 = (cx + int(pc2[0]*15), cy + int(pc2[1]*15))
            cv2.line(img, (cx, cy), p1, 125, 1)
            cv2.line(img, (cx, cy), p2, 125, 1)
            cv2.circle(img, (cx, cy), 2, 0)
            cv2.circle(img, (int(n_cols/2), int(n_rows/2)), 1, 0)

            pose = os.path.join(GLOBAL_SAMPLE_POSE_DIR, '%d.png'%np.random.randint(0, 12345))
            im = Image.new('L', (n_cols, n_rows))
            im.paste(Image.fromarray(img), (0, 0, n_cols, n_rows))
            im.save(pose)

    unprojected_z = np.vstack(unprojected_z)
    unprojected_y = np.vstack(unprojected_y)
    estimated_poses = np.vstack(estimated_poses)

    return estimated_poses, unprojected_z, unprojected_y



def convert_grasp_frame(frame2matrix, matrix2grasp):
    """Function for converting from one grasp frame to another.

    This is useful as transforming grasp positions requires multiplication of
    4x4 matrix, while contact normals (orientation) are multiplication of 3x3
    components (i.e. without positional components.
    """


    if frame2matrix.ndim == 1:
        frame2matrix = reshape(frame2matrix, (3, 4))
        frame2matrix = np.vstack([frame2matrix, [0, 0, 0, 1]])

    # A grasp is contacts, normals, and forces (3), and has (x,y,z) components
    num_fingers = int(matrix2grasp.shape[1]/9)
    contact_points = reshape(matrix2grasp[0, :num_fingers*3])
    contact_normals = reshape(matrix2grasp[0, num_fingers*3:num_fingers*6])
    contact_forces = reshape(matrix2grasp[0, num_fingers*6:])

    # Append a 1 to end of contacts for easier multiplication
    contact_points = np.hstack([contact_points, np.ones((3, 1))])

    # Convert positions, normals, and forces to object reference frame
    points = np.zeros((num_fingers, 3))
    forces = np.zeros(points.shape)
    normals = np.zeros(points.shape)

    for i in xrange(num_fingers):

        points[i] = np.dot(frame2matrix, contact_points[i:i+1].T)[:3].T
        forces[i] = np.dot(frame2matrix[:3, :3], contact_forces[i:i+1].T).T
        normals[i] = np.dot(frame2matrix[:3, :3], contact_normals[i:i+1].T).T

    grasp = np.vstack([points, normals, forces]).reshape(1, -1)

    return grasp


def decode_grasp(grasp_line, object_mask, object_depth_image):

    #if grasp_line['NumDiffObjectsColliding']>0:
    if grasp_line['AllTipsInContact'] != 1:
        return None
    elif grasp_line['NumDiffObjectsColliding'] != 0:
        return None

    # Check that the force sensor was active for this trial
    fs0 = grasp_line['forceSensorStatus0']
    fs1 = grasp_line['forceSensorStatus1']
    fs2 = grasp_line['forceSensorStatus2']
    if fs0 != 1 or fs1 != 1 or fs2 != 1:
        raise Exception('Force sensor was broken. Should not have happened.')

    #workspace_to_base = g['BarrettHand'][i]
    work2obj = grasp_line['object_matrix'].reshape(3, 4)
    work2obj = format_htmatrix(work2obj)
    obj2work = invert_htmatrix(work2obj)

    world2work = grasp_line['wrtObjectMatrix'].reshape(3, 4)
    world2work = format_htmatrix(world2work)
    work2world = invert_htmatrix(world2work)

    work2cam = grasp_line['cameraMatrix'].reshape(3, 4)
    work2cam = format_htmatrix(work2cam)
    cam2work = invert_htmatrix(work2cam)

    work2cam_var = np.atleast_2d(grasp_line['rot_variant_matrix'])
    work2cam_invar = np.atleast_2d(grasp_line['rot_invariant_matrix'])

    obj2cam = np.dot(obj2work, work2cam)
    obj2world = np.dot(obj2work, work2world)
    cam2world = np.dot(cam2work, work2world)
    world2obj = invert_htmatrix(obj2world)

    # Encode the grasp WRT estimated coordiante frame attached to img
    cam2img, unproj_z, unproj_y = \
        estimate_object_pose(object_mask, object_depth_image, GLOVAL_FOV)

    # Contact points, normals, and forces should be WRT world frame
    contact_points = np.hstack(
        [grasp_line['contactPoint0'],
         grasp_line['contactPoint1'],
         grasp_line['contactPoint2']])

    contact_normals = np.hstack(
        [grasp_line['contactNormal0'],
         grasp_line['contactNormal1'],
         grasp_line['contactNormal2']])

    contact_forces = np.hstack(
        [grasp_line['contactForce0'],
         grasp_line['contactForce1'],
         grasp_line['contactForce2']])

    world2grasp = np.hstack([contact_points, contact_normals, contact_forces])
    world2grasp = np.atleast_2d(world2grasp)

    work2grasp = convert_grasp_frame(work2world, world2grasp)

    work2com = np.atleast_2d(format_point(grasp_line['com']))
    work2mass = reshape(grasp_line['mass'], (1, 1))
    work2inertia = reshape(grasp_line['inertia'], (1, 9))

    # Make sure everything is the right dimension so we can later concatenate
    obj2cam = np.atleast_2d(obj2cam[:3].flatten())
    world2cam = invert_htmatrix(cam2world)
    world2cam = np.atleast_2d(world2cam[:3].flatten())
    world2obj = np.atleast_2d(world2obj[:3].flatten())
    world2work = np.atleast_2d(world2work[:3].flatten())

    return {'grasp_wrt_work':work2grasp,
            'inertia_wrt_work':work2inertia,
            'mass_wrt_work':work2mass,
            'com_wrt_work':work2com,
            'obj_wrt_world':world2obj,
            'workspace_wrt_world':world2work,
            'cam_wrt_work_variant':work2cam_var,
            'cam_wrt_work_invariant':work2cam_invar,
            'cam_wrt_obj':obj2cam,
            'cam_wrt_world':world2cam,
            'img_wrt_cam':cam2img,
            'unproj_z':unproj_z,
            'unproj_y':unproj_y}


def parse_grasp(line, header):
    """Parses a line of information following size convention in header."""

    # Make sure our data is always a 2-d array
    line = np.atleast_2d(line)

    grasp = {}
    current_pos = 0

    # Decode all the data into a python dictionary
    for i in range(0, len(header), 2):

        name = str(header[i])
        n_items = int(header[i+1])
        subset = line[:, current_pos:current_pos + n_items]

        try:
            subset = subset.astype(np.float32)
        except Exception as e:
            subset = subset.astype(str)

        grasp[name] = subset.ravel()
        current_pos += n_items

    return grasp


def parse_image(image_as_list, depth=False, mask=False):
    """Parses an image as either: RGB, Depth, or a mask."""

    image_pixels = GLOBAL_IMAGE_WIDTH*GLOBAL_IMAGE_HEIGHT
    if len(image_as_list)/image_pixels < 1:
        print 'Invalid image size (require %dx%dxn)'%(GLOBAL_IMAGE_HEIGHT,
                                                      GLOBAL_IMAGE_WIDTH)
        return None

    # Convert the list into an array
    image = np.asarray(image_as_list, dtype=np.float32)

    # Make sure the number of channels is at the front of the matrix
    image = image.reshape(GLOBAL_IMAGE_HEIGHT, GLOBAL_IMAGE_WIDTH, -1)

    # Decode depth info
    if depth is True:
        image = GLOBAL_NEAR_CLIP + image*(GLOBAL_FAR_CLIP - GLOBAL_NEAR_CLIP)
    # Convert 3 channel RGB to grayscale / binary image
    elif mask is True:
        image[image > 0] = 1.0
        image = 1.0 -image[:, :, 0:1]

    # Need to flip each channel upside down, due to encoding by VREP
    for i in xrange(image.shape[2]):
        image[:, :, i] = np.flipud(image[:, :, i])

    # Make a placeholder at beginning of array, and make channels be second spot
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis]

    return image


def decode(all_data):
    """Primary function that decodes collected simulated data."""

    # Initialize elements to be None
    keys = ['header', 'depth', 'colour', 'mask', 'pregrasp',
            'postgrasp', 'direct_depth', 'direct_colour']

    misc_keys = ['depth', 'colour', 'matrix']

    # We need to set these to be None, or else (as a list) they share memory
    decoded = {key: list() for key in keys}
    elems = dict.fromkeys(keys, None)
    top_elems = dict.fromkeys(misc_keys, None)

    count = 0
    for i, line in enumerate(all_data):

        # First item of a line is always what the line represents (e.g.
        # image/grasp/header)
        data_type = line[0]
        data = line[1:-1]

        if 'TOPDOWN_DEPTH' in data_type: # Same orientation as manipulator
            if top_elems['depth'] is None:
                top_elems['depth'] = []

            # Once we hit this, we should not have any more generic images
            if len(decoded['depth']) == 0:
                top_elems['depth'].append(parse_image(data, depth=True))

        elif 'TOPDOWN_COLOUR' in data_type: # Same orientation as manipulator
            if top_elems['colour'] is None:
                top_elems['colour'] = []

            # Once we hit this, we should not have any more generic images
            if len(decoded['depth']) == 0:
                top_elems['colour'].append(parse_image(data))

        elif 'TOPDOWN_MATRIX' in data_type: # Same orientation as manipulator
            if top_elems['matrix'] is None:
                top_elems['matrix'] = []

            # Once we hit this, we should not have any more generic images
            if len(decoded['depth']) == 0:
                top_elems['matrix'].append(np.atleast_2d(data))

        # Prefix 'DIRECT' indicates same orientation as gripper
        elif data_type == 'DIRECT_DEPTH':
            elems['direct_depth'] = parse_image(data, depth=True)

        elif data_type == 'DIRECT_COLOUR':
            elems['direct_colour'] = parse_image(data)

        # No prefix inficates that image y-direction always points upwards
        elif data_type == 'GRIPPER_HEADER':
            elems['header'] = data

        elif data_type == 'GRIPPER_IMAGE': # Depth image
            elems['depth'] = parse_image(data, depth=True)

        elif data_type == 'GRIPPER_IMAGE_COLOUR':
            elems['colour'] = parse_image(data)

        elif data_type == 'GRIPPER_MASK_IMAGE':
            elems['mask'] = parse_image(data, mask=True)

        elif data_type == 'GRIPPER_PREGRASP':
            grasp = parse_grasp(data, elems['header'])
            preg = decode_grasp(grasp, elems['mask'], elems['depth'])
            elems['pregrasp'] = preg

        elif data_type == 'GRIPPER_POSTGRASP':
            grasp = parse_grasp(data, elems['header'])
            postg = decode_grasp(grasp, elems['mask'], elems['depth'])
            elems['postgrasp'] = postg

            # Check we've retrieved an element for each component
            # This is where we'll catch whether or not the grasp was
            #   successful, as the 'postgrasp' should not be None
            count += 1
            if all(elems[k] is not None for k in keys):

                # Postgrasp
                work2grasp = elems['postgrasp']['grasp_wrt_work']
                for i, mtx in enumerate(top_elems['matrix']):
                    work2top = format_htmatrix(mtx.reshape(3, 4))
                    top2work = invert_htmatrix(work2top)
                    top2grasp = convert_grasp_frame(top2work, work2grasp)

                    elems['postgrasp']['grasp_wrt_topdown_%d'%i] = top2grasp

                work2cam_var = elems['postgrasp']['cam_wrt_work_variant']
                work2cam_var = format_htmatrix(work2cam_var.reshape(3, 4))
                cam_var_2_work = invert_htmatrix(work2cam_var)
                elems['postgrasp']['grasp_wrt_cam_variant'] = \
                    convert_grasp_frame(cam_var_2_work, work2grasp)

                work2cam_invar = elems['postgrasp']['cam_wrt_work_invariant']
                work2cam_invar = format_htmatrix(work2cam_invar.reshape(3, 4))
                cam_invar_2_work = invert_htmatrix(work2cam_invar)
                elems['postgrasp']['grasp_wrt_cam_invariant'] = \
                    convert_grasp_frame(cam_invar_2_work, work2grasp)


                # Pregrasp
                work2grasp = elems['pregrasp']['grasp_wrt_work']
                for i, mtx in enumerate(top_elems['matrix']):
                    work2top = format_htmatrix(mtx.reshape(3, 4))
                    top2work = invert_htmatrix(work2top)
                    top2grasp = convert_grasp_frame(top2work, work2grasp)

                    elems['pregrasp']['grasp_wrt_topdown_%d'%i] = top2grasp

                work2cam_var = elems['pregrasp']['cam_wrt_work_variant']
                work2cam_var = format_htmatrix(work2cam_var.reshape(3, 4))
                cam_var_2_work = invert_htmatrix(work2cam_var)
                elems['pregrasp']['grasp_wrt_cam_variant'] = \
                    convert_grasp_frame(cam_var_2_work, work2grasp)

                work2cam_invar = elems['pregrasp']['cam_wrt_work_invariant']
                work2cam_invar = format_htmatrix(work2cam_invar.reshape(3, 4))
                cam_invar_2_work = invert_htmatrix(work2cam_invar)
                elems['pregrasp']['grasp_wrt_cam_invariant'] = \
                    convert_grasp_frame(cam_invar_2_work, work2grasp)


                for k in elems.keys():
                    if 'header' in k:
                        continue
                    decoded[k].append(elems[k])

                decoded_len = len(decoded['depth'])
                if decoded_len % 50 == 0:
                    print 'Successful grasp #%4d/%4d'%(decoded_len, count)

            # Reset the elements to be None
            elems.update(dict.fromkeys(elems.keys(), None))
        else:
            raise Exception('Data type: %s not understood'%data_type)

    # Quick check to see that we've decoded something
    if len(decoded['depth']) == 0:
        return False

    # Go through the collected pregrasp/postgrasp arrays, and combine each of
    # the elements that share the same header together
    pregrasp_dict = {}
    postgrasp_dict = {}
    grasps = zip(decoded['pregrasp'], decoded['postgrasp'])

    for i, (pregrasp, postgrasp) in enumerate(grasps):

        if i == 0:
            for key in pregrasp.keys():
                key_size = pregrasp[key].shape[1]
                pregrasp_dict[key] = np.empty((len(grasps), key_size))
                postgrasp_dict[key] = np.empty((len(grasps), key_size))

        for key in pregrasp.keys():
            pregrasp_dict[key][i] = pregrasp[key]
            postgrasp_dict[key][i] = postgrasp[key]


    return {'depth_images':np.vstack(decoded['depth']),
            'mask_images':np.vstack(decoded['mask']),
            'colour_images':np.vstack(decoded['colour']),
            'direct_depth':np.vstack(decoded['direct_depth']),
            'direct_colour':np.vstack(decoded['direct_colour']),
            'pregrasps':pregrasp_dict,
            'postgrasps':postgrasp_dict}


def postprocess(data, object_name):
    """Standardizes data by removing outlier grasps."""

    def remove_from_dataset(dataset, indices):
        """Convenience function for filtering bad indices from dataset."""

        for key, value in dataset.iteritems():
            if isinstance(value, dict):
                for subkey in dataset[key].keys():
                    dataset[key][subkey] = dataset[key][subkey][indices]
            else:
                dataset[key] = dataset[key][indices]

        return dataset

    # ------------------- Clean the dataset --------------------------

    # Remove any duplicate items (i.e. using camera pose matrix)
    unique = get_unique_idx(data['pregrasps']['cam_wrt_obj'], 2, 1e-5)
    data = remove_from_dataset(data, unique)

    if data['depth_images'].shape[0] > 50:

        # -- Image minimum values (i.e. don't want to look through table)
        image_minvals = np.amin(data['depth_images'], axis=(1, 2, 3))

        image_outlier_mask = get_outlier_mask(image_minvals, m=3)
        data = remove_from_dataset(data, image_outlier_mask)

        # -- Image variance (don't want image filling entire screen)
        image_var_mask = np.var(data['depth_images'], axis=(1, 2, 3)) > 1e-3
        data = remove_from_dataset(data, image_var_mask)

        # -- Encoded grasp outliers (i.e. don't want extremely different gr)
        grasp_mask = get_outlier_mask(data['pregrasps']['grasp_wrt_work'], m=3)
        data = remove_from_dataset(data, grasp_mask)

        if data['depth_images'].shape[0] == 0:
            return


    pregrasp_size = data['pregrasps']['grasp_wrt_work'].shape[0]
    postgrasp_size = data['postgrasps']['grasp_wrt_work'].shape[0]

    to_check = ['depth_images', 'colour_images', 'mask_images']
    assert(all(data[key].shape[0] == pregrasp_size for key in to_check))

    assert(all(pregrasp_size == data['pregrasps'][k].shape[0] for \
            k in data['pregrasps'].keys()) and \
           all(postgrasp_size == data['postgrasps'][k].shape[0] for \
            k in data['postgrasps'].keys()))


    # ------------------- Save the dataset --------------------------
    save_path = os.path.join(GLOBAL_PROCESSED_DIR, object_name+'.hdf5')
    datafile = h5py.File(save_path, 'w')
    datafile.create_dataset('GRIPPER_IMAGE', data=data['depth_images'])
    datafile.create_dataset('GRIPPER_IMAGE_MASK', data=data['mask_images'])
    datafile.create_dataset('GRIPPER_IMAGE_COLOUR', data=data['colour_images'])
    datafile.create_dataset('DIRECT_IMAGE', data=data['direct_depth'])
    datafile.create_dataset('DIRECT_COLOUR', data=data['direct_colour'])


    gr = datafile.create_group('GRIPPER_PREGRASP')
    for key in data['pregrasps'].keys():
        gr.create_dataset(key, data=data['pregrasps'][key])

    gr = datafile.create_group('GRIPPER_POSTGRASP')
    for key in data['postgrasps'].keys():
        gr.create_dataset(key, data=data['postgrasps'][key])

    datafile.create_dataset('OBJECT_NAME', data=[object_name]*postgrasp_size)

    sample_images(datafile, object_name, GLOBAL_SAMPLE_IMAGE_DIR)
    datafile.close()

    print 'Number of objects: ', postgrasp_size


def merge_files(directory):
    """Merges all files within a directory.

    This is used to join all the trials for a single given object, and assumes
    that all files have the same number of variables.
    """

    data = []
    for o in os.listdir(directory):

        if not '.txt' in o or o == 'commands':
            continue

        # Open the datafile, and find the number of fields
        object_path = os.path.join(directory, o)
        content_file = open(object_path, 'r')
        reader = csv.reader(content_file, delimiter=',')
        for line in reader:
            data.append(line)
        content_file.close()
    return data


def main():
    """Performs post-processing on grasps collected during simulation

    Notes
    -----
    This is a pretty beefy file, that does a lot of things. The data should be
    saved by the simulator in a structure similar to:
    class_objectName (folder for a specific object)
       |-> Grasp attempts 1:N (file)
       |-> Grasp attempts N:M (file)
       ...
       |-> Grasp attempts M:P (file)
    """

    if not os.path.exists(GLOBAL_SAMPLE_IMAGE_DIR):
        os.makedirs(GLOBAL_SAMPLE_IMAGE_DIR)
    if not os.path.exists(GLOBAL_PROCESSED_DIR):
        os.makedirs(GLOBAL_PROCESSED_DIR)
    if not os.path.exists(GLOBAL_SAMPLE_POSE_DIR):
        os.makedirs(GLOBAL_SAMPLE_POSE_DIR)


    # If we call the file just by itself, we assume we're going to perform
    # processing on each of objects tested during simulation.
    # Else, pass in a specific object/folder name, which can be found in
    # GLOBAL_RAW_DATA_DIR
    if len(sys.argv) == 1:
        object_directory = os.listdir(GLOBAL_RAW_DATA_DIR)
    else:
        object_directory = sys.argv[1]
        object_directory = [object_directory.split('/')[-1]]

    num_objects = len(object_directory)

    for i, object_name in enumerate(object_directory):

        try:
            print 'Processing object %d/%d: %s'%(i, num_objects, object_name)
            direct = os.path.join(GLOBAL_RAW_DATA_DIR, object_name)

            # Path to .txt file and hdf5 we want to save
            save_path = os.path.join(GLOBAL_PROCESSED_DIR, object_name+'.hdf5')
            if os.path.exists(save_path):
                os.remove(save_path)

            # Open up all individual files, merge them into a single file
            merged_data = merge_files(direct)

            decoded = decode(merged_data)

            # Check if the decoding returned successfully
            if isinstance(decoded, dict):
                postprocess(decoded, object_name)

        except Exception as e:
            print 'Exception occurred: ', e

if __name__ == '__main__':
    main()

    #fpath = '/mnt/data/datasets/grasping/scene_v40/train'
    #fname = '41_jar_and_lid_final-05-Apr-2016-14-14-31'
    #df = os.path.join(fpath, fname+'.hdf5')
    #dset = h5py.File(df, 'r')
    #sample_images(dset, fname, 'temp_images')


