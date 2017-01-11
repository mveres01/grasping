import os
import sys
sys.path.append('..')

import csv
import h5py

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

import cv2

from lib.utils import (format_htmatrix, invert_htmatrix,
                       sample_images, get_unique_idx)

# Object/simulation properties
from lib.python_config import (config_image_width, config_image_height,
                               config_near_clip, config_far_clip, config_fov)
# Save/data directories
from lib.python_config import (config_collected_data_dir, config_processed_data_dir,
                               config_sample_image_dir, config_sample_pose_dir)


def get_outlier_mask(data_in, sigma=3):
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
    mask = abs(data_in - mean) < sigma*std
    mask = np.sum(mask, axis=1)

    # Want samples where all variables are within a 'good' region
    return mask == data_in.shape[1]


def unproject_2d(point, depth, fov_y, image_shape):
    """Converts a 2D point (in pixel space) to real-world 3D space.

    Notes
    -----
    Assumes field of view (config_fov) is given in radians
    """

    # Make sure the values are encoded as floating point numbers
    px = np.float32(point[0])
    py = np.float32(point[1])
    scene_y = -(2.*(py-0.)/image_shape[0] - 1.0)*depth*np.tan(fov_y/2.)
    scene_x = -(2.*(px-0.)/image_shape[1] - 1.0)*depth*np.tan(fov_y/2.)

    return np.asarray([scene_x, scene_y, depth])


def get_image_matrix(y_axis, z_axis, center):
    """Calculates the transformation matrix from camera to image plane."""

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
    """Given a binary image mask and depth image, calculate the object pose.

    Notes
    -----
    This function uses PCA to compute the major/minor axis of the object, then
    unprojects the point back into a 3D coordinate system by building a
    homogeneous transofmration matrix.
    """

    _, _, n_rows, n_cols = mask.shape

    if mask.ndim > 2:
        mask_2d = np.squeeze(mask)
    else:
        mask_2d = mask

    if depth_image.ndim > 2:
        depth_2d = np.squeeze(depth_image)
    else:
        depth_2d = depth_image


    cx, cy, coords = get_image_centroid(np.squeeze(mask_2d))

    # Estimate the major axis via principle components of the binary image
    pca = PCA(n_components=2).fit(coords)
    pc1 = np.asarray(pca.components_[0])
    pc2 = np.asarray(pca.components_[1])

    # Use PCA estimates to compute major axes
    c = 10.0 # arbitrary
    dir_z = (np.round(cx + pc1[0]*c), np.round(cy + pc1[1]*c))
    dir_y = (np.round(cx + pc2[0]*c), np.round(cy + pc2[1]*c))

    # Get the depth information at estimated object center of mass
    # Then convert the image pixel to coordinateed realtive to camera
    depth = depth_2d[int(cy), int(cx)]
    com = unproject_2d([cx, cy], depth, config_fov, [n_rows, n_cols])
    unproj_z = unproject_2d(dir_z, depth, fov_y, [n_rows, n_cols])
    unproj_y = unproject_2d(dir_y, depth, fov_y, [n_rows, n_cols])

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
        raise Exception('Computed determinant (%2.4f) not equal to 1.'%det)

    unproj_z = np.atleast_2d(unproj_z)
    unproj_y = np.atleast_2d(unproj_y)
    cam2img = cam2img[:3].flatten()


    # Check whether the selected pixel is on object using ground truth
    if np.random.randn() > 0.99:
        img = np.uint8((1.0 - mask_2d)*255.0).copy()
        p1 = (cx + int(pc1[0]*15), cy + int(pc1[1]*15))
        p2 = (cx + int(pc2[0]*15), cy + int(pc2[1]*15))
        cv2.line(img, (cx, cy), p1, 125, 1)
        cv2.line(img, (cx, cy), p2, 125, 1)
        cv2.circle(img, (cx, cy), 2, 0)
        cv2.circle(img, (int(n_cols/2), int(n_rows/2)), 1, 0)

        pose = os.path.join(config_sample_pose_dir,
                            '%d.png'%np.random.randint(0, 12345))

        pil_image = Image.new('L', (n_cols, n_rows))
        pil_image.paste(Image.fromarray(img), (0, 0, n_cols, n_rows))
        pil_image.save(pose)


    return cam2img, unproj_z, unproj_y


def convert_grasp_frame(frame2matrix, matrix2grasp):
    """Function for converting from one grasp frame to another.

    This is useful as transforming grasp positions requires multiplication of
    4x4 matrix, while contact normals (orientation) are multiplication of 3x3
    components (i.e. without positional components.
    """

    # A grasp is contacts, normals, and forces (3), and has (x,y,z) components
    n_fingers = int(matrix2grasp.shape[1]/9)
    contact_points = matrix2grasp[0, :n_fingers*3].reshape(3, 3)
    contact_normals = matrix2grasp[0, n_fingers*3:n_fingers*6].reshape(3, 3)
    contact_forces = matrix2grasp[0, n_fingers*6:].reshape(3, 3)

    # Append a 1 to end of contacts for easier multiplication
    contact_points = np.hstack([contact_points, np.ones((3, 1))])

    # Convert positions, normals, and forces to object reference frame
    points = np.zeros((n_fingers, 3))
    forces = np.zeros(points.shape)
    normals = np.zeros(points.shape)

    for i in xrange(n_fingers):

        points[i] = np.dot(frame2matrix, contact_points[i:i+1].T)[:3].T
        forces[i] = np.dot(frame2matrix[:3, :3], contact_forces[i:i+1].T).T
        normals[i] = np.dot(frame2matrix[:3, :3], contact_normals[i:i+1].T).T

    grasp = np.vstack([points, normals, forces]).reshape(1, -1)

    return grasp


def decode_grasp(grasp_line, object_mask, object_depth_image):
    """Extracts different homogeneous transform matrices and grasp components."""

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

    # Encode the grasp into the workspace frame
    world2grasp = np.hstack([contact_points, contact_normals, contact_forces])
    world2grasp = np.atleast_2d(world2grasp)

    # These object properties are encoded WRT workspace
    work2com = grasp_line['com']
    work2mass = grasp_line['mass']
    work2inertia = grasp_line['inertia']

    #workspace_to_base = g['BarrettHand'][i]
    work2obj = grasp_line['object_matrix'].reshape(3, 4)
    work2obj = format_htmatrix(work2obj)
    obj2work = invert_htmatrix(work2obj)

    world2work = grasp_line['wrtObjectMatrix'].reshape(3, 4)
    world2work = format_htmatrix(world2work)
    work2world = invert_htmatrix(world2work)

    # Camera frame that doesn't change rotation (one image many grasps)
    work2cam_otm = grasp_line['rot_invariant_matrix'].reshape(3, 4)
    work2cam_otm = format_htmatrix(work2cam_otm)
    cam2work_otm = invert_htmatrix(work2cam_otm)

    # Camera frame that changes rotation (one image to one grasp)
    work2cam_oto = grasp_line['rot_variant_matrix'].reshape(3, 4)
    work2cam_oto = format_htmatrix(work2cam_oto)
    cam2work_oto = invert_htmatrix(work2cam_oto)

    # Some misc frames that will be helpful later
    obj2world = np.dot(obj2work, work2world)
    world2obj = invert_htmatrix(obj2world)
    obj2cam = np.dot(obj2work, work2cam_otm)

    # Encode the grasp WRT estimated coordiante frame attached to img
    # unproj_z/y are points in 3D space showing the direction of x/y axes
    cam2img, unproj_z, unproj_y = \
        estimate_object_pose(object_mask, object_depth_image, config_fov)

    # ### finally, convert the grasp frames
    work2grasp = convert_grasp_frame(work2world, world2grasp)
    cam2grasp_oto = convert_grasp_frame(cam2work_oto, work2grasp)
    cam2grasp_otm = convert_grasp_frame(cam2work_otm, work2grasp)

    # Make sure everything is the right dimension so we can later concatenate
    work2com = np.atleast_2d(grasp_line['com'])
    work2mass = np.atleast_2d(work2mass)
    work2inertia = np.atleast_2d(work2inertia)

    cam2img = np.atleast_2d(cam2img)
    obj2cam = np.atleast_2d(obj2cam[:3].flatten())
    world2obj = np.atleast_2d(world2obj[:3].flatten())
    world2work = np.atleast_2d(world2work[:3].flatten())
    cam2work_oto = np.atleast_2d(cam2work_oto[:3].flatten())
    cam2work_otm = np.atleast_2d(cam2work_otm[:3].flatten())
    work2cam_oto = np.atleast_2d(work2cam_oto[:3].flatten())
    work2cam_otm = np.atleast_2d(work2cam_otm[:3].flatten())

    return {'work2grasp':work2grasp,
            'cam2grasp_oto':cam2grasp_oto,
            'cam2grasp_otm':cam2grasp_otm,
            'work2inertia':work2inertia,
            'work2mass':work2mass,
            'work2com':work2com,
            'frame_world2obj':world2obj,
            'frame_world2work':world2work,
            'frame_work2cam_otm':work2cam_otm,
            'frame_work2cam_oto':work2cam_oto,
            'frame_cam2work_otm':cam2work_otm,
            'frame_cam2work_oto':cam2work_oto,
            'frame_cam2img_otm':cam2img,
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

    image_pixels = config_image_width*config_image_height

    if len(image_as_list)/image_pixels < 1:
        print 'Invalid image size (need %dx%dxn)'%(config_image_width,
                                                   config_image_height)
        return None

    # Convert the list into an array
    image = np.asarray(image_as_list, dtype=np.float32)
    image = image.reshape(config_image_height, config_image_width, -1)

    # Decode the depth information using near + far clipping planes
    if depth is True:
        image = config_near_clip + image*(config_far_clip - config_near_clip)

    # Since the "binary mask" saaved from simulator is 3-channeled, convert
    # this to a single channel image
    elif mask is True:
        image[image > 0] = 1.0
        image = 1.0 -image [:, :, 0:1]

    # Need to flip the image upside down due to encoding from sim 
    for i in xrange(image.shape[2]):
        image[:, :, i] = np.flipud(image[:, :, i])

    # Make a placeholder at beginning of array, and make channels be second spot
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis]

    return image


def decode_raw_data(all_data):
    """Primary function that decodes collected simulated data."""

    # Initialize elements to be None
    # Note that "OTO" means the image and gripper share a one-to-one mapping
    # Note that "OTM" means the image and gripper share a one-to-many mapping
    keys = ['header', 'depth_oto', 'colour_oto', 'depth_otm',
            'colour_otm', 'mask_otm', 'pregrasp', 'postgrasp']

    misc_keys = ['depth', 'colour', 'matrix']

    # We need to set these to be None, or else (as a list) they share memory
    decoded = {key: list() for key in keys}
    elems = dict.fromkeys(keys, None)
    top_elems = dict.fromkeys(misc_keys, None)


    # ---------------- Loop through all recorded data ------------------------

    count = 0 # which attempt
    successful = 0 # how many successful attempts
    for i, line in enumerate(all_data):

        # First item of a line is always what the line represents
        # (e.g. an image/grasp/header)
        data_type = line[0]
        data = line[1:-1]


        # Before we collect grasp data, we collect some images of the object
        # from cameras in fixed locations above the workspace
        if 'TOPDOWN_DEPTH' in data_type:
            if top_elems['depth'] is None:
                top_elems['depth'] = []

            # Once we hit this, we should not have any more generic images
            if len(decoded['depth_otm']) == 0:
                top_elems['depth'].append(parse_image(data, depth=True))

        elif 'TOPDOWN_COLOUR' in data_type:
            if top_elems['colour'] is None:
                top_elems['colour'] = []

            if len(decoded['depth_otm']) == 0:
                top_elems['colour'].append(parse_image(data))

        elif 'TOPDOWN_MATRIX' in data_type:
            if top_elems['matrix'] is None:
                top_elems['matrix'] = []

            if len(decoded['depth_otm']) == 0:
                top_elems['matrix'].append(np.atleast_2d(data))

        # Once we're collecting data, we encoded it using the following terms.
        # Prefix 'DIRECT' indicates same orientation as gripper
        elif data_type == 'DIRECT_DEPTH':
            elems['depth_oto'] = parse_image(data, depth=True)

        elif data_type == 'DIRECT_COLOUR':
            elems['colour_oto'] = parse_image(data)

        # No prefix inficates that image y-direction always points upwards
        elif data_type == 'GRIPPER_HEADER':
            elems['header'] = data

        elif data_type == 'GRIPPER_IMAGE': # Depth image
            elems['depth_otm'] = parse_image(data, depth=True)

        elif data_type == 'GRIPPER_IMAGE_COLOUR':
            elems['colour_otm'] = parse_image(data)

        elif data_type == 'GRIPPER_MASK_IMAGE':
            elems['mask_otm'] = parse_image(data, mask=True)

        elif data_type == 'GRIPPER_PREGRASP':
            grasp = parse_grasp(data, elems['header'])
            preg = decode_grasp(grasp, elems['mask_otm'], elems['depth_otm'])
            elems['pregrasp'] = preg

        elif data_type == 'GRIPPER_POSTGRASP':
            grasp = parse_grasp(data, elems['header'])
            postg = decode_grasp(grasp, elems['mask_otm'], elems['depth_otm'])
            elems['postgrasp'] = postg

            # Check we've retrieved an element for each component
            # This is where we'll catch whether or not the grasp was
            #   successful, as the 'postgrasp' should not be None
            count += 1
            if all(elems[k] is not None for k in keys):

                for k in elems.keys():
                    if 'header' not in k:
                        decoded[k].append(elems[k])

                successful += 1
                if successful % 50 == 0:
                    print 'Successful grasp #%4d/%4d'%(successful, count)

            # Reset the elements to be None
            elems.update(dict.fromkeys(elems.keys(), None))
        else:
            raise Exception('Data type: %s not understood'%data_type)


    # Quick check to see that we've decoded something
    if len(decoded['depth_otm']) == 0:
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

    return {'image_depth_oto':np.vstack(decoded['depth_oto']),
            'image_colour_oto':np.vstack(decoded['colour_oto']),
            'image_depth_otm':np.vstack(decoded['depth_otm']),
            'image_colour_otm':np.vstack(decoded['colour_otm']),
            'image_mask_otm':np.vstack(decoded['mask_otm']),
            'pregrasp':pregrasp_dict,
            'postgrasp':postgrasp_dict}


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

    # -- Remove duplicate grasps using workspace --> camera frame
    unique = get_unique_idx(data['pregrasp']['frame_work2cam_oto'], -1, 1e-5)
    data = remove_from_dataset(data, unique)

    if data['image_depth_otm'].shape[0] > 50:

        # -- Remove any images with no interesting information
        good_indices = np.var(data['image_depth_otm'], axis=(1, 2, 3)) > 1e-3
        data = remove_from_dataset(data, good_indices)

        # -- Image minimum values (i.e. don't want to look through table)
        image_minvals = np.amin(data['image_depth_otm'], axis=(1, 2, 3))
        good_indices = get_outlier_mask(image_minvals, sigma=3)
        data = remove_from_dataset(data, good_indices)

        # -- Remove any super wild grasps
        good_indices = get_outlier_mask(data['pregrasp']['work2grasp'], sigma=4)
        data = remove_from_dataset(data, good_indices)

        if data['image_depth_otm'].shape[0] == 0:
            return

    # Make sure we have the same number of samples for all data elements
    to_check = ['image_depth_otm', 'image_colour_otm', 'image_mask_otm',
                'image_depth_oto', 'image_colour_oto']

    keys = data['pregrasp'].keys()
    pregrasp_size = data['pregrasp']['work2grasp'].shape[0]
    postgrasp_size = data['postgrasp']['work2grasp'].shape[0]

    assert all(data[key].shape[0] == pregrasp_size for key in to_check)
    assert all(pregrasp_size == data['pregrasp'][k].shape[0] for k in keys)
    assert all(postgrasp_size == data['postgrasp'][k].shape[0] for k in keys)

    # ------------------- Save the dataset --------------------------
    save_path = os.path.join(config_processed_data_dir, object_name+'.hdf5')

    datafile = h5py.File(save_path, 'w')
    datafile.create_dataset('image_depth_otm', data=data['image_depth_otm'], 
                            compression='gzip')

    datafile.create_dataset('image_depth_oto', data=data['image_depth_oto'], 
                            compression='gzip')

    datafile.create_dataset('image_colour_otm', data=data['image_colour_otm'], 
                            compression='gzip')

    datafile.create_dataset('image_colour_oto', data=data['image_colour_oto'], 
                            compression='gzip')

    datafile.create_dataset('image_mask_otm', data=data['image_mask_otm'], 
                            compression='gzip')

    datafile.create_dataset('object_name', data=[object_name]*postgrasp_size,
                            compression='gzip')

    # Add the pregrasp to the dataset
    grasp_group = datafile.create_group('pregrasp')
    for key in data['pregrasp'].keys():
        grasp_group.create_dataset(key, data=data['pregrasp'][key],
                                   compression='gzip')
    grasp_group.create_dataset('object_name', data=[object_name]*postgrasp_size,
                               compression='gzip')

    # Add the postgrasp to the dataset
    grasp_group = datafile.create_group('postgrasp')
    for key in data['postgrasp'].keys():
        grasp_group.create_dataset(key, data=data['postgrasp'][key],
                                   coompression='gzip')
    grasp_group.create_dataset('object_name', data=[object_name]*postgrasp_size)

    # We'll save some images to visualize what we've collected/decoded
    sample_images(datafile, config_sample_image_dir)
    datafile.close()

    print 'Number of objects: ', postgrasp_size


def merge_files(directory):
    """Merges all files within a directory.

    This is used to join all the trials for a single given object, and assumes
    that all files have the same number of variables.
    """

    data = []
    for object_file in os.listdir(directory):

        if not '.txt' in object_file or object_file == 'commands':
            continue

        # Open the datafile, and find the number of fields
        object_path = os.path.join(directory, object_file)
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

    if not os.path.exists(config_sample_image_dir):
        os.makedirs(config_sample_image_dir)
    if not os.path.exists(config_processed_data_dir):
        os.makedirs(config_processed_data_dir)
    if not os.path.exists(config_sample_pose_dir):
        os.makedirs(config_sample_pose_dir)


    # If we call the file just by itself, we assume we're going to perform
    # processing on each of objects tested during simulation.
    # Else, pass in a specific object/folder name, which can be found in
    # collected_dir
    if len(sys.argv) == 1:
        object_directory = os.listdir(config_collected_data_dir)
    else:
        object_directory = [sys.argv[1].split('/')[-1]]
    num_objects = len(object_directory)


    for i, object_name in enumerate(object_directory):

        try:
            print 'Processing object %d/%d: %s'%(i, num_objects, object_name)
            direct = os.path.join(config_collected_data_dir, object_name)

            # Path to .txt file and hdf5 we want to save
            save_path = os.path.join(config_processed_data_dir, object_name+'.hdf5')
            if os.path.exists(save_path):
                os.remove(save_path)

            # Open up all individual files, merge them into a single file
            merged_data = merge_files(direct)

            decoded = decode_raw_data(merged_data)

            # Check if the decoding returned successfully
            if isinstance(decoded, dict):
                postprocess(decoded, object_name)

        except Exception as e:
            print 'Exception occurred: ', e

if __name__ == '__main__':
    main()



