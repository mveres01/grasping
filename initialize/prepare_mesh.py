import os
import sys
sys.path.append('..')

import csv
import numpy as np

import trimesh
from trimesh.io.export import export_mesh

from lib.utils import get_unique_idx

from lib.python_config import (config_object_dir, config_param_dir,
                               config_mesh_dir, config_object_mass,
                               config_object_density)

# All meshes we process should have 24 total values associated with it
GLOBAL_N_MORPH_PARAMS = 24


def merge_files(param_dir, param_list, app='.obj'):
    """Merges all single-lined parameter files into a single file.

    Parameters
    ----------
    param_dir : directory where all the parameter files are held.

    Returns
    -------
    Array containing all merged parameters.

    Notes
    -----
    Each datafile should contain 24 elements, following the convention:
    'coeffs':data[0:5],
    'origin':data[5:8],
    'axis':data[8:11],
    'mass':data[11],
    'com':data[12:15],
    'inertia':data[15:24]}
    """

    mesh_names = np.zeros((len(param_list), ), dtype=object)
    mesh_coeffs = np.zeros((len(param_list), 5), dtype=np.float32)

    file_count = 0
    for mesh_name in param_list:

        # Each file should have a 1-d array of values
        path = os.path.join(param_dir, mesh_name + app)
        with open(path, 'r') as content_file:
            reader = csv.reader(content_file, delimiter=',')
            morph_params = reader.next()
        morph_params = np.asarray(morph_params)

        if morph_params.shape[0] == GLOBAL_N_MORPH_PARAMS:

            # Append the filename to the dataframe
            mesh_names[file_count] = mesh_name
            mesh_coeffs[file_count] = morph_params[:5]
            file_count += 1
        else:
            print 'Morph file %s contains wrong # parameters (need %d)'%\
                  (mesh_name, GLOBAL_N_MORPH_PARAMS)

    # We'll estimate the mesh properties using the trimesh library, so for now
    # ignore the other mesh properties
    return mesh_names, mesh_coeffs


def get_unique_objects(mesh_names, mesh_coeffs):
    """Finds which objects within a class are unique, given the transforms.

    Parameters
    ----------
    mesh_names : An array of strings, specifiying the meshes to process.
    mesh_coeffs : An (n,5) array of transformation coefficients.

    Returns
    -------
    a list of indices specifying the unique objects in collection.
    """

    # Find all the unique "classes" of objects
    mesh_classes = [str(f).split('-')[0] for f in mesh_names]
    unique_classes = list(set(mesh_classes))

    names_matrix = np.asarray(mesh_classes, dtype=object)
    unique_idx = []

    for unique in unique_classes:

        # Find all objects that belong to a specific class
        class_idx = np.where(names_matrix == unique)[0]

        if len(class_idx) > 1:
            indices = get_unique_idx(mesh_coeffs[class_idx], thresh=1e-4)
            unique_idx.append(class_idx[indices])
        else:
            unique_idx.append(class_idx)

    unique_idx = np.hstack(unique_idx).flatten()

    return mesh_names[unique_idx]


def process_mesh(mesh_input_path, mesh_output_dir):
    """Given the path to a mesh, make sure its watertight & estimate params."""

    # Holds the processed name (1), mass  (1), center of mass (3) & inertia (9)
    processed = np.zeros((1, 14), dtype=object)

    try:
        mesh = trimesh.load_mesh(mesh_input_path)
    except Exception as e:
        print 'Exception: Unable to load mesh %s (%s): '%(mesh_input_path, e)
        return None

    full_mesh_name = mesh_input_path.split('/')[-1]
    mesh_name = full_mesh_name.split('.')[0]

    # Can visualize the mesh by uncommenting the line below
    #mesh.show()

    # Fix any issues with the mesh
    if not mesh.is_watertight:
        mesh.process()
        if not mesh.is_watertight:
            print 'Mesh (%s) cannot be made watertight'%mesh_name
            return None

    # Then export as an STL file
    mesh_output_name = os.path.join(mesh_output_dir, mesh_name + '.stl')
    export_mesh(mesh, mesh_output_name, 'stl')

    # Calculate mesh properties using build-in functions
    mesh_properties = mesh.mass_properties()
    com = np.array(mesh_properties['center_mass'])
    inertia = np.array(mesh_properties['inertia'])

    # Need to format the inertia based on object density.
    # We un-do the built-in calculation (density usually considered
    # as '1'), and multiply by our defined density
    inertia /= mesh_properties['density']
    inertia *= config_object_density

    # We don't want an unreasonable inertia
    inertia = np.clip(inertia, -1e-5, 1e-5)
    processed[0, 0] = mesh_name
    processed[0, 1] = config_object_mass
    processed[0, 2:5] = com
    processed[0, 5:14] = inertia.flatten()

    return processed


def main(mesh_input_dir, mesh_output_dir, mesh_param_dir=None):
    """Saves a copy of each of the meshes, fixing any issues along the way."""

    if not os.path.exists(mesh_output_dir):
        os.makedirs(mesh_output_dir)

    # Get a list of the meshes in a directory
    mesh_list = os.listdir(mesh_input_dir)

    if len(mesh_list) == 0:
        raise Exception('No meshes found in dir %s'%mesh_input_dir)

    # Meshes can be encoded through a few different formats, so grab xtn
    mesh_extension = '.' + mesh_list[0].split('.')[-1]
    mesh_list = [f.split('.')[0] for f in mesh_list]

    if mesh_param_dir is not None:
        # If we want to check for duplicate meshes, we'll do this using the
        # coefficients the meshes were morphed by. First we need to ensure that
        # each 'mesh' is associated with its own 'morph' file
        postfix = '-params.csv'
        param_list = os.listdir(mesh_param_dir)
        param_list = [f.split(postfix)[0] for f in param_list if postfix in f]
        valid_meshes = list(set(param_list) & set(mesh_list))

        # Since we assume that there's a single morph file per object, we'll
        # first merge them into a single file.
        names_, coeffs_ = merge_files(mesh_param_dir, valid_meshes, postfix)

        # Filter unique objects in a given class by looking at morph parameters
        mesh_names = get_unique_objects(names_, coeffs_)
    else:
        mesh_names = mesh_list

    processed_mesh_list = []
    for name in mesh_names:
        mesh_path = os.path.join(mesh_input_dir, name + mesh_extension)
        processed = process_mesh(mesh_path, mesh_output_dir)
        if processed is not None:
            processed_mesh_list.append(processed)

    print '%4d/%4d meshes successfully processed.'%\
            (len(processed_mesh_list), len(mesh_names))

    # Write each row to file
    csvfile = open(os.path.join(mesh_output_dir, 'mesh_object_properties.txt'), 'wb')
    writer = csv.writer(csvfile, delimiter=',')
    for to_write in processed_mesh_list:
        writer.writerow(to_write)
    csvfile.close()


if __name__ == '__main__':
    main(config_object_dir, config_mesh_dir, config_param_dir)

