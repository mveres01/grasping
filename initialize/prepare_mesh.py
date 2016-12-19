import os
import sys
sys.path.append('..')

import csv
import numpy as np
import pandas as pd

import trimesh
from trimesh.io.export import export_mesh

from lib.python_config import (config_object_dir, config_param_dir, 
                               config_mesh_dir, config_object_mass, 
                               config_object_density)

# All meshes we process should have 24 total values associated with it
GLOBAL_N_MORPH_PARAMS = 24

def get_unique_objects(names, coeffs):
    """Finds which objects within a class are unique, given the transforms.

    Parameters
    ----------
    names : A list of strings (size 'n'), specifiying the meshes to process.
    coeffs : An (n,5) array of transformation coefficients.

    Returns
    -------
    a list of indices specifying the unique objects in collection.
    """

    # Find all the unique "classes" of objects
    names = [f.split('-')[0] for f in names]
    names_matrix = np.asarray(names, dtype=str)
    classes = list(set(names))

    unique_idx = []

    for unique in classes:

        # Find all objects that belong to a specific class
        class_idx = np.where(names_matrix == unique)[0]

        # Remove any class objects that were morphed using the same parameters
        coefficients_df = pd.DataFrame(coeffs[class_idx]).drop_duplicates()
    
        # We'll save these unique indices
        unique_indices = coefficients_df.index.values
        unique_idx.append(class_idx[unique_indices])

    unique_idx = np.hstack(unique_idx).flatten()
    return unique_idx


def merge_parameter_files(paramdir, postfix='-params.csv'):
    """Merges all single-lined parameter files into a single file.

    Parameters
    ----------
    config_param_dir : directory where all the parameter files are held

    Returns
    -------
    Array containing all merged parameters

    Notes
    -----
    Each datafile should contain 24 elements, following the convention:
    'names':data[0],
    'coeffs':data[1:6],
    'origin':data[6:9],
    'axis':data[9:12],
    'mass':data[12],
    'com':data[13:16],
    'inertia':data[16:25]}
    """

    # Make sure we have both the object file and morph file
    object_files = os.listdir(config_object_dir)
    object_files = [f.split('.')[0] for f in object_files]

    morph_files = os.listdir(config_param_dir)
    morph_files = [f.split(postfix)[0] for f in morph_files if postfix in f]
    morph_files = list(set(morph_files) & set(object_files))


    file_count = 0
    morph_data = np.zeros((len(morph_files), GLOBAL_N_MORPH_PARAMS + 1), dtype=object)
    for morph_file in morph_files:

        # Each file should have a 1-d array of values
        fp = os.path.join(paramdir, morph_file + postfix)
        morph_params = pd.read_csv(fp, index_col=False, header=None).values
        morph_params = morph_params.reshape((-1,))

        if morph_params.shape[0] == GLOBAL_N_MORPH_PARAMS:

            # Append the filename to the dataframe
            row = np.hstack([morph_file.split(postfix)[0], morph_params])
            morph_data[file_count] = row
            file_count += 1

    # We'll estimate the mesh properties using the trimesh library, so for now
    # ignore the other mesh properties
    names = morph_data[:file_count, 0]
    coeffs = morph_data[:file_count, 1:6]

    return names, coeffs


def process_meshes():
    """Saves a copy of each of the meshes, fixing any issues along the way."""

    if not os.path.exists(config_mesh_dir):
        os.makedirs(config_mesh_dir)

    # Each mesh has its own file, so merge them into a single structure
    morph_names, morph_coeffs = merge_parameter_files(config_param_dir)

    # Filter unique objects in a given class by looking at morph parameters
    unique_idx = get_unique_objects(morph_names, morph_coeffs)
    morph_names = morph_names[unique_idx]
    morph_coeffs = morph_coeffs[unique_idx]

    # Holds the processed name (1), mass  (1), center of mass (3) & inertia (9)
    processed = np.zeros((morph_names.shape[0], 14), dtype=object)

    good_mesh_cnt = 0
    for i, morph_name in enumerate(morph_names):

        if len(morph_names) % 0.1*len(morph_names) == 0:
            print 'Preprocessing mesh %d/%d'%(i, len(morph_names))

        path = os.path.join(config_object_dir, morph_name+'.obj')

        try:
            trimesh.constants.tol.merge = 1e-12
            mesh = trimesh.load_mesh(path)
        except Exception as e:
            print 'Exception occurred processed mesh %s (%s): '%(morph_name, e)
            continue

        # Fix any issues with the mesh
        if not mesh.is_watertight:
            mesh.process()
            if not mesh.is_watertight:
                print 'Mesh %s cannot be made watertight'%morph_name
                continue

        # Then export as an STL file
        fn = os.path.join(config_mesh_dir, morph_name.split('.obj')[0])
        export_mesh(mesh, fn + '.stl', 'stl')

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
        processed[good_mesh_cnt, 0] = morph_name
        processed[good_mesh_cnt, 1] = config_object_mass
        processed[good_mesh_cnt, 2:5] = com
        processed[good_mesh_cnt, 5:14] = inertia.flatten()

        # Two different ways to visualize the mesh. One is prebuilt with
        # the library, the other is by using matplotlib and polycollection
        #mesh.show()
        good_mesh_cnt += 1

    print 'Number of processed mesh files: ', good_mesh_cnt
    processed = processed[:good_mesh_cnt]

    # Write each row to file
    csvfile = open(os.path.join(config_mesh_dir, 'mesh_object_properties.txt'), 'wb')
    writer = csv.writer(csvfile, delimiter=',')
    for to_write in processed:
        writer.writerow(to_write)
    csvfile.close()


if __name__ == '__main__':
    process_meshes()
