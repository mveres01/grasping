import os, sys, csv
import numpy as np
import pandas as pd
import trimesh

from trimesh.io.export import export_mesh

# Set a constant object mass and density
mass = 1.0
density = 1000.
mesh_xtn = '.obj'

# Directories where meshes and morph parameters can be found
object_dir = '/scratch/mveres/object_files'
param_dir = '/scratch/mveres/morph_files'


def get_unique_objects(names, coeffs):
    """Finds which objects within a class are unique, given the transforms
    
    Parameters
    ----------
    names : A list of strings (size 'n'), specifiying the meshes to process
    coeffs : An (n,5) array of transformation coefficients

    Returns
    -------
    a list of indices specifying the unique objects in collection
    """
 
    # Find all the unique "classes" of objects
    classes = list(set([f.split('-')[0] for f in names]))

    unique_idx = []
    arr = np.atleast_2d(np.arange(names.shape[0])).T

    for unique in classes:

        # For each unique class of objects, gather all object that belong
        # to that class 
        cidx = [idx for idx in xrange(names.shape[0]) if unique in names[idx]]
    
        # Find the objects that were morphed using the same coefficients 
        # (which would be duplicate shapes)
        df = pd.DataFrame(coeffs[cidx])
        df = df.drop_duplicates()

        unique_indices = df.index.values
        unique_idx.append(arr[cidx][unique_indices])
    unique_idx = np.vstack(unique_idx).flatten()
    return unique_idx


def merge_parameter_files(paramdir, postfix='-params.csv'):
    """Merges all single-lined parameter files into a single file

    Parameters
    ----------
    param_dir : directory where all the parameter files are held

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

    # List all the parameter files we have
    files = os.listdir(paramdir)
    files = [f for f in files if postfix in f]

    n_var = 24
    file_count = 0
    data = np.zeros((len(files), n_var+1), dtype=object)
    for f in files:

        # Each file should have a 1-d array of values
        fp = os.path.join(paramdir, f)
        df = pd.read_csv(fp, index_col=False, header=None).values
        df = df.reshape((-1,))

        # Make sure the file contains the right number of items
        if df.shape[0] == n_var:
            # Append the filename to the dataframe
            row = np.hstack([f.split(postfix)[0], df])
            data[file_count] = row
            file_count += 1

    # We'll estimate the mesh properties using the trimesh library
    names = data[:file_count,0]
    coeffs = data[:file_count,1:6]
    
    return names, coeffs


def main():

    # We'll save a copy of each of the meshes, fixing any issues along the way
    file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(file_path))
    save_dir = os.path.join(base_dir, 'meshes')
       
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Each mesh has its own file, so merge them into a single structure
    names, coeffs = merge_parameter_files(param_dir)

    # Filter unique objects in a given class by looking at morph parameters
    unique_idx = get_unique_objects(names, coeffs)
    names, coeffs = names[unique_idx], coeffs[unique_idx]

    # Holds the processed name (1), mass  (1), center of mass (3) & inertia (9)
    processed = np.zeros((names.shape[0],14), dtype=object)

    good_mesh_cnt = 0
    for i, f in enumerate(names):

        if len(names) % 0.1*len(names) == 0:
            print 'Preprocessing mesh %d/%d'%(i,len(names))
 
        path = os.path.join(object_dir, f+mesh_xtn)

        try:
            mesh = trimesh.load_mesh(path)
        except Exception as e:
            continue
        mesh.remove_degenerate_faces()

        # Fix any issues with the mesh
        if not mesh.is_watertight:
            mesh.process()
            mesh.fill_holes()

        if mesh.is_watertight:
            fn = os.path.join(save_dir, f.split('.obj')[0])
            export_mesh(mesh, fn+'.stl', 'stl')  
        else:
            continue

        # Calculate mesh properties using build-in functions
        mesh_properties = mesh.mass_properties()
        com = np.array(mesh_properties['center_mass'])
        inertia = np.array(mesh_properties['inertia'])

        # Need to format the inertia based on object density. 
        # We un-do the built-in calculation (density usually considered 
        # as '1'), and multiply by our defined density
        inertia /= mesh_properties['density']
        inertia *= density

        # We don't want an unreasonable inertia
        inertia = np.clip(inertia, -1e-1, 1e-1)
        processed[good_mesh_cnt,0] = f
        processed[good_mesh_cnt,1] = mass
        processed[good_mesh_cnt,2:5] = com
        processed[good_mesh_cnt,5:14] = inertia.flatten()

        # Two different ways to visualize the mesh. One is prebuilt with
        # the library, the other is by using matplotlib and polycollection
        #mesh.show()
        good_mesh_cnt+=1

    processed = processed[:good_mesh_cnt]

    # Write each row to file
    fname = os.path.join(save_dir, 'objects.txt')
    csvfile = open(fname,'wb')
    writer = csv.writer(csvfile, delimiter=',')
    a = [writer.writerow(to_write) for to_write in processed]
    csvfile.close()

if __name__ == '__main__':
    main()
