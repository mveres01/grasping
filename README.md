# grasping

This project contains the code used for generating multi-modal grasps in V-REP, and described in the paper "An Integrated Simulator and Data Set that Combines Grasping and Vision for Deep Learning" (TBA). 

## Requirements
* Python 2.7
* V-REP from http://www.coppeliarobotics.com/downloads.html
* matrix.lua from https://github.com/davidm/lua-matrix/tree/master/lua
* Python trimesh library from https://pypi.python.org/pypi/trimesh
* Mesh files from  https://uwaterloo.ca/neurorobotics-lab/g3db
* (optional) GNU Parallel & linux for parallelization

## Initialize paths
* In lib/python_config.py change the variable *project_dir* to point to where this project was downloaded to. This file is used for controlling basic parameters within all python files
* In lib/lua_config.py change the different directory lists to point to where the project was downloaded to. This file is used for controlling parameters within the different V-REP simulations
* Open scenes/get_initial_poses.ttt in V-REP. Modify the threaded script by double-clicking the blue page icon beside 'GraspCollection', and change the variable *local file_dir* to point to where the lua_config.lua file is found
* Open scenes/scene_collect_grasps.ttt in V-REP. Modify the threaded script by double-clicking the blue page icon beside 'GraspCollection', and change the variable *local file_dir* to point to where the lua_config.lua file is found

## Download meshes
* Download object meshes as either the Wavefront .obj file format, or .stl file format, and place them in data/meshes/object_files. To obtain the mesh files used in the work, you can download them from the [Waterloo G3DB](https://uwaterloo.ca/neurorobotics-lab/g3db) project. In this work, we only used a subset of these meshes, and the list can be found within the  /lib folder.
* Note 1: In this work, the meshes were labeled according to the convention 'XXX_yyyyyyy', where 'XXX' is the object class (e.g. 42, 25), and 'yyyyyy' is the name of the object name (e.g. 'wineglass', 'mug'). Example: '42_wineglass'.
* Note 2: The simulation works best with simple meshes; For complex meshes, you may need to manually process them to reduce the number of triangles or complexity before running in the simulation. The more complex the mesh is, the more unstable the simulations will be

## Step 1: Prepare all meshes / mesh properties for simulation
* First, we need to preprocess all the meshes we'll use to identify their relative properties, and fix any meshes that are not watertight. This process will create a file called 'mesh_object_properties.txt' in the data/ folder, containing information about each mesh, including mass, center of mass, and inertia.
```unix
$: cd initialize
$: python prepare_meshes.py
```
* Open V-REP and run scenes/get_initial_poses.ttt. This will create a file data/initial_poses.txt that contains all information on the starting pose of the object and gripper, and will be used for generating potential grasp candidates.
* Run initialize/prepare_candidates.py. This will read the pose information collected by V-REP, and generate a list of candidates for each object that will be tested in the simulator. Note that these candidates will be saved under collect/candidates
```unix
$: cd initialize
$: cat ../data/initial_poses.txt | parallel python prepare_meshes.py
```
* Once the candidates have been generated, you can either run each of them manually through the simulation, or create a "commands" file that will continuously all of them through the simulator. These commands will be saved under collect/commands
```unix
$: cd initialize
$: python prepare_commands.py
```
## Step 2: run the generated grasps through the simulator
* Launch the simulations using generated command files. Note that there may be some specific simulator variables you may be interested in changing (i.e. camera near/far clipping plances, which contacts are part of the gripper), which can be found inside the 'config.lua' file. The simulation will save successful grasps to the data/collected folder.
* Assuming you are running with linux and using GNU Parallel, you can launch the simulations with: 
```unix
$: cd collect
$: screen
$: cat commands/mainXXX.txt | parallel -j N_JOBS 
```
where XXX is a specific file to be run on a compute node, and N_JOBS is a number (i.e. 8), which specifies the number of jobs you want to run in parallel. If no number is specified, GNU parallel will use the maximum number of cores available. If you are running on windows, you can sequentially collect grasps by manually launching the simulation, and changing which file is used within lib/lua_config.lua file

* Once simulations are done, decode the collected data
```unix
$: cd collect
```
.. For decoding data sequentially (class by class): 
```unix
$: python decode_grasp_data.py
```
.. For decoding data in parallel (multiple classes at once): 
```unix
$: ls ../data/collected/ > files.txt
$: cat files.txt | parallel python decode_grasp_data.py
```
* Run split_train_test.py to move one file from each object class to a test directory
```unix
$: cd collect
$: python split_train_test.py
```
* Create a list of items you want to consider in the train set, by looking at the top N classes (in the generated bar plot of grasp statistics), and save these in the file collect/train_items.txt. Run postprocess.py to generate the final grasping dataset
```unix
$: cd collect
$: python postprocess_split_data.py
```

# Miscellaneous resources
* lib/sample_load.py is a standalone script that shows how data can be loaded and grasps can be plotted
* lib/sample_clean_list.txt is a list of items in the train/test/validation set that have been checked over for correctness. This represents only a small sample of overall grasps in the dataset, and there may be other improper/incorrect grasps representing noise in the dataset.
* lib/sample_mesh_list.txt is a list of meshes that were used in this project
