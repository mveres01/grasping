# grasping

## Requirements
* Python 2.7
* V-REP from http://www.coppeliarobotics.com/downloads.html
* matrix.lua from https://github.com/davidm/lua-matrix/tree/master/lua
* Python trimesh library from https://pypi.python.org/pypi/trimesh
* Mesh files (tested with .stl and .obj)
* GNU Parallel

## Initialize paths
* In lib/python_config.py change the project dir to point to where this project was downloaded to. This file is used for controlling basic parameters within all python files
* In lib/lua_config.py change the different directory lists to point to where the project was downloaded to. This file is used for controlling parameters within the different V-REP simulations
* Open scenes/get_initial_poses.ttt in V-REP. Modify the threaded script by double-clicking the blue page icon beside 'GraspCollection', and change the variable to "local file_dir" to point to where the lua_config.lua file is found
* Open scenes/scene_collect_grasps.ttt in V-REP. Modify the threaded script by double-clicking the blue page icon beside 'GraspCollection', and change the variable to "local file_dir" to point to where the lua_config.lua file is found

## Download meshes
* Download meshes as either the Wavefront .obj file format, or .stl file format. Place them in data/meshes/object_files
* Note that the simulation works best with simple meshes; For complex meshes, you may need to manually process them to reduce the number of triangles or complexity before running in the simulation. The more complex the mesh is, the more unstable the simulations will be

## Step 1: Prepare all meshes / mesh properties for simulation
* First load all the meshes we'll use. This will create a file called 'mesh_object_properties.txt' in the data/ folder. This file contains information about each mesh, including mass, center of mass, and inertia.
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
## Step 2: run the commands within the simulator
* Launch the simulations using generated command files. Note that there may be some specific simulator variables you may be interested in changing (i.e. camera near/far clipping plances, which contacts are part of the gripper), which can be found inside the 'config.lua' file. The simulation will save successful grasps to the data/collected folder; launch the simulations with: 
```unix
$: cd collect
$: screen
$: cat commands/mainXXX.txt | parallel -j N_JOBS 
```
where XXX is a specific file to be run on a compute node, and N_JOBS is a number (i.e. 8), which specifies the number of jobs you want to run in parallel. If no number is specified, GNU parallel will use the maximum number of cores available
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
