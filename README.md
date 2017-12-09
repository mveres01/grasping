# grasping

This project contains the code used for generating multi-modal grasps in V-REP, and described in the paper "An Integrated Simulator and Data Set that Combines Grasping and Vision for Deep Learning" (TBA). 

---------------------------
__EDIT 2017/12/09__: This repository is being depracated in favour of a [newer version](https://github.com/mveres01/multi-contact-grasping), supporting the major following changes:

* __Simplified pipeline__: Put meshes in a directory and immediately begin grasp experiments. This replaces the previous strategy of repeatedly switching between python scripts and the simulator for different phases of collection
* __Communication via Python Remote API__: The new repository supports communication through a python remote API. No more need to communicate through .csv files! Major processes ("dropping" an object, "grasping" an object, and "lifting" an object) are segmented into seperate threaded simulation scripts, and launched by setting a simulation flag from a python script. Additionally, custom functions can be implemented on the server side, and called through generic remote API functions with ease. Samples implemented are: set / get functions for specifying joint angles and object poses, loading objects, and collecting images.
* __Domain Randomization__: Visual scene properties (such as object colour / texture, table colour / texture, lighting, and camera pose) can be modified easily to collect a significant amount of sensory experience per grasp attempt. It can also be used to arbitrarily take images of the scene without any grasping experience, if someone is interested in e.g. segmentation algorithms. Domain randomization was introduced by [Tobin et. al.](https://arxiv.org/abs/1703.06907)
* __View Images Immediately__: Images (RGB of object, RGB of object + gripper, object mask & depth) are saved in a dedicated folder whenever queried from the simulator. Visualize what images you're capturing right away!
* __Definition of Grasp Success__: Previously, all fingers were required to be in contact with the object at the height of the object lift. In the new version, we use a proximity sensor attached to the robotic gripper to measure whether an object is present in the gripper or not.
* __Grasp Candidate Generation__: Grasp candidates are now generated following the surface normals of a mesh object, with random orientations around the grippers local z-direction. Experimentally, this tends to give grasps with a higher probability of success then the pre- and post- multiplication method.

-----------------------------

## Requirements
* Python 2.7
* V-REP from http://www.coppeliarobotics.com/downloads.html
* matrix.lua from https://github.com/davidm/lua-matrix/tree/master/lua
* Python trimesh library from https://pypi.python.org/pypi/trimesh
* Mesh files from  https://uwaterloo.ca/neurorobotics-lab/g3db
* (optional) an Xserver (such as [Xorg](https://wiki.archlinux.org/index.php/xorg)) if running V-REP in headless mode. Note that default behaviour requires this, but can be changed if you wish to use the GUI (see Aside: Dissecting the commands, and running with / without headless mode)
* (optional) GNU Parallel & linux for parallelization

## Initialize paths
* In lib/python_config.py change the variable *project_dir* to point to where this project was downloaded to. This file is used for controlling basic parameters within all python files
* In lib/lua_config.py change the different directory lists to point to where the project was downloaded to. This file is used for controlling parameters within the different V-REP simulations
* Open scenes/get_initial_poses.ttt in V-REP. Modify the threaded script by double-clicking the blue page icon beside 'GraspCollection', and change the variable *local file_dir* to point to where the lua_config.lua file is found
* Open scenes/scene_collect_grasps.ttt in V-REP. Modify the threaded script by double-clicking the blue page icon beside 'GraspCollection', and change the variable *local file_dir* to point to where the lua_config.lua file is found

## Download meshes
* Download object meshes as either the Wavefront .obj file format, or .stl file format, and place them in data/meshes/object_files. To obtain the mesh files used in the work, you can download them from the [Waterloo G3DB](https://uwaterloo.ca/neurorobotics-lab/g3db) project. In this work, we only used a subset of these meshes, and the list can be found within the  /lib folder.
* __Note 1__: In this work, the meshes were labeled according to the convention 'XXX_yyyyyyy', where 'XXX' is the object class (e.g. 42, 25), and 'yyyyyy' is the name of the object name (e.g. 'wineglass', 'mug'). Example: '42_wineglass'.
* __Note 2__: The simulation works best with simple meshes; For complex meshes, you may need to manually process them to reduce the number of triangles or complexity before running in the simulation. Some meshes in the above file *are* complex, and note that the more complex the mesh is, the more unstable the simulations will be.

## Step 1: Prepare all meshes / mesh properties for simulation
* First, we need to preprocess all the meshes we'll use to identify their relative properties, and fix any meshes that are not watertight. This process will create a file called 'mesh_object_properties.txt' in the data/ folder, containing information about each mesh, including mass, center of mass, and inertia.
```unix
$: cd initialize
$: python prepare_mesh.py
```
* Open V-REP and run scenes/get_initial_poses.ttt. This will create a file data/initial_poses.txt that contains all information on the starting pose of the object and gripper, and will be used for generating potential grasp candidates.
* Run initialize/prepare_candidates.py. This will read the pose information collected by V-REP, and generate a list of candidates for each object that will be tested in the simulator. Note that these candidates will be saved under collect/candidates
```unix
$: cd initialize
$: cat ../data/initial_poses.txt | parallel python prepare_candidates.py
```
* Once the candidates have been generated, you can either run each of them manually through the simulation, or create a "commands" file that will aid in autonomously running them through the simulator. If you plan on running in headless mode (and the following line works for you), you can skip the subsequent aside.

```unix
$: cd initialize
$: python prepare_commands.py
```

### Aside: Dissecting the commands, and running with / without headless mode
On Line 72 of prepare_commands.py, we have the following code:

```unix
commands[i] = \
            'ulimit -n 4096; export DISPLAY=:1; vrep.sh -h -q -s -g%s -g%s -g%s %s '\
            %(sub_cmd[0], sub_cmd[1], sub_cmd[2], config_simulation_path)
```

Each element serves the following purpose:

* __ulimit -n 4096__ : Setting the number of available processes a user can run. Note that you may not need this, but we found it useful.
* __export DISPLAY=:1__ : For running these V-REP simulations in headless mode, an Xserver is needed for handling the vision information captured through the cameras. Here, we index a specific server attached to a display, but note that depending on how yours has been set up, you may need to change the index accordingly.
* __vrep.sh -h -q -s -g%s -g%s -g%s %s__ : V-REP has a number of command-line options available (see [here](http://www.coppeliarobotics.com/helpFiles/en/commandLine.htm)). __-h__ specifies headless mode (i.e. without the GUI), __-q__ tells the program to quit after simulation has ended, __-s__ is to start the simulation, and __-g__ are arguments that get passed inside of the simulation script. The first -g specifies a specific set of grasps we've generated (e.g. 40_carafe_final-11-Mar-2016-17-09-17.txt), while the second and third -g's specify a range of grasps we wish to use (e.g. lines 1 --> 1500 of the specified file). The final element of the command specifies where the V-REP scene we wish to run can be found.

Notice that the -h flag assumes you will be running the simulations in headless mode, which require an Xorg server for handling vision information. If you are not running in headless mode, you can replace the above line with the following:

```unix
commands[i] = \
            'ulimit -n 4096; vrep.sh -q -s -g%s -g%s -g%s %s '\
            %(sub_cmd[0], sub_cmd[1], sub_cmd[2], config_simulation_path)
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

* (optional) To get a sense of what the simulation is doing, you can also run commands *not* using headless mode. Adjust the following command accordingly, with the object and grasp range you wish to use:
```
$: vrep.sh -q -s -g40_carafe_final-11-Mar-2016-17-09-17.txt -g1 -g1501 /path/to/grasping/collect/scene_collect_grasps.ttt
```
where /path/to/grasping/collect/scene_collect_grasps.ttt represents the config_simulation_path variable in lib/python_config.py

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
