import os

# Set a constant object mass and density
#project_dir = '/scratch/mveres/grasping'
project_dir = '/home/robot/Documents/grasping'

# Used for prepare_mesh.py
config_object_mass = 1.0
config_object_density = 1000.
config_object_dir = os.path.join(project_dir, 'data/meshes/object_files')
config_mesh_dir = os.path.join(project_dir, 'data/meshes/meshes')

# Used for prepare_candidates.py
config_candidate_dir = os.path.join(project_dir, 'collect/candidates')
config_pose_path = os.path.join(project_dir, 'data/initial_poses.txt')

# Used for prepare_commands.py
config_compute_nodes = 5 # How many compute nodes are available
config_chunk_size = 1500
config_max_trials = 10000
config_command_dir = os.path.join(project_dir, 'collect/commands')
config_simulation_path = os.path.join(project_dir, 'collect/scene_collect_grasps.ttt')

# Used for decode_grasp_data.py
config_image_width = 128
config_image_height = 128
config_near_clip = 0.01
config_far_clip = 0.70
config_fov = 50.0*(3.14159265/180.0)
config_collected_data_dir = os.path.join(project_dir, 'data/collected')
config_processed_data_dir = os.path.join(project_dir, 'data/processed')
config_sample_image_dir = os.path.join(config_processed_data_dir, 'sample_images')
config_sample_pose_dir = os.path.join(config_processed_data_dir, 'sample_poses')

# Used for split_dataset.py, postprocess.py
# train_items contains a list of object classes to be used in train set
config_train_item_list = 'train_items.txt'
config_train_dir = os.path.join(config_processed_data_dir, 'train')
config_test_dir = os.path.join(config_processed_data_dir, 'test')
config_dataset_path = os.path.join(config_processed_data_dir, 'grasping.hdf5')
