myconf = {

    --[[
    -- Basic configuration properties
    config = 'laptop';
    path_sep = '\\';
    data_dir = 'C:\\Users\\Matt\\Documents\\grasping\\initialize\\candidates\\';
    working_dir = 'C:\\Users\\Matt\\Documents\\grasping\\initialize\\collected\\';
    object_file = '82_robot_final-28-Mar-2016-09-43-31.txt';
    mesh_dir = 'C:\\Users\\Matt\\Documents\\vrep-scenes\\object_files\\stl_meshes\\';
    ]]

    --[[
    config = 'school';
    path_sep = '/';
    data_dir = '/home/robot/Documents/grasping/initialize/';
    working_dir = '/home/robot/Documents/grasping/initialize/';
    object_file = '93_snake_final-25-Feb-2016-20-11-05.txt';
    mesh_dir = '/home/robot/Documents/grasping/data/meshes/meshes/';
    ]]

    config = 'gpu';
    path_sep = '/';
    data_dir = '/scratch/mveres/grasping/grasping/initialize';
    working_dir = '/scratch/mveres/grasping/grasping/collect';
    object_file = '93_snake_final-25-Feb-2016-20-10-47.txt';
    mesh_dir = '/home/robot/Documents/grasping/data/meshes/meshes/';


    -- Gripper properties
    gripper_base = 'BarrettHand';
    gripper_palm = 'BarrettHand_PACF';
    gripper_contacts = {'BarrettHand_fingerTipSensor_respondable0',
                        'BarrettHand_fingerTipSensor_respondable1',
                        'BarrettHand_fingerTipSensor_respondable2'};

    gripper_force_sensors = {'BarrettHand_fingerTipSensor0',
                             'BarrettHand_fingerTipSensor1',
                             'BarrettHand_fingerTipSensor2'};

    gripper_finger_angles = {0, 0, 0}; -- Can be arbitrary angles
    palm_distances = {0.04, 0.07, 0.10};
    contact_proximity_sensor = 'contactPointProximitySensor';


    -- Object properties
    object_material = 'usr_sticky';
    object_start_position = {0,0,0.3};

    -- Camera properties
    camera_resolution_x = 128;
    camera_resolution_y = 128;
    camera_near_clip = 0.01;
    camera_far_clip = 0.7;
    camera_fov = 50*math.pi/180; -- camera field of view
    camera_contact_offset = 0.3; -- Position of camera away from  gripper


    -- Visualization properties
    display_num_points = 5000;
    display_point_density = 0.001;
    display_point_size = 0.0005;
    display_vector_width = 0.0005;

    -- Parameters controlling the object lift
    maxVel={0.2,0.2,0.2,1.0};
    maxAccel={0.05,0.05,0.05,0.05};
    maxJerk={0.2,0.2,0.2,1.0};

    -- Data initialization config
    object_list_file = 'objects.txt';
}


