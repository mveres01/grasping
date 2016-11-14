import os, sys
import csv
import math
import random
import time
import logging
import itertools
import theano
import lasagne

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np
import cPickle as pickle
import theano.tensor as T

import lasagne.layers as nn

from sklearn import preprocessing

# Project-dependencies
from utils import (load_dataset_hdf5, split_similar_objects,
                   get_network_info, get_maximized_prediction)

from visualize import (visualize_mu_logsigma_output, display_grasps)

from visualize_grasps_mesh import tile_grasps_mesh

from loss import build_conditional_loss
from loss import estimate_cll
from networks import build_cgm_network

from visualize import  plot_tsne_generated_grasps_gmm


scaler = None
image_min = []
image_max = []
image_std = []
image_mean = []


GLOBAL_SAVE_DIR = '/scratch/mveres/predictions'
GLOBAL_DATAFILE = '/scratch/mveres/ttv_localencode_v40.hdf5'

# NOTE: Images were clipped in sim to be between (0.1, 0.75)
def transform_images(images, compute_new=False):

    global image_min, image_max, image_std, image_mean

    # Need to place channel columns first
    if images.ndim == 4:
        images = np.transpose(images,(1,0,2,3))

    # Compute the statistics
    if compute_new == True:
        image_mean = []
        for i, channel in enumerate(images):
            minv = np.min(channel)
            maxv = np.max(channel)

            image_mean.append((minv + maxv)/2.)
    
    # Apply transformation to image channels
    for i, channel in enumerate(images):
        images[i] = (images[i]/image_mean[i]) - 1.

    if images.ndim==4:
        images = np.transpose(images, (1,0,2,3))
    elif images.ndim == 3:
        images = images[:,np.newaxis,:,:]

    images = images.astype(theano.config.floatX)
    return images


# minal / maxval are the ranges set in simulator (i.e. known vals)
def inverse_transform_images(images):

    global image_min, image_max, image_std, image_mean

    images = np.transpose(images, (1,0,2,3))    
    for i, channel in enumerate(images):
        images[i] = (images[i] + 1.)*image_mean[i]

    images = np.transpose(images, (1,0,2,3))

    return images


def transform_grasps(grasps, compute_new = False):

    global scaler

    # If data hasn't been fit yet, fit a scaler (global value)
    if compute_new == True:
        scaler = preprocessing.StandardScaler().fit(grasps)
    grasps = scaler.transform(grasps)

    n_samples, n_var = grasps.shape
    grasps = grasps.astype(theano.config.floatX)

    return grasps


def inverse_transform_grasps(grasps):

    global scaler

    shape = grasps.shape
    if len(shape) != 2:
        grasps = grasps.reshape(shape[0], -1)

    grasps = scaler.inverse_transform(grasps)
    grasps = grasps.reshape(shape)

    return grasps


# ######################### Batch iterator #######################
def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


def iterate_mb_class(inputs, labels, batchsize, shuffle=False):

    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)

    # Get a list of the unique object classes
    names = labels[:,0]
    classes = [l.split('/')[-1].split('_')[0] for l in names]
    unique_classes = list(set(classes))

    # Find the indices where these classes are in the label set
    classes = np.asarray(classes)
    sorted_idx = [indices[classes==u] for u in unique_classes]
    max_batches = np.max([len(s) for s in sorted_idx])
   
    # Put different classes after each other, then return a batch
    # e.g. indices = [class1, class2 ... classn, class1, class2 ... ] 
    indices = zip(*sorted_idx)
    indices = list(itertools.chain(*indices))
    for start_idx in range(0, len(indices) - batchsize+1, batchsize):
        excerpt = indices[start_idx:start_idx+batchsize]
        yield inputs[excerpt]


# Convenience function for computing conditional log-likelihood
def conditional_likelihood(cll_fcn, images, grasps, batch_size=1):

    valid_cll = 0
    indices = np.arange(images.shape[0])
    for batch in iterate_minibatches(indices, batch_size, False):
        est  =  cll_fcn(images[batch],  grasps[batch]) 
        valid_cll += est

    valid_cll = valid_cll / (batch_size*len(indices))
    return valid_cll


def sample_grasps(images, grasps, labels, sample_fcn, epoch, save_dir, n_samples=6):

    n_samples_per_gen = 10
    for i in xrange(images.shape[0]):

        sampled_grasps = np.zeros((n_samples, 18))
        images_tiled = np.repeat(images[i:i+1], n_samples_per_gen, axis=0)

        for j in xrange(n_samples):

            gr_mu, gr_ls = sample_fcn(images_tiled)

            y, success = get_maximized_prediction(gr_mu, gr_ls)

            if scaler is not None:
                y = np.atleast_2d(y)
                y = scaler.inverse_transform(y)
            sampled_grasps[j] = y

        img_name = 'epoch%d_grasps%d.png'%(epoch, i) 
        save_path = os.path.join(save_dir, img_name)

        grasp_orig = inverse_transform_grasps(grasps[i:i+1])
        fig = tile_grasps_mesh(grasp_orig, sampled_grasps, labels[i])

        plt.tight_layout()
        plt.savefig(save_path)


# Builds network (using parameter values in 'params' and defines functions
# for sampling from network
def build_network(params):
 
    # Between fully-connected and conv-images
    params['X1'] = T.tensor4('x1')
    params['Y'] = T.matrix('y')
    params['Y_tgt'] = T.matrix('y_tgt')
    
    # Create VAE model and (optionally) load previously saved weights
    print '  Building Model ... '
    network = build_cgm_network(**params)

    params['net_outputs'] =  [network['l_gener_x'], network['l_prior_z']]

    # Optionally load a previously saved weight file
    if params['weight_file'] is not None:
        f = open(params['weight_file'], 'rb')
        weights = pickle.load(f)
        f.close()

        lasagne.layers.set_all_param_values(params['net_outputs'], weights)

    # Print the network structure / write to file for reference
    logger = params['logger']
    net_info = get_network_info(network['l_gener_x'])
    if logger is not None:
        [logger.info(layer) for layer in net_info]
        print 'network: '
        for layer in net_info: print layer

    return network


def build_train_fcns(network, X1, Y, lrate, n_cll, beta1, beta2, **kwargs):


    print '  Compiling conditional LL function ... '    
    cll_sym_loss = estimate_cll(network, X1, Y, n_samples=n_cll)

    cll_fcn = theano.function([X1, Y], cll_sym_loss)

    # Generates a sample using the mean predictions of the network
    genz = nn.get_output(network['l_prior_z'], deterministic=True)
    genx = nn.get_output(network['l_gener_x'], 
                        {network['l_recog_z']:genz}, 
                         deterministic=True)
    gen_fcn = theano.function([X1], genx) 


    print '  Compiling training function ... ' 
    net_outputs= [network['l_gener_x'], network['l_prior_z']]
    net_params = lasagne.layers.get_all_params(net_outputs, trainable=True)

    loss, _ = build_conditional_loss(False, network, Y)

    grads = T.grad(loss, net_params)
    updates = lasagne.updates.adam(grads, net_params, 
                                   learning_rate=lrate, 
                                   beta1=beta1, beta2=beta2)

    train_fcn = theano.function([X1, Y], loss, updates=updates)   

    print '  Compiling validation function ... ' 
    val_loss, val_pred = build_conditional_loss(True, network, Y)
    
    # Function for validating losses given both inputs simultaneously
    val_fcn = theano.function([X1, Y], [val_loss, val_pred])

    stoch_vars = lasagne.layers.get_output(
              [network['l_recog_mu'], network['l_recog_ls'],
               network['l_prior_mu'], network['l_prior_ls'],
               network['l_gener_mu'], network['l_gener_ls']],
               deterministic=True)
    stoch_vis_fcn = theano.function([X1, Y], stoch_vars)



    # Get the first parameterizing distribution
    prior_z = lasagne.layers.get_output(
        [network['l_prior_mu'], network['l_prior_ls']], deterministic=True)

    # Sample stochastically
    code = lasagne.layers.get_output(
        network['l_prior_z'], 
        {network['l_prior_mu']:prior_z[0], 
        network['l_prior_ls']:prior_z[1]}, deterministic=False)

    # Get the generators parameterizing distribution
    grasp_vals = lasagne.layers.get_output(
        [network['l_gener_mu'], network['l_gener_ls']],
        {network['l_recog_z']:code}, deterministic=True)

    # Sample stochastically
    grasp_stoch = lasagne.layers.get_output(
        network['l_gener_x'],
        {network['l_gener_mu']:grasp_vals[0],
        network['l_gener_ls']:grasp_vals[1]}, deterministic=False)

    # For predicting the mu and log(sigma) of sampled inputs
    mu_sigma_fcn = theano.function([X1], grasp_vals)

    train_fcns = {}
    train_fcns['train_fcn'] = train_fcn
    train_fcns['valid_fcn'] = val_fcn
    train_fcns['cll_fcn'] = cll_fcn
    train_fcns['gen_fcn'] = gen_fcn
    train_fcns['stoch_vis_fcn'] = stoch_vis_fcn
    train_fcns['mu_sigma_fcn'] = mu_sigma_fcn

    return train_fcns


# Train the network
def train(network, X_train, X_test, X_val, params, train_fcns):

    save_dir = params['save_dir']
    save_weights = params['save_weights'] 

    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    net_outputs = params['net_outputs']

    cll_fcn = train_fcns['cll_fcn']
    gen_fcn = train_fcns['gen_fcn']
    train_fcn = train_fcns['train_fcn']
    valid_fcn = train_fcns['valid_fcn']
    stoch_vis_fcn = train_fcns['stoch_vis_fcn']
    mu_sigma_fcn = train_fcns['mu_sigma_fcn']

    # Get the training data
    train_grasp, train_image, train_labels = X_train
    test_grasp,  test_image,  test_labels  = X_test
    valid_grasp, valid_image, valid_labels = X_val
 

    # ------------------------- Training  --------------------------------
 
    # For reference, plot what the original grasp space looks like 
    grasp_orig_arr = inverse_transform_grasps(valid_grasp)
    display_grasps(grasp_orig_arr, save_dir+'/grasp_orig.png')

    count =  0
    best_params = None
    best_test_err = np.inf
 
    print 'Starting joint training ... '
    for epoch in range(num_epochs):

        start_time = time.time()

        train_err, valid_err = 0, 0
        train_batches, valid_batches = 0, 0
        indices = np.arange(train_image.shape[0])

        '''
        for batch in iterate_minibatches(indices, batch_size, True):
           
            err = train_fcn(train_image[batch], train_grasp[batch])
            train_err += err
            train_batches += 1
        train_loss=train_err/(train_batches*batch_size)
        '''

        for batch in iterate_mb_class(indices, train_labels, batch_size, True):
          
            err = train_fcn(train_image[batch],  train_grasp[batch])
            train_err += err
            train_batches += 1
        train_loss=train_err/(train_batches*batch_size) 

        # Check validation stats
        valid_loss = 10000
        '''
        indices = np.arange(valid_image.shape[0])
        for batch in iterate_minibatches(indices, batch_size, False):
           
            err, _ = valid_fcn(valid_image[batch], valid_grasp[batch])
            valid_err += err
            valid_batches += 1
        valid_loss=valid_err/(valid_batches*batch_size)
        '''

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  train loss:\t\t{:.6f}".format(train_loss))
        #print("  valid loss:\t\t{:.6f}".format(valid_loss))


        # Estimate the conditional likelihood and use for early stopping
        check_validation = epoch >= 30 and epoch%2 == 0
        valid_cll = 100000
        if check_validation:
            valid_cll = conditional_likelihood(cll_fcn, valid_image, valid_grasp)
            print("CLL took: {:.3f}s".format(time.time()-start_time))
            print("  valid CLL: \t\t{:.6f}".format(valid_cll))

        if any(np.isnan([train_loss, valid_loss, valid_cll])):
            return (np.inf, np.inf, np.inf)


        # Save some info every few epochs
        if epoch % 5 == 0:

            n = valid_image.shape[0]

            # TODO: Check that grasps plotting with just 'scatter', give same 
            # positions / locations as those of grasps + 3d mesh

            # Generate a sample from our network using test info 
            sample_grasps(valid_image[:10], valid_grasp[:10], valid_labels[:10], 
                          mu_sigma_fcn, epoch, params['save_dir'], n_samples=6)

            latent_outputs = []
            rz_mu, rz_ls, pz_mu, pz_ls, gz_mu, gz_ls = \
                stoch_vis_fcn(valid_image[:1000], valid_grasp[:1000])
            latent_outputs.append([rz_mu, rz_ls, 'Recognition'])
            latent_outputs.append([pz_mu, pz_ls, 'Prior'])
            latent_outputs.append([gz_mu, gz_ls, 'Generator'])
           
            visualize_mu_logsigma_output(\
                latent_outputs, save_dir+'/mu_logsigma_output%d.png'%epoch) 

        if valid_cll > best_test_err*0.99:
            count +=1 
        else:
            count = 0
            best_test_err = valid_cll
            best_params = lasagne.layers.get_all_param_values(net_outputs)
        if check_validation and count >= 6:
            lasagne.layers.set_all_param_values(net_outputs, best_params)
            break  

    test_cll = \
        conditional_likelihood(cll_fcn, test_image,test_grasp)
    
    # ------------ SAVE NETWORK PARAMETERS ------------------- 
    if save_weights == True:
        fn =save_dir+'/params_{:.6f}.pkl'.format(test_cll)
        params = lasagne.layers.get_all_param_values(net_outputs)

        f = open(fn, 'wb')
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    # Estimate the conditional likelihood of tesst set
    n = train_image.shape[0]
    train_cll = conditional_likelihood(
        cll_fcn, train_image[:n], train_grasp[:n])
    valid_cll = conditional_likelihood(\
        cll_fcn, valid_image,  valid_grasp)

    print("")
    print("  train CLL:\t\t{:.6f}".format(train_cll))
    print("  valid CLL:\t\t{:.6f}".format(valid_cll))
    print("  test  CLL:\t\t{:.6f}".format(test_cll))

    return network, (train_cll, test_cll, valid_cll)



# Object type is either "similar" or "different" depending on
# whether we are testing objects similar to train set or not
def visualize_network(network, data_in, params, train_fcns, objtype='similar'):

    global scaler
    
    n_samples_per_gen = 10 
    image_var  = params['X1']
    grasp_var  = params['Y']
    grasp_target = params['Y_tgt']
    cll_fcn = train_fcns['cll_fcn']
    gen_fcn = train_fcns['gen_fcn']
    save_dir   = params['save_dir']
    model_type = params['model_type']

    test_batch_size = 1

    pred_dir = os.path.join(save_dir, objtype)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir) 

    grasps, images, labels, props  = data_in

    print 'Computing statistics on %s objects'%objtype
    print 'Grasps/Images: ',grasps.shape, images.shape
 
    # Get the first parameterizing distribution
    prior_z = lasagne.layers.get_output(
        [network['l_prior_mu'], network['l_prior_ls']], deterministic=True)

    # Sample stochastically
    code = lasagne.layers.get_output(
        network['l_prior_z'], 
        {network['l_prior_mu']:prior_z[0], 
        network['l_prior_ls']:prior_z[1]}, deterministic=False)

    # Get the generators parameterizing distribution
    grasp_vals = lasagne.layers.get_output(
        [network['l_gener_mu'], network['l_gener_ls']],
        {network['l_recog_z']:code}, deterministic=True)

    # Sample stochastically
    grasp_stoch = lasagne.layers.get_output(
        network['l_gener_x'],
        {network['l_gener_mu']:grasp_vals[0],
        network['l_gener_ls']:grasp_vals[1]}, deterministic=False)

    # For making deterministic predictions (MEAN value only) 
    mean_pred = lasagne.layers.get_output(
        network['l_gener_x'], {network['l_recog_z']:code}, deterministic=True)

    # Generate an output using DETERMINISTIC z, DETERMINISTIC x 
    mean_gen_fcn = theano.function([params['X1'] ], grasp_vals[0])

    # Generate an output using STOCHASTIC z, STOCHASTIC x
    stoch_gen_fcn = theano.function([params['X1']], grasp_stoch)

    # Generate an output using STOCHASTIC z, STOCHASTIC x
    latent_gen_fcn = theano.function([params['X1']],[code,grasp_stoch])
    
    # For sampling from 'mean' of distributions through network
    pred_fcn = theano.function([params['X1']], mean_pred)

    # For predicting the mu and log(sigma) of sampled inputs
    mu_sigma_fcn = theano.function([params['X1']], grasp_vals)


    # Make deterministic test-set predictions
    cnt = 0
    grasp_pred = np.empty(grasps.shape)
    indices = np.arange(images.shape[0])
    for batch in iterate_minibatches(indices, test_batch_size, False):
        grasp_pred[cnt]  = pred_fcn(images[batch])
        cnt += 1

    # Compute the conditional log-likelihood of the grasps
    MSE = np.mean( (grasp_pred - grasps)**2)
    conditional_ll = conditional_likelihood(cll_fcn, images, grasps)

    print 'CLL: ',conditional_ll
    print 'MSE: ',MSE

    # Un-standardize the data
    grasp_pred_arr = inverse_transform_grasps(grasp_pred)

   
    # Save the predictions to file 
    with open(pred_dir+'/'+model_type+'-'+objtype+'.csv','wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        a = [writer.writerow(np.hstack((label, data, prop)))\
            for label, data, prop in zip(encoded_labels, encoded_grasps, props)]

    with open(pred_dir+'/conditional_likelihood.txt','wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([conditional_ll])

    with open(pred_dir+'/MSE.txt','wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([MSE])

    # Randomly choose an image to generate samples for
    np.random.seed(1234)
    rtg = np.random.randint(1, grasps.shape[0], size=10)
    
    for r in rtg:
        gr_pred = [latent_gen_fcn(images[r:r+1]) for _ in xrange(4000)]

        gr_pred = zip(*gr_pred)
        code = np.vstack(gr_pred[0])
        output_pred = np.vstack(gr_pred[1])

        n_samples=5
        sampled_grasps = np.zeros((n_samples, 18))
        for i in xrange(n_samples):

            images_tiled = np.repeat(images[r:r+1], n_samples_per_gen, axis=0)
            gr_mu, gr_ls = mu_sigma_fcn(images_tiled)

            y, success = get_maximized_prediction(gr_mu, gr_ls)

            if scaler is not None:
                y = np.atleast_2d(y)
                y = scaler.inverse_transform(y)
            sampled_grasps[i] = y


        # Use t-SNE to make a plot of our WHOLE data distribution
        plot_tsne_generated_grasps_gmm(
            output_pred, labels[r], pred_dir+'/'+str(r), scaler) 

        plt.close('all')
        save_path = os.path.join(pred_dir, '%s_sampled_grasps.png'%(r))
        fig = tile_grasps_mesh(None, sampled_grasps, labels[r]) 
        plt.savefig(save_path)


def init_paths_and_logger(model_type):

    # Make sure all the directories we need exist
    print '  Creating save paths'
    dtime = time.strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(GLOBAL_SAVE_DIR, model_type)
    save_dir = os.path.join(model_dir, dtime)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print '  Initializing Logger ... '
    logger = logging.getLogger('sample')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(save_dir+'/log.txt')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return save_dir, logger


def main(params):

    save_dir, logger = init_paths_and_logger(params['model_type'])
    params['save_dir'] = save_dir
    params['logger'] = logger

    for key, value in params.iteritems():
        logger.info(key+', '+str(value))

    # Load data
    data = load_dataset_hdf5(GLOBAL_DATAFILE, n_train=params['n_train'])

    train_grasps, train_images, train_labels, train_props = data[0]
    test_grasps, test_images,  test_labels, test_props = data[1]
    valid_grasps, valid_images,  valid_labels, valid_props = data[2]

    params['train_data_shape: '] = train_images.shape[0]

    # Preprocess grasps - calc new scaler for training set 
    train_grasps = transform_grasps(train_grasps, True)
    test_grasps  = transform_grasps(test_grasps,  False)
    valid_grasps = transform_grasps(valid_grasps, False)

    # Preprocess images - calc tform params for training set
    train_images = transform_images(train_images,True)
    test_images  = transform_images(test_images, False)
    valid_images = transform_images(valid_images,False)

    # Get a list of all the object classes within training set
    train_classes = list(set(train_labels[:,0]))
    for i, cls in enumerate(train_classes):
        train_classes[i] = cls.split('_')[0] + cls.split('_')[1]
    train_classes = list(set(train_classes))

    '''
    valid_grasps = valid_grasps[:2000]
    valid_images = valid_images[:2000]
    valid_labels = valid_labels[:2000]
    valid_props = valid_props[:2000]
    '''

    # NOTE: X_val has no real use in Leave One Out
    X_train = (train_grasps, train_images, train_labels)
    X_test  = (test_grasps,  test_images, test_labels)
    X_val   = (valid_grasps, valid_images, valid_labels) 

    print 'Train/Test/Valid shape: \n',\
        train_images.shape, test_images.shape, valid_images.shape

    # Build the network
    network = build_network(params)
    train_fcns = build_train_fcns(network, **params)

    network, err = train(network, X_train, X_test, X_val, params, train_fcns)
    logger.info('Error: ')
    logger.info('---------------------------------------')
    logger.info('Train Loss: \t\t{:.6f}'.format(err[0]))
    logger.info('Valid Loss: \t\t{:.6f}'.format(err[2]))
    logger.info('Test  Loss: \t\t{:.6f}'.format(err[1])) 
    logger.info('')

    # Find which objects in the test set is similar to training set
    similar, different = split_similar_objects(train_classes, test_labels) 

    

    X_similar = (test_grasps[similar], 
                 test_images[similar],
                 test_labels[similar],
                 test_props[similar])

    X_different = (test_grasps[different], 
                   test_images[different], 
                   test_labels[different],
                   test_props[different])

    visualize_network(network, X_similar, params, train_fcns, objtype='similar')
    visualize_network(network, X_different, params, train_fcns, objtype='different')
        

if __name__ == '__main__':

    #search_hyperparams()

    seed = np.random.randint(1, 123456789)
    np.random.seed(seed)
        

    z_dim =6 
    y_dim = 6 
    x1_filts = [16, 16, 32, 32, 64, 64, 64, 64, 64, 64]
    x1_fsize = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    n_channels = 4
    n_train = int(2**15)

    param_list = {
            'model_type':'cvae-%s'%n_train,
            'save_weights':True,
            'batch_size':200,
            'lrate':1e-3,
            'num_epochs':100,
            'beta1':0.9,
            'beta2':0.99,
            'x1_filts':x1_filts,
            'x1_fsize':x1_fsize,
            'r_hid_x1':[100],
            'r_hid_y':[y_dim],
            'r_hid_shared':[64, z_dim],  
            'p_hid_x1':[100, z_dim], 
            'd_hid_x1':[100, 18],
            'g_hid_z':[z_dim, 9, 18],
            'x1_shape':(None, n_channels, 128, 128),
            'y_shape':(None, 18),
            'nonlinearity':lasagne.nonlinearities.LeakyRectify(0.2),
            'regularization':'dropout',
            'p':0.1, 
            'rng':None,
            'weight_file':None,
            'W_init':lasagne.init.GlorotUniform(),
            'n_train':n_train,
            'n_cll':100,
            'max_norm':5}

    main(param_list)

