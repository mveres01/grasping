import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import tanh, sigmoid, rectify
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import get_network_info


def LogSumExp(x, axis=None, keepdims=True):
    ''' Numerically stable theano version of the Log-Sum-Exp trick'''
    x_max = T.max(x, axis=axis, keepdims=True)
  
    preres = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=keepdims))
    return preres + x_max.reshape(preres.shape)


# Log likelihood for cost function
def log_likelihood(tgt, mu, ls, eps=1e-6, axis=None):
  
    c = -np.float32(0.5*np.log(2.0*np.pi))
    ll = c - ls - T.sqr(tgt-mu) / (2.0*T.exp(2.0*ls)+eps)
    return T.sum(ll, axis=axis)


# KL Divergence of two Gaussians, assuming diagonal covariance
def kldiv(mu1, ls1, mu2, ls2, eps=1e-6, axis=None):
    
    kl = ls2 - ls1 - 0.5 + \
         0.5*T.exp(2*ls1)/(T.exp(2*ls2)+eps) + \
         0.5*T.sqr(mu1-mu2)/(T.exp(2*ls2)+eps)
    return T.sum(kl, axis=axis)


# Used to build the loss function for conditional network
def build_conditional_loss(deterministic, network, grasps_target):

    l_recog_mu = network['l_recog_mu']
    l_recog_ls = network['l_recog_ls']
    l_recog_z  = network['l_recog_z']

    l_gener_mu = network['l_gener_mu']
    l_gener_ls = network['l_gener_ls']
    l_gener_x  = network['l_gener_x']
    
    l_prior_mu = network['l_prior_mu']
    l_prior_ls = network['l_prior_ls']
    l_prior_z  = network['l_prior_z']

    params = lasagne.layers.get_output(
        [l_recog_mu, l_recog_ls, l_recog_z, l_gener_mu, 
         l_gener_ls, l_gener_x, l_prior_mu, l_prior_ls], 
         deterministic=deterministic)

    rec_mu, rec_ls, rec_z, gen_mu, gen_ls, gen_x, pri_mu, pri_ls=params

    pri_ls = T.clip(pri_ls, -3, 1)
    rec_ls = T.clip(rec_ls, -3, 1)
    gen_ls = T.clip(gen_ls, -3, 1)

    # KL( p(z|x,y) || p(z|x) )
    kl_div = kldiv(rec_mu, rec_ls, pri_mu, pri_ls)

    # p(y|x,z)
    py_xz = log_likelihood(grasps_target, gen_mu, gen_ls)
    
    loss = T.sum(kl_div - py_xz)
    return loss, gen_x


# Used to build the loss function for conditional network
def build_recurrent_loss(deterministic, network, grasps_target):

    l_recog_mu = network['l_recog_mu']
    l_recog_ls = network['l_recog_ls']
    l_recog_z  = network['l_recog_z']

    l_gener_mu = network['l_gener_mu']
    l_gener_ls = network['l_gener_ls']
    l_gener_x  = network['l_gener_x']
   
    l_prior_mu = network['l_prior_mu']
    l_prior_ls = network['l_prior_ls']
    l_prior_z  = network['l_prior_z']

    params = lasagne.layers.get_output(
              [l_recog_mu, l_recog_ls, l_recog_z, l_gener_mu, 
              l_gener_ls, l_gener_x, l_prior_mu, l_prior_ls],
              deterministic=deterministic)

    rec_mu, rec_ls, rec_z, gen_mu, gen_ls, gen_x, pri_mu, pri_ls=params
 
    pri_ls = T.clip(pri_ls, -3, 1)
    rec_ls = T.clip(rec_ls, -3, 1)
    gen_ls = T.clip(gen_ls, -3, 1)

    # KL( p(z|x,y) || p(z|x) )
    kl_div = kldiv(rec_mu, rec_ls, pri_mu, pri_ls, axis=1)

    # p(y|x,z)
    py_xz = log_likelihood(grasps_target, gen_mu, gen_ls, axis=1)
    
    loss = T.sum(kl_div - py_xz)

    return loss, gen_x


# Used to build the loss function for conditional network
def build_gsnn_loss(deterministic, network, grasps_target):

    l_prior_mu = network['l_prior_mu']
    l_prior_ls = network['l_prior_ls']
    l_prior_z  = network['l_prior_z']
    l_gener_mu = network['l_gener_mu']
    l_gener_ls = network['l_gener_ls']
    l_gener_x  = network['l_gener_x']
    
    params = lasagne.layers.get_output(
              [l_prior_mu, l_prior_ls, l_prior_z, 
               l_gener_mu, l_gener_ls, l_gener_x], 
              deterministic=deterministic)

    pri_mu, pri_ls, pri_z, gen_mu, gen_ls, gen_x =params
   
    pri_ls = T.clip(pri_ls, -3, 1)
    gen_ls = T.clip(gen_ls, -3, 1)
 
    # p(y|x,z)
    py_xz  = -log_likelihood(grasps_target, gen_mu, gen_ls)
 
    return py_xz, gen_x


def build_prediction_loss(deterministic, network, target_var):

    l_x2y_out= network['l_x']
    
    x2y_out = lasagne.layers.get_output(
            l_x2y_out, deterministic=deterministic)
    
    # p(y|x)
    # Assume output distribution is 0 mean Gaussian 
    py_x = -log_likelihood(target_var, x2y_out, 0) 

    return py_x, x2y_out


# Used to estimate conditional likelihood using importance sampling from conditional VAE
# Estimate of the conditional log-likelihood using important sampling
def estimate_cll(network, image_var,  grasp_var, n_samples=100):

    l_recog_x1 = network['l_recog_x1']
    l_recog_y  = network['l_recog_y']
    l_prior_x1 = network['l_prior_x1']
    l_pred_x1 = network['l_pred_x1']

    l_recog_mu = network['l_recog_mu']
    l_recog_ls = network['l_recog_ls']
    l_recog_z  = network['l_recog_z']
    l_gener_mu = network['l_gener_mu']
    l_gener_ls = network['l_gener_ls']
    l_prior_mu = network['l_prior_mu']
    l_prior_ls = network['l_prior_ls']
    l_prior_z  = network['l_prior_z']

    x1 = T.extra_ops.repeat(image_var, n_samples, axis=0)
    y  = T.extra_ops.repeat(grasp_var, n_samples, axis=0)

    # Draw sample z's from recognition distribution
    rec_mu, rec_ls  = lasagne.layers.get_output(
        [l_recog_mu, l_recog_ls], {l_recog_x1:x1, 
        l_recog_y:y}, deterministic=True)

    # Now sample stochastically across Gaussian layer
    rec_z = lasagne.layers.get_output(l_recog_z,
        {l_recog_mu:rec_mu, l_recog_ls:rec_ls}, deterministic=False)

    # Deterministic skips the regularization layers
    pri_mu, pri_ls, gen_mu, gen_ls = lasagne.layers.get_output(
        [l_prior_mu, l_prior_ls, l_gener_mu, l_gener_ls], 
        {l_prior_x1:x1, l_pred_x1:x1, l_recog_z:rec_z}, deterministic=True)

    py_xz = log_likelihood(y, gen_mu, gen_ls, axis=1)
    pz_x  = log_likelihood(rec_z, pri_mu, pri_ls, axis=1)
    qz_xy = log_likelihood(rec_z, rec_mu, rec_ls, axis=1)

    logp = py_xz + pz_x - qz_xy
    cll = -T.log(T.cast(n_samples, 'float32')) + LogSumExp(logp)
    cll = -T.sum(cll)

    return cll


# Used to estimate conditional likelihood using importance sampling from conditional VAE
# Estimate of the conditional log-likelihood using important sampling
def estimate_recurrent_cll(network, image_var,  grasp_var, n_samples=10):

    l_recog_x1 = network['l_recog_x1']
    l_recog_y  = network['l_recog_y']
    l_prior_x1 = network['l_prior_x1']
    l_pred_x1 = network['l_pred_x1']

    l_recog_mu = network['l_recog_mu']
    l_recog_ls = network['l_recog_ls']
    l_recog_z  = network['l_recog_z']
    l_gener_mu = network['l_gener_mu']
    l_gener_ls = network['l_gener_ls']
    l_prior_mu = network['l_prior_mu']
    l_prior_ls = network['l_prior_ls']
    l_prior_z  = network['l_prior_z']

    x1 = T.extra_ops.repeat(image_var, n_samples, axis=0)
    y  = T.extra_ops.repeat(grasp_var, n_samples, axis=0)

    # Draw sample z's from recognition distribution
    rec_mu, rec_ls  = lasagne.layers.get_output(
        [l_recog_mu, l_recog_ls], {l_recog_x1:x1, 
        l_recog_y:y}, deterministic=True)

    # Now sample stochastically across Gaussian layer
    rec_z = lasagne.layers.get_output(l_recog_z,
        {l_recog_mu:rec_mu, l_recog_ls:rec_ls}, deterministic=False)

    # Deterministic skips the regularization layers
    pri_mu, pri_ls, gen_mu, gen_ls = lasagne.layers.get_output(
        [l_prior_mu, l_prior_ls, l_gener_mu, l_gener_ls], 
        {l_prior_x1:x1, l_pred_x1:x1, l_recog_z:rec_z}, deterministic=True)

    py_xz = log_likelihood(y, gen_mu, gen_ls, axis=1)
    pz_x  = log_likelihood(rec_z, pri_mu, pri_ls, axis=1)
    qz_xy = log_likelihood(rec_z, rec_mu, rec_ls, axis=1)

    logp = py_xz + pz_x - qz_xy
    cll = -T.log(T.cast(n_samples, 'float32')) + LogSumExp(logp)
    cll = -T.sum(cll)

    return cll




# Used to estimate conditional likelihood using Monte-Carlo sampling from conditional VAE
# Estimate of the conditional log-likelihood using important sampling
def estimate_cll_montecarlo(network, image_var, grasp_var, n_samples=10):


    l_gener_in = network['l_gener_in']
    l_recog_z  = network['l_recog_z']
    l_gener_mu = network['l_gener_mu']
    l_gener_ls = network['l_gener_ls']
    l_gener_in = network['l_gener_in']
    l_prior_in = network['l_prior_in']
    l_prior_mu = network['l_prior_mu']
    l_prior_ls = network['l_prior_ls']
    l_prior_z  = network['l_prior_z']

    x1 = T.extra_ops.repeat(image_var, n_samples, axis=0)
    x2 = T.extra_ops.repeat(grasp_var, n_samples, axis=0)

    
    # Need to sample deterministically to avoid network regularization
    pri_mu, pri_ls = lasagne.layers.get_output(
        [l_prior_mu, l_prior_ls],
        {l_prior_in:x1}, deterministic=True)

    pri_z = lasagne.layers.get_output(l_prior_z,
        {l_prior_mu:pri_mu, l_prior_ls:pri_ls}, deterministic=False) 

    gen_mu, gen_ls = lasagne.layers.get_output(
        [l_gener_mu, l_gener_ls],
        {l_recog_z:pri_z, l_gener_in:x1},deterministic=True)


    py_xz = log_likelihood(x2, gen_mu, gen_ls, axis=1)

    cll = T.log(1./T.cast(n_samples, 'float32')) + LogSumExp(py_xz)
    cll = -cll.sum()

    return cll


# Used to estimate conditional likelihood using MC sampling for GSNN
def estimate_gsnn_cll(network, image_var, grasp_var, n_samples=10000):

    l_prior_x1 = network['l_prior_x1']
    l_pred_x1 = network['l_pred_x1']
    l_gener_mu = network['l_gener_mu']
    l_gener_ls = network['l_gener_ls']
    l_prior_mu = network['l_prior_mu']
    l_prior_ls = network['l_prior_ls']
    l_prior_z  = network['l_prior_z']


    x1 = T.extra_ops.repeat(image_var, n_samples, axis=0)
    y = T.extra_ops.repeat(grasp_var, n_samples, axis=0)
    
    # Need to sample deterministically to avoid network regularization
    pri_mu, pri_ls = lasagne.layers.get_output(
        [l_prior_mu, l_prior_ls],
        {l_prior_x1:x1}, deterministic=True)

    pri_z = lasagne.layers.get_output(l_prior_z,
        {l_prior_mu:pri_mu, l_prior_ls:pri_ls}, deterministic=False) 

    gen_mu, gen_ls = lasagne.layers.get_output(
        [l_gener_mu, l_gener_ls],
        {l_prior_z:pri_z, l_pred_x1:x1},deterministic=True)


    py_xz = log_likelihood(y, gen_mu, gen_ls, axis=1)

    cll = T.log(1./T.cast(n_samples,'float32')) + LogSumExp(py_xz)
    cll = -cll.sum()

    return cll


# Used to estimate conditional likelihood for CNN
def estimate_conv_cll(network, image_var, grasp_tgt, n_samples=1):

    x1 = T.extra_ops.repeat(image_var, n_samples, axis=0)
    y = T.extra_ops.repeat(grasp_tgt, n_samples, axis=0)
    grasp_x = lasagne.layers.get_output(
        network['l_x'], {network['l_input_x1']:x1}, deterministic=True)
   
    py_x = -log_likelihood(y, grasp_x, 0., axis=None)

    return T.sum(py_x)

