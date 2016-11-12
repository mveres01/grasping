import numpy as np
import lasagne
import lasagne.layers as nn
from lasagne.layers import dnn
import theano
import theano.tensor as T
from lasagne.nonlinearities import tanh, sigmoid, rectify
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import get_network_info


# ##################### Custom layer for middle of VCAE #################
# This layer takes the mu and sigma (both DenseLayers) and combines 
# them with a random vector epsilon to sample values for a multivariate
# Gaussian
class GaussianSampleLayer(nn.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1,21474679))
        super(GaussianSampleLayer, self).__init__([mu,logsigma],**kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs

        n_samples = self.input_shapes[0][0] or inputs[0].shape[0]

        # Conv vs dense layer
        if inputs[0].ndim == 4:
            shape=(n_samples,
                   self.input_shapes[0][1] or inputs[0].shape[1],
                   self.input_shapes[0][2] or inputs[0].shape[2],
                   self.input_shapes[0][3] or inputs[0].shape[3])
        else:
            shape=(n_samples,
                   self.input_shapes[0][1] or inputs[1].shape[1]) 

        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)


# https://github.com/TimSalimans/weight_norm/blob/master/nn.py
# i.e. developer of the technique along with D. Kingma
class WeightNormLayer(nn.Layer):
    def __init__(self, incoming, 
                 b=lasagne.init.Constant(0.), 
                 g=lasagne.init.Constant(1.), 
                 W=lasagne.init.GlorotUniform(), 
                 nonlinearity=None, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity

        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g")
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]
        
        # scale weights in layer below
        incoming.W_param = incoming.W
        incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim==4:
            W_axes_to_sum = (1,2,3)
            W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]
        if g is not None:
            incoming.W = incoming.W_param * (self.g/T.sqrt(T.sum(T.square(incoming.W_param),
                axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / T.sqrt(T.sum(T.square(incoming.W_param),
                axis=W_axes_to_sum,keepdims=True))        


    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = T.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            stdv = T.sqrt(T.mean(T.square(input),axis=self.axes_to_sum))
            input /= stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m/stdv), (self.g, self.g/stdv)]
        elif hasattr(self,'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(input)
       

# Wrapper function / utility function 
def weight_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)


# Code from https://github.com/TimSalimans/weight_norm/blob/master/nn.py
class GlobalAvgLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(GlobalAvgLayer, self).__init__(incoming, **kwargs)
    def get_output_for(self, input, **kwargs):
        return T.mean(input, axis=(2,3))
    def get_output_shape_for(self, input_shape):
        return input_shape[:2]


# Avoids having messy statements inside of network build
def apply_regularization(layer_in, regularization ='batch_norm', p=0.5):

    if regularization == 'dropout':
        return nn.DropoutLayer(layer_in,p=p,name='drop')
    elif regularization == 'batch_norm':
        return nn.batch_norm(layer_in)
    elif regularization == 'weight_norm':
        return weight_norm(layer_in, name='wn')

    return layer_in


# Regular convolutional net - NO SAMPLING 
def build_prediction_net(X1, x1_filts, x1_fsize, d_hid_x1, 
                         x1_shape, nonlinearity, regularization, p,
                         rng=None, W_init=lasagne.init.Orthogonal(),
                         b_init = lasagne.init.Constant(0.), stride=2):


    # ------------------ Build graph for X1 -------------------------
    l_in_x1 = nn.InputLayer(input_var=X1, shape=x1_shape, name='d_x1_in')

    layer = l_in_x1
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[0], x1_fsize[0], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[1], x1_fsize[1], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 64x64
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[2], x1_fsize[2], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[3], x1_fsize[3], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 32x32
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[4], x1_fsize[4], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[5], x1_fsize[5], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[6], x1_fsize[6], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 16x16
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[7], x1_fsize[7], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[8], x1_fsize[8], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[9], x1_fsize[9], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 8x8
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[7], x1_fsize[7], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[8], x1_fsize[8], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[9], x1_fsize[9], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 4x4
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[9], x1_fsize[9], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')


    layer = nn.DenseLayer(layer, num_units=d_hid_x1[0], W=W_init, b=b_init, 
                          nonlinearity=nonlinearity)
    layer = nn.DropoutLayer(layer, p=0.1, name='drop')
    #layer = weight_norm(layer, name='wn')
    #layer = nn.batch_norm(layer)


    l_x = nn.DenseLayer(layer, num_units=d_hid_x1[1], W=W_init, b=b_init, 
                          nonlinearity=None, name='d_x')


    return {'l_input_x1':l_in_x1, 'l_x':l_x}


# Takes an input X and Y, and produces a map to sample a gaussian
# lantent variable z
def build_recognition_net(X1, Y, x1_filts, x1_fsize,
                          r_hid_x1, r_hid_y, r_hid_shared,
                          x1_shape, y_shape, nonlinearity,
                          regularization, p, rng=None,
                          W_init=lasagne.init.Orthogonal(),
                          b_init=lasagne.init.Constant(0.), stride=2):

    # ------------ Build the graph for x1  --------------------

    l_in_x1 = nn.InputLayer(input_var=X1, shape=x1_shape, name='r_x1_in')
   

    layer = l_in_x1
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[0], x1_fsize[0], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[1], x1_fsize[1], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 64x64
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[2], x1_fsize[2], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[3], x1_fsize[3], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 32x32
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[4], x1_fsize[4], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[5], x1_fsize[5], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[6], x1_fsize[6], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 16x16
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[7], x1_fsize[7], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[8], x1_fsize[8], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[9], x1_fsize[9], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 8x8
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[7], x1_fsize[7], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[8], x1_fsize[8], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[9], x1_fsize[9], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 4x4
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[9], x1_fsize[9], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # layer = nn.DropoutLayer(layer, p=0.5, name='drop')
    #layer = weight_norm(layer, name='wn')
    #layer = nn.batch_norm(layer)
    layer = nn.ReshapeLayer(layer, (-1, x1_filts[9]*4*4))


    layer = nn.DenseLayer(layer, num_units=r_hid_x1[0], W=W_init, 
                         b=b_init, nonlinearity=nonlinearity)
    x1_fuse = nn.DropoutLayer(layer, p=0.1, name='drop')


    # ----------------- Apply network to grasp input -----------------------
    if type(Y) != nn.dense.DenseLayer:
        l_in_y = nn.InputLayer(input_var=Y, shape=y_shape)
    else:
        l_in_y = Y


    layer = nn.DenseLayer(l_in_y, num_units=r_hid_y[0], W=W_init, 
                         b=b_init, nonlinearity=nonlinearity)
    layer = nn.DropoutLayer(layer, p=0.1, name='drop')
    #layer = weight_norm(layer, name='wn')
    #layer = nn.batch_norm(layer)

    layer = nn.ConcatLayer([x1_fuse,  layer], axis=1, name='r_concat') 

    layer = nn.DenseLayer(layer, num_units=r_hid_shared[0], W=W_init, 
                         b=b_init, nonlinearity=nonlinearity)
    layer = nn.DropoutLayer(layer, p=0.1, name='drop')
    #layer = weight_norm(layer, name='wn')
    #layer = nn.batch_norm(layer)


    layer = nn.DenseLayer(layer, num_units=r_hid_shared[1], W=W_init, 
                         b=b_init, nonlinearity=nonlinearity)
    layer = nn.DropoutLayer(layer, p=0.1, name='drop')
    #layer = weight_norm(layer, name='wn')
    #layer = nn.batch_norm(layer)

    z_mu = nn.DenseLayer(layer, num_units=r_hid_shared[-1], W=W_init, 
                         b=b_init, nonlinearity=None, name='r_z_mu')

    z_ls = nn.DenseLayer(layer, num_units=r_hid_shared[-1], W=W_init, 
                         b=b_init, nonlinearity=None, name='r_z_ls')

    z  = GaussianSampleLayer(z_mu, z_ls, rng=rng, name='r_z')

    return {'l_input_x1':l_in_x1, 'l_input_y':l_in_y, 
            'l_z_mu':z_mu, 'l_z_ls':z_ls, 'l_z':z}



   
# Prior network that predicts p(z|x)
def build_prior_net(X1, x1_filts, x1_fsize, p_hid_x1,
                    x1_shape, nonlinearity, regularization, p, 
                    rng=None, W_init=lasagne.init.Orthogonal(),
                    b_init=lasagne.init.Constant(0.), stride=2):


    # ------------ Build the graph for x1  --------------------

    l_in_x1 = nn.InputLayer(input_var=X1, shape=x1_shape, name='r_x1_in')
   

    layer = l_in_x1
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[0], x1_fsize[0], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[1], x1_fsize[1], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 64x64
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[2], x1_fsize[2], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[3], x1_fsize[3], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 32x32
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[4], x1_fsize[4], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[5], x1_fsize[5], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[6], x1_fsize[6], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 16x16
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[7], x1_fsize[7], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[8], x1_fsize[8], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[9], x1_fsize[9], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 8x8
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[7], x1_fsize[7], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[8], x1_fsize[8], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')
    layer = dnn.Conv2DDNNLayer(layer, x1_filts[9], x1_fsize[9], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')

    # 4x4
    layer = dnn.MaxPool2DDNNLayer(layer, pool_size=2, stride=stride)

    layer = dnn.Conv2DDNNLayer(layer, x1_filts[9], x1_fsize[9], W=W_init,
                               b=b_init, nonlinearity=nonlinearity, pad='same')


    layer = nn.DenseLayer(layer, num_units=p_hid_x1[0], W=W_init, b=b_init, 
                          nonlinearity=nonlinearity)
    layer = nn.DropoutLayer(layer, p=0.1, name='drop')
    #layer = weight_norm(layer, name='wn')
    #layer = nn.batch_norm(layer)

    z_mu = nn.DenseLayer(layer, num_units=p_hid_x1[-1], W=W_init, b=b_init, 
                         nonlinearity=None, name='p_z_mu')

    z_ls = nn.DenseLayer(layer, num_units=p_hid_x1[-1], W=W_init, b=b_init, 
                         nonlinearity=None, name='p_z_ls')

    z  = GaussianSampleLayer(z_mu, z_ls, rng=rng, name='p_z')

    return {'l_input_x1':l_in_x1,  'l_z_mu':z_mu, 'l_z_ls':z_ls, 'l_z':z}


# Decoder network, X = latent "codes"
# Last element in g_z_hid should be the y-dimension
def build_generation_net(Z, g_hid_z, nonlinearity, regularization, p, 
                         rng=None, W_init=lasagne.init.Orthogonal(),
                         b_init=lasagne.init.Constant(0.)):

    layer = Z
    for i, n_hid in enumerate(g_hid_z[:-1]):
        
        layer = nn.DenseLayer(
            layer, num_units=n_hid, W=W_init, b=b_init,
            nonlinearity=nonlinearity, name='g_z_hid%d'%i)

    layer = apply_regularization(layer, regularization, p=p)

    l_x = nn.DenseLayer( 
        layer, num_units=g_hid_z[-1], W=W_init, b=b_init,
        nonlinearity=nonlinearity, name='g_x')
        
    return {'l_x':l_x}


# p = dropout prob
def build_cgm_network(X1, Y, x1_filts, x1_fsize,
                      r_hid_x1, r_hid_y, r_hid_shared, 
                      p_hid_x1, d_hid_x1,  g_hid_z, x1_shape,  
                      y_shape, nonlinearity,
                      regularization, p, rng, W_init, **kwargs ):

    # x1_filts, x1_fsize are similar between all networks
    if len(x1_filts) != len(x1_fsize):
        raise Exception('# filts x1 != # convolutions x1')

    # Weight initializations all use the same format
    b_init = lasagne.init.Constant(0.)

    prediction = build_prediction_net(
        X1, x1_filts, x1_fsize, d_hid_x1,   
        x1_shape, nonlinearity, regularization, p, rng, W_init)

    recognition = build_recognition_net(
        X1, Y, x1_filts, x1_fsize, r_hid_x1, r_hid_y, r_hid_shared, 
        x1_shape, y_shape, nonlinearity, regularization, p, rng, W_init)

    generation = build_generation_net(recognition['l_z'], g_hid_z, 
        nonlinearity, regularization, p, rng, W_init)

    prior_net = build_prior_net(X1, x1_filts, x1_fsize, 
        p_hid_x1, x1_shape,  nonlinearity, regularization, p, rng, W_init)

    # Generation and prediction network outputs a prediction of size (Y_dim),
    # and both predictions need to be combined together before sampling out 
    predx = prediction['l_x']
    genx  = generation['l_x']

    layer = nn.ConcatLayer([predx,  genx], axis=1, name='r_concat') 

    initial_z = nn.DenseLayer(
        layer, num_units=d_hid_x1[-1], W=W_init, b=b_init,
        nonlinearity=nonlinearity, name='e_sum_nonlin')

    initial_z = apply_regularization(initial_z, regularization, p=p)

    l_x_mu = nn.DenseLayer(
        initial_z, num_units=d_hid_x1[-1], W=W_init, b=b_init,
        nonlinearity=None, name='GMu')

    l_x_ls = nn.DenseLayer(
        initial_z, num_units=d_hid_x1[-1], W=W_init, b=b_init,
        nonlinearity=None ,name='GLs')

    l_x  = GaussianSampleLayer(l_x_mu, l_x_ls, rng=rng, name='GenX')


    return {'l_recog_x1':recognition['l_input_x1'],
            'l_recog_y':recognition['l_input_y'],
            'l_recog_mu':recognition['l_z_mu'],
            'l_recog_ls':recognition['l_z_ls'],
            'l_recog_z' :recognition['l_z'],
            'l_prior_x1':prior_net['l_input_x1'],
            'l_prior_mu':prior_net['l_z_mu'],
            'l_prior_ls':prior_net['l_z_ls'],
            'l_prior_z':prior_net['l_z'],
            'l_pred_x1':prediction['l_input_x1'],
            'l_pred_x':prediction['l_x'],
            'l_gener_mu':l_x_mu,
            'l_gener_ls':l_x_ls,
            'l_gener_x':l_x}    
            

# p = dropout prob
def build_recurrent_network(X1, Y, x1_filts, x1_fsize,
                      r_hid_x1, r_hid_y, r_hid_shared, p_hid_x1, 
                      d_hid_x1, g_hid_z, x1_shape, y_shape, nonlinearity,
                      regularization, p, rng, W_init, **kwargs ):

    # x1_filts, x1_fsize are similar between all networks
    if len(x1_filts) != len(x1_fsize):
        raise Exception('# filts x1 != # convolutions x1')

    # Weight initializations all use the same format
    b_init = lasagne.init.Constant(0.)

    prediction = build_prediction_net(
        X1, x1_filts, x1_fsize, d_hid_x1,   
        x1_shape, nonlinearity, regularization, p, rng, W_init)

    recognition = build_recognition_net(
        X1, Y, x1_filts, x1_fsize, r_hid_x1, r_hid_y, r_hid_shared, 
        x1_shape, y_shape, nonlinearity, regularization, p, rng, W_init)

    generation = build_generation_net(recognition['l_z'], g_hid_z, 
        nonlinearity, regularization, p, rng, W_init)

    prior_net = build_recognition_net(
        X1, prediction['l_x'], x1_filts, x1_fsize, r_hid_x1, 
        r_hid_y, r_hid_shared, x1_shape, y_shape, 
        nonlinearity, regularization, p, rng, W_init)


    # Generation and prediction network outputs a prediction of size (Y_dim),
    # and both predictions need to be combined together before sampling out 
    predx = prediction['l_x']
    genx  = generation['l_x']

    layer = nn.ConcatLayer([predx,  genx], axis=1, name='r_concat') 

    initial_z = nn.DenseLayer(
        layer, num_units=d_hid_x1[-1], W=W_init, b=b_init,
        nonlinearity=nonlinearity, name='e_sum_nonlin')

    initial_z = apply_regularization(initial_z, regularization, p=p)

    l_x_mu = nn.DenseLayer(
        initial_z, num_units=d_hid_x1[-1], W=W_init, b=b_init,
        nonlinearity=None, name='GMu')

    l_x_ls = nn.DenseLayer(
        initial_z, num_units=d_hid_x1[-1], W=W_init, b=b_init,
        nonlinearity=None ,name='GLs')

    l_x  = GaussianSampleLayer(l_x_mu, l_x_ls, rng=rng, name='GenX')


    return {'l_recog_x1':recognition['l_input_x1'],
            'l_recog_y':recognition['l_input_y'],
            'l_recog_mu':recognition['l_z_mu'],
            'l_recog_ls':recognition['l_z_ls'],
            'l_recog_z' :recognition['l_z'],
            'l_prior_x1':prior_net['l_input_x1'],
            'l_prior_mu':prior_net['l_z_mu'],
            'l_prior_ls':prior_net['l_z_ls'],
            'l_prior_z':prior_net['l_z'],
            'l_pred_x1':prediction['l_input_x1'],
            'l_pred_x':prediction['l_x'],
            'l_gener_mu':l_x_mu,
            'l_gener_ls':l_x_ls,
            'l_gener_x':l_x}   


# p = dropout prob
def build_gsnn_network(X1, Y, x1_filts, x1_fsize,
                      r_hid_x1, r_hid_y, r_hid_shared, 
                      p_hid_x1, d_hid_x1, g_hid_z, x1_shape, 
                      y_shape, nonlinearity,
                      regularization, p, rng, W_init, **kwargs ):

    # x1_filts, x1_fsize are similar between all networks
    if len(x1_filts) != len(x1_fsize):
        raise Exception('# filts x1 != # convolutions x1')

    # Weight initializations all use the same format
    b_init = lasagne.init.Constant(0.)

    prediction = build_prediction_net(
        X1, x1_filts, x1_fsize, d_hid_x1,  
        x1_shape,  nonlinearity, regularization, p, rng, W_init)

    prior_net = build_prior_net(X1, x1_filts, x1_fsize, 
        p_hid_x1, x1_shape, nonlinearity, regularization, p, rng, W_init)

    generation = build_generation_net(prior_net['l_z'], g_hid_z, 
        nonlinearity, regularization, p, rng, W_init)


   # Generation and prediction network outputs a prediction of size (Y_dim),
    # and both predictions need to be combined together before sampling out 
    predx = prediction['l_x']
    genx  = generation['l_x']
    initial_z = nn.ElemwiseSumLayer([predx,genx], name='ElemwiseSum')

    initial_z = nn.DenseLayer(
        initial_z, num_units=d_hid_x1[-1], W=W_init, b=b_init,
        nonlinearity=nonlinearity, name='e_sum_nonlin')

    initial_z = apply_regularization(initial_z, regularization, p=p)

    l_x_mu = nn.DenseLayer(
        initial_z, num_units=d_hid_x1[-1], W=W_init, b=b_init,
        nonlinearity=None, name='GMu')

    l_x_ls = nn.DenseLayer(
        initial_z, num_units=d_hid_x1[-1], W=W_init, b=b_init,
        nonlinearity=None ,name='GLs')

    l_x  = GaussianSampleLayer(l_x_mu, l_x_ls, rng=rng, name='GenX')

    return {'l_prior_x1':prior_net['l_input_x1'],
            'l_prior_mu':prior_net['l_z_mu'],
            'l_prior_ls':prior_net['l_z_ls'],
            'l_prior_z':prior_net['l_z'],
            'l_pred_x1':prediction['l_input_x1'],
            'l_pred_x':prediction['l_x'],
            'l_gener_mu':l_x_mu,
            'l_gener_ls':l_x_ls,
            'l_gener_x':l_x}  


# Regular convolutional net - NO SAMPLING 
def build_pred_network(X1, x1_filts, x1_fsize, d_hid_x1, 
                  x1_shape, nonlinearity, regularization, p, 
                  rng, W_init, **kwargs):

    prediction = build_prediction_net(
        X1, x1_filts, x1_fsize, d_hid_x1,  
        x1_shape,  nonlinearity, regularization, p, rng, W_init)

    return {'l_input_x1':prediction['l_input_x1'], 'l_x':prediction['l_x']}


