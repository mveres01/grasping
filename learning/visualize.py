import os
import lasagne
import numpy as np
import pandas as pd
import csv


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.collections import PolyCollection

from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# Import seaborn for nice plotting
import seaborn as sns

# We import sklearn.
import sklearn
from sklearn import mixture
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


# Plots a hitogram of weights and biases for each layer in the network
def visualize_weight_histograms(network, output_layers=None, save_name=None):

    plt.close('all')

    if output_layers == None:
        output_layers = network['l_x']

    # Collect all layers in our models
    all_layers = lasagne.layers.get_all_layers(output_layers)
    layers = [l for l in all_layers if hasattr(l,'W') or hasattr(l,'b')]
    
    n_layers = len(layers)
    fig = plt.figure(figsize=(20,10*len(layers)))
      
    # For each layer, try to plot histogram of weights and biases 
    count = 1
    for i, layer in enumerate(layers):
        try: # Plot weights
            ax = fig.add_subplot(n_layers,2,count)
            vals = layer.W.get_value().flatten(2)
            
            norm = np.sqrt(np.sum(vals**2))
            ax.hist(vals)
            ax.set_title(layer.name + ' W (norm: %s)'%norm)
        except Exception: pass
        try: #Plot biases
            ax = fig.add_subplot(n_layers,2,count+1)
            vals = layer.b.get_value().flatten(2)

            norm = np.sqrt(np.sum(vals**2))
            ax.hist(vals)
            ax.set_title(layer.name+ ' b (norm: %s)'%norm)
        except Exception: pass
        count+=2
   
    # (Optional) choose to save the figure 
    if save_name is not None:
        plt.savefig(save_name)


# Plots the activation of mu and log(sigma) layers
# Vals = [ (mu_1, logsigma_1) ... (mu_n, logsigma_n)]
def visualize_mu_logsigma_output(vals, save_name=None):

    plt.close('all')
    n_layers = len(vals)
    fig = plt.figure(figsize=(20,10*n_layers))
         
    count = 1 
    for i, layer in enumerate(vals): 
        mu, ls, title = layer

        ax = fig.add_subplot(n_layers,2,count, projection='3d')
        ax = visualize_waterfall(mu, ax=ax)
        ax.set_title(title+' mu value')
        
        ax = fig.add_subplot(n_layers,2,count+1, projection='3d')
        ax = visualize_waterfall(ls, ax=ax)
        ax.set_title(title + ' ls value')
        count +=2 

    fig.tight_layout()

    # (Optional) choose to save the figure 
    if save_name is not None:
        plt.savefig(save_name)


# Plots the activation of mu and log(sigma) layers
# Vals = [ (mu_1, logsigma_1) ... (mu_n, logsigma_n)]
def visualize_waterfall(vals, save_name=None, ax=None):

    plt.rc('font',family='serif')

    n_bins = 100
    range_min = np.min(vals)
    range_max = np.max(vals)
    hist_range = (range_min, range_max)
    n_var = vals.shape[1]

    sns.set_palette('muted')
    palette = np.array(sns.color_palette("hls", 5))

    # Create a figure and get current axis
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    # Cycle through the list of colours, and take the first element
    colors = [np.roll(palette, i, 0)[0] for i in xrange(n_var)]

    # For each variable, bin the input and record (bin, count)
    maxval = 0
    hists = []
    for row in vals.T:
        xy = np.histogram(row, bins=n_bins, range=hist_range)
        x = np.atleast_2d(xy[0]).T
        y = np.atleast_2d(xy[1]).T

        maxval = np.maximum(maxval, np.max(x))
        hists.append(np.squeeze(np.asarray(zip(y,x))))

    # To make a waterfall graph, need to interface with PolyCollection
    poly = PolyCollection(hists, facecolors=colors)
    poly.set_alpha(0.7)

    # Assigns an index to each histogram
    zs = list(np.arange(vals.shape[1]))
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_ylim([-1, vals.shape[1]])
    ax.set_xlim([range_min+0.1*range_min, range_max+0.1*range_max])
    ax.set_zlim([0, maxval])
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])
    #ax.set_xlabel('X')
    #ax.set_ylabel('Variable')
    #ax.set_zlabel('Histogram Count')

    # (Optional) choose to save the figure 
    if save_name is not None:
        plt.savefig(save_name, transparent=True)

    return ax

# Displays/Plots grasp configurations
def display_grasps(data, saveName):

    plt.close("all")

    df = data
    np.random.shuffle(df)
    n_samples, n_var = df.shape

    # so we don't plot too much
    n = slice(0,int(df.shape[0]*0.75))
    zeros = np.zeros((n_samples,1))

    # Create figure and set ranges
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylim([-0.25,0.25])
    ax.set_xlim([-0.25,0.25])
    ax.set_zlim([-0.25,0.25])

    # Gaussian KDE plots frequency of position using colour, but
    for i in range(0,9,3):
        ax.scatter(df[n,i], df[n,i+1], df[n,i+2], marker='o')
   
    if saveName is not None: 
        plt.savefig(saveName)  


# Displays/Plots grasp configurations
def overlay_grasps(data_test, data_recon, saveName=None):

    plt.close("all")

    df_1 = data_test
    df_2 = data_recon

    # Create figure and set ranges
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylim([-0.10,0.10])
    ax.set_xlim([-0.10,0.10])
    ax.set_zlim([-0.10,0.10])

    # Plot first finger
    for i in range(0,9,3):
    
        # Plot position (scatter) and direction (quiver) of normal    
        ax.scatter(df_1[i], df_1[i+1], df_1[i+2], c='red', marker='o')
        ax.quiver( df_1[i], df_1[i+1], df_1[i+2], 
                   df_1[i+9], df_1[i+10],df_1[i+11], length=0.02)

        ax.scatter(df_2[i], df_2[i+1], df_2[i+2], c='blue', marker='o')
        ax.quiver( df_2[i], df_2[i+1], df_2[i+2], 
                   df_2[i+9], df_2[i+10],df_2[i+11], length=0.02)

    if saveName is not None: 
        plt.savefig(saveName)

# Plots the training objective 
# NOTE:  Lrate_array is an n_epochsx2 array
def plot_accuracy(lrate_array, epoch, save_dir):

    plt.close('all')

    timesteps = np.arange(epoch)
    plt.plot(timesteps, lrate_array[:epoch, 0], color='blue',
             linestyle='-', label = 'Train LB')
    plt.plot(timesteps, lrate_array[:epoch, 1], color='green',
             linestyle='-', label = 'Valid LB')
    
    plt.legend(loc='upper right')
    plt.savefig(save_dir+'/loss_curve.png')

# Tiles images into an (n_row x n_column) image
def tile_images(inputs, rows, cols, save_path):

    plt.close('all')

    height = inputs.shape[2]
    width = inputs.shape[3]
    count = 0

    im = Image.new('L',(height*rows, width*cols))
    for (x,y), val in np.ndenumerate(np.zeros((cols, rows))):
        img = inputs[count].reshape(width, height)*255.0
        im.paste(Image.fromarray(img),(y*width, x*height))
        count +=1

    im.save(save_path)

# Plots test set, train set nearest images using method above
def tile_closest_images(teX, trX, nToPlot, save_path):

    plt.close('all')

    size = int(np.sqrt(teX.shape[-1]))

    if teX.ndim == 2:
        teX = teX.reshape(teX.shape[0], -1, size,size)
    if trX.ndim == 3:
        trX = trX.reshape(trX.shape[0], trX.shape[1], size,size)
    n_samples, n_closest, height, width = trX.shape

    # width, height
    im = Image.new('L',(width*(n_closest+1), height*nToPlot))
   
    # NOTE: Plot 5 closest samples 
    for i in xrange(nToPlot):
        img = teX[i,0]*255.0
        im.paste(Image.fromarray(img),(0, i*height, width, (i+1)*height))

        for j in xrange(n_closest):
            img = trX[i,j]*255.0
            im.paste(Image.fromarray(img),((j+1)*width, i*height,
                 (j+2)*width, (i+1)*height))

    im.save(save_path)


# Displays/Plots grasp configurations
# NOTE: PIL does this differently then matplotlib
def tile_grasps(teX, rows, cols, save_path):

    plt.close("all")

    # Import seaborn for nice plotting
    import seaborn as sns

    # We import seaborn to make nice plots.
    sns.set_style('darkgrid')
    sns.set_palette('muted')

    fig = plt.figure(figsize=(15*rows,5*cols))

    count = 0
    for i in xrange(rows*cols):

        count += 1
        ax = fig.add_subplot(rows, cols, count, projection='3d')
        ax.autoscale(False)

        for j in range(0,9,3):
            ax.scatter(teX[i,j], teX[i,j+1], teX[i,j+2], marker='o')
            ax.quiver( teX[i,j+0], teX[i,j+1], teX[i,j+2], 
                       teX[i,j+9], teX[i,j+10],teX[i,j+11], length=0.02)


        ax.set_xlim([-0.10,0.10])
        ax.set_ylim([-0.10,0.10])
        ax.set_zlim([-0.10,0.10])
        ax.set_xlabel('X', linespacing=6.)
        ax.set_ylabel('Y', linespacing=6.)
        ax.set_zlabel('Z', linespacing=6.)

        #ax.dist = 50

    plt.tight_layout() 
    plt.savefig(save_path, dpi=120)


# Displays/Plots grasp configurations
def tile_closest_grasps(teX, trX, nToPlot, save_path):

    plt.close("all")
    n_samples, n_closest, datasize = trX.shape

    # Create figure and set ranges
    fig = plt.figure(figsize=(3*(n_closest+1),3*nToPlot))

    count = 0
   
    # Plot the "Test set" column, then "N" closest train set samples
    for i in xrange(nToPlot):

        count += 1
        ax = fig.add_subplot(nToPlot,n_closest+1,count, projection='3d')
        ax.set_ylim([-0.10,0.10])
        ax.set_xlim([-0.10,0.10])
        ax.set_zlim([-0.10,0.10])

        # Plot first finger
        mList = ['o','^','*']
        for j in range(0,9,3):
            m = mList[int(j/3)]
            ax.scatter(teX[i,j], teX[i,j+1], teX[i,j+2], marker=m)
            ax.quiver(teX[i,j], teX[i,j+1], teX[i,j+2], 
                      teX[i,j+9], teX[i,j+10],teX[i,j+11], length=0.02)
    
        for j in xrange(n_closest):

            count += 1
            ax = fig.add_subplot(nToPlot,n_closest+1,count,projection='3d')
            ax.set_ylim([-0.10,0.10])
            ax.set_xlim([-0.10,0.10])
            ax.set_zlim([-0.10,0.10])

            ax.scatter(trX[i,j,0], trX[i,j,1], trX[i,j,2], 
                        c='blue', marker='o')
            ax.scatter(trX[i,j,3], trX[i,j,4], trX[i,j,5], 
                        c='blue', marker='^')
            ax.scatter(trX[i,j,6], trX[i,j,7], trX[i,j,8], 
                        c='blue', marker='*')

            ax.quiver( trX[i,j,0], trX[i,j,1], trX[i,j,2], 
                       trX[i,j,9], trX[i,j,10],trX[i,j,11], length=0.02)
            ax.quiver( trX[i,j,3], trX[i,j,4], trX[i,j,5], 
                       trX[i,j,12],trX[i,j,13],trX[i,j,14], length=0.02)
            ax.quiver( trX[i,j,6], trX[i,j,7], trX[i,j,8], 
                       trX[i,j,15],trX[i,j,16],trX[i,j,17], length=0.02)

    plt.tight_layout()
    plt.savefig(save_path)


# https://github.com/oreillymedia/t-SNE-tutorial
def plot_tsne_generated_grasps(X, label, prefix, scaler=None, state=1234):
    
    # Set the colours
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    palette = np.array(sns.color_palette("hls", 10))

    # Make a TSNE plot
    proj = TSNE(random_state=state, learning_rate=100, perplexity=15)
    proj = proj.fit_transform(X)

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax = sns.jointplot(proj[:,0], proj[:,1], cmap='Blues', shade=True)
    ax.plot_joint(plt.scatter, s=1)

    plt.savefig(prefix+'_tsne.png', dpi=120)
    plt.close('all')


    if scaler is not None:
        X = scaler.inverse_transform(X)

    # Plot a scatterplot of generated fingertip positions
    plt.close('all')
    f = plt.figure(figsize=(8, 8))
    ax = f.add_subplot(1,1,1, projection='3d')

    # Give each finger a different colour
    mList = ['r','g','b']
    for j, colour in enumerate(mList):
        ax.scatter(X[:, j*3], X[:, j*3+1], X[:, j*3+2], c=colour, s=2)

    ax.axis('tight')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.savefig(prefix+'_scatterplot.png', dpi=120)


# https://github.com/oreillymedia/t-SNE-tutorial
def plot_tsne_generated_grasps_gmm(X, label, prefix, scaler=None, state=1234):

    # Set the colours
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    palette = np.array(sns.color_palette("hls", 10))

    # Fit a 2-element GMM
    np.random.seed(state)
    g = mixture.GMM(n_components=2)
    g.fit(X)

    # Apply t-SNE to visualize our data distribution
    tsne = TSNE(random_state=state, learning_rate=100, perplexity=15)
    proj = tsne.fit_transform(X)

    # Colour-code the samples as to which cluster they belond to
    n_nbrs = 5
    cluster_centers = g.means_
    colours = ['r' if i==0 else 'g' for i in g.predict(X)]

    f = plt.figure(figsize=(4, 4))
    ax = plt.subplot(aspect='equal')
    ax = sns.jointplot(proj[:,0], proj[:,1], cmap='Blues')
    ax.plot_joint(plt.scatter, s=1)
    plt.savefig(prefix+'_tsne-gmm-coded.pdf', dpi=120)
    plt.close('all')


def visualize_images(image, save_dir):

    height = images.shape[2]
    width = images.shape[3]
    
    im = Image.new('RGB',(height, width))
    colour_image = inverse_transform_images(images[r:r+1])
    colour_image = np.squeeze(colour_image[:,1:4]*255.0).transpose(1,2,0)
    colour_image = colour_image.astype(np.uint8)

    img = Image.fromarray(colour_image)
    im.paste(img)
    im.save(save_dir)



