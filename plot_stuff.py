import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

from utils import TileRows

"""
Load a pickled data model from the GBRBM
"""
def load_data(filename):
    data = pickle.load(open(filename,'rb'))
    return data

"""
Plot one HE signature with its reconstruction
"""
def plot_signature(data,i):
    truth = data['truth']
    mask = data['truth_mask']
    recon = data['recon']

    m = mask[i,:]
    t = truth[i,:]
    r = recon[i,:]
    
    x = np.arange(truth.shape[1]) 

    fig = plt.figure(figsize=(25,10))
    plt.plot(x,t,'b-',linewidth=3)
    plt.plot(x,r,'r-',linewidth=3)
    for i in xrange(12):
        if np.sum(m[i*202:(i+1)*202]) > 0:
            plt.plot(x[i*202:(i+1)*202],t[i*202:(i+1)*202],'g-',linewidth=3)
    for i in xrange(12):
        plt.axvline(x=(i+1)*202,color='#c6c6c6',linewidth=1)
    plt.legend(['Ground Truth','Reconstruction','Input to GBRBM'],loc='upper right',fontsize=24)
    plt.xlabel('Visible Unit',fontsize=24)
    plt.ylabel('Standardized Intensity',fontsize=24)
    plt.xlim(0,truth.shape[1]-1)
    plt.xticks(np.arange(12)*202,fontsize=16)
    plt.yticks(fontsize=16)

    if not os.path.isdir('figs'):
        os.makedirs('figs')
    plt.savefig('figs/signature.png')

"""
Plot the MSE from different sparsity levels
"""
def plot_mse():
    d9 = load_data('data_0.9.p')
    d7 = load_data('data_0.7.p')
    d5 = load_data('data_0.5.p')
    d3 = load_data('data_0.3.p')

    n = 200
    x = np.arange(n)

    fig = plt.figure(figsize=(9,5))
    plt.plot(x,d9['mse'][:n],'-')
    plt.plot(x,d7['mse'][:n],'-')
    plt.plot(x,d5['mse'][:n],'-')
    plt.plot(x,d3['mse'][:n],'-')
    plt.xlabel('epoch',fontsize=12)
    plt.ylabel('MSE',fontsize=12)
    plt.legend(['10%','30%','50%','70%'],loc='upper right',fontsize=12)
    
    if not os.path.isdir('figs'):
        os.makedirs('figs')
    plt.savefig('figs/mse.png')

"""
Plot the learned variance as a 1D signature
"""
def plot_1d_variance(data):
    var = np.exp(data['vlogvar'])
    x = np.arange(len(var))

    fig = plt.figure(figsize=(25,10))
    plt.plot(x,var,'b-',linewidth=3)
    for i in xrange(12):
        plt.axvline(x=(i+1)*202,color='#c6c6c6',linewidth=1)
    plt.xlabel('Visible Unit',fontsize=24)
    plt.ylabel('Variance',fontsize=24)
    plt.xlim(0,len(var)-1)
    plt.xticks(np.arange(12)*202,fontsize=16)
    plt.yticks(fontsize=16)

    if not os.path.isdir('figs'):
        os.makedirs('figs')
    plt.savefig('figs/1d_variance.png')

"""
Plot the learned variance from a CIFAR model as a single 32x32 image
"""
def plot_cifar_variance(data):
    var = np.exp(data['vlogvar'])
    var = np.tile(var,(4,1))

    ImTiler = TileRows(gray=False, img_shape=(32,32), tile_shape=(1,1))
    ImTiler.imsave(var,'figs/cifar_variance.png')

"""
Plot 36 of the learned filters with the largest L2-norms as 1d signatures
"""
def plot_filters_1d(data):
    # sort learned filters by largest to smallest L2-norm
    W = data['W']
    filters = W.T
    filters_sort = np.argsort(np.linalg.norm(filters,ord=2,axis=1))[::-1]
    filters = filters[filters_sort,:]

    nplots = 36
    assert W.shape[1] >= nplots
    x = np.arange(W.shape[0])

    fig = plt.figure(figsize=(18,14))
    for i in xrange(nplots):
        plt.subplot(nplots**0.5,nplots**0.5,i+1)
        plt.plot(x, filters[i,:],'-')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if not os.path.isdir('figs'):
        os.makedirs('figs')
    plt.savefig('figs/filters_1d.png')

"""
Plot all rows of an ndarray for visualization
"""
def plot_all(X):
    n_samples = X.shape[0]
    x = np.arange(X.shape[-1])

    fig = plt.figure(figsize=(18,14))
    for i in xrange(n_samples):
        plt.plot(x,X[i,:])

    plt.show()
