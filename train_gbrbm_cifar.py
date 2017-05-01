from __future__ import print_function

import timeit
import numpy as np
import pickle

import theano
import theano.tensor as T
import os
import sys

from utils import TileRows, scale_rows_to_unit_interval
from load_data import center_data, standardize_data
from load_data import load_cifar_data

from gbrbm import GBRBM


"""
Learning rate scheduler
"""
def adjust_lr(lr_spec, epoch):
    if lr_spec['type'] == 'anneal':
        num = lr_spec['init'] * lr_spec['start']
        denom = np.max([lr_spec['start'],lr_spec['slope']*(epoch+1)])
        lr = np.max([lr_spec['floor'], num/denom])
    else:
        lr = np.float32(lr_spec['init'])
    return lr.astype(theano.config.floatX)

"""
Compute MSE between two ndarrays
"""
def compute_MSE(recon, truth):
    recon = scale_rows_to_unit_interval(recon)
    truth = scale_rows_to_unit_interval(truth)
    mse = np.mean(((recon - truth)**2)[:])
    return mse

def test_gbrbm(data,
              lr_spec,
              gray=True,
              momentum=0.0,
              training_epochs=15,
              batch_size=20,
              n_samples=100,
              output_folder='rbm_plots/',
              n_hidden=500,
              ks=1,
              seed=1234,
              model_savename='data.p'):
    """
    Train an GBRBM on the CIFAR-10 dataset

    :param data: tuple of ndarrays. len(data) = 6
        data[0] = training features
        data[1] = missing data mask, data[0].shape == data[1].shape
        data[2] = class labels, not applicable in this case
        data[3] = test features
        data[4] = missing data mask, data[3].shape == data[4].shape
        data[5] = class labels, not applicable in this case
    :param lr_spec: dict containing info about learning rate schedule
    :param gray: bool value on whether to convert CIFAR images to gray
    :param momentum: momentum term for controlling learning rate
    :param training_epochs: number of epochs used for training
    :param batch_size: size of a batch used to train the RBM
    :param n_samples: number of test samples to use for plotting results
    :param output_folder: folder to store filter visualizations and results
    :param n_hidden: number of hidden units
    :param ks: number of Gibbs sampling steps to use for contrastive divergence
    :param seed: random number generator seed for repeatable results
    :param model_savename: name of pickle file to save the learned parameters
    """

    """ Convert to theano shared arrays """
    train_set_x = theano.shared((data[0]*data[1]).astype(theano.config.floatX),borrow=True)
    train_set_m = theano.shared(data[1].astype(theano.config.floatX),borrow=True)
    train_set_y = theano.shared(data[2].astype(theano.config.floatX),borrow=True)
    test_set_x = theano.shared((data[3]*data[4]).astype(theano.config.floatX),borrow=True)
    test_set_m = theano.shared(data[4].astype(theano.config.floatX),borrow=True)
    test_set_y = theano.shared(data[5].astype(theano.config.floatX),borrow=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    learning_rate = T.scalar(dtype=theano.config.floatX)
    x = T.matrix('x')  # the data is presented as batches of rasterized images
    m = T.matrix('m')  # the sparsity mask corresponding to input data
    lr = T.scalar('lr',dtype=theano.config.floatX)

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    gbrbm = GBRBM(input=x, 
                mask=m, 
                learning_rate=lr,
                lr_spec=lr_spec,
                momentum=momentum,
                n_visible=data[0].shape[1],
                n_hidden=n_hidden,
                numpy_rng=np.random.RandomState(seed))

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = gbrbm.get_cost_updates(persistent=persistent_chain,
                                         k=ks)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    ImTiler = TileRows(gray=gray, img_shape = (32,32), tile_shape = (10,10))
    ImTiler.imsave(data[0],output_folder+"example_train_data.png",data[1])
    ImTiler.imsave(data[3],output_folder+"example_test_data.png",data[4])

    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index,learning_rate],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            m: train_set_m[index * batch_size: (index + 1) * batch_size],
            lr: learning_rate
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    mse = []
    for ep in range(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in range(n_train_batches):
            sys.stdout.write("\r{:.2f}%".format(100*(float(batch_index+1))/float(n_train_batches)))
            sys.stdout.flush()
            mean_cost += [train_rbm(batch_index,adjust_lr(lr_spec,ep))]

        presig_v, vis_mf, vis_sample = gbrbm.get_reconstructions(test_set_x)
        mse.append(compute_MSE(vis_mf.copy(),data[3][:100]))

        sys.stdout.write("\r")
        print("Training epoch {}/{}, cost is {}, lr is {}".format(ep+1, training_epochs, np.mean(mean_cost), adjust_lr(lr_spec,ep)))

        # sort filters by largest to smallest L2-norm
        filters = gbrbm.W.get_value(borrow=True).T
        filters_sort = np.argsort(np.linalg.norm(filters,ord=2,axis=1))[::-1]
        filters = filters[filters_sort,:]

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        ImTiler.imsave(filters,output_folder+"filters_at_epoch_%i.png" % ep)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    #################################
    #     Sampling from the RBM     #
    #################################
    presig_v, vis_mf, vis_sample = gbrbm.get_reconstructions(test_set_x)

    print('Plotting reconstructions')
    recon = vis_mf.copy()
    recon = scale_rows_to_unit_interval(recon)
    ImTiler.imsave(recon,output_folder+'recon.png')

    print("Plotting associated ground truth")
    truth = data[3][:100]
    truth = scale_rows_to_unit_interval(truth)
    truth_mask = data[4][:100]
    truth_degraded = truth*truth_mask
    ImTiler.imsave(truth,output_folder+'recon_truth.png')

    print("Plotting degraded ground truth")
    ImTiler.imsave(truth,output_folder+'recon_truth_degraded.png',mask=truth_mask)

    print("Plotting combined ground truth and reconstructions")
    combined = truth.copy()
    combined[~truth_mask.astype(bool)] = recon[~truth_mask.astype(bool)]
    ImTiler.imsave(combined,output_folder+'recon_truth_combined.png')

    recon_error = np.mean(((recon - truth)**2)[:])
    print("Reconstruction MSE = {}".format(recon_error))

    data = {'truth':truth,
            'truth_mask':truth_mask,
            'recon':recon,
            'mse':mse,
            'W':gbrbm.W.get_value(borrow=True),
            'hbias':gbrbm.hbias.get_value(borrow=True),
            'vbias':gbrbm.vbias.get_value(borrow=True),
            'vlogvar':gbrbm.vlogvar.get_value(borrow=True)}

    if not os.path.isdir('models'):
        os.makedirs('models')
    pickle.dump(data,open("models/"+model_savename,"wb"))

if __name__ == '__main__':
    data = load_cifar_data(cls="car",
                           center=True,
                           standardize=True,
                           whiten=False,
                           gray=False,
                           channel_sparsity=None,
                           spatial_sparsity=0.9,
                           degrade_training = True)

    """ 'constant' or 'anneal' """
    lr_spec = {'type':'anneal',
               'start':1.0,
               'slope':0.05,
               'floor':0.0,
               'init':1e-3}

    test_gbrbm(data,
              lr_spec,
              gray=False,
              momentum=0.0,
              training_epochs=100,
              batch_size=20,
              n_hidden=1024,
              n_samples=100,
              ks=1,
              seed=1234,
              model_savename="data_cifar.p")
