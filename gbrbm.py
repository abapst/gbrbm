"""
This is an implementation of a Gaussian Restricted Boltzmann Machine with
Gaussian visible units and Bernoulli (binomial) hidden units. We learn both
the mean and variance of the Gaussian distribution for each visible unit
in order to model real-valued data.

This code was modified from the binary RBM tutorial found here:

http://deeplearning.net/tutorial/rbm.html
"""
import numpy as np
import theano
import theano.tensor as T
import os
import sys

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class GBRBM(object):
    """
       Restricted Boltzmann Machine (RBM) with Gaussian visible units
       and Bernoulli hidden units
    """
    def __init__(
        self,
        input=None,
        mask=None,
        learning_rate=None,
        lr_spec=None,
        momentum=0.0,
        n_visible=1024,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        vlogvar=None,
        numpy_rng=None,
        theano_rng=None
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :lr_spec: a dict of configuration parameters for the learning rate
         schedule. TODO: adapative learning rate not yet implemented.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias

        :param vlogvar: None for standalone RBMs or a symbolic variable
        pointing to a shared log-variance for Gaussian visible units
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.momentum = momentum

        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized from uniform distribution and 
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-1. / (n_hidden + n_visible),
                    high=1. / (n_hidden + n_visible),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
        else:
            W = theano.shared(
                 value=W.astype(theano.config.floatX),
                 name='W',
                 borrow=True)
     
        if hbias is None:
            hbias = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )
        else:
            hbias = theano.shared(value=hbias.astype(theano.config.floatX), name='hbias', borrow=True)

        if vbias is None:
            vbias = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )
        else:
            vbias = theano.shared(value=vbias.astype(theano.config.floatX), name='vbias', borrow=True)

        if vlogvar is None:
            # create shared variable for log-variance of visible units
            vlogvar = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vlogvar',
                borrow=True
            )
        else:
            vlogvar = theano.shared(value=vlogvar.astype(theano.config.floatX), name='vlogvar', borrow=True)

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.vlogvar = vlogvar

        prev_gw = np.zeros(shape=(n_visible, n_hidden), dtype=np.float32)
        self.prev_gw = theano.shared(value=prev_gw, borrow=True, name='g_w')

        prev_gh = np.zeros(n_hidden, dtype=np.float32)
        self.prev_gh = theano.shared(value=prev_gh, borrow=True, name='g_h')

        prev_gv = np.zeros(n_visible, dtype=np.float32)
        self.prev_gv = theano.shared(value=prev_gv, borrow=True, name='g_v')

        prev_gvlogvar = np.zeros(n_visible, dtype=np.float32)
        self.prev_gvlogvar = theano.shared(value=prev_gvlogvar, borrow=True, name='g_vv')

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')
        self.mask = mask
        if not mask:
            self.mask = T.matrix('mask')
        self.lr = learning_rate
        if not learning_rate:
            self.lr = T.scalar('lr',theano.config.floatX)

        self.theano_rng = theano_rng

        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias, self.vlogvar]
        self.prev_gparams = [self.prev_gw, self.prev_gh, self.prev_gv, self.prev_gvlogvar]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        h_input = T.dot(v_sample/T.exp(self.vlogvar), self.W) + self.hbias
        fe = -T.sum(0.5 * ((v_sample-self.vbias)**2)/T.exp(self.vlogvar), axis=1)
        fe -= T.sum(T.nnet.softplus(h_input), axis=1)
        return fe

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot((vis/T.exp(self.vlogvar)), self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = pre_sigmoid_v1.copy()
        """
        v1_sample = self.theano_rng.normal(size=v1_mean.shape,
                                           avg=pre_sigmoid_v1,
                                           std=T.exp(self.vlogvar),
                                           dtype=theano.config.floatX)
        """
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(self.mask*chain_end))

        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        # constructs the update dictionary
        for prev_gparam, gparam, param in zip(self.prev_gparams, gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - \
                             gparam * T.cast(self.lr,dtype=theano.config.floatX) - \
                             prev_gparam * T.cast(self.momentum,dtype=theano.config.floatX) 
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.
        """

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy

    """
    Test the GBRBM model on a set of test examples by reconstructing
    them over a single Gibbs Visible->Hidden->Visible pass.
    :param test_set_x: a theano shared array of test features, with missing
                       data zero'd out if applicable
    :param nsamples: number of samples to reconstruct
    :param nsteps: number of Gibbs VHV passes to run (should just use 1 for
                   normal reconstruction)
    Returns:
        presig_v: reconstructed sample activations before sigmoid is applied
        vis_mf: mean-field sample activations after sigmoid is applied 
                (these are the predicted reconstructions)
        vis_sample: for a GBRBM, these are the same as presig_v.
    """
    def get_reconstructions(self,test_set_x,nsamples=100,nsteps=1):
        nsamples = np.min((test_set_x.get_value(borrow=True).shape[0],
                           nsamples)) 

        persistent_vis_chain = theano.shared(
            np.asarray(
                test_set_x.get_value(borrow=True)[:nsamples],
                dtype=theano.config.floatX
            )
        )

        (
            [
                presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_vhv,
            outputs_info=[None, None, None, None, None, persistent_vis_chain],
            n_steps=nsteps,
            name="gibbs_vhv"
        )

        updates.update({persistent_vis_chain: vis_samples[-1]})
        sample_fn = theano.function(
            [],
            [
                presig_vis[-1],
                vis_mfs[-1],
                vis_samples[-1]
            ],
            updates=updates,
            name='sample_fn'
        )

        presig_v, vis_mf, vis_sample = sample_fn()

        return (presig_v, vis_mf, vis_sample)
