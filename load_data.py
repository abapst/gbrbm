import cPickle
import numpy as np
import theano
import theano.tensor as T


""" Subtract mean of all nonzero entries in each row """
def center_data(X,mask=None):
    if mask is None:
        mask = np.ones(X.shape).astype(np.bool)

    for r in xrange(X.shape[0]):
        row = X[r,:]
        X[r,:] = row - np.mean(row[mask[r,:].astype(np.bool)])
    return X

""" Divide by standard deviation of all nonzero entries in each row """
def standardize_data(X,mask=None,sqrt_bias=0.,scale=1.0):
    assert len(X.shape) == 2
    ddof=1
    if X.shape[1] == 1:
        ddof = 0

    if mask is None:
        mask = np.ones(X.shape).astype(np.bool)

    for r in xrange(X.shape[0]):
        row = X[r,:]
        model_std = np.sqrt(sqrt_bias + row[mask[r,:].astype(np.bool)].var(ddof=ddof))/scale
        X[r,:] = row/model_std
    return X

""" Get whitening transform from data """
def get_zca_matrix(X):
        cov = np.cov(X, rowvar=False)   # cov is (N,N)
        U,S,V = np.linalg.svd(cov)		# U is (N,N), S is (N,) 
        eps = 1e-5
        zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + eps)), U.T))
        return zca_matrix

""" Generate mask with randomly degraded values """
def get_mask(X,nchannels,spatial_sparsity=1.0,channel_sparsity=(False,1)):
    if nchannels == 1:
        npix = X.shape[1]
    else:
        npix = X.shape[1]/nchannels

    """ Introduce in-channel sparsity by randomly zeroing out pixels """
    n = int(np.round(npix*spatial_sparsity))
    masks = np.zeros((X.shape[0],npix),dtype=bool)
    masks[:,:n] = True
    for i in range(masks.shape[0]):
        np.random.shuffle(masks[i,:])

    if nchannels > 1:
        masks = np.tile(masks,(1,nchannels))

        """ Add channel sparsity by keeping only nchannels at a time """
        if channel_sparsity is not None:
            assert channel_sparsity <= nchannels
            channels = np.arange(nchannels)
            for i in xrange(masks.shape[0]):
                np.random.shuffle(channels)

                """ mark channels that will be masked """
                rowmask = np.ones((nchannels*npix,)).astype(bool)
                for c in channels[:channel_sparsity]:
                    rowmask[c*npix:c*npix+npix] = False
                masks[i,rowmask] = 0.

    return masks

def load_cifar_data(path="data/cifar-10-batches-py/",
                    cls="all",
                    gray=True,
                    center=False,
                    standardize=False,
                    whiten=False,
                    spatial_sparsity=1.0,
                    channel_sparsity=None,
                    degrade_training=True):
    """
    Params:
        :param path: path to directory containing cifar-10 batches
        :param cls: string, if not "all" then a single class to keep e.g. "car"
        :param gray: bool value on whether to convert images to gray or not
        :param center: bool value on whether to center data rows or not
        :param standardize: bool value on whether to divide rows by std deviation
        :param whiten: bool value on whether to zcf whiten data features
        :param spatial_sparsity: float amount of in-channel data to keep
        :param channel_sparsity: if not None, number of channels to keep
        :param degrade_training: bool value on whether to generate masks for training          data or keep it whole.

    Returns:
        (trainx,trainm,trainy,testx,testm,testy)
        :*x: ndarray, shape = (n_samples,nfeatures)
        :*m: ndarray degradation mask, shape = (n_samples,nfeatures)
        :*y: ndarray labels, shape = (n_samples,)
    """

    labels={"plane":0,
            "car":  1,
            "bird": 2,
            "cat":  3,
            "deer": 4,
            "dog":  5,
            "frog": 6,
            "horse":7,
            "ship": 8,
            "truck":9}

    def convert_to_gray(X):
        dim = X.shape

        assert(len(dim) == 2)
        assert(dim[1] % 3 == 0)

        X = np.split(X,3,axis=1)
        return np.round(np.dstack(X).mean(axis=2))
            
    """ Load training data """
    trainx = []
    trainy = []
    for i in range(5):
        fo = open(path+"data_batch_"+str(i+1),"rb")
        data = cPickle.load(fo)
        trainx.append(data['data'])
        trainy.append(data['labels'])
        fo.close()
    trainy = np.hstack(trainy).astype(np.float32)

    """ Load test data """
    fo = open(path+"test_batch","rb")
    testx = data['data']
    testy = data['labels']
    fo.close()
    testy = np.asarray(testy,dtype=np.float32)

    if gray is True:
        trainx = convert_to_gray(np.vstack(trainx).astype(np.float32))
        testx = convert_to_gray(testx.astype(np.float32))
        nchannels = 1
    else:
        trainx = np.vstack(trainx).astype(np.float32)
        testx = testx.astype(np.float32)
        nchannels = 3

    """ Select individual class for reconstruction """ 
    if cls in labels.keys():
        idx = labels[cls]
        trainx = trainx[trainy == idx]
        trainy = trainy[trainy == idx]
        testx = testx[testy == idx]
        testy = testy[testy == idx]

    """ Get degradation masks """
    if degrade_training is True:
        trainm = get_mask(trainx,nchannels,spatial_sparsity,channel_sparsity)
    else:
        trainm = np.ones(trainx.shape)
    testm = get_mask(testx,nchannels,spatial_sparsity,channel_sparsity)

    if center is True:
        trainx = center_data(trainx,trainm)
        testx = center_data(testx,testm)

    if standardize is True:
        trainx = standardize_data(trainx,trainm)
        testx = standardize_data(testx,testm)

    if whiten is True:
        zca_matrix = get_zca_matrix(trainx)
        trainx = np.dot(trainx, zca_matrix).astype(np.float32)
        testx = np.dot(testx, zca_matrix).astype(np.float32)

    return (trainx,trainm,trainy,testx,testm,testy)

"""
Randomly shift the rows of an ndarray left or right by a set amount
"""
def random_shift_data(X,amount=30,noise=1e-6):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    shift = np.random.randint(low=-amount,high=amount,size=(n_samples,))

    for i in xrange(n_samples):
        row = X[i,:]
        # generate random noise
        row_shifted = np.random.uniform(low=-noise,high=noise,size=(n_features,))
        if shift[i] > 0:
            row_shifted[shift[i]:] = row[:-shift[i]]
            X[i,:] = row_shifted
        elif shift[i] < 0:
            row_shifted[:shift[i]] = row[-shift[i]:]
            X[i,:] = row_shifted
        else:
            continue
    return X
