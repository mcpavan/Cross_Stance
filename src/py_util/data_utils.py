import numpy as np

def load_vectors(vecfile, dim=300, unk_rand=True, seed=0):
    '''
    Loads saved vectors;
    :param vecfile: the name of the file to load the vectors from.
    :return: a numpy array of all the vectors.
    '''
    vecs = np.load(vecfile)
    np.random.seed(seed)

    if unk_rand:
        vecs = np.vstack((vecs, np.random.randn(dim))) # <unk> -> V-2 ??
    else:
        vecs = np.vstack((vecs, np.zeros(dim))) # <unk> -> V - 2??
    vecs = np.vstack((vecs, np.zeros(dim))) # pad -> V-1 ???
    vecs = vecs.astype(float, copy=False)

    return vecs