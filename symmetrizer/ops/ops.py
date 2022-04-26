import numpy as np
import torch
import torch.nn.functional as F


def symmetrize_invariant_out(W, group):
    """
    Create invariant weight matrix
    """
    Wsym = 0
    for parameter in group.parameters:
        Wsym += group._output_transformation(W, parameter)
    return Wsym


def symmetrize(W, group):
    """
    Create equivariant weight matrix
    """
    Wsym = 0
    for parameter in group.parameters:
        input_trans = group._input_transformation(W, parameter)
        Wsym += group._output_transformation(input_trans, parameter)
    return Wsym


def get_basis(size, group, new_size, space="equivariant"):
    """
    Get basis using symmetrization algorithm
    """
    w = np.random.randn(*size)

    # If space = random, do not symmetrize (only project)
    if group is not None and space != "random":
        w = symmetrize(w, group)

    # Vectorize W
    wvec = np.reshape(w, [w.shape[0], -1])

    # Get basis
    __, s, vh = np.linalg.svd(wvec)
    rank = np.linalg.matrix_rank(wvec)

    if space=="nullspace":
        # If nullspace, remove the first r=rank vectors
        rnk = vh.shape[0] - rank
        vh = vh[rank:]
        rank = rnk

    indices = [-1] + new_size

    # Unvectorize W
    w = np.reshape(vh[:rank, ...], indices)
    basis = torch.tensor(w.astype(np.float32), requires_grad=False)
    return basis, rank


def get_invariant_basis(size, group, new_size, space="equivariant"):
    """
    Get invariant basis using invariant version of symmetrization algorithm
    """
    w = np.random.randn(*size)

    # If space = random, do not symmetrize (only project)
    if group is not None and space != "random":
        w = symmetrize_invariant_out(w, group)

    # Vectorize W
    wvec = np.reshape(w, [w.shape[0], -1])

    # Get basis
    __, s, vh = np.linalg.svd(wvec)
    rank = np.linalg.matrix_rank(wvec)

    if space=="nullspace":
        # If nullspace, remove the first r=rank vectors
        rnk = vh.shape[0] - rank
        vh = vh[rank:]
        rank = rnk

    indices = [-1] + new_size

    # Unvectorize W
    w = np.reshape(vh[:rank, ...], indices)
    basis = torch.tensor(w.astype(np.float32), requires_grad=False)
    return basis, rank


def get_coeffs(size, gain):
    """
    Initialize basis coefficients (the trainable parameters)
    """
    coeffs = torch.randn(*size)
    coeffs *= gain
    coeffs = torch.nn.Parameter(coeffs, requires_grad=True)
    return coeffs


def compute_gain(gain_type, rank, channels_in, channels_out, gr_in, gr_out):
    """
    Compute gain depending on initialization method
    """
    if gain_type == "xavier":
        gain = np.sqrt(2./(float(rank * channels_out * gr_out)))
    elif gain_type == "he":
        gain = np.sqrt(2./(float(rank * channels_in * gr_in)))
    return gain


def c2g(tensor, g, invert=False):
    """
    Reshape from groups*channels to groups, channels
    """
    if g is None:
        return tensor
    if len(tensor.shape) == 4:
        b, gc, h, w = tensor.shape
        if not invert:
            tensor = tensor.reshape(b, gc//g, g, h, w)
        elif invert:
            tensor = tensor.reshape(b, g, gc//g, h, w)
    elif len(tensor.shape) == 3:
        b, gc, h = tensor.shape
        if not invert:
            tensor = tensor.reshape(b, gc//g, g, h)
        elif invert:
            tensor = tensor.reshape(b, g, gc//g, h)
    return tensor


def g2c(tensor):
    """
    Reshape from groups, channels to groups*channels
    """
    if len(tensor.shape) == 5:
        b, c, g, h, w = tensor.shape
        tensor = tensor.reshape(b, c*g, h, w)
    elif len(tensor.shape) == 3:
        b, c, g = tensor.shape
        tensor = tensor.reshape(b, c*g)
    return tensor


class GroupRepresentations:
    """
    Class to hold representations
    """
    def __init__(self, trans_set, name):
        """
        """
        self.representations = trans_set
        self.name = name

    def __len__(self):
        """
        """
        return len(self.representations)

    def __getitem__(self, idx):
        """
        """
        return self.representations[idx]


def direct_sum_group(rep_list):
    """
    Return the direct sum of each group representation in rep_list
    """
    dims = [0, 0]
    for rep in rep_list:
        group_order = len(rep)
        rep_dim = rep[0].shape
        dims[0] += rep_dim[0]
        dims[1] += rep_dim[1]
    direct_sum_list = []
    for g in range(group_order):
        direct_sum = np.zeros(dims)
        i = j = 0
        for rep in rep_list:
            rep_g = rep[g]
            x, y = rep_g.shape
            direct_sum[i:i+x, j:j+y] = rep_g
            i += x
            j += y
        direct_sum_list.append(torch.FloatTensor(direct_sum))
    return GroupRepresentations(direct_sum_list, "DirectSumGroupRepr")


def direct_sum_features(feature_list):
    """
    Return the direct sum of each feature in feature list
    """
    return torch.cat(feature_list, 1)
