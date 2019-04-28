import numpy as np


def proj_max(v, to=1):
    """
    Projection of the given vector onto the unit ball under the max norm.
    Alters the numpy array.

    :param v: shape = (n, 1).
    :param to: Scalar, target value of the box constraint [-to, to].

    :return: shape = (n, 1).
    """
    for i in range(v.shape[0]):
        if v[i] >= to:
            v[i] = to
        if v[i] <= -to:
            v[i] = -to

    return v


def torch_proj(v, left, right):
    """
    Project v to [left, right].

    :param v: shape = (n, 1).
    :param left: Scalar.
    :param right: Scalar.

    :return: shape = (n, 1).
    """
    if left > right:
        raise ValueError("Left needs to be <= right.")

    for i in range(v.size(0)):
        if v[i] >= right:
            v[i] = right
        if v[i] <= left:
            v[i] = left

    return v


def t_torch(a):
    return a.t()


def t_numpy(a):
    return a.T


def dot_torch(a, b):
    return a.mm(b)


def dot_numpy(a, b):
    return a.dot(b)


def sqrt_torch(a):
    return a.sqrt()


def sqrt_numpy(a):
    return np.sqrt(a)


def ttype(tensor_type):
    if tensor_type == 'numpy':
        return t_numpy, dot_numpy, sqrt_numpy
    elif tensor_type == 'pytorch':
        return t_torch, dot_torch, sqrt_torch
