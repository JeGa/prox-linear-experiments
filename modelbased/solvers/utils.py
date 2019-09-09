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


def ttype(tensor_type):
    if tensor_type == 'numpy':
        F = NumpyFunctions
    elif tensor_type == 'pytorch':
        F = PytorchFunctions
    else:
        raise NameError("Unknown tensor_type.")

    return F.t, F.dot, F.sqrt


class NumpyFunctions:
    @staticmethod
    def t(a):
        return a.T

    @staticmethod
    def dot(a, b):
        return a.dot(b)

    @staticmethod
    def sqrt(a):
        return np.sqrt(a)


class PytorchFunctions:
    @staticmethod
    def t(a):
        return a.t()

    @staticmethod
    def dot(a, b):
        return a.mm(b)

    @staticmethod
    def sqrt(a):
        return a.sqrt()
