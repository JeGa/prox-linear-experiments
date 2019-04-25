def proj_max(v, to=1):
    """
    Projection of the given vector onto the unit ball under the max norm.
    Alters the numpy array.

    :param v: shape = (n, 1).

    :return: shape = (n, 1).
    """
    for i in range(v.shape[0]):
        if v[i] >= to:
            v[i] = to
        if v[i] <= -to:
            v[i] = -to

    return v
