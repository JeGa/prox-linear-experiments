def proj_max(v):
    """
    Projection of the given vector onto the unit ball under the max norm.
    Alters the numpy array.

    :param v: shape = (n, 1).

    :return: shape = (n, 1).
    """
    for i in range(v.shape[0]):
        if v[i] >= 1:
            v[i] = 1
        if v[i] <= -1:
            v[i] = -1

    return v
