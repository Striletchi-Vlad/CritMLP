import numpy as np


def nu(gamma, af, neg_slope=None, pos_slope=None):
    """
    Compute the parameter "nu" required for criticality calculations.
    gamma: weight of the residual connections
    af: activation function
    neg_slope: slope of the negative part of the activation function
    pos_slope: slope of the positive part of the activation function
    """

    if af != 'relu-like' and (neg_slope is not None or pos_slope is not None):
        raise ValueError("neg_slope and pos_slope only valid for relu-like")

    # K* = 0 universality class
    if af in ['tanh', 'sin']:
        return 2/3 * (1 - gamma**4)

    # Scale invariant universality class
    if af == 'relu':
        neg_slope = 0
        pos_slope = 1

    if af == 'linear':
        neg_slope = 1
        pos_slope = 1

    # Relu-like (still scale invariant)
    if af == "relu-like":
        if neg_slope is None:
            neg_slope = 0
        if pos_slope is None:
            pos_slope = 1

    # Half-stable universality class
    # TODO implement half-stable universality class
    if af == 'gelu':
        return nu(gamma, 'relu')

    A2 = (pos_slope**2 + neg_slope**2) / 2
    A4 = (pos_slope**4 + neg_slope**4) / 2
    g1 = 1 - gamma**2
    g2 = 4 * gamma**2
    return g1 * (g1 * (3*A4 / A2**2 - 1) + g2)


def gamma_from_nu(nu, af, neg_slope=None, pos_slope=None):
    """
    Reverse engineer the gamma parameter, governing the residual
    connections, given nu.
    af: activation function
    neg_slope: slope of the negative part of the activation function
    pos_slope: slope of the positive part of the activation function
    """

    if af != 'relu-like' and (neg_slope is not None or pos_slope is not None):
        raise ValueError("neg_slope and pos_slope only valid for relu-like")

    # K* = 0 universality class
    if af in ['tanh', 'sin']:
        return [np.sqrt(np.sqrt(1 - 3*nu/2))]

    # Half-stable universality class
    # TODO implement half-stable universality class
    if af == 'gelu':
        return gamma_from_nu(nu, 'relu')

    # Scale invariant universality class
    if af == 'relu':
        neg_slope = 0
        pos_slope = 1

    if af == 'linear':
        neg_slope = 1
        pos_slope = 1

    # Relu-like (still scale invariant)
    if af == "relu-like":
        if neg_slope is None:
            neg_slope = 0
        if pos_slope is None:
            pos_slope = 1

    A2 = (pos_slope**2 + neg_slope**2) / 2
    A4 = (pos_slope**4 + neg_slope**4) / 2

    const1 = 3*A4 / A2**2 - 1
    a = const1 - 4
    b = 4 - 2*const1
    c = const1 - nu
    d = b**2 - 4*a*c
    x1 = (-b + np.sqrt(d)) / (2*a)
    x2 = (-b - np.sqrt(d)) / (2*a)
    sols = []
    if x1 >= 0 and x1 <= 1:
        sols.append(x1)
    if x2 >= 0 and x2 <= 1:
        sols.append(x2)

    return sols


def optimal_aspect_ratio(gamma, nL, af):
    """
    Return the optimal aspect ratio for a given gamma, output
    width, and activation function
    gamma: weight of the residual connections
    nL: width at the end of the network
    af: activation function
    """
    return (4 / (20 + 3*nL)) * 1/nu(gamma, af)


def gamma_from_ratio(ratio, nL, af):
    """
    Reverse engineer the gamma parameter, governing the residual
    connections, given the optimal aspect ratio.
    ratio: optimal aspect ratio
    nL: width at the end of the network
    af: activation function
    """
    c = 4 / (20 + 3*nL)
    res = gamma_from_nu(c/ratio, af)
    if len(res) == 0:
        return 0
    return res[0]


def distribution_hyperparameters(gamma, af, neg_slope=None, pos_slope=None):
    """
    Calculate the critical distribution hyperparameters (Cb, CW)
    gamma: weight of the residual connections
    af: activation function
    neg_slope: slope of the negative part of the activation function
    pos_slope: slope of the positive part of the activation function
    """

    if af != 'relu-like' and (neg_slope is not None or pos_slope is not None):
        raise ValueError("neg_slope and pos_slope only valid for relu-like")

    Cb = 0
    g1 = 1 - gamma**2
    # Scale invariant universality class
    if af in ['relu', 'linear', 'relu-like']:
        if af == 'relu':
            A2 = 1/2
        if af == 'linear':
            A2 = 1
        if af == 'relu-like':
            if neg_slope is None:
                neg_slope = 0
            if pos_slope is None:
                pos_slope = 1

            A2 = (pos_slope**2 + neg_slope**2) / 2
        CW = 1 / A2 * g1
    # K* = 0 universality class
    if af in ['tanh', 'sin']:
        CW = g1
    # Half-stable universality class
    if af == 'gelu':
        Cb = 0.17292239
        CW = 1.98305826
    if af == 'swish':
        Cb = 0.055514317
        CW = 1.98800468

    return Cb, CW
