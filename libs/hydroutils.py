import keras.backend as K
import numpy as np
from scipy import stats
import logging
import tensorflow as tf


def nse_loss(y_true, y_pred):
    y_true = y_true[:, 365:,: ]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, 365:,: ]  # Omit values in the spinup period (the first 365 days)
    
    numerator = K.sum(K.square(y_pred - y_true), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)

    return numerator / denominator

def nse_metrics(y_true, y_pred):
    y_true = y_true[:, 365:,: ]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, 365:,: ]  # Omit values in the spinup period (the first 365 days)
    
    numerator = K.sum(K.square(y_pred - y_true), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)
    rNSE =  numerator / denominator

    return 1.0 - rNSE


def batch_nse_loss(y_true, y_pred):
    y_true = y_true[:, 365:,: ]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, 365:,: ]  # Omit values in the spinup period (the first 365 days)
    
    square = K.square(y_pred - y_true)
    std = K.std(y_true, axis = 1, keepdims=True)
    
    #K.mean(K.sum(square / K.square(std + 0.1), axis=1, keepdims=False))

    return K.mean(square / K.square(std + 0.01), axis=1, keepdims=False)

def batch_nse_metrics(y_true, y_pred):
    y_true = y_true[:, 365:,: ]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, 365:,: ]  # Omit values in the spinup period (the first 365 days)
    
    square = K.square(y_pred - y_true)
    std = K.std(y_true, axis = 1, keepdims=True)
    
    #K.mean(K.sum(square / K.square(std + 0.1), axis=1, keepdims=False))

    return 1- K.mean(square / K.square(std + 0.01), axis=1, keepdims=False)


def nse(y_true, y_pred):
    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - numerator / denominator

    
def agreementindex(evaluation, simulation):
    """
    Agreement Index (d) developed by Willmott (1981)
        .. math::   
         d = 1 - \\frac{\\sum_{i=1}^{N}(e_{i} - s_{i})^2}{\\sum_{i=1}^{N}(\\left | s_{i} - \\bar{e} \\right | + \\left | e_{i} - \\bar{e} \\right |)^2}  
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Agreement Index
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        simulation, evaluation = np.array(simulation), np.array(evaluation)
        Agreement_index = 1 - (np.sum((evaluation - simulation)**2)) / (np.sum(
            (np.abs(simulation - np.mean(evaluation)) + np.abs(evaluation - np.mean(evaluation)))**2))
        return Agreement_index
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan
    


def fdc_fhv(obs: np.ndarray, sim: np.ndarray, h: float = 0.01) -> float:
    """Peak flow bias of the flow duration curve (Yilmaz 2018).
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    h : float, optional
        Fraction of the flows considered as peak flows. Has to be in range(0,1), by default 0.02
    
    Returns
    -------
    float
        Bias of the peak flows
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If `h` is not in range(0,1)
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (h <= 0) or (h >= 1):
        raise RuntimeError("h has to be in the range (0,1)")

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # subset data to only top h flow values
    obs = obs[:np.round(h * len(obs)).astype(int)]
    sim = sim[:np.round(h * len(sim)).astype(int)]

    fhv = np.sum(sim - obs) / (np.sum(obs) + 1e-6)

    return fhv *100
