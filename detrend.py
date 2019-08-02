import pandas as pd
import numpy as np


def constant_detrend(val):
    # Does a constant detrend of the data.
    # Inputs: val - The data that you want to detrend.
    # Returns the detrended data, along with the average used for detrending.

    # Calculates the average, and subtracts it from the data.
    val = val - val.mean()
    # Divides the new data by its standard deviation, normalized by N.
    return val/(val.std(ddof=0)), val.mean()


def lin_detrend(time, val, label):
    # Does a linear detrend of the data.
    # Inputs: time - Time series data.
    #         val - The data that you want to detrend.
    #         label - The associated labels with each signal.
    # Returns the detrended data, along with the linear function that was used for detrending.

    # Initializes a data frame to store the detrended data.
    detrend_data = pd.DataFrame()
    # Gets the linear polynomial associated with each signal.
    lin_poly = np.polynomial.polynomial.polyfit(time, val, 1)
    # Initializes a dictionary to store standard deviations.
    std_list = {}
    # Iteratively detrends each signal by its' linear fit.
    for i in range(lin_poly.shape[1]):
        # Gets the current signal label.
        curr_sig = label[i]
        # Gets the linear fit associated with the current signal, and evaluates it for our time steps.
        poly_func = lin_poly[:, i]
        poly_eval = np.polynomial.polynomial.polyval(time, poly_func)
        # Subtracts the linear fit from the original data, and renames the series for storage.
        difference = val.iloc[:, i] - poly_eval
        difference = difference.rename(curr_sig)
        # Gets the standard deviation. We need this to rescale the data later.
        std_diff = difference.std(ddof=0)
        std_list[curr_sig] = std_diff
        # Divides the data by the standard deviation for normalization.
        detrend = difference/std_diff
        # Stores the data in a new list.
        detrend_data = detrend_data.append(detrend)
    return detrend_data, lin_poly, std_list


def quad_detrend(time, val, label):
    # Does a quadratic detrend of the data.
    # Inputs: time - Time series data.
    #         val - The data that you want to detrend.
    #         label - The associated labels with each signal.
    # Returns the detrended data, along with the quadratic polynomial used for detrending.

    # Initializes a data frame to store the detrended data.
    detrend_data = pd.DataFrame()
    # Gets the quadratic polynomial associated with each signal.
    quad_poly = np.polynomial.polynomial.polyfit(time, val, 2)

    # Iteratively detrends each signal by its' quadratic fit.
    for i in range(quad_poly.shape[1]):
        # Gets the current signal label.
        curr_sig = label[i]
        # Gets the quadratic fit associated with the current signal, and evaluates it for our time steps.
        poly_func = quad_poly[:, i]
        poly_eval = np.polynomial.polynomial.polyval(time, poly_func)
        # Subtracts the quadratic fit from the original data, and renames the series for storage.
        difference = val.iloc[:, i] - poly_eval
        difference = difference.rename(curr_sig)
        # Divides the data by the standard deviation for normalization.
        detrend = difference/difference.std(ddof=0)
        # Stores the data in a new list.
        detrend_data = detrend_data.append(detrend)
    return detrend_data, quad_poly
