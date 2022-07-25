import pandas as pd
import numpy as np


def main_detrend(detrend_degree: int, time: pd.Series, val: pd.DataFrame, label: list):
    """
    This is the primary function for parsing the detrending of the data. The objective here is to
    take the inputs, and pass them to the appropriate type of detrending.
    :param detrend_degree: This is the degree of the polynomial that we want for detrending.
    :param time: The time series of the data we're looking at.
    :param val: The associated values for each time point of the data we're looking at.
    :param label: The name of all the signals in the data set.
    :return:
    detrend_data: pd.DataFrame containing the detrended data (values only!)
    trend: This is a Numpy Array containing the coefficients of the detrending, which we use for
    re-calculating approximations.
    std_list: Same as above, except as a pd.Series. Standard deviation is used as another part
    of the detrending, so we use it later for re-calculating the approximations.
    """
    # We pass the function through our general polynomial fitting function. Technically, if the
    # detrend_degree is 0, there are faster ways of manipulating the data, but this is easier for
    # reference and formatting later.
    detrend_data, trend, std_list = poly_fit(time, val, label, detrend_degree)

    return detrend_data, trend, std_list


def constant_detrend(val: pd.DataFrame):
    """
    Does a constant detrend of the data. This function is faster than a poly_fit with degree 0, but goes unused
    so that we can have a consistent output format. I'm keeping it here just in case.
    :param val: Input data from transient stability.
    :return:
    detrend_data: pd.DataFrame that contains the detrended data.
    mean_data: np.NdArray that contains the coefficients of the linear detrend function.
    std_list: dictionary that contains the standard deviation used for normalization.
    """

    # Gets the means for all the signals.
    mean_data = val.mean()
    # Subtracts the mean from each of the corresponding signals.
    difference_data = val - mean_data
    # Gets the standard deviation for each signal.
    std_list = difference_data.std(ddof=0)
    # Gets the detrended data.
    detrend_data = difference_data / std_list

    return detrend_data, mean_data, std_list


def poly_fit(time_series: pd.Series, val: pd.DataFrame, labels: list, degree: int):
    """
    This is a generic polynomial fitting function, along with detrending the data.
    :param time_series: The time series of the data we're looking at.
    :param val: The associated values for each time point of the data we're looking at.
    :param labels: The name of all the signals in the data set.
    :param degree: This is the degree of the polynomial that we want for detrending.
    :return:
    detrend_data: pd.DataFrame containing the detrended data (values only!)
    poly: This is a Numpy Array containing the coefficients of the detrending, which we use for
    re-calculating approximations.
    std_dict: Same as above, except as a pd.Series. Standard deviation is used as another part
    of the detrending, so we use it later for re-calculating the approximations.
    """
    # Does a polynomial fit over the entire data set.
    poly = np.polynomial.polynomial.polyfit(time_series, val, degree)

    # Each column of poly_df refers to the coefficients of the equation c0 + c1*x + c2*x^2 + ... + cN * x^N
    poly_df = pd.DataFrame(poly, columns=labels)

    # To take advantage of itertuples and its speed, we transpose the DataFrame.
    poly_df = poly_df.T

    # This is where we'll store our standard deviations and detrended data.
    std_dict = {}
    detrend_dict = {}
    # Iterates through each row of the polynomial fit DataFrame.
    for row in poly_df.itertuples():
        # The index gets you the name of the signal.
        label = row.Index
        # The remaining information stored in the row are the coefficients of the polynomial fit.
        poly_func = np.array(list(row[1:]))
        poly_eval = np.polynomial.polynomial.polyval(time_series, poly_func)
        # Subtracts the linear fit from the original data, and renames the series for storage.
        difference = val[label].values - poly_eval
        difference = difference.rename(label)
        # Gets the standard deviation. We need this to rescale the data later.
        std_diff = difference.std(ddof=0)
        std_dict[label] = std_diff
        # Divides the data by the standard deviation for normalization.
        detrend = difference / std_diff
        # Stores the data in a new list.
        detrend_dict[label] = detrend

    # Converts the resulting dictionary into a new DataFrame.
    detrend_data = pd.DataFrame(detrend_dict)

    return detrend_data, poly_df, pd.Series(std_dict)
