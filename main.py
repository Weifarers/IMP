import os
import time
# noinspection PyUnresolvedReferences
import pandas as pd
# noinspection PyUnresolvedReferences
from detrend import *
from IMP import *
import options
import statistics
import random


def main():
    # This initializes some variables so Python doesn't yell at us.
    parent_path = None
    file_name = None

    # We set a loop to ensure that the path that's input is a valid one.
    isdir = False
    # First, we prompt the user for a directory for their data.
    while not isdir:
        # We distinguish the directory where all the data is stored and the raw data is stored so that
        # the code can create a new folder for the filtered data within the directory specified, such that
        # everything is consolidated in one place.
        parent_path = input("Input Folder where your data is stored: ")
        # If it's a valid path in the machine, accept the path and move on.
        if os.path.isdir(parent_path):
            isdir = True
        # Otherwise, continue to prompt the user for a path.
        else:
            print("That path is not a valid path, please try again.")
            continue

    # We set another loop to ensure the file exists.
    isfile = False
    while not isfile:
        # Prompts the user for a file name.
        file_name = input("Input the name of the file, with extension: ")
        if os.path.isfile(parent_path + '\\' + file_name):
            # This is just a checker for the extension. The code only accepts csv/txt, although
            # it could be modified to accept xlsx files.
            if file_name[-3:] != 'csv' and file_name[-3:] != 'txt':
                print('Please use either a CSV or txt file.')
                continue
            # If it passes all the checks, exit the loop.
            else:
                isfile = True
        # If the file doesn't exist, prompt them for another one.
        else:
            print("That file does not exist, please try again.")
            continue

    read_start = time.time()
    data = {}
    # Reads in the data. This is a simple if-else to check what file type it is, and adjust
    # the delimiter accordingly.
    if file_name[-3:] == 'csv':
        data = pd.read_csv(parent_path + '\\' + file_name, sep=',')
    elif file_name[-3:] == 'txt':
        data = pd.read_csv(parent_path + '\\' + file_name, sep='\t')
    read_end = time.time()
    print('Data has been read in: ', read_end - read_start)

    # Imports the options from the options.py file.
    svd_threshold, num_iter, initial_signal, detrend_degree, random_flag = options.options()

    # Applies our data pre-processing.
    detrend_start = time.time()
    detrend_data, trend, std_list, time_series, delta_t = pre_process(data, detrend_degree)
    detrend_end = time.time()
    print('Data has been pre-processed in: ', detrend_end - detrend_start)

    # Applies the IMP method using the given options.
    imp_start = time.time()
    f_list, b_per, y_hat_data, cost_list, imp_data, ave_cost\
        = imp_method(detrend_data, num_iter, time_series, initial_signal,
                     svd_threshold, delta_t, trend, std_list, False)
    imp_end = time.time()
    print('IMP done (no vector) in: ', imp_end - imp_start)

    # random_imp(detrend_data, num_iter, time_series, initial_signal, svd_threshold, delta_t, trend, std_list)

    # varying_user_input(detrend_data, num_iter, time_series, initial_signal, svd_threshold,
    #                    delta_t, trend, std_list, random_flag)


def random_imp(detrend_data, num_iter, time_series, initial_signal, svd_threshold, delta_t, trend, std_list):
    i = 0
    random_dict = {}
    while i < 300:
        f_list, b_per, y_hat_data, cost_list, imp_data, ave_cost \
            = imp_method(detrend_data, num_iter, time_series, initial_signal,
                         svd_threshold, delta_t, trend, std_list, True)
        random_dict[i] = list(ave_cost.values())
        print('Random Selection %d done' % i)
        i += 1

    random_df = pd.DataFrame.from_dict(random_dict)
    random_df.to_csv('random_df.csv')


def varying_user_input(detrend_data, num_iter, time_series, initial_signal, svd_threshold,
                       delta_t, trend, std_list, random_flag):
    # Applies the IMP method to the data set, but applies it by iterating over one choice of option.
    # This lets us do a sensitivity analysis for any of our input parameters.

    # Getting some output statistics:
    max_cost_dict = {}
    min_cost_dict = {}
    average_cost_dict = {}
    time_dict = {}
    signal_select = {}

    # Generates a list of equally spaced SVD thresholds.
    svd_thresh_list = np.arange(0.025, 0.2005, 0.0005).tolist()
    # Generates a list of integers for random starting point selection.
    rng = np.random.default_rng(seed=308)
    random_ints = rng.choice(detrend_data.shape[1], size=300, replace=False)

    # Then iterating over some variable of interest.
    for i in random_ints:
        imp_start = time.time()
        f_list, b_per, y_hat_data, cost_list, imp_data, ave_cost \
            = imp_method(detrend_data, num_iter, time_series, i,
                         svd_threshold, delta_t, trend, std_list, random_flag)
        imp_end = time.time()
        print("---------- Signal Selection: ", i, "overview:----------")
        # print('IMP done in: ', imp_end - imp_start)
        # We'll store the relevant values of interest into dictionaries.
        max_cost_dict[i] = [max(cost_list.values())]
        min_cost_dict[i] = [min(cost_list.values())]
        # average_cost_dict[i] = [statistics.mean(cost_list.values())]
        average_cost_dict[detrend_data.iloc[:, i].name] = [statistics.mean(cost_list.values())]
        signal_select[i] = list(imp_data.keys())
        time_dict[i] = [imp_end - imp_start]

    # # Then we output the results into DataFrames, and then into CSVs.
    max_cost = pd.DataFrame(max_cost_dict)
    min_cost = pd.DataFrame(min_cost_dict)
    ave_cost = pd.DataFrame(average_cost_dict)
    # time_df = pd.DataFrame(time_dict)
    #
    max_cost.to_csv("max_cost.csv")
    min_cost.to_csv("min_cost.csv")
    ave_cost.to_csv("ave_cost.csv")
    # time_df.to_csv("time.csv")

    signal_select_df = pd.DataFrame(signal_select)
    signal_select_df.to_csv("signal_select.csv")


def pre_process(data: pd.DataFrame, detrend_degree: int):
    """
    This function pre-processes our data; this involves detrending the data, and splitting the data into the
    appropriate values.
    :param data: Input data from the CSV.
    :param detrend_degree: The order of polynomial that we'll be using to detrend.
    :return:
    detrend_data: pd.DataFrame that contains the detrended data.
    trend: pd.DataFrame that contains the coefficients for our polynomial fit.
    std_list: pd.Series that contains the standard deviation for each signal.
    time_series: pd.Series that contains the time series information.
    delta_t: Float that tells us the time step between data points.
    """
    # Splits the data into it's time series.
    time_series = data.iloc[:, 0]
    # Gets the associated values of the data we're looking at.
    values = data.iloc[:, 1:]
    # Using the values, gets the labels for each of the signals.
    labels = list(values)

    # Normalizes the time series so it always starts as t = 0. This is particularly important for PowerWorld data,
    # where the contingency does not start at t = 0, but normally starts at some time in the future, say t = 1.
    time_series = time_series - time_series[0]

    # Also gets the time step.
    delta_t = time_series[1] - time_series[0]

    # Applies the data detrending to the data.
    detrend_data, trend, std_list = main_detrend(detrend_degree, time_series, values, labels)

    return detrend_data, trend, std_list, time_series, delta_t


def imp_method(detrend_data: pd.DataFrame, num_iter: int, time_series: pd.Series, initial_index: int,
               svd_thresh: float, delta_t: float, trend: pd.DataFrame, std_list: pd.Series, random_flag: bool):
    """
    Function that applies the Iterative Matrix Pencil method.
    :param detrend_data: pd.DataFrame of detrended data that the IMP will be applied to.
    :param num_iter: Integer for number of iterations.
    :param time_series: pd.Series that contains the time series information.
    :param initial_index: Integer that tells us what signal to start with.
    :param svd_thresh: Float that acts as our filter for singular value decomposition.
    :param delta_t: Float that tells us the time step between data points.
    :param trend: pd.DataFrame that contains the coefficients for our polynomial fit.
    :param std_list: pd.Series that contains the standard deviation for each signal.
    :param random_flag: boolean that indicates whether or not we're using the random IMP.
    :return:
    f_list: np.ndarray that contains the frequencies for each mode.
    b_per: np.ndarray that contains the damping percentages for each mode.
    y_hat_data: pd.DataFrame that contains the reproduced data.
    cost_list: Dictionary that contains the cost function for each signal at the end.
    """
    # Getting some initial values to pass down to the other functions.
    #   - num_data = number of data points per signal.
    #   - num_signal = number of signals.
    num_data = detrend_data.shape[0]
    num_signal = detrend_data.shape[1]
    # Also checks if the number of iterations is greater than the number of signals.
    if num_iter > num_signal:
        num_iter = num_signal

    # Iterative Matrix Pencil Method
    # Initializations of variables that we'll be storing information in later.
    imp_data = {}
    eigenvalues = np.array([])
    y_hat_data_ns = {}
    mode_mag = {}
    mode_ang = {}
    ave_cost = {}

    # Sets the initial signal of interest based on the index provided by the options.
    initial_signal = detrend_data.iloc[:, initial_index]
    imp_data[initial_signal.name] = initial_signal.values

    # Then begins the iterative part. Since Python indexing would go from 0 to num_iter - 1, which is 1 iteration
    # less than what we want,
    for i in range(num_iter + 1):
        # print("----------Iteration", i, "overview----------")
        iter_start = time.time()
        # Sets a flag for use when checking cost functions.
        in_imp = True

        matrix_pencil_start = time.time()
        # Applies the matrix pencil method to the set of signals.
        eigenvalues = matrix_pencil(num_data, imp_data, svd_thresh)
        matrix_pencil_end = time.time()
        # print('Time for Matrix Pencil method: ', matrix_pencil_end - matrix_pencil_start)

        mode_shape_start = time.time()
        # Gets the mode shapes for all of the data.
        mode_mag, mode_ang = mode_shapes(detrend_data, eigenvalues)
        mode_shape_end = time.time()
        # print('Time for calculating mode shapes: ', mode_shape_end - mode_shape_start)

        imp_reproduce_start = time.time()
        # Reconstructs all of the data.
        y_hat_data_ns = imp_reproduce(eigenvalues, delta_t, trend, std_list, time_series, num_data,
                                      mode_mag, mode_ang)
        imp_reproduce_end = time.time()
        # print('Time for reproducing data: ', imp_reproduce_end - imp_reproduce_start)

        cost_start = time.time()
        # Gets the cost functions for all of the data.
        cost_list = cost_function(detrend_data, y_hat_data_ns, num_data)
        cost_end = time.time()
        # print('Time for calculating cost functions: ', cost_end - cost_start)

        # Chooses the signal with the largest cost function.
        max_cost = max(cost_list, key=lambda key: cost_list[key])
        # print("Signal Selected: ", max_cost, cost_list[max_cost])
        # Some statistics per iteration.
        min_cost = min(cost_list, key=lambda key: cost_list[key])
        # print("Minimum Cost: ", min_cost, cost_list[min_cost])

        # This is for the implementation of a random IMP method.
        if random_flag:
            # Generates a list of valid choices for buses by removing any buses that we've already selected.
            valid_buses = list(set(list(cost_list.keys())) - set(list(imp_data.keys())))

            # Randomly selects a bus from that list.
            random_bus = random.choice(valid_buses)

            # Adds that bus to the list of data to process.
            imp_data[random_bus] = pd.Series(detrend_data[random_bus].values)
            # print("Signal Selected:", random_bus)

        elif not random_flag:
            # Includes a conditional that checks whether or not the signal has already been included. If it has,
            # move to the next highest one. We use a flag to keep this up until we've run out of signals.
            while in_imp:
                # This if statement checks if the signal we just got with the largest cost function is already in the
                # list.
                if max_cost in imp_data.keys():
                    temp_dict = cost_list
                    # If it is, remove it from the existing list of cost functions, and then grab a new one.
                    del temp_dict[max_cost]
                    # This is a check to see if the list is empty. If it isn't, which returns true, we can just get the
                    # next signal.
                    if bool(temp_dict):
                        max_cost = max(temp_dict, key=lambda key: temp_dict[key])
                    # If the list is empty, then we've taken the last signal, so nothing else has to be done and we
                    # can exit the while loop and move on to the next iteration.
                    else:
                        in_imp = False
                # If it isn't, exit the while loop and move on to the next iteration.
                else:
                    in_imp = False

            # Updates the imp_data list with that signal.
            imp_data[max_cost] = pd.Series(detrend_data[max_cost].values)

            # print("Signal Selected:", max_cost)

        # Measuring the average cost function for the system over each iteration.
        ave_cost[i] = statistics.mean(cost_list.values())
        iter_end = time.time()
        print("Iteration", i, "total time:", iter_end - iter_start)

    # Gets the frequencies and damping percentages associated with the last set of eigenvalues from IMP.
    f_list, b_per = imp_mode_present(eigenvalues, delta_t)
    cost_list = cost_function(detrend_data, y_hat_data_ns, num_data)
    # Also reproduces all of the data with the data from the last iteration.
    y_hat_data = imp_reproduce_final(eigenvalues, delta_t, trend, std_list, time_series, num_data,
                                     mode_mag, mode_ang)

    return f_list, b_per, y_hat_data, cost_list, imp_data, ave_cost


if __name__ == "__main__":
    main()
