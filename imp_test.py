from detrend import *
import numpy as np
import scipy as sp
import time
from scipy.linalg import eig
import matplotlib
import matplotlib.pyplot as plt


def main():
    global file
    # Initializes a start time for the code, so we can time individual portions.
    start_time = time.time()
    # Prompts users to get a CSV file of data.
    filepath = '5 Signal - Python.csv'
    # Gets the name of the file to display to users.
    if len(filepath) > 0:
        file = pd.read_csv(filepath, sep=',')
    # Establishes some global variables to track computation times.
    # Gets the time series data.
    time_data = file['Time']
    # Gets the remaining values from the data.
    val_data = file.iloc[:, 1:]
    # Gets the signal labels.
    signal_label = list(val_data)
    # Defines the SVD threshold.
    svd_thresh = 0.025
    num_iter = 10
    # Applies the iterative matrix pencil method.
    f_list, b_per, y_hat_data, detrend_data, cost_list, detrend_time, imp_time, iter_time = \
        imp(time_data, val_data, signal_label, svd_thresh, num_iter)

    # Displays the results.
    modal_df = pd.DataFrame({"Frequency": f_list, "Damping": b_per})
    print('Modal Analysis Results: \n', modal_df)
    cost_df = pd.DataFrame({"Number": list(cost_list.keys()), "Cost Function": list(cost_list.values())})
    print('Cost Function Results: \n', cost_df)
    print('Time Taken to Detrend Data: %.2f s \n' % (detrend_time - start_time))
    print('Time Taken to do IMP: %.2f s \n' % (imp_time - start_time))

    # Plotting the reconstructed data.
    plt.figure(1)
    recon_data = y_hat_data.iloc[:, 0]
    orig_data = val_data.iloc[:, 0]
    plt.plot(time_data, orig_data, time_data, recon_data, 'r--')
    # Setting the plot options.
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Reconstructed Data vs Original Data')

    # Plotting the modes.
    plt.figure(2)
    plt.plot(f_list, b_per, 'o')
    # Setting the plot options.
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Damping %')
    plt.title('Plot of Modes')
    plt.show()


def imp(time_series, val, label, svd_thresh, num_iter):
    # Calls the Iterative Matrix Pencil method.
    # Inputs: time - Time series
    #         val - The data that you want to detrend.
    #         label - The associated labels with each signal.
    #         svd_thresh - The threshold used to filter the singular values.
    #         num_iter - Number of iterations provided by user.
    #         start_time - Initialization of starting time to time individual components of IMP.
    # Outputs: f_list - List of all the modal frequencies.
    #          b_per - List of all the damping percentages for the modal frequencies.
    #          y_hat_data - DataFrame containing the reproduced data.
    #          detrend_data - DataFrame containing the original data that was detrended.
    #          cost_list - List containing all the cost functions for each signal.

    # Getting some initial values to pass down to the other functions.
    #   - num_data = number of data points per signal.
    #   - num_signal = number of signals.
    num_data = val.shape[0]
    num_signal = val.shape[1]
    # Also checks if the number of iterations is greater than the number of signals.
    if num_iter > num_signal:
        num_iter = num_signal
    # Initializes some constants for later use.
    eigs = {}
    y_hat_data = pd.DataFrame()
    y_hat_data_ns = pd.DataFrame()
    iter_time = {}

    # Data Pre-processing
    time_series = time_series - time_series[0]
    # Gets the time step.
    delta_t = time_series[1] - time_series[0]
    detrend_data, lin_poly, std_list = lin_detrend(time_series, val, label)
    # Transposes the result so each column is a signal.
    detrend_data = detrend_data.transpose()
    detrend_time = time.time()

    # Iterative Matrix Pencil Method
    # Initializes the data set we'll be applying the matrix pencil method to.
    imp_data = pd.DataFrame()
    # Sets the initial signal to be the first signal in the data set. This gives us flexibility in choosing
    # our starting point.
    initial_signal = detrend_data.iloc[:, 0]
    imp_data[initial_signal.name] = initial_signal.values
    for i in range(num_iter):
        # Starts a timer for each iteration of the IMP method, along with a dictionary to store values.
        iter_start = time.time()
        # Sets a flag for use when checking cost functions.
        in_imp = True
        # Applies the matrix pencil method to the set of signals.
        eigs = matrix_pencil(imp_data, svd_thresh)
        # Times the matrix pencil method for each iteration.
        matrix_pencil_time = time.time()
        # Gets the mode shapes for all of the data.
        mode_mag, mode_ang = mode_shapes(detrend_data, eigs)
        # times how long it takes to get the mode shapes.
        mode_shape_time = time.time()
        # Reconstructs all of the data.
        y_hat_data, y_hat_data_ns = imp_reproduce(mode_mag, mode_ang, eigs, delta_t, num_signal,
                                                  num_data, time_series, lin_poly, std_list)
        # Times how long it takes to reconstruct the data.
        reconstruct_time = time.time()
        # Gets the cost functions for all of the data.
        cost_list = cost_function(detrend_data, y_hat_data_ns, num_signal, num_data)
        # Times how long it takes to calculate the cost functions.
        cost_time = time.time()
        # Chooses the signal with the largest cost function.
        max_cost = max(cost_list, key=lambda key: cost_list[key])
        # Includes a conditional that checks whether or not the signal has already been included. If it has, move to the
        # next highest one. We use a flag to keep this up until we've run out of signals.
        while in_imp:
            if max_cost in imp_data.keys():
                del cost_list[max_cost]
                # This is a check to see if the list is empty. If it isn't, which returns true, we can just get the
                # next signal.
                if bool(cost_list):
                    max_cost = max(cost_list, key=lambda key: cost_list[key])
                # If the list is empty, then we've taken the last signal, so nothing else has to be done and we
                # can exit the loop.
                else:
                    in_imp = False
            else:
                in_imp = False
        # Updates the imp_data list with that signal.
        imp_data[max_cost] = pd.Series(detrend_data[max_cost].values)
        # Generates a dictionary for each iteration that stores the times for each iteration.
        iter_time[i+1] = {'Matrix Pencil: %.3f' % (matrix_pencil_time - iter_start),
                          'Mode Shape Calculation: %.3f' % (mode_shape_time - matrix_pencil_time),
                          'Reconstruction: %.3f' % (reconstruct_time - mode_shape_time),
                          'Cost Function Calculation: %.3f' % (cost_time - reconstruct_time)}
    # Gets the frequencies and damping percentages associated with the last set of eigenvalues from IMP.
    f_list, b_per = imp_mode_present(eigs, delta_t)
    cost_list = cost_function(detrend_data, y_hat_data_ns, num_signal, num_data)
    imp_time = time.time()
    return f_list, b_per, y_hat_data, detrend_data, cost_list, detrend_time, imp_time, iter_time


def matrix_pencil(y, svd_thresh):
    # Calls the matrix pencil method.
    # Inputs: y - Signal data.
    #         svd_thresh - The threshold used to filter the singular values.
    # Returns the eigenvalues of the matrix pair {Y2, Y1}
    # Gets the N and L (pencil parameter) used to create the Hankel matrix.
    num_data = y.shape[0]
    l_val = np.floor(num_data / 2)
    l_val = l_val.astype(int)
    # Initializes an empty Hankel matrix.
    hankel = np.array(())
    # Creates a temporary Hankel matrix for every signal.
    for column in y:
        temp_hankel = np.zeros((num_data - l_val, l_val + 1))
        # Iterates through every row of the Hankel matrix.
        for j in range(num_data - l_val):
            temp_hankel[j] = y[column][j:j + l_val + 1]
        # Vertically stacks the Hankel matrices, to increase its' size with more signals.
        # The if statement is there to check for the first instance of the Hankel matrix, where the size is 0.
        hankel = np.vstack([hankel, temp_hankel]) if hankel.size else temp_hankel

    # Takes a SVD of the Hankel matrix..
    u, s_val, vh = np.linalg.svd(hankel)
    # Converts vH into v.
    v = vh.conj().transpose()
    # Gets the largest singular value.
    s_first = s_val[0]
    # Scales the values by the large singular value.
    s_val = s_val / s_first
    # Only keeps entries where the scaled value is greater than the threshold.
    s_val = s_val[s_val > svd_thresh]
    # Rescales the singular values back to their original value.
    s_val = s_val * s_first
    # Creates a diagonal matrix with the singular values we kept.
    new_s = np.diag(s_val)

    # Scales the size of the v matrix from SVD based on how many singular values we kept.
    num_cols = new_s.shape[1]
    new_v = v[:, 0:num_cols]

    # Gets the v1 and v2 matrices.
    v_1 = new_v[:-1, :]
    v_2 = new_v[1:, :]

    # Constructs the y1 and y2 matrices from v1/v2
    y_1 = np.dot(v_1.transpose(), v_1)
    y_2 = np.dot(v_2.transpose(), v_1)

    # Gets the eigenvalues of the matrix pair.
    eigs = sp.linalg.eig(y_2, y_1)[0]

    return eigs


def imp_mode_present(z, delta_t):
    # Converts the discrete-time modes into continuous-time modes for presentation purposes.
    # Note that when presenting the results, we ignore the complex conjugates, and only show the
    # eigenvalues with positive complex component.
    # Inputs: z - The eigenvalues of the matrix pair {Y2,Y1}.
    #         delta_t - Time step between data points.
    # Returns the modes by frequency and damping percentage.

    # Converts eigenvalues to continuous time ones.
    ct_lambda = np.log(z) / delta_t
    # Removes all complex conjugates with negative imaginary components.
    ct_lambda = np.extract(np.imag(ct_lambda) >= 0, ct_lambda)
    # Gets the list of frequencies.
    f_list = np.imag(ct_lambda) / (2 * np.pi)
    # Gets the list of damping percentages.
    b_per = -100 * (np.real(ct_lambda)) / (np.sqrt(np.square(np.real(ct_lambda)) + np.square(np.imag(ct_lambda))))
    return f_list, b_per


def mode_shapes(y, z):
    # Calculates the mode shapes associated with each signal.
    # Inputs: y - Detrended data that was used to calculate the modes.
    #         z - The eigenvalues of the matrix pair {Y2,Y1}.
    # Returns the mode shapes split into two lists, magnitude and phase.
    exp = np.arange(y.shape[0])
    exp_col = np.asmatrix(exp).transpose()
    # Constructs the Z matrix.
    z_mat = np.power(np.asmatrix(z), exp_col)
    # Solves for the mode shapes.
    mode_shape = np.linalg.lstsq(z_mat, y, rcond=None)[0]
    # Gets the associated magnitudes and phases.
    mode_mag = np.absolute(mode_shape)
    mode_theta = np.angle(mode_shape)

    return mode_mag, mode_theta


def imp_reproduce(mode_mag, mode_theta, z, delta_t, num_signal, num_data, time_series, lin_poly, std_list):
    # Reproduces each of the signals given the set of modes.
    # Inputs: mode_mag - The list of the magnitudes for each mode shape.
    #         mode_theta - The list of the angles for each mode shape.
    #         z - The eigenvalues of the matrix pair {Y2, Y1}.
    #         delta_t - The time step between data points.
    #         num_signal - The number of signals in your data set.
    #         num_data - The number of data points per signal.
    #         time_series - List of all time points.
    #         lin_poly - List of linear polynomials used for data detrending.
    #         std_list - List of standard deviations for each signal.
    # Outputs: y_hat_data - Reproduced data, rescaled.
    #          y_hat_data_ns - Reproduced data, but unscaled. This is used for cost function calculations.
    # Converts poles to the modes of the signal, which contain the damping and frequency.
    ct_lambda = np.log(z) / delta_t
    # Getting the number of modes.
    num_modes = ct_lambda.shape[0]
    # Breaks down the lambda into its real (damping) and imaginary (angular frequency) components.
    b_list = np.real(ct_lambda)
    w_list = np.imag(ct_lambda)
    # Initializing the data frame where we'll store the data.
    y_hat_data = pd.DataFrame()
    # Initializing another DataFrame where we store the unscaled data, so we can calculate the cost function.
    y_hat_data_ns = pd.DataFrame()
    for i in range(num_signal):
        # Creates a name for each new signal.
        signal_name = 'Signal ' + str(i + 1)
        # Gets the standard deviation of the associated signal.
        curr_std = std_list[signal_name]
        # Gets the detrend associated with the signal as well.
        curr_detrend = lin_poly[:, i]
        # Evaluates the detrend.
        detrend_data = curr_detrend[1] * time_series + curr_detrend[0]
        # Initializes a temporary y_hat that we'll use to store the results.
        y_hat_temp = np.zeros((num_data, 1))
        # Iterating through each mode.
        for j in range(num_modes):
            # The data is split into two components, an exponential one and a cosine one.
            exp_comp = mode_mag[j, i] * np.exp(b_list[j] * time_series)
            cos_comp = np.cos(np.add(w_list[j] * time_series, mode_theta[j, i]))
            # Gets the values, and forces them into a shape such that you can combine them.
            exp_comp = exp_comp.values
            exp_comp = exp_comp.reshape((num_data, 1))
            cos_comp = cos_comp.values
            cos_comp = cos_comp.reshape((num_data, 1))
            # We update the values every iteration.
            y_hat_temp = np.add(y_hat_temp, np.multiply(exp_comp, cos_comp))

        # Adds this data to our entire reconstructed data set. Note that here we scale the data by the standard
        # deviation and add back in the detrend, effectively reversing what we had initially done to the data set.
        y_hat_data[signal_name] = curr_std * pd.Series(y_hat_temp.transpose()[0]) + detrend_data
        # Stores the unscaled data so we can use it to calculate the cost function for each signal.
        y_hat_data_ns[signal_name] = pd.Series(y_hat_temp.transpose()[0])
    return y_hat_data, y_hat_data_ns


def cost_function(y, y_hat, num_signal, num_data):
    # Initializes a dictionary to store the cost functions.
    cost_list = {}
    # For each signal in our system, calculate the cost function. There's probably a faster way of
    # doing this via matrices, tbh.
    for i in range(num_signal):
        # Gets a signal name.
        signal_name = 'Signal ' + str(i + 1)
        # Calculates the residual.
        residual = np.subtract(y[signal_name], y_hat[signal_name])
        # Calculates the cost function.
        temp_cost = np.linalg.norm(residual) / num_data
        # Adds this cost function to the list.
        cost_list[signal_name] = temp_cost

    return cost_list


if __name__ == '__main__':
    main()
