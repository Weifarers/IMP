from detrend import *
import numpy as np
import scipy as sp
from scipy.linalg import eig


def matrix_pencil(num_data: int, imp_data: dict, svd_thresh: float):
    """
    This function applies the Matrix Pencil method to any given set of input data.
    :param num_data: Integer containing the number of data points per signal.
    :param imp_data: Dictionary containing the signals that we are passing through the Matrix Pencil method.
    :param svd_thresh: Float that has the SVD threshold we will use to filter the singular values.
    :return:
    eigenvalues: np.ndarray that contains all the eigenvalues after the filtering process of the Matrix Pencil method.
    """
    # Using the number of data points per signal, gets the pencil parameter.
    l_val = np.floor(num_data / 2)
    l_val = l_val.astype(int)

    # Initializes an empty Hankel matrix.
    hankel = np.array(())

    # Creates a temporary Hankel matrix for every signal.
    for column in imp_data:
        temp_hankel = np.zeros((num_data - l_val, l_val + 1))
        # Iterates through every row of the Hankel matrix.
        for j in range(num_data - l_val):
            temp_hankel[j] = imp_data[column][j:j + l_val + 1]
        # Vertically stacks the Hankel matrices, to increase its' size with more signals.
        # The if statement is there to check for the first instance of the Hankel matrix, where the size is 0.
        hankel = np.vstack([hankel, temp_hankel]) if hankel.size else temp_hankel

    # Takes a SVD of the Hankel matrix. Note that in some versions of PyCharm or other IDEs, this might spit
    # out a "Tuple assignment balance is incorrect" warning. I'm not entirely sure what the warning is meant
    # to fend off, but I think this is a bug with some IDEs, so you can ignore it. This shouldn't pose a
    # problem during the actual execution of the code.
    # noinspection PyTupleAssignmentBalance
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
    eigenvalues = sp.linalg.eig(y_2, y_1)[0]

    return eigenvalues


def imp_mode_present(z: np.ndarray, delta_t: float):
    """
    Converts the discrete-time modes into continuous-time modes for presentation purposes.
    Note that when presenting the results, we ignore the complex conjugates, and only show the
    eigenvalues with positive complex component.
    :param z: Eigenvalues of the matrix pair {Y2, Y1}
    :param delta_t: Time step between data points.
    :return:
    f_list: np.ndarray that has all the modal frequencies.
    b_per: np.ndarray that has all the damping percentages.
    """
    # Converts eigenvalues to continuous time ones.
    ct_lambda = np.log(z) / delta_t
    # Removes all complex conjugates with negative imaginary components.
    ct_lambda = np.extract(np.imag(ct_lambda) >= 0, ct_lambda)
    # Gets the list of frequencies.
    f_list = np.imag(ct_lambda) / (2 * np.pi)
    # Gets the list of damping percentages.
    b_per = -100 * (np.real(ct_lambda)) / (np.sqrt(np.square(np.real(ct_lambda)) + np.square(np.imag(ct_lambda))))
    return f_list, b_per


def mode_shapes(y: pd.DataFrame, z: np.ndarray):
    """
    Calculates the mode shapes associated with each signal.
    :param y: Detrended data that was used to calculate the modes.
    :param z: Eigenvalues of the matrix pair {Y2, Y1}
    :return:
    mode_mag: pd.DataFrame containing all the magnitudes for each of the modes.
    mode_theta: pd.DAtaFrame containing all the phase angles for each of the modes.
    """
    # Gets the names of all the signals.
    names = y.columns
    # Gets all the exponents for the construction of the Z matrix.
    exp = np.arange(y.shape[0])
    exp_col = np.asmatrix(exp).transpose()
    # Constructs the Z matrix.
    z_mat = np.power(np.asmatrix(z), exp_col)
    # Solves for the mode shapes.
    mode_shape = np.linalg.lstsq(z_mat, y, rcond=None)[0]
    # Gets the associated magnitudes and phases.
    mode_mag = np.absolute(mode_shape)
    mode_theta = np.angle(mode_shape)

    # Builds the data into DataFrames.
    mode_mag = pd.DataFrame(data=mode_mag, columns=names)
    mode_theta = pd.DataFrame(data=mode_theta, columns=names)

    return mode_mag, mode_theta


def imp_reproduce(z: np.ndarray, delta_t: float, trend: pd.DataFrame, std_list: pd.Series,
                  time_series: pd.Series, num_data: int, mode_mag: pd.DataFrame, mode_theta: pd.DataFrame):
    """
    This function takes the eigenvalues and mode shapes, and reproduces all the signals.
    :param z: np.ndarray containing all the eigenvalues from the Matrix Pencil method.
    :param delta_t: Float telling us the time step between time points.
    :param trend: pd.DataFrame containing all the coefficients of the detrending from
    :param std_list: pd.Series containing the standard deviations of the detrended data.
    :param time_series: pd.Series containing the time series data.
    :param num_data: Integer containing the number of data points per signal.
    :param mode_mag: pd.DataFrame containing the mode shape magnitudes for every mode and every signal.
    :param mode_theta: pd.DataFrame containing the mode shape angles for every mode and every signal.
    :return:
    y_hat_data_ns_df: pd.DataFrame containing the reproduced data, but without the any of the detrending.
    """

    # Converts poles to the modes of the signal, which contain the damping and frequency.
    ct_lambda = np.log(z) / delta_t

    # Breaks down the lambda into its real (damping) and imaginary (angular frequency) components.
    b_list = np.real(ct_lambda)
    w_list = np.imag(ct_lambda)

    # Creates fixed matrices that we'll use in conjunction with matrix manipulation to solve for y_hat.
    # Each matrix here is a N x M matrix, where N is the number of data points, and M is the number of modes.
    # The goal is to evaluate the summation in y_hat by making each column it's own mode, and then summing
    # across the columns.
    b_mat = np.tile(b_list, (num_data, 1))
    w_mat = np.tile(w_list, (num_data, 1))
    # Creates a similar matrix for the time series data.
    time_mat = np.tile(np.array([time_series.values]).transpose(), (1, len(b_list)))

    # Initializes the cosine and exponential components, since they're the same values for all the signals.
    exp_comp = np.exp(np.multiply(b_mat, time_mat))
    cos_ang_comp = np.multiply(w_mat, time_mat)

    # Initializes dictionaries to story the reproduced data.
    y_hat_data_ns = {}

    # Iteratively jumps through every row of the detrending DataFrame.
    for trend_row in trend.itertuples():
        # Gets the name of the signal, and the standard deviation associated with it.
        signal_name = trend_row.Index

        # Now we need the mode shapes for the particular signal of interest.
        r_list = mode_mag[signal_name]
        theta_list = mode_theta[signal_name]

        # We convert these into matrices, similar to how we did the modes.
        r_mat = np.tile(r_list, (num_data, 1))
        theta_mat = np.tile(theta_list, (num_data, 1))

        # Then we expand upon our original exponential and cosine component.
        temp_exp = np.multiply(r_mat, exp_comp)
        temp_cos = np.cos(np.add(cos_ang_comp, theta_mat))

        # Then evaluate the function to get y_hat.
        y_hat_temp = np.multiply(temp_exp, temp_cos)

        # Sums along the columns.
        y_hat_temp = y_hat_temp.sum(axis=1)

        # Stores the unscaled data so we can use it to calculate the cost function for each signal.
        y_hat_data_ns[signal_name] = y_hat_temp

    y_hat_data_ns_df = pd.DataFrame(y_hat_data_ns)

    return y_hat_data_ns_df


def imp_reproduce_final(z: np.ndarray, delta_t: float, trend: pd.DataFrame, std_list: pd.Series,
                        time_series: pd.Series, num_data: int, mode_mag: pd.DataFrame, mode_theta: pd.DataFrame):
    """
    This function differs from IMP reproduce, by only reproducing all the data as it originally was. This is only
    necessary *after* the IMP method is done, since before then, we compare the detrended data. So to save on
    computation time, instead of reproducing the data fully during the IMP, we just wait until the end. The code
    is functionally almost exactly the same as the imp_reproduce() function.
    :param z: np.ndarray containing all the eigenvalues from the Matrix Pencil method.
    :param delta_t: Float telling us the time step between time points.
    :param trend: pd.DataFrame containing all the coefficients of the detrending from
    :param std_list: pd.Series containing the standard deviations of the detrended data.
    :param time_series: pd.Series containing the time series data.
    :param num_data: Integer containing the number of data points per signal.
    :param mode_mag: pd.DataFrame containing the mode shape magnitudes for every mode and every signal.
    :param mode_theta: pd.DataFrame containing the mode shape angles for every mode and every signal.
    :return:
    y_hat_data_df: pd.DataFrame containing the reproduced data, y_hat.
    """

    # Converts poles to the modes of the signal, which contain the damping and frequency.
    ct_lambda = np.log(z) / delta_t

    # Breaks down the lambda into its real (damping) and imaginary (angular frequency) components.
    b_list = np.real(ct_lambda)
    w_list = np.imag(ct_lambda)

    # Creates fixed matrices that we'll use in conjunction with matrix manipulation to solve for y_hat.
    # Each matrix here is a N x M matrix, where N is the number of data points, and M is the number of modes.
    # The goal is to evaluate the summation in y_hat by making each column it's own mode, and then summing
    # across the columns.
    b_mat = np.tile(b_list, (num_data, 1))
    w_mat = np.tile(w_list, (num_data, 1))
    # Creates a similar matrix for the time series data.
    time_mat = np.tile(np.array([time_series.values]).transpose(), (1, len(b_list)))

    # Initializes the cosine and exponential components, since they're the same values for all the signals.
    exp_comp = np.exp(np.multiply(b_mat, time_mat))
    cos_ang_comp = np.multiply(w_mat, time_mat)

    # Initializes dictionaries to story the reproduced data.
    y_hat_data = {}

    # Iteratively jumps through every row of the detrending DataFrame.
    for trend_row in trend.itertuples():
        # Gets the name of the signal, and the standard deviation associated with it.
        signal_name = trend_row.Index
        curr_std = std_list[signal_name]
        # The remaining information stored in the row are the coefficients of the polynomial fit.
        poly_func = np.array(list(trend_row[1:]))
        # Evaluates the detrended polynomial over the time series.
        detrend_eval = np.polynomial.polynomial.polyval(time_series, poly_func)

        # Now we need the mode shapes for the particular signal of interest.
        r_list = mode_mag[signal_name]
        theta_list = mode_theta[signal_name]

        # We convert these into matrices, similar to how we did the modes.
        r_mat = np.tile(r_list, (num_data, 1))
        theta_mat = np.tile(theta_list, (num_data, 1))

        # Then we expand upon our original exponential and cosine component.
        temp_exp = np.multiply(r_mat, exp_comp)
        temp_cos = np.cos(np.add(cos_ang_comp, theta_mat))

        # Then evaluate the function to get y_hat.
        y_hat_temp = np.multiply(temp_exp, temp_cos)

        # Sums along the columns.
        y_hat_temp = y_hat_temp.sum(axis=1)

        # Adds this data to our entire reconstructed data set. Note that here we scale the data by the standard
        # deviation and add back in the detrend, effectively reversing what we had initially done to the data set.
        y_hat_data[signal_name] = np.add((curr_std * y_hat_temp), detrend_eval)

    # We then convert everything back into DataFrames for later use.
    y_hat_data_df = pd.DataFrame(y_hat_data)

    return y_hat_data_df





def cost_function(y: pd.DataFrame, y_hat: pd.DataFrame, num_data: int):
    """
    Calculates the cost functions for each signal.
    :param y: Original data gathered from transient stability.
    :param y_hat: Reproduced data from the matrix pencil method.
    :param num_data: Number of the data points in the system.
    :return: cost_dict: Dictionary that contains the cost function for each signal.
    """
    # Takes the difference between the original data and the reproduced data.
    residual = y - y_hat
    # Applies a norm function over each column of the residual, and then divides it by the number of data points.
    cost = np.linalg.norm(residual, axis=0)**2 / (2*num_data)
    # Constructs a dictionary where each key is the bus, and the value is the associated cost function.
    cost_dict = dict(zip(residual.columns, cost))

    return cost_dict
