def options():
    # This defines the SVD threshold.
    svd_threshold = 0.025
	
    # This defines the number of iterations.
    num_iter = 10
	
    # This defines where the first signal is picked from, by index.
    initial_signal = 0
	
    # This defines the type of detrending used.
    # 0 = Constant detrending
    # 1 = Linear detrending
    # 2 = Quadratic detrending
    detrend_degree = 1
	
    # This flag is used for the random signal selection. If true, the IMP method
    # will randomly pick signals each iteration instead of the one with the highest cost function.
    random = False

    return svd_threshold, num_iter, initial_signal, detrend_degree, random
