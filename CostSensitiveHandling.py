import numpy as np

cost_m = [[0.1, 2], [1, 0]]


def stratification_undersample(X, y, per=0.66, dimensions=2):
    """Under-sampling.
       Parameters
       ----------
           X : array-like of shape = [n_samples, n_features]
                or [n_samples, n_features, n_channels]
               The input samples.
           y : array-like of shape = [n_samples]
               Ground truth (correct) labels.
           per: float, optional (default = 0.5)
               Percentage of the minority class in the under-sampled data
           dimensions: int, dimensions of X
       """
    n_samples = X.shape[0]

    filter_1 = np.where(y >= .5, True, False)
    filter_0 = np.where(y < .5, True, False)

    y1 = y[filter_1]
    y0 = y[filter_0]

    filter_0 = filter_0.reshape(1, -1)
    if dimensions == 3:
        x1 = X[filter_1, :, :]
        x0 = X[filter_0, :, :]
    elif dimensions == 2:
        x1 = X[filter_1, :]
        x0 = X[filter_0, :]
    else:
        print("Error with the dimensions")
        exit()

    num_y1 = y1.shape[0]

    num_y0 = n_samples - num_y1
    num_y0_new = num_y1 * 1.0 / per - num_y1
    perc_to_keep = num_y0_new / num_y0

    rej_rand = np.random.rand(num_y0)
    filter_ = rej_rand <= perc_to_keep

    if dimensions == 3:
        x0 = x0[filter_, :, :]
    elif dimensions == 2:
        x0 = x0[filter_, :]
    else:
        print("error with the dimensions")
        exit()

    y0 = y0[filter_]

    return np.concatenate((x0, x1)), np.concatenate((y0, y1))


def rejection_sampling(X, y):
    """Rejection sampling. This method was taken and modified from https://github.com/albahnsen/CostSensitiveClassification/blob/master/costcla/sampling/cost_sampling.py
       Parameters
       ----------
           X : array-like of shape = [n_samples, n_features]
               The input samples.
           y : array-like of shape = [n_samples]
               Ground truth (correct) labels.
    """
    tn, fn, fp, tp = cost_m[0][0], cost_m[0][1], cost_m[1][0], cost_m[1][1]
    cost_mat = np.tile(np.array([fp, fn, tp, tn]), (len(y), 1))

    cost_mis = cost_mat[:, 0]
    filter_1 = np.where(y >= .5, True, False)
    cost_mis[filter_1] = cost_mat[filter_1, 1]

    wc = cost_mis / cost_mis.max()

    n_samples = X.shape[0]

    rej_rand = np.random.rand(n_samples)

    filter_ = rej_rand <= wc

    x_cps = X[filter_]
    y_cps = y[filter_]

    return x_cps, y_cps


def example_weighting(y):
    weights = np.where(y >= .5, 2.0, 1.0)

    return weights


if __name__ == '__main__':
    # testing
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]])
    y = np.array([.1, .2, .7, .8, .5, .6, .1, .2, .1, .2])

    # print(stratification_undersample(X, y))
    # print(example_weighting(y))
    print(rejection_sampling(X, y))
