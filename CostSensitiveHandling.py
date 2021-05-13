import numpy as np

cost_m = [[0.1, 2], [1, 0]]


def stratification_undersample(X, y, per=0.66):
    """Under-sampling.This method was taken and modified from https://github.com/albahnsen/CostSensitiveClassification/blob/master/costcla/sampling/sampling.py
       Parameters
       ----------
           X : array-like of shape = [n_samples, n_features]
               The input samples.
           y : array-like of shape = [n_samples]
               Ground truth (correct) labels.
           per: float, optional (default = 0.5)
               Percentage of the minority class in the under-sampled data
       """
    n_samples = X.shape[0]
    num_y1 = np.where(y >= .5, 1, 0).sum()
    num_y0 = n_samples - num_y1

    filter_rand = np.random.rand(int(num_y1 + num_y0))

    if num_y1 < num_y0:
        num_y0_new = num_y1 * 1.0 / per - num_y1
        num_y0_new_per = num_y0_new * 1.0 / num_y0
        filter_0 = np.logical_and(y < .5, filter_rand <= num_y0_new_per)
        filter_ = np.nonzero(np.logical_or(y >= .5, filter_0))[0]
    else:
        num_y1_new = num_y0 * 1.0 / per - num_y0
        num_y1_new_per = num_y1_new * 1.0 / num_y1
        filter_1 = np.logical_and(y >= .5, filter_rand <= num_y1_new_per)
        filter_ = np.nonzero(np.logical_or(y < .5, filter_1))[0]

    X_u = X[filter_, :]
    y_u = y[filter_]

    return X_u, y_u


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
    cost_mis[y >= 0.5] = cost_mat[y >= 0.5, 1]

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

    # print(stratification_sample(X, y))
    print(example_weighting(y))
    # print(rejection_sampling(X, y))
