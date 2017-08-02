#based on the code of https://github.com/keyonvafa/DeepGP



from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import value_and_grad
from scipy.optimize import minimize


def build_tanh_dataset():
    n_data = 120
    X = np.linspace(-6, 6, n_data)
    y = np.tanh(X)
    X = X.reshape((len(X),1))
    return X,y

if __name__ == '__main__':

    x, y = build_tanh_dataset()

    def unpack_kernel_params(params):
        
        mean        = params[0]
        cov_params  = params[2:]
        noise_scale = np.exp(params[1]) + 0.001
        return mean, cov_params, noise_scale

    def rbf_covariance(kernel_params, x, xp):
        output_scale = np.exp(kernel_params[0])
        lengthscales = np.exp(kernel_params[1:])

        sqdist = (np.sum(x/lengthscales, 1).reshape(-1,1) - np.sum(xp/lengthscales,1))**2
        return output_scale * np.exp(-.5 * sqdist)

    def minus_log_marginal_likelihood(params):
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        cov_y_y = rbf_covariance(cov_params, x, x) + noise_scale * np.eye(len(y))
        prior_mean = mean * np.ones(len(y))
        mllk = -mvn.logpdf(y, prior_mean, cov_y_y+1e-6*np.eye(len(cov_y_y)))
        return mllk

    def predict(params, x, y, xstar):
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        

        cov_f_f = rbf_covariance(cov_params, xstar, xstar)
        cov_y_f = rbf_covariance(cov_params, x, xstar)
        cov_y_y = rbf_covariance(cov_params, x, x) + noise_scale * np.eye(len(y))
        pred_mean = mean +   np.dot(solve(cov_y_y, cov_y_f).T, y - mean)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov 

    # Set up figure.
    #fig = plt.figure(figsize=(12,8), facecolor='white')
    #ax = fig.add_subplot(111, frameon=False)
    #plt.show(block=False)

    def callback(params):
        #plt.cla()

        mean        = params[0]
        cov_params  = params[2:]
        noise_scale = np.exp(params[1])

        print("params ",mean, cov_params, params[1])
        print(minus_log_marginal_likelihood(params))

        # Show posterior marginals.
        plot_xs = np.reshape(np.linspace(-7, 7, 300), (300,1))
        pred_mean, pred_cov = predict(params, x, y, plot_xs)
        marg_std = np.sqrt(np.diag(pred_cov))
        #ax.plot(plot_xs, pred_mean, 'b')
        #ax.fill(np.concatenate([plot_xs, plot_xs[::-1]]),
        #        np.concatenate([pred_mean - 1.96 * marg_std,
        #                      (pred_mean + 1.96 * marg_std)[::-1]]),
        #        alpha=.15, fc='Blue', ec='None')

        # Show samples from posterior.
        rs = npr.RandomState(0)
        sampled_funcs = rs.multivariate_normal(pred_mean, pred_cov, size=10)
        #ax.plot(plot_xs, sampled_funcs.T)

        #ax.plot(x, y, 'kx')
        #ax.set_ylim([-1.5, 1.5])
        #ax.set_xticks([])
        #ax.set_yticks([])
        #plt.draw()
        #plt.pause(1.0/60.0)

    # Initialize covariance parameters
    rs = npr.RandomState(0)
    num_params = 4
    init_params = 0.1 * rs.randn(num_params)

    print("Optimizing covariance parameters...")
    cov_params = minimize(value_and_grad(minus_log_marginal_likelihood), init_params, jac=True,
                          method='CG', callback=callback)

    print(cov_params['x'])
    # plt.pause(10.0)

