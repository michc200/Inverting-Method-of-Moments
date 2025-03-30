# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:03:45 2021

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import utils
import generate_dataset as gen

np.random.seed(10)

L = 8
N = 1.23 * 10**7
N = int(N)
gamma = 0.2

x = np.random.rand(L) # TODO: maybe replace this with a gaussian function
x = x / np.linalg.norm(x)
# y_clean = utils.generate_micrograph_1d(x, gamma, L, N)
y_clean = gen.create_clean_observation(base_signal=x, observation_length=N, gamma=gamma)

SNR = 0.5
sigma2 = (np.linalg.norm(x) ** 2) / (L * SNR)
y = gen.add_gaussian_noise(signal=y_clean, sigma=np.sqrt(sigma2))

shifts_2nd = utils.shifts_2nd(L)

shifts_3rd = utils.shifts_3rd_reduced(L)


L2 = len(shifts_2nd)
L3 = len(shifts_3rd)
W = utils.calc_W_heuristic(shifts_2nd, shifts_3rd)


gamma0 = 0.18

estimations_mom = []
estimations_gmm = []

x0 = np.random.rand(L) # TODO: maybe replace this with a gaussian function as well - need to check
x0 = (np.linalg.norm(x) ** 2) * x0 / np.linalg.norm(x0)

x_gamma0 = np.concatenate((x0, np.array([gamma0])))
    
sizes = np.array([10**5, 10**6, 10**7])
for sz in sizes:
    sz = int(sz)
    yi = y[:sz]

    ac1_yi = np.mean(yi) 
    
    ac2_yi = gen.get_autocorrelation(signal=yi, length=L2)
    # ac2_yi_old = np.zeros((len(shifts_2nd), )) # TODO: replace with my function, run both and verify that they are the same
    # for (i, shift) in enumerate(shifts_2nd):
    #     ac2_yi_old[i] = utils.ac2(yi, shift)
        
    ac3_yi = gen.get_third_moment(signal=yi, length=L)
    ac3_yi_old = np.zeros((len(shifts_3rd), )) # TODO: replace with my function, run both and verify that they are the same
    for (i, shifts) in enumerate(shifts_3rd):
        ac3_yi_old[i] = utils.ac3(yi, shifts[0], shifts[1])
    
    samplesi = utils.sample(yi, L)
    del yi
    
    estimation_mom = utils.opt(x_gamma0, ac1_yi, ac2_yi, ac3_yi, shifts_2nd, shifts_3rd, sigma2, W)
    estimations_mom.append(estimation_mom)
    
    f_gmm = utils.calc_function_gmm(samplesi, gamma0, x0, shifts_2nd, shifts_3rd, sigma2)
    del samplesi
    cov_f = np.cov(f_gmm)
    del f_gmm
    W_gmm = np.linalg.inv(np.sqrt(N) * cov_f)
    W_gmm = W_gmm
    
    estimation_gmm = utils.opt(x_gamma0, ac1_yi, ac2_yi, ac3_yi, shifts_2nd, shifts_3rd, sigma2, W_gmm)
    estimations_gmm.append(estimation_gmm)
    
errs_mom = [np.linalg.norm(estimations_mom[i].x[:-1] - x) / np.linalg.norm(x) for i in range(3)]
errs_gmm = [np.linalg.norm(estimations_gmm[i].x[:-1] - x) / np.linalg.norm(x) for i in range(3)]
# %% plots
plt.close("all")
with plt.style.context('ggplot'):
    fig = plt.figure()
    plt.plot(x, label='Ground truth', lw=2)
    # TODO: fix the weird inverted graphs
    plt.plot(estimations_mom[0].x[:-1], lw=2, label='Autocorrelation analysis')
    plt.plot(estimations_gmm[0].x[:-1], lw=2, label='Generalized autocorrelation analysis')
    plt.ylim((0, 0.6))
    plt.legend(loc=1, fontsize=6)
    plt.xlim((0, 7))
    fig.tight_layout()
    plt.show()

    fig = plt.figure()
    plt.plot(x, label='Ground truth')
    plt.plot(estimations_mom[1].x[:-1], lw=2, label='Autocorrelation analysis')
    plt.plot(estimations_gmm[1].x[:-1], lw=2, label='Generalized autocorrelation analysis')
    plt.legend()
    plt.ylim((0, 0.6))
    plt.legend(loc=1, fontsize=6)
    plt.xlim((0, 7))
    fig.tight_layout()
    plt.show()
    
    fig = plt.figure()
    plt.plot(x, label='Ground truth')
    plt.plot(estimations_mom[2].x[:-1], lw=2, label='Autocorrelation analysis')
    plt.plot(estimations_gmm[2].x[:-1], lw=2, label='Generalized autocorrelation analysis')
    plt.legend()
    plt.ylim((0, 0.6))
    plt.legend(loc=1, fontsize=6)
    plt.xlim((0, 7))
    fig.tight_layout()
    plt.show()
    
    fig = plt.figure()
    plt.loglog(sizes, errs_mom, 'ro-', lw=2, label='Autocorrelation analysis')
    plt.loglog(sizes, errs_gmm, 'bo--', lw=2, label='Generalized autocorrelation analysis')
    plt.xticks(sizes)
    plt.xlabel('N')
    plt.ylabel('recovery error')
    plt.legend(loc=1, fontsize=6)
    fig.tight_layout()
    plt.show()
    
