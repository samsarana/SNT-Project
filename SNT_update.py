import numpy as np
from math import log, exp
from numbers import Real
from random import choice as random_choice
import copy
import yaml
import csv
from functools import partial
import matplotlib.pyplot as plt
import scipy.stats as stats
from cycler import cycler
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import theano
import pymc3 as pm
from SNT import fit_beta, fit_lognorm, fit_norm, myStatsLognorm

# Inputs
propagated = np.load("propagated.npy")
#print(propagated.item())

file_handle = open('my_priors.yml')
my_priors = yaml.safe_load(file_handle)
file_handle.close()

def update_lognorm_with_lognorm(mu_T, sd_T, mu_M, sd_M):
	'''See p. 4 of https://oxpr.io/blog/2017/5/20/bayesian-updating-for-lognormal-distributions'''
	# print('here!')
	# print(evidence_params)
	E_mu = (mu_T*sd_T**-2 + mu_M*(sd_M - sd_T)**-2) / (sd_T**-2 + (sd_M - sd_T)**-2)
	sigma = (sd_T**-2 + (sd_M + sd_T)**-2)**-0.5
	return E_mu, sigma

for career, priors in my_priors.items():
	if priors['Personal_fit']: # i.e. if I've entered data for/am thinking of working on this cause area
		personal_fit = fit_lognorm(priors['Personal_fit'])
		for bucket in ['LT', 'STH', 'STA']:
			if priors[bucket]: # i.e. if I've entered data for my prior effectiveness according to this bucket (I care about this bucket)
				mu_T, sd_T = fit_lognorm(priors[bucket]) # lognorm fits well because product of 3 independent RVs
				if career == 'Earn-to-give':
					donation_target, evidence = propagated.item()[career][bucket]
				else:
					evidence = propagated.item()[career][bucket]['Direct'] # propagated is an .npy object which has to be loaded with .item()
				personal_evidence = np.random.lognormal(*personal_fit, len(evidence)) * evidence
				s, loc, scale = stats.lognorm.fit(personal_evidence) # fit lognormal to personal fit-adjusted evidence to then do an analytical update
				mu_M = log(scale)
				sd_M = s
				assert(abs(loc) < 1e-4) # loc is an idiosyncratic scipy.stats parameter, that determines the x-value where the lognormal "takes off". I want this to be ~0 s.t. scale and s can give me standard lognormal params
				mu_P, sd_P = update_lognorm_with_lognorm(mu_T, sd_T, mu_M, sd_M)
				fig, ax = plt.subplots()
				plt.xscale('log')
				min_x = min([myStatsLognorm.ppf(1e-6, *params) for params in [(mu_T, sd_T), (mu_M, sd_M), (mu_P, sd_P)]])
				max_x = max([myStatsLognorm.ppf(0.999, *params) for params in [(mu_T, sd_T), (mu_M, sd_M), (mu_P, sd_P)]])
				x = np.geomspace(min_x, max_x, 1e4)
				ax.plot(x, myStatsLognorm.pdf(x, mu_T, sd_T), 'b', alpha=0.6, label='Prior')
				ax.plot(x, myStatsLognorm.pdf(x, mu_M, sd_M), 'r', alpha=0.6, label='Personal fit adjusted evidence')
				ax.plot(x, myStatsLognorm.pdf(x, mu_P, sd_P), 'g', alpha=0.6, label='Posterior')
				if career == 'Earn-to-give':
					ax.set_title('Update for {} career in {} bucket, donating to {}'.format(career, bucket, donation_target))
				else:
					ax.set_title('Updating for {} career in {} bucket'.format(career, bucket))
				x_labels = {'LT': "Reduction in p(extinction) per extra person",
							'STH': "Person-affecting human QALYs saved per extra person",
							'STA': "Near-term animal-inclusive HEWLAYs saved per extra person"}
				ax.set_xlabel(x_labels[bucket])
				ax.set_ylabel("Probablity density")
				ax.legend()
plt.show()