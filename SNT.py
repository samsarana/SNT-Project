import numpy as np
from math import log, exp
from numbers import Real
from random import choice as random_choice
import itertools
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
import sys

# Globals: Z-scores for different confidence intervals; number of samples for MC methods; moral parameters for animal bucket
Z_50 = 0.67449
Z_80 = 1.28
Z_90 = 1.645
NUM_SAMPLES_MCMC_AGGREGATION = 2000
NUM_SAMPLES_MC_PROPAGATION = 10000
SENTIENCE_ADJ = 0.0381
WELLBEING_ADJ = -5
LOWER_BOUND_DONATIONS = 50e3 * 20 # donating an average of $50,000/year for 20 years
UPPER_BOUND_DONATIONS = 250e3 * 25 # donating an average of $250,000/year for 25 years

# Inputs
file_handle = open("SNT_estimates.yml")
estimates = yaml.safe_load(file_handle)
file_handle.close()

# Fit distributions to percentiles
# Import functions from R for fitting when >2 percentiles given (least squares)
rriskDistributions = importr('rriskDistributions')
fit_norm_r = robjects.r['get.norm.par']
fit_lognorm_r = robjects.r['get.lnorm.par']
fit_beta_r = robjects.r['get.beta.par']

class myStatsLognorm:
	'''Wrapper for the scipy.stats.lognorm object
	   to allow it to be called with (mu, sd) instead of the
	   quirky way scipy parameterise it
	'''
	def pdf(x, mu, sd):
		return stats.lognorm.pdf(x,scale=exp(mu), s=sd)
	def ppf(x, mu, sd):
		return stats.lognorm.ppf(x,scale=exp(mu), s=sd)
	def median(mu, sd):
		return stats.lognorm.median(scale=exp(mu), s=sd)


def my_random_lognormal(mu, sd, sign, num_samples):
	'''Like np.random.lognormal except will return -1*realised variable
	   if sign == -1
	'''
	return sign * np.random.lognormal(mu, sd, num_samples)

# Functions for fitting distributions
def fit_norm(estimate):
	'''Given estimate of 10th, 90th percentiles and optional median,
	   fits a normal distribution
	'''
	if isinstance(estimate, Real): # no uncertainty, just return point estimate instead of dist parameters
		return estimate

	tenth, *median, ninetieth = estimate
	if not median:
		mu = 0.5 * (tenth + ninetieth)
		sigma = (ninetieth - tenth) / (2 * Z_80)
	else:
		median = median[0]
		p = robjects.FloatVector([0.1, 0.5, 0.9])
		q = robjects.FloatVector([tenth, median, ninetieth])
		dist_as_str = fit_norm_r(p=p, q=q, show_output=False, plot=False).r_repr()
		if dist_as_str == 'NA':
			print(estimate)
			raise ValueError('Distribtion could not be fitted to parameters!')
		params_as_str = dist_as_str.strip('c(mean = )').partition(', sd = ')
		mu, sigma = [float(params_as_str[i]) for i in (0,2)]
	return mu, sigma

def fit_lognorm(estimate):
	'''Given estimate of 10th, 90th percentiles and optional median,
	   fits a lognormal distribution
	'''
	if isinstance(estimate, Real): # no uncertainty
		return estimate
	sign = +1
	tenth, *median, ninetieth = estimate
	if not median:
		if tenth < 0:
			if not ninetieth < 0:
				raise NotImplementedError("Haven't implemented lognormal-shape dist when support is [-inf, inf]")
			else:
				tenth, ninetieth = abs(ninetieth), abs(tenth)
				sign = -1
		mu = 0.5 * log(tenth * ninetieth)
		sigma = log(ninetieth/tenth) / (2*Z_80)
	else:
		median = median[0]
		if tenth < 0:
			if not ninetieth < 0 and median < 0:
				raise NotImplementedError("Haven't implemented lognormal-shape dist when support is [-inf, inf]")
			else:
				tenth, median, ninetieth = abs(ninetieth), abs(median), abs(tenth)
				sign = -1
		p = robjects.FloatVector([0.1, 0.5, 0.9])
		q = robjects.FloatVector([tenth, median, ninetieth])
		dist_as_str = fit_lognorm_r(p=p, q=q, show_output=False, plot=False).r_repr()
		if dist_as_str == 'NA':
			print(estimate)
			raise ValueError('Distribtion could not be fitted to parameters!')
		params_as_str = dist_as_str.strip('c(meanlog = )').partition(', sdlog = ')
		mu, sigma = [float(params_as_str[i]) for i in (0,2)]
	return sign*mu, sign*sigma # really what I will do here is output the PyMC compatible distribution, and when it's negative, I'll just output (-distribution(mu, sigma))

def fit_beta(estimate):
	'''Given estimate of 10th, 90th percentiles and optional median,
	   fits a beta distribution
	'''
	if isinstance(estimate, Real): # no uncertainty
		return estimate
	tenth, *median, ninetieth = estimate
	if not median:
		p = robjects.FloatVector([0.1, 0.9])
		q = robjects.FloatVector([tenth, ninetieth])
	else:
		median = median[0]
		p = robjects.FloatVector([0.1, 0.5, 0.9])
		q = robjects.FloatVector([tenth, median, ninetieth])
	dist_as_str = fit_beta_r(p=p, q=q, show_output=False, plot=False).r_repr()
	if dist_as_str == 'NA':
		print(estimate)
		raise ValueError('Distribtion could not be fitted to parameters!')
	alpha_str, beta_str = [dist_as_str.partition(', shape2 = ')[i] for i in (0,2)]
	alpha = float(alpha_str.partition('c(shape1 = ')[2])
	beta = float(beta_str.strip(')'))
	return alpha, beta

# Functions for Bayesian updating
def generate_observations(parameters, likelihood_sampler, n):
	'''Given a list of tuples (theta1, theta2) of distribution parameters fitted to
	   confidence intervals from each source, return a generator for n samples
	   split evenly between the distributions
	'''
	observations = iter(())
	for theta1, theta2 in parameters.values():
		itertools.chain(observations, likelihood_sampler(theta1, theta2, size=int(n/len(parameters))))
	return observations

def bayesian_update(likelihood_dist, observations, prior_parameters):
	'''Perform a bayesian update on uniform priors over specified ranges using evidence
	   drawn from likelihood distribution of given type
	   Return mean of posterior values for parameters
	'''
	theta1_lower, theta1_upper, theta2_lower, theta2_upper = prior_parameters
	with pm.Model() as model:
		prior_theta1 = pm.Uniform('prior_theta1', theta1_lower, theta1_upper)
		prior_theta2 = pm.Uniform('prior_theta2', theta2_lower, theta2_upper)
		likelihood = likelihood_dist('likelihood', prior_theta1, prior_theta2, observed=list(observations))
		trace = pm.sample(NUM_SAMPLES_MCMC_AGGREGATION)
		posterior_theta1 = np.mean(trace['prior_theta1'])
		posterior_theta2 = np.mean(trace['prior_theta2'])
		return posterior_theta1, posterior_theta2

# Function for plotting evidence aggregation graphs
def my_plot(distr, evidence, median, posterior, aggregated, cause_area, estimate_type, estimates, log_scale=False, sign=+1):
	pdf, ppf = distr.pdf, distr.ppf
	if log_scale:
		min_x = min([np.log10(ppf(1e-6, *params)) for params in list(evidence.values()) + [median, posterior, aggregated]])
		max_x = max([np.log10(ppf(0.999, *params)) for params in list(evidence.values()) + [median, posterior, aggregated]])
		x = np.logspace(min_x, max_x, 1e4)
	else:
		min_x = min([ppf(1e-6, *params) for params in list(evidence.values()) + [median, posterior, aggregated]])
		max_x = max([ppf(0.999, *params) for params in list(evidence.values()) + [median, posterior, aggregated]])
		x = np.linspace(min_x, max_x, 1e4)
	fig, ax = plt.subplots() # defaults to just 1 graph
	ax.set_prop_cycle(cycler('color', ['b', 'g', 'c', 'y', 'r', 'm', 'k']))
	for source, evidence_params in evidence.items():
		ax.plot(sign*x, pdf(x, *evidence_params), alpha=0.6, label=str(source))
		if not source == 'PS21 Report': # this report was not from confidence interval
			pcentiles = estimates[cause_area][estimate_type][source]
			if len(pcentiles) == 3: # then we want to plot/check fit (if just 2 pcentiles, fit will be perfect, we don't care)
				for pcentile in pcentiles:
					ax.plot(pcentile, 0, 'rx')
				ax.plot(distr.ppf(0.1, *evidence_params), 0, 'go')
				ax.plot(distr.ppf(0.5, *evidence_params), 0, 'go')
				ax.plot(distr.ppf(0.9, *evidence_params), 0, 'go')
	ax.plot(sign*x, pdf(x, *median), alpha=0.6, label='Median of evidence parameters')
	ax.plot(sign*x, pdf(x, *posterior), alpha=0.6, label='Posterior')
	ax.plot(sign*x, pdf(x, *aggregated), alpha=0.6, label='Average of median and posterior')
	x_labels = {'scale_LT': 'Total X-risk associated with {} adjusting for the work that has already been done',
			'scale_ST-Human': 'Presentist person-affecting human QALYs saved if remainder of {} solved',
			'scale_ST-Animal': 'Farmed animal years saved in next 100 years if remainder of {} solved',
			'crowdedness_people': 'Full-time staff',
			'crowdedness_dollars': 'Annual spending',
			'tractability_people': 'Δ fraction remaining problem solved by doubling full-time staff',
			'tractability_dollars': 'Δ fraction remaining problem solved by doubling annual spending'}
	ax.set_xlabel(x_labels[estimate_type].format(cause_area))
	ax.set_ylabel('probability density')
	# Chosen a global y scale to go up to 3*minimum of (probability corresponding to median value for each pdf) (kinda hacky)
	#y_max = 8*min([pdf(distr.median(*params), *params) for params in list(evidence.values()) + [median, posterior, aggregated]])
	y_max = 2*max([pdf(distr.median(*params), *params) for params in list(evidence.values()) + [median, posterior, aggregated]]) # changed this
	ax.set_ylim(bottom=0, top=y_max)
	ax.set_title("Plot of aggregation of estimates for {} for {}".format(estimate_type, cause_area))
	#fig.canvas.set_window_title('My title')
	ax.legend()
	if log_scale:
		plt.xscale('symlog')
	fig.savefig('Aggregation Graphs/{} for {}.png'.format(estimate_type, cause_area), bbox_inches='tight')
	plt.close(fig)

# Functions for computing SNT using Monte Carlo simulation
def X_risk_effects(reduction_p_extinction_dist_sampler):
	'''Returns estimated presentist persona-affecting human QALYs saved
	   by a given reduction in the proabbility of extinction
	'''
	global_population = 7.6E6
	global_life_expectancy = 70.5
	global_mean_age = 38
	global_average_quality_adjustment = 0.75
	return lambda : reduction_p_extinction_dist_sampler() * global_population * (global_life_expectancy - global_mean_age) * global_average_quality_adjustment

def compute_SNT(scale, crowdedness, tractability):
	'''My way of truncating the normal distribution for crowdedness is not quite
		correct on two fronts. Firstly, fitting percentiles to a truncated normal
		distribution and fitting percentiles to a normal distribution, truncating and
		renormalising is not the same; strictly I should do the former, but it's harder
		and the error seems neglibible compared to the error in my model
		Secondly, I am doing the "truncating and renormalising at the simulation stage
		and to do this I should reject any negative samples unit my sample vector is all
		positive
		But I can't see a way to implement this without cripling numpy's speed
		Before I had: 
		c = [-1]
		while not all(c_i > 0 for c_i in c):
			c = crowdedness()
		But the simulations then took way too long to run
		I think my method approximates this well enough
	 '''
	return scale() * 1 / abs(crowdedness()) * tractability()

if __name__ == '__main__':
	# Fit distributions
	distributions = copy.deepcopy(estimates)
	for cause_area, cause_area_estimates in distributions.items(): # exclude the funding_constraint parameter for now
		del distributions[cause_area]['funding_constraint']
	for cause_area, cause_area_estimates in estimates.items():
		for estimate_type, parameter_estimates in cause_area_estimates.items():
			if parameter_estimates and isinstance(parameter_estimates, dict):
				for source, estimate in parameter_estimates.items():
					dist = None
					print('fitting', cause_area, estimate_type, source)
					if estimate_type.startswith('scale_LT') or estimate_type.startswith('tractability'):
						dist = fit_beta(estimate)
					elif estimate_type.startswith('scale_ST'):
						dist = fit_lognorm(estimate)
					elif estimate_type.startswith('crowdedness'):
						dist = fit_norm(estimate)
					distributions[cause_area][estimate_type][source] = dist

	# extra datum
	distributions['Nuclear']['scale_ST-Human']['PS21 Report'] = (18.52052, 1.033587)

	aggregated = copy.deepcopy(distributions)
	for cause_area, cause_area_dists in distributions.items():
		for estimate_type, parameter_dists in cause_area_dists.items():
			if parameter_dists: # i.e. if we have data for this estimate
				print(cause_area, estimate_type, parameter_dists)
				if estimate_type.startswith('scale_LT') or estimate_type.startswith('tractability'):# and cause_area == 'AIS' and estimate_type.startswith('scale_LT'):
					theta1_list, theta2_list = zip(*parameter_dists.values())
					# Aggregation method 1: calculate median of parameters
					median_thetas = np.median(theta1_list), np.median(theta2_list)
					# Aggregation method 2: Bayesian updating (for each type of likelihood dist)
					alpha_upper = 2 * np.mean(theta1_list)
					beta_upper = 2 * np.mean(theta2_list)
					prior_parameters = (0, alpha_upper, 0, beta_upper)
					observations = generate_observations(parameter_dists, np.random.beta, NUM_SAMPLES_MCMC_AGGREGATION)
					posterior_thetas = bayesian_update(pm.Beta, observations, prior_parameters)
					agg_thetas = np.mean([median_thetas, posterior_thetas], axis=0)
					aggregated[cause_area][estimate_type] = partial(np.random.beta, *agg_thetas, size=NUM_SAMPLES_MC_PROPAGATION)
					my_plot(stats.beta, parameter_dists, median_thetas, posterior_thetas, agg_thetas, cause_area, estimate_type, estimates)
				elif estimate_type.startswith('scale_ST'):# and cause_area == 'Biorisk' 
					sign = +1 # only for scale_ST i hackily make support for lognormal dist to have probabilty mass only over negative support
					if list(parameter_dists.values())[0][1] < 0: # since I have either only +ve or only -ve lognormal distributions, this hack of checking if sigma is -ve (for which lognormal is undefined i.e. I've cooked it up) will do fine for now
						#print(cause_area)
						sign = -1
						for source in parameter_dists.keys():
							parameter_dists[source] = (-1*parameter_dists[source][0], -1*parameter_dists[source][1]) # change back to positive, then change back later. quite hacky
					theta1_list, theta2_list = zip(*parameter_dists.values())
					# Aggregation method 1: calculate median of parameters
					median_thetas = np.median(theta1_list), np.median(theta2_list)
					# Aggregation method 2: Bayesian updating (for each type of likelihood dist)
					mu_upper = 2 * np.mean(theta1_list) # sign * -- I think... should just roll with the positive values until the very end
					sd_upper = 2 * np.mean(theta2_list) # sign *
					prior_parameters = (0, mu_upper, 0, sd_upper)
					observations = generate_observations(parameter_dists, np.random.lognormal, NUM_SAMPLES_MCMC_AGGREGATION)

					posterior_thetas = bayesian_update(pm.Lognormal, observations, prior_parameters)
					agg_thetas = np.mean([median_thetas, posterior_thetas], axis=0)
					aggregated[cause_area][estimate_type] = partial(my_random_lognormal, *agg_thetas, sign, NUM_SAMPLES_MC_PROPAGATION)
					my_plot(myStatsLognorm, parameter_dists, median_thetas, posterior_thetas, agg_thetas, cause_area, estimate_type, estimates, log_scale=True, sign=sign)
				elif estimate_type.startswith('crowdedness'):# and cause_area == 'Health' and estimate_type.startswith('crowdedness_dollars'):
					theta1_list, theta2_list = zip(*parameter_dists.values())
					# Aggregation method 1: calculate median of parameters
					median_thetas = np.median(theta1_list), np.median(theta2_list)
					# Aggregation method 2: Bayesian updating (for each type of likelihood dist)
					mu_upper = 2 * np.mean(theta1_list)
					sd_upper = 2 * np.mean(theta2_list)
					prior_parameters = (0, mu_upper, 0, sd_upper)
					observations = generate_observations(parameter_dists, np.random.normal, NUM_SAMPLES_MCMC_AGGREGATION)
					posterior_thetas = bayesian_update(pm.Normal, observations, prior_parameters)
					agg_thetas = np.mean([median_thetas, posterior_thetas], axis=0)
					aggregated[cause_area][estimate_type] = partial(np.random.normal, *agg_thetas, size=NUM_SAMPLES_MC_PROPAGATION)
					my_plot(stats.norm, parameter_dists, median_thetas, posterior_thetas, agg_thetas, cause_area, estimate_type, estimates)
			else:
				print('error!')
				assert(False)
				#aggregated[cause_area][estimate_type] = lambda : 0

	np.save("aggregated.npy", aggregated)

	propagated = dict()
	mu_donations, sigma_donations = fit_lognorm((LOWER_BOUND_DONATIONS, UPPER_BOUND_DONATIONS))
	total_donations = stats.lognorm.rvs(s=sigma_donations, scale=exp(mu_donations), size=NUM_SAMPLES_MC_PROPAGATION)
	fig_LT, ax_LT = plt.subplots()
	plt.xscale('log')
	fig_STH, ax_STH = plt.subplots()
	plt.xscale('log')
	fig_STA, ax_STA = plt.subplots()
	plt.xscale('symlog', linthreshx=100)
	for cause_area, dists in aggregated.items():
		propagated[cause_area] = dict()
		snt = compute_SNT(dists['scale_LT'], dists['crowdedness_people'], dists['tractability_people'])
		propagated[cause_area]['LT'] = dict()
		propagated[cause_area]['LT']['Direct'] = snt
		ax_LT.hist(snt, bins=np.geomspace(1e-14, 1e-2, 80), label=cause_area, histtype='step')

		# fit beta distribution to funding_constraint ratio
		alpha_funding_constr, beta_funding_constr = fit_beta(estimates[cause_area]['funding_constraint'])
		funding_constr = stats.beta.rvs(a=alpha_funding_constr, b=beta_funding_constr, size=NUM_SAMPLES_MC_PROPAGATION)

		# LT x 1/c dollars x t dollars
		snt = compute_SNT(dists['scale_LT'], dists['crowdedness_dollars'], dists['tractability_dollars']) * total_donations * funding_constr
		propagated[cause_area]['LT']['Earn'] = snt
		
		# (STH + X-effects) x 1/c people x t people
		scale_ST_Human_total = lambda: dists['scale_ST-Human']() + X_risk_effects(dists['scale_LT'])()
		snt = compute_SNT(scale_ST_Human_total, dists['crowdedness_people'], dists['tractability_people'])
		propagated[cause_area]['STH'] = dict()
		propagated[cause_area]['STH']['Direct'] = snt
		ax_STH.hist(snt, bins=np.geomspace(1e-2, 1e10, 100), label=cause_area, histtype='step')
		# (STH + X-effects) x 1/c dollars x t dollars
		snt_per_dollar = compute_SNT(scale_ST_Human_total, dists['crowdedness_dollars'], dists['tractability_dollars'])
		# extra datum from Givewell 2018 G Sheet: Health Cost-effectiveness / QALYs saved per marginal $: lognormal(-3.045886, 0.750309)
		# Give this equal weight as other estimates combined (sample equally from GiveWell and rest)
		if cause_area == 'Health':
			np.append(snt, np.random.lognormal(-3.045886, 0.750309, NUM_SAMPLES_MC_PROPAGATION))
		# fit beta distribution to funding_constraint ratio
		alpha_funding_constr, beta_funding_constr = fit_beta(estimates[cause_area]['funding_constraint'])
		funding_constr = stats.beta.rvs(a=alpha_funding_constr, b=beta_funding_constr, size=NUM_SAMPLES_MC_PROPAGATION)
		snt = snt_per_dollar * total_donations * funding_constr
		propagated[cause_area]['STH']['Earn'] = snt
		
		# (adj*STA + STH) x 1/c people x t people # forget the X-effects here
		scale_ST_Animal_total = lambda: scale_ST_Human_total() - SENTIENCE_ADJ * WELLBEING_ADJ * dists['scale_ST-Animal']() # subtract because SENTIENCE_ADJ is _negative_ 5 
		snt = compute_SNT(scale_ST_Animal_total, dists['crowdedness_people'], dists['tractability_people'])
		propagated[cause_area]['STA'] = dict()
		propagated[cause_area]['STA']['Direct'] = snt
		ax_STA.hist(snt, bins=np.append(np.geomspace(-1e10, -1e-2, 120), np.geomspace(1e-2, 1e18, 120)), label=cause_area, histtype='step')
		# (adj*STA + STH) x 1/c dollars x t dollars
		snt = compute_SNT(scale_ST_Animal_total, dists['crowdedness_dollars'], dists['tractability_dollars']) * total_donations * funding_constr
		propagated[cause_area]['STA']['Earn'] = snt
	# identify top earn-to-give cause and add to plot
	best_earn_to_give = {'LT': {'cause': None, 'impact_mean': 0, 'impact_dist': None},
						'STH': {'cause': None, 'impact_mean': 0, 'impact_dist': None},
						'STA': {'cause': None, 'impact_mean': 0, 'impact_dist': None}}
	for cause_area, CA_results in propagated.items():
		for bucket, bucket_results in CA_results.items():
			mean_impact = np.mean(bucket_results['Earn'])
			if mean_impact > best_earn_to_give[bucket]['impact_mean']:
				best_earn_to_give[bucket]['impact_mean'] = mean_impact
				best_earn_to_give[bucket]['impact_dist'] = bucket_results['Earn']
				best_earn_to_give[bucket]['cause'] = cause_area

	propagated['Earn-to-give'] = dict()
	propagated['Earn-to-give']['LT'] = dict()
	propagated['Earn-to-give']['STH'] = dict()
	propagated['Earn-to-give']['STA'] = dict()

	propagated['Earn-to-give']['LT'][str(best_earn_to_give['LT']['cause'])] = best_earn_to_give['LT']['impact_dist']
	propagated['Earn-to-give']['STH'][str(best_earn_to_give['STH']['cause'])] = best_earn_to_give['STH']['impact_dist']
	propagated['Earn-to-give']['STA'][str(best_earn_to_give['STA']['cause'])] = best_earn_to_give['STA']['impact_dist']
	
	np.save("propagated.npy", propagated)

	ax_LT.hist(best_earn_to_give['LT']['impact_dist'], bins=np.geomspace(1e-14, 1e-2, 80), label='Earn-to-give for {}'.format(best_earn_to_give['LT']['cause']), histtype='step')
	ax_STH.hist(best_earn_to_give['STH']['impact_dist'], bins=np.geomspace(1e-2, 1e10, 100), label='Earn-to-give for {}'.format(best_earn_to_give['STH']['cause']), histtype='step')
	ax_STA.hist(best_earn_to_give['STA']['impact_dist'], bins=np.append(np.geomspace(-1e10, -1e-2, 120), np.geomspace(1e-2, 1e18, 120)), label='Earn-to-give for {}'.format(best_earn_to_give['STA']['cause']), histtype='step')
	ax_LT.set_title("Long-termist bucket")
	ax_LT.set_xlabel("Reduction in X-risk per extra person")
	ax_LT.set_ylabel("Probablity density")
	ax_LT.legend()
	ax_STH.set_title("Short-termist, human-centric bucket")
	ax_STH.set_xlabel("Person-affecting human QALYs saved per extra person")
	ax_STH.set_ylabel("Probablity density")
	ax_STH.legend()
	ax_STA.set_title("Short-termist, animal-inclusive bucket")
	ax_STA.set_xlabel("Near-term animal-inclusive HEWLAYs saved per extra person")
	ax_STA.set_ylabel("Probablity density")
	ax_STA.legend()
	fig_LT.savefig('Distributions over estimated reduction in X-risk per extra person', dpi=400)
	fig_STH.savefig('Distributions over estimated person-affecting human QALYs saved per extra person', bbox_inches='tight', dpi=400)
	fig_STA.savefig('Distributions over estimated near-term animal-inclusive HEWLAYs saved per extra person', bbox_inches='tight', dpi=400)

	with open('results.csv', mode='w', newline='') as res_file:
	    res_writer = csv.writer(res_file)
	    for cause_area, buckets in propagated.items():
	    	for bucket, career_modes in buckets.items():
	    		percentiles = []
	    		if cause_area == 'Earn-to-give':
	    			donation_target, impact = list(career_modes.items())[0]
	    			percentiles = [np.percentile(impact, 5), np.mean(impact), np.percentile(impact, 95), str(donation_target)]
	    		else:
		    		impact = career_modes['Direct']
		    		percentiles = [np.percentile(impact, 5), np.mean(impact), np.percentile(impact, 95)]
	    		res_writer.writerow(percentiles)
	plt.show()