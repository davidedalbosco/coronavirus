#%% Have a look at the progession of the coronavirus in Italy just for fun

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pymc3 as pm
import scipy.stats as stats
import csv
import dateutil.parser
from matplotlib import rcParams

params = {'legend.fontsize' : 20,
          'axes.labelsize'  : 20, 
          'axes.titlesize'  : 24,
          'xtick.labelsize' : 14,
          'ytick.labelsize' : 14,
          'lines.linewidth' : 3,
          'lines.markersize': 12}

rcParams.update(params)

#%% Import the data
# File to read the file officially given by the Italian goverment on the page 
# https://github.com/pcm-dpc/COVID-19/tree/master/dati-andamento-nazionale

with open('data.txt') as f:
    data = list(csv.reader(f))
        
keys = data[0]
values = []
for i in range(len(keys)):
    numbers = []
    if keys[i] == 'data':
        for j in range(1, len(data)):
            numbers.append(dateutil.parser.parse(data[j][i]))
    elif keys[i] == 'stato':
        for j in range(1, len(data)):
            numbers.append(data[j][i])
    else:
        for j in range(1, len(data)):
            numbers.append(int(data[j][i]))
    values.append(numbers)
    
data_dict = dict(zip(keys, values))

cases = data_dict['totale_casi']
    
days = np.arange(0, len(cases))
lim_time = 50
time = np.linspace(0, lim_time, 1000)   # More dense time array to plot the fits
    
# Plot the data
plt.figure()
plt.plot(days, cases, linewidth=False, marker='o')
plt.grid()
plt.xlabel('Days from %s' % data_dict['data'][0])
plt.ylabel('Total number of cases')
plt.title('Data')

# Exponential fit: weights proportional to y of the fit. See 
# https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly

popt_exp, cov_exp = np.polyfit(days, np.log(cases), 1, cov=True)
popt_exp_w, cov_exp_w = np.polyfit(days, np.log(cases), 1, cov=True, w=cases)

plt.figure()
plt.plot(days, cases, linewidth=False, marker='o', label='Data')
plt.plot(time, np.exp(time*popt_exp[0] + popt_exp[1]), 
         label='Exponential fit (bias towards small values)')
plt.plot(time, np.exp(time*popt_exp_w[0] + popt_exp_w[1]), 
         label='Exponential fit (less bias)')
plt.grid()
plt.legend()
plt.title('Fits of the data')


# Fit with a logistic function
def logistic_func(x, L, k, x0):
   return L/(1+np.exp(-k*(x-x0)))

popt, pcov = curve_fit(logistic_func, days, cases, 
                       bounds=([1e4, 0.2, 1], [1e6, 0.3, 100]))

# Standard deviation error on parameters
perr = np.sqrt(np.diag(pcov))

# Plot the fit
plt.plot(time, logistic_func(time, *popt), label='Logistic function fit')
min_logistic = np.min(logistic_func(time, *popt))
max_logistic = np.max(logistic_func(time, *popt))
plt.ylim([min_logistic - 0.05*max_logistic, 1.05*max_logistic])

plt.xlabel('Days from %s' % data_dict['data'][0])
plt.ylabel('Total number of cases')

#%% Bayesian model with 2 straight lines in the semilogarithmic plane
# Inspired from https://docs.pymc.io/notebooks/GLM-linear.html

log_cases = np.log(cases)
mu_slope = 0.2
sigma_slope = 0.2

mu_intercept = 5
sigma_intercept = 4

# Build up the model
with pm.Model() as model:
    # Define priors
    slope1 = pm.Normal('slope1', mu=mu_slope, sigma=sigma_slope)
    slope2 = pm.Normal('slope2', mu=mu_slope, sigma=sigma_slope)
    intercetp1 = pm.Normal('intercept1', mu=mu_intercept, sigma=sigma_intercept)
    intercept2 = pm.Normal('intercept2', mu=mu_intercept, sigma=sigma_intercept)
    
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    
    tau = pm.DiscreteUniform('tau', lower=0, upper=len(cases) - 1)
    
    idx = np.arange(len(cases))    # Index
    slope = pm.math.switch(tau > idx, slope1, slope2)
    intercept = pm.math.switch(tau > idx, intercetp1, intercept2)
  
    # Define likelihood
    likelihood = pm.Normal('log_cases', mu=intercept + slope * days,
                        sigma=sigma, observed=log_cases)
    
    step = pm.Metropolis()
    trace = pm.sample(30000, tune=5000, step=step, cores=1)
    
# Results
intercept1_samples = trace['intercept1']
intercept2_samples = trace['intercept2']

slope1_samples = trace['slope1']
slope2_samples = trace['slope2']

tau_samples = trace['tau']

#%% Plot output of analysis

x_slope = np.linspace(mu_slope - 3*sigma_slope, mu_slope + 3*sigma_slope)
x_intercept = np.linspace(mu_intercept - 3*sigma_intercept, 
                          mu_intercept + 3*sigma_intercept)

plt.figure()
ax = plt.subplot(221)
plt.hist(slope1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label='Posterior of first slope', density=True)
plt.plot(x_slope, stats.norm.pdf(x_slope, mu_slope, sigma_slope), 
         label='Prior of first slope')
plt.grid()
plt.xlabel('First slope value')
plt.legend()

ax = plt.subplot(222)

plt.hist(slope2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label='Posterior of second slope', density=True)
plt.plot(x_slope, stats.norm.pdf(x_slope,  mu_slope, sigma_slope), 
         label='Prior of second slope')
plt.grid()
plt.xlabel('Second slope value')
plt.legend()


plt.subplot(223)
plt.hist(intercept1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label='Posterior of first intercept', density=True)
plt.plot(x_intercept, stats.norm.pdf(x_intercept, mu_intercept, sigma_intercept), 
         label='Prior of first intercept')
plt.grid()
plt.xlabel('First intercept value')
plt.legend()


plt.subplot(224)
plt.hist(intercept2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label='Posterior of second intercept', density=True)
plt.plot(x_intercept, stats.norm.pdf(x_intercept, mu_intercept, sigma_intercept), 
         label='Prior of second intercept')
plt.grid()
plt.xlabel('Second intercept value')
plt.legend()
plt.tight_layout()

plt.figure()
# Find unique values and their frequency 
labels, counts = np.unique(tau_samples, return_counts=True)
norm = np.sum(counts)

plt.bar(labels, counts/norm, align='center', width=0.3, color='C2',
        label=r'Posterior of $\tau$')

plt.xticks(np.arange(len(cases)))
plt.legend(loc='upper left')
plt.grid()
plt.xlabel(r'$\tau$ (days from %s)' % data_dict['data'][0])
plt.ylabel('Probability');

#%% Different plot for the posteriors

plt.figure()
ax = plt.subplot(311)
plt.hist(slope1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label='Posterior of first slope', density=True)
plt.hist(slope2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label='Posterior of second slope', density=True)
plt.plot(x_slope, stats.norm.pdf(x_slope,  mu_slope, sigma_slope), 
         label='Slope prior')
plt.grid()
plt.xlabel('Slope value')
plt.ylabel('Probability density')
plt.xlim([0.1, 0.45])
plt.legend()

plt.subplot(312)
plt.hist(intercept1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label='Posterior of first intercept', density=True)
plt.hist(intercept2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label='Posterior of second intercept', density=True)
plt.plot(x_intercept, stats.norm.pdf(x_intercept, mu_intercept, sigma_intercept), 
         label='Intercept prior')
plt.grid()
plt.xlabel('Intercept value')
plt.ylabel('Probability density')
plt.xlim([5, 7])
plt.legend()

plt.subplot(313)
plt.bar(labels, counts/norm, align='center', width=0.3, color='C2',
        label=r'Posterior of $\tau$')

plt.xticks(np.arange(len(cases)))
plt.legend(loc='upper left')
plt.grid()
plt.xlabel(r'$\tau$ (days from %s)' % data_dict['data'][0])
plt.ylabel('Probability')
plt.tight_layout()
plt.show()

#%% Look at how the model compares with the data

N = len(tau_samples)
t_bayes = np.linspace(0, 30, 1000)

slope_bayes = np.zeros(len(t_bayes))
intercept_bayes = np.zeros(len(t_bayes))

for i in range(len(t_bayes)):
    ix = (t_bayes[i] < tau_samples)   # Boolean variable
    slope_bayes[i] = (slope1_samples[ix].sum() + slope2_samples[~ix].sum())/N
    intercept_bayes[i] = (intercept1_samples[ix].sum() + 
                          intercept2_samples[~ix].sum())/N

plt.figure()
plt.plot(days, cases, linewidth=False, marker='o')
plt.plot(t_bayes, np.exp(t_bayes*slope1_samples.mean() + 
                         intercept1_samples.mean()), label='First component', 
         linestyle='--')
    
plt.plot(t_bayes, np.exp(t_bayes*slope_bayes + intercept_bayes), 
         label='Two components model')
plt.grid()
plt.xlabel('Days from %s' % data_dict['data'][0])
plt.ylabel('Total number of cases')
plt.title('Prediction of the Bayesian model')
plt.legend()

# Semilog scale
plt.figure()
plt.semilogy(days, cases, linewidth=False, marker='o')
plt.semilogy(t_bayes, np.exp(t_bayes*slope1_samples.mean() + 
                         intercept1_samples.mean()), label='First component', 
             linestyle='--')
    
plt.plot(t_bayes, np.exp(t_bayes*slope_bayes + intercept_bayes), 
         label='Two components model')
plt.grid(which='both')
plt.grid(which='minor', linestyle=':', linewidth=0.5)
plt.xlabel('Days from %s' % data_dict['data'][0])
plt.ylabel('Total number of cases')
plt.title('Prediction of the Bayesian model')
plt.legend()
