from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

plt.style.use('ggplot')

from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from functools import partial


# =============================================================================
# Bayesian Data Analysis
# 
# The fundamental objective of Bayesian data analysis is to determine the posterior distribution of the parameters θ based on a measured data set X
# 
# p(θ|X)=p(X|θ)p(θ)p(X)
# 
# where the denominator is the so-called evidence, which is an integral of the posterior probability over all possible parameters
# 
# p(X)=∫dθ∗p(X|θ∗)p(θ∗)
# 
# Here,
# 
#     p(θ|X) is the likelihood,
#     p(θ) is the prior and
#     p(X) is a normalizing constant also known as the evidence or marginal likelihood
# 
# The computational issue is the difficulty of evaluating the integral in the denominator. There are many ways to address this difficulty, inlcuding:
# 
#     In cases with conjugate priors (with conjugate priors, the posterior has the same distribution as the prior), we can get closed form solutions
#     We can use numerical integration
#     We can approximate the functions used to calculate the posterior with simpler functions and show that the resulting approximate posterior is “close” to true posterior (variational Bayes)
#     We can use Monte Carlo methods, of which the most important is Markov Chain Monte Carlo (MCMC)
# 
# Motivating example
# 
# We will use the toy example of estimating the bias of a coin given a sample consisting of n tosses to illustrate a few of the approaches.
# Analytical solution
# 
# If we use a beta distribution as the prior, then the posterior distribution has a closed form solution. This is shown in the example below. Some general points:
# 
#     We need to choose a prior distribtution family (i.e. the beta here) as well as its parameters (here a=10, b=10)
#     The prior distribution may be relatively uninformative (i.e. more flat) or informative (i.e. more peaked)
#     The posterior depends on both the prior and the data
#         As the amount of data becomes large, the posterior approximates the MLE
#         An informative prior takes more data to shift than an uninformative one
#     Of course, it is also important the model used (i.e. the likelihood) is appropriate for the fitting the data
#     The mode of the posterior distribution is known as the maximum a posteriori (MAP) estimate (cf MLE which is the mode of the likelihood)
# 
# =============================================================================

n = 1000
h = 640
p = h/n
rv = st.binom(n, p)
mu = rv.mean()

a, b = 200, 200
prior = st.beta(a, b)
post = st.beta(h+a, n-h+b)
ci = post.interval(0.95)

thetas = np.linspace(0, 1, 200)
plt.figure(figsize=(12, 9))
plt.style.use('ggplot')
plt.plot(thetas, prior.pdf(thetas), label='Prior', c='blue')
plt.plot(thetas, post.pdf(thetas), label='Posterior', c='red')
plt.plot(thetas, n*st.binom(n, thetas).pmf(h), label='Likelihood', c='green')
plt.axvline((h+a-1)/(n+a+b-2), c='red', linestyle='dashed', alpha=0.4, label='MAP')
plt.axvline(mu/n, c='green', linestyle='dashed', alpha=0.4, label='MLE')
plt.xlim([0, 1])
plt.axhline(0.3, ci[0], ci[1], c='black', linewidth=2, label='95% CI');
plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel('Density', fontsize=16)
plt.legend();

# =============================================================================
# MCMC
# 
# This lecture will only cover the basic ideas of MCMC in the most common variant - Metropolis-Hastings. All code will be built from the ground up to ilustrate what is involved in fitting an MCMC model, but only toy examples will be shown since the goal is conceptual understanding.
# 
# In Bayesian statistics, we want to estiamte the posterior distribution, but this is often intractable due to the high-dimensional integral in the denominator (marginal likelihood). A few other ideas we have encountered that are also relevant here are Monte Carlo integration with inddependent samples and the use of proposal distributions (e.g. rejection and importance sampling). As we have seen from the Monte Carlo inttegration lectures, we can approximate the posterior p(θ|X) if we can somehow draw many samples that come from the posterior distribution. With vanilla Monte Carlo integration, we need the samples to be independent draws from the posterior distribution, which is a problem if we do not actually know what the posterior distribution is .
# 
# With MCMC, we draw samples from a (simple) proposal distribution so that each draw depends only on the state of the previous draw (i.e. the samples form a Markov chain). Under certain condiitons, the Markov chain will have a unique stationary distribution. In addition, not all samples are used - instead we set up acceptance criteria for each draw based on comparing successive states with respect to a target distribution that enusre that the stationary distribution is the posterior distribution of interest. The nice thing is that this target distribution only needs to be proportional to the posterior distribution, which means we don’t need to evaluate the potentially intractable marginal likelihood, which is just a normalizing constant. We can find such a target distribution easily, since posterior∝likelihood×prior. After some time, the Markov chain of accepted draws will converge to the staionary distribution, and we can use those samples as (correlated) draws from the posterior distribution, and find functions of the posterior distribution in the same way as for vanilla Monte Carlo integration.
# 
# There are several flavors of MCMC, but the simplest to understand is the Metropolis-Hastings random walk algorithm, and we will start there.
# 
# To carry out the Metropolis-Hastings algorithm, we need to draw random samples from the folllowing distributions
# 
#     the standard uniform distribution
#     a proposal distriution p(x) that we choose to be N(0,σ)
#     the target distribution g(x) which is proportional to the posterior probability
# 
# Given an initial guess for θ with positive probability of being drawn, the Metropolis-Hastings algorithm proceeds as follows
# 
#     Choose a new proposed value θp such that θp=θ+Δθ where Δθ∼N(0,σ)
# 
#     Caluculate the ratio
#     ρ=g(θp|X))g(θ|X)
# 
# where g is the posterior probability.
# 
#     If the proposal distribution is not symmetrical, we need to weigh the accceptanc probablity to maintain detailed balance (reversibilty) of the stationary distribution, and instead calculate
# 
# ρ=g(θp|X))p(θ|θp)g(θ|X)p(θ)
# 
#     If ρ≥1 then set θ=θp
# 
#     If ρ<1, then set θ=θp with probability ρ, otherwise set θ=θ (this is where we use the standard uniform distribution)
# 
#     Repeat the earlier steps
# 
# After some number of iterations k, the samples θk+1,θk+2,… will be samples from the posterior distributions. Here are initial concepts to help your intuition about why this is so:
# 
#     We accept a proposed move to θk+1 whenever the density of the (unnormalzied) target distribution at θk+1 is larger than the value of θk - so θ will more often be found in places where the target distribution is denser
#     If this was all we accepted, θ would get stuck at a local mode of the target distribution, so we also accept occasional moves to lower density regions - it turns out that the correct probability of doing so is given by the ratio ρ
#     The acceptance criteria only looks at ratios of the target distribution, so the denominator cancels out and does not matter - that is why we only need samples from a distribution proprotional to the posterior distribution
#     So, θ will be expected to bounce around in such a way that its spends its time in places proportional to the density of the posterior distribution - that is, θ is a draw from the posterior distribution.
# 
# Additional notes:
# 
# Different propsoal distributions can be used for Metropolis-Hastings:
# 
#     The independence sampler uses a proposal distribtuion that is independent of the current value of θ. In this case the propsoal distribution needs to be similar to the posterior distirbution for efficiency, while ensuring that the acceptance ratio is bounded in the tail region of the posterior.
#     The random walk sampler (used in this example) takes a random step centered at the current value of θ - efficiency is a trade-off between small step size with high probability of acceptance and large step sizes with low probaiity of acceptance. Note that the random walk may take a long time to traverse narrow regions of the probabilty distribution. Changing the step size (e.g. scaling Σ for a multivariate normal proposal distribution) so that a target proportion of proposals are accepted is known as tuning.
#     Much research is being conducted on different proposal distributions for efficient sampling of the posterior distribution.
# 
# We will first see a numerical example and then try to understand why it works.
# 
# =============================================================================

def target(lik, prior, n, h, theta):
    if theta < 0 or theta > 1:
        return 0
    else:
        return lik(n, theta).pmf(h)*prior.pdf(theta)

n = 10
h = 9
a = 100
b = 100
lik = st.binom
prior = st.beta(a, b)
sigma = 0.01 # width of the proposal distribution, determines the step-size

naccept = 0
theta = 0.1 # starting value of the chain
niters = 1000
samples = np.zeros(niters+1)
samples[0] = theta
for i in range(niters):
    theta_p = theta + st.norm(0, sigma).rvs()
    rho = min(1, target(lik, prior, n, h, theta_p)/target(lik, prior, n, h, theta ))
    u = np.random.uniform()
    if u < rho:
        naccept += 1
        theta = theta_p
    samples[i+1] = theta
nmcmc = len(samples)//2
print("Efficiency = ", naccept/niters)

post = st.beta(h+a, n-h+b)

plt.figure(figsize=(12, 9))
plt.hist(samples[nmcmc:], 40, histtype='step', density=True, linewidth=1, label='Distribution of posterior samples');
plt.hist(prior.rvs(nmcmc), 40, histtype='step', density=True, linewidth=1, label='Distribution of prior samples');
plt.plot(thetas, post.pdf(thetas), c='red', linestyle='--', alpha=0.5, label='True posterior')
plt.xlim([0,1]);
plt.legend(loc='best');

# =============================================================================
# Exercises:
# 
# 1) Change the values for the prior distribution (a, b) and for the outcome of the experiment h. Explain qualitatively how the true posterior shifts.
# 
# 2) Create a situation where the "distribution of the posterior samples" does not recover the "true posterior"
# 
# 3) Create a situation where the efficiency is extremely low, but the results (distribution of posterior matching the true posterior) are still very good.
# 
# Trace plots are often used to informally assess for stochastic convergence. Rigorous demonstration of convergence is an unsolved problem, but simple ideas such as running mutliple chains and checking that they are converging to similar distribtions are often employed in practice.
# 
# =============================================================================

def mh_coin(niters, n, h, theta, lik, prior, sigma):
    samples = [theta]
    while len(samples) < niters:
        theta_p = theta + st.norm(0, sigma).rvs()
        rho = min(1, target(lik, prior, n, h, theta_p)/target(lik, prior, n, h, theta ))
        u = np.random.uniform()
        if u < rho:
            theta = theta_p
        samples.append(theta)
    return samples

n = 100
h = 61
lik = st.binom
prior = st.beta(a, b)
sigma = 0.005
niters = 1000

sampless = [mh_coin(niters, n, h, theta, lik, prior, sigma) for theta in np.arange(0.1, 1, 0.2)]

# Convergence of multiple chains

for samples in sampless:
    plt.plot(samples, '-o')
plt.xlim([0, niters])
plt.ylim([0, 1]);

# =============================================================================
# There are two main ideas - first that the samples generated by MCMC constitute a Markov chain, and that this Markov chain has a unique stationary distribution that is always reached if we generate a very large number of samples. The second idea is to show that this stationary distribution is exactly the posterior distribution that we are looking for.
# Exercises:
# 
# 1) Explain the Figure above. What is shown on the axes? What is the behavior at the beginning?
# 
# 2) How does the Figure change when changing sigma and/or niters?
# Gibbs sampler
# 
# Suppose we have a vector of parameters θ=(θ1,θ2,…,θk) and we want to estimate the joint posterior distribution p(θ|X). Suppose we can find and draw random samples from all the conditional distributions
# 
# p(θ1|θ2,…θk,X)
# p(θ2|θ1,…θk,X)
# …
# p(θk|θ1,θ2,…,X)
# 
# With Gibbs sampling, the Markov chain is constructed by sampling from the conditional distribution for each parameter θi in turn, treating all other parameters as observed. When we have finished iterating over all parameters, we are said to have completed one cycle of the Gibbs sampler. Where it is difficult to sample from a conditional distribution, we can sample using a Metropolis-Hastings algorithm instead - this is known as Metropolis wihtin Gibbs.
# 
# Gibbs sampling is a type of random walk thorugh parameter space, and hence can be thought of as a Metroplish-Hastings algorithm with a special proposal distribtion. At each iteration in the cycle, we are drawing a proposal for a new value of a particular parameter, where the propsal distribution is the conditional posterior probability of that parameter. This means that the propsosal move is always accepted. Hence, if we can draw ssamples from the ocnditional distributions, Gibbs sampling can be much more efficient than regular Metropolis-Hastings.
# Advantages of Gibbs sampling
# 
#     No need to tune proposal distribution
#     Proposals are always accepted
# 
# Disadvantages of Gibbs sampling
# 
#     Need to be able to derive conditional probability distributions need to be able to draw random samples from contitional probability distributions
#     Can be very slow if paramters are coorelated becauce you cannot take “diagonal” steps (draw picture to illustrate)
# 
# Motivating example
# 
# We will use the toy example of estimating the bias of two coins given sample pairs (z1,n1) and (z2,n2) where zi is the number of heads in ni tosses for coin i.
# 
# =============================================================================

def bern(theta, z, N):
    """Bernoulli likelihood with N trials and z successes."""
    return np.clip(theta**z * (1-theta)**(N-z), 0, 1)

def bern2(theta1, theta2, z1, z2, N1, N2):
    """Bernoulli likelihood with N trials and z successes."""
    return bern(theta1, z1, N1) * bern(theta2, z2, N2)

def make_thetas(xmin, xmax, n):
    xs = np.linspace(xmin, xmax, n)
    widths =(xs[1:] - xs[:-1])/2.0
    thetas = xs[:-1]+ widths
    return thetas

def make_plots(X, Y, prior, likelihood, posterior, projection=None):
    fig, ax = plt.subplots(1,3, subplot_kw=dict(projection=projection), figsize=(12,3))
    if projection == '3d':
        ax[0].plot_surface(X, Y, prior, alpha=0.3, cmap=plt.cm.jet)
        ax[1].plot_surface(X, Y, likelihood, alpha=0.3, cmap=plt.cm.jet)
        ax[2].plot_surface(X, Y, posterior, alpha=0.3, cmap=plt.cm.jet)
    else:
        ax[0].contour(X, Y, prior)
        ax[1].contour(X, Y, likelihood)
        ax[2].contour(X, Y, posterior)
    ax[0].set_title('Prior')
    ax[1].set_title('Likelihood')
    ax[2].set_title('Posterior')
    plt.tight_layout()

thetas1 = make_thetas(0, 1, 101)
thetas2 = make_thetas(0, 1, 101)
X, Y = np.meshgrid(thetas1, thetas2)

a = 2
b = 3

z1 = 11
N1 = 14
z2 = 7
N2 = 14

prior = stats.beta(a, b).pdf(X) * stats.beta(a, b).pdf(Y)
likelihood = bern2(X, Y, z1, z2, N1, N2)
posterior = stats.beta(a + z1, b + N1 - z1).pdf(X) * stats.beta(a + z2, b + N2 - z2).pdf(Y)
make_plots(X, Y, prior, likelihood, posterior)
make_plots(X, Y, prior, likelihood, posterior, projection='3d')

a = 2
b = 3

z1 = 11
N1 = 14
z2 = 7
N2 = 14

prior = lambda theta1, theta2: stats.beta(a, b).pdf(theta1) * stats.beta(a, b).pdf(theta2)
lik = partial(bern2, z1=z1, z2=z2, N1=N1, N2=N2)
target = lambda theta1, theta2: prior(theta1, theta2) * lik(theta1, theta2)

theta = np.array([0.5, 0.5])
niters = 10000
burnin = 500
sigma = np.diag([0.2,0.2])

thetas = np.zeros((niters-burnin, 2), np.float64)
for i in range(niters):
    new_theta = stats.multivariate_normal(theta, sigma).rvs()
    p = min(target(*new_theta)/target(*theta), 1)
    if np.random.rand() < p:
        theta = new_theta
    if i >= burnin:
        thetas[i-burnin] = theta

kde = stats.gaussian_kde(thetas.T)
XY = np.vstack([X.ravel(), Y.ravel()])
posterior_metroplis = kde(XY).reshape(X.shape)
make_plots(X, Y, prior(X, Y), lik(X, Y), posterior_metroplis)
make_plots(X, Y, prior(X, Y), lik(X, Y), posterior_metroplis, projection='3d')

a = 2
b = 3

z1 = 11
N1 = 14
z2 = 7
N2 = 14

prior = lambda theta1, theta2: stats.beta(a, b).pdf(theta1) * stats.beta(a, b).pdf(theta2)
lik = partial(bern2, z1=z1, z2=z2, N1=N1, N2=N2)
target = lambda theta1, theta2: prior(theta1, theta2) * lik(theta1, theta2)

theta = np.array([0.5, 0.5])
niters = 10000
burnin = 500
sigma = np.diag([0.2,0.2])

thetas = np.zeros((niters-burnin,2), np.float64)
for i in range(niters):
    theta = [stats.beta(a + z1, b + N1 - z1).rvs(), theta[1]]
    theta = [theta[0], stats.beta(a + z2, b + N2 - z2).rvs()]

    if i >= burnin:
        thetas[i-burnin] = theta

kde = stats.gaussian_kde(thetas.T)
XY = np.vstack([X.ravel(), Y.ravel()])
posterior_gibbs = kde(XY).reshape(X.shape)
make_plots(X, Y, prior(X, Y), lik(X, Y), posterior_gibbs)
make_plots(X, Y, prior(X, Y), lik(X, Y), posterior_gibbs, projection='3d')

# =============================================================================
# Exercise
# 
# Write your own Metropolis Hastings (and Gibbs) sampler. Feel free to use some of the code above:
# 
#     Choose a bivariate normal (Gaussian) distribution as the likelihood of the 2-dimensional data vector p(D|θ1,θ2)
# 
# p(D|θ1,θ2)∝exp(−12[(D−M)tC−1(D−M)]χ2(θ1,θ2))
# 
#     Assume θ1 ranges from 0.1-0.5, and θ2 ranges from 0.6-1.0. The true values are θ1=0.3 and θ2=0.8
# 
#     Assume a simple mapping of the underlying parameters θ1, θ2 onto the data vector D=(d1,d2), e.g. the most simple mapping is d1=θ1 and d2=θ2. Note that usually the mapping of underlying parameters onto data is a complex function involving ODEs, PDEs, intergrals, interpolation routines, differentiation, etc. More importantly, each data point is usually a function of all underlying parameters; there is not a 1-1 mapping of a parameter onto 1 data point (although there are transformations on the data vector that can achieve this).
# 
#     Define a Gaussian prior for your parameters. Choose a very large (co)variance to begin with (trying to get an uninformative prior). Later choose more informative priors.
# 
#     Define a (co)variance matrix in data space, and play with the values of N and ρ (you can set both to zero at first)
#     [d21+Nρρd22+N]
# 
#     With all this information, generate a data vector D at the true values θ1 and θ2 and then sample θ1 and θ2 (using your model definition and your covariance matrix) based on a Metropolis Hastings (and Gibbs) to estimate the posterior probability.
# 
# =============================================================================







