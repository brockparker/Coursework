import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

plt.rc('axes', labelsize=14)
plt.rc('figure', titlesize=30)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
fig, ax = plt.subplots(1, 1, figsize=(6,4), layout='tight')
# BP Stylization parameters.

# BP Binomial.

n = 50
p = 0.05
n_samp = 100
n_samp_large = 100000
k = np.arange(0, 15)
# BP Necessary parameters for the binomial distribution. 

ax.hist(st.binom.rvs(n, p, size=n_samp), color = 'darkviolet', alpha=0.5, histtype = 'stepfilled', density = True, bins = k.max()+1, align = 'left', range = (k.min(),k.max()+1), label = '{} Samples'.format(n_samp))
ax.hist(st.binom.rvs(n, p, size=n_samp_large), color = 'gold', alpha=0.5, histtype = 'stepfilled', density = True, bins = k.max()+1, align = 'left', range = (k.min(),k.max()+1), label = '{} Samples'.format(n_samp_large))
# BP Plotting binomial distribution at different number of samplings.

ax.plot(k, st.binom.pmf(k, n, p), color = 'k', label = 'True PDF')
# BP Plotting true binomial pdf.
ax.legend(loc = 'upper right')
ax.set_xlabel('x')
ax.set_ylabel('PDF')
ax.set_title('Binomial Distribution ($n={}$, $p={}$)'.format(n,p))
fig.tight_layout()
###plt.savefig(r'/home/baparker/GitHub/Coursework/ASTR-513/HW/HW2/HW2_2_a.png',dpi = 250)
plt.savefig(r'C:/Users/Brock/Documents/Git/Coursework/ASTR-513/HW/HW2/HW2_2_a.png',dpi = 250)
plt.show()
# BP Plotting parameters.

# BP Poisson.

fig, ax = plt.subplots(1, 1, figsize=(6,4), layout='tight')
# BP Generating plotting axis.

lam = 4
# BP Necessary parameters for the poisson distribution. 

ax.hist(st.poisson.rvs(lam, size=n_samp), color = 'darkviolet', alpha=0.5, histtype = 'stepfilled', density = True, bins = k.max()+1, align = 'left', range = (k.min(),k.max()+1), label = '{} Samples'.format(n_samp))
ax.hist(st.poisson.rvs(lam, size=n_samp_large), color = 'gold', alpha=0.5, histtype = 'stepfilled', density = True, bins = k.max()+1, align = 'left', range = (k.min(),k.max()+1), label = '{} Samples'.format(n_samp_large))
# BP Plotting poisson distribution at different number of samplings.

ax.plot(k, st.poisson.pmf(k, lam), color = 'k', label = 'True PDF')
# BP Plotting true poisson pdf.
ax.legend(loc = 'upper right')
ax.set_xlabel('x')
ax.set_ylabel('PDF')
ax.set_title('Poisson Distribution ($\lambda={}$)'.format(lam))
fig.tight_layout()
###plt.savefig(r'/home/baparker/GitHub/Coursework/ASTR-513/HW/HW2/HW2_2_b.png',dpi = 250)
plt.savefig(r'C:/Users/Brock/Documents/Git/Coursework/ASTR-513/HW/HW2/HW2_2_b.png',dpi = 250)
plt.show()
# BP Plotting parameters.

# BP Gaussian.

fig, ax = plt.subplots(1, 1, figsize=(6,4), layout='tight')
# BP Generating plotting axis.

mu = 3
sig = 3
x = np.linspace(-10, 20, 1000)
# BP Necessary parameters for the gaussain distribution. 

sample = st.norm.rvs(mu, sig, size=n_samp)
sample_large = st.norm.rvs(mu, sig, size=n_samp_large)
# BP Sampling the Gaussian distribution with two different sample numbers.

ax.hist(sample, color = 'darkviolet', alpha=0.5, histtype = 'stepfilled', density = True, bins = 25, label = '{} Samples'.format(n_samp))
ax.hist(sample_large, color = 'gold', alpha=0.5, histtype = 'stepfilled', density = True, bins = 25, label = '{} Samples'.format(n_samp_large))
# BP Plotting gaussian distribution at different number of samplings.

ax.plot(x, st.norm.pdf(x, mu, sig), color = 'k', label = 'True PDF')
# BP Plotting true gaussian pdf.
ax.legend(loc = 'upper right')
ax.set_xlabel('x')
ax.set_ylabel('PDF')
ax.set_title('Gaussian Distribution ($\mu={}$, $\sigma={}$)'.format(mu, sig))
fig.tight_layout()
###plt.savefig(r'/home/baparker/GitHub/Coursework/ASTR-513/HW/HW2/HW2_2_c.png',dpi = 250)
plt.savefig(r'C:/Users/Brock/Documents/Git/Coursework/ASTR-513/HW/HW2/HW2_2_c.png',dpi = 250)
plt.show()
# BP Plotting parameters.

mom = st.moment(sample, moment=[0,1,2,3,4])
mom_large = st.moment(sample_large, moment = [0,1,2,3,4])

print('----Gaussian Distribution----')
print('Small Distribution: mu_0={:.3f}, mu_1={:.3f}, mu_2={:.3f}, mu_3={:.3f}, mu_4={:.3f}'.format(*mom))
print('Large Distribution: mu_0={:.3f}, mu_1={:.3f}, mu_2={:.3f}, mu_3={:.3f}, mu_4={:.3f}'.format(*mom_large))


# BP Exponential.

fig, ax = plt.subplots(1, 1, figsize=(6,4), layout='tight')
# BP Generating plotting axis.

lam = 0.65
x = np.linspace(0, 15, 1000)
# BP Necessary parameters for the exponential distribution.

sample = st.expon.rvs(scale = 1/lam, size=n_samp)
sample_large = st.expon.rvs(scale = 1/lam, size=n_samp_large)
# BP Sampling distribution at two different sizes.

ax.hist(sample, color = 'darkviolet', alpha=0.5, histtype = 'stepfilled', density = True, bins = 25, label = '{} Samples'.format(n_samp))
ax.hist(sample_large, color = 'gold', alpha=0.5, histtype = 'stepfilled', density = True, bins = 25, label = '{} Samples'.format(n_samp_large))
# BP Plotting exponential distribution at different number of samplings.

ax.plot(x, st.expon.pdf(x, scale = 1/lam), color = 'k', label = 'True PDF')
# BP Plotting true exponential pdf.
ax.legend(loc = 'upper right')
ax.set_xlabel('x')
ax.set_ylabel('PDF')
ax.set_title('Exponential Distribution ($\lambda={}$)'.format(lam))
fig.tight_layout()
###plt.savefig(r'/home/baparker/GitHub/Coursework/ASTR-513/HW/HW2/HW2_2_d.png',dpi = 250)
plt.savefig(r'C:/Users/Brock/Documents/Git/Coursework/ASTR-513/HW/HW2/HW2_2_d.png',dpi = 250)
plt.show()
# BP Plotting parameters.

mom = st.moment(sample, moment=[0,1,2,3,4])
mom_large = st.moment(sample_large, moment = [0,1,2,3,4])

print('----Exponential Distribution----')
print('Small Distribution: mu_0={:.3f}, mu_1={:.3f}, mu_2={:.3f}$, mu_3={:.3f}, mu_4={:.3f}'.format(*mom))
print('Large Distribution: mu_0={:.3f}, mu_1={:.3f}, mu_2={:.3f}$, mu_3={:.3f}, mu_4={:.3f}'.format(*mom_large))

# BP Uniform.

fig, ax = plt.subplots(1, 1, figsize=(6,4), layout='tight')
# BP Generating plotting axis.

a = 0
b = 1
x = np.linspace(a, b, 1000)
# BP Necessary parameters for the uniform distribution. 

ax.hist(st.uniform.rvs(size=n_samp), color = 'darkviolet', alpha=0.5, histtype = 'stepfilled', density = True, bins = 25, label = '{} Samples'.format(n_samp))
ax.hist(st.uniform.rvs(size=n_samp_large), color = 'gold', alpha=0.5, histtype = 'stepfilled', density = True, bins = 25, label = '{} Samples'.format(n_samp_large))
# BP Plotting binomial uniform at different number of samplings.

ax.plot(x, st.uniform.pdf(x), color = 'k', label = 'True PDF')
# BP Plotting true uniform pdf.
ax.legend(loc = 'upper right')
ax.set_xlabel('x')
ax.set_ylabel('PDF')
ax.set_title('Uniform Distribution ($a={}$, $b={}$)'.format(a, b))
fig.tight_layout()
###plt.savefig(r'/home/baparker/GitHub/Coursework/ASTR-513/HW/HW2/HW2_2_e.png',dpi = 250)
plt.savefig(r'C:/Users/Brock/Documents/Git/Coursework/ASTR-513/HW/HW2/HW2_2_e.png',dpi = 250)
plt.show()
# BP Plotting parameters.