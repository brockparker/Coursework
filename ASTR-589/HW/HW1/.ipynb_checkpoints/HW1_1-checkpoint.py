import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt
# BP Necessary imports.

def integrand(t,u):
    top = np.exp(-(t/u))
    bottom = np.exp(1/(t**2)) - 1
    return top / bottom
# BP Defining function to integrate.

I_nu = []
mu = np.linspace(1e-5, 1, 1000)
# BP Creating arrays to loop over.
    
for x in mu:
    result = int.quad(lambda t: integrand(t,x), 1e-4, 10)[0] / x
    I_nu.append(result)
# BP Integrating the inner function for each value of mu and appending to result array.

plt.rc('axes', labelsize=14)
plt.rc('figure', titlesize=30)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
fig, ax = plt.subplots(1, 1, figsize=(6,4), layout='tight')
# BP Stylization parameters.

ax.plot(mu, I_nu, color='deeppink', label='Specific Intensity')
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\frac{{I_{{\nu}}({{\mu}})}}{{A}}$ (Arbitrary Units)')
ax.set_title('Specific Intensity versus Viewing Angle')
ax.tick_params(axis='both', direction='in', which='both')
ax.legend(loc = 'upper left')
fig.tight_layout()
plt.savefig(r'/home/baparker/GitHub/Coursework/ASTR-589/HW/HW 1/HW1_1.png',dpi = 250)
plt.show()
# BP Plotting and labelling.