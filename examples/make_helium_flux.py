import matplotlib.pyplot as plt
from galpy.PlotGalprop import galpropDATA
import math
import pandas as pd

# Setup Figure:
fig = plt.figure(figsize=(13,8))
ax = plt.gca()
#ax.tick_params(axis='both',which='both',direction='in',pad=8,tick2On=True,gridOn=False)
#ax.ticklabel_format(style='sci', axis='y', scilimits=(6,15))

p = 2.7 #power to mult. CR flux 
m_h = 3728.423 * (1.0 / 1e3) # He4 mass in GeV

# GALPROP data
instance_new = galpropDATA("/project/ckarwin/astro/chris/annihilation_in_flight/GALPROP_Sims/WebRun/results_54_0780000h","54_0780000h.gz","spectra")
output_new  = instance_new.CRspectra(2,-1,joined=True,phi=0)
output_2  = instance_new.CRspectra(2,-1,joined=True,phi=650)

# Plot GALPROP CR flux with solar modulation
energy = output_2[0]
spectra = output_2[1][0]

standard_flux_list = []
for i in range(0,len(energy)):
    standard_flux = spectra[i] * energy[i]**p
    standard_flux_list.append(standard_flux)
ax.semilogx(energy,standard_flux_list,zorder=10,label='WebRun ($\mathregular{\phi}$ = 650 MV)',ls='--',lw=3,color='black')

# Plot GALPROP CR spectrum (WebRun)
energy = output_new[0]
spectra = output_new[1][0]

standard_flux_list = []
for i in range(0,len(energy)):
    standard_flux = spectra[i] * energy[i]**p
    standard_flux_list.append(standard_flux)

ax.semilogx(energy,standard_flux_list,alpha=0.8,zorder=10,label='WebRun',lw=6,color='black',ls='-')

# Plot parameters KE
plt.xlabel('Energy [MeV/n]', fontsize=16)
plt.ylabel('E$\mathregular{^{2.7}}$ df/dE [(MeV/n)$\mathregular{^{1.7}}$ cm$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$ sr$\mathregular{^{-1}}$]',fontsize=16)
plt.xlim(1e2,1e8)
#plt.ylim(1e-2,5e1) #for helium E^2.0
plt.ylim(0,14e3) #for helium E^2.7
plt.title('CR Helium LIS',fontsize=16)
plt.legend(loc=4,frameon=False,prop={'size':12})
ax.tick_params(axis='both', labelsize=14)
plt.grid(ls=":",color="grey",alpha=0.5)
plt.savefig('CR_helium_flux_KE_2.7_simple.pdf',bbox_inches='tight')
plt.show()
