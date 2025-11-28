import matplotlib.pyplot as plt
from galpy.PlotGalprop import galpropDATA
import math
import pandas as pd
import numpy as np

# setup figure
fig = plt.figure(figsize=(13,8))
ax = plt.gca()

p = 3 # power to mult. CR flux 
m_e = 0.511 * (1.0 / 1e3) #proton mass [GeV]

# WebRun IEM:
instance_new = galpropDATA("/project/ckarwin/astro/chris/annihilation_in_flight/GALPROP_Sims/WebRun/results_54_0780000h","54_0780000h.gz","spectra")
e_output_new  = instance_new.CRspectra(-1,0,joined=True,phi=0)
e_output_2_new  = instance_new.CRspectra(-1,0,joined=True,phi=650)
p_output_new  = instance_new.CRspectra(1,0,joined=True,phi=0)
p_output_2_new  = instance_new.CRspectra(1,0,joined=True,phi=650)

# Plot GALPROP CR flux with solar modulation (WebRun)
energy = e_output_new[0]
e_spectra = e_output_2_new[1][0]
p_spectra = p_output_2_new[1][0]

spectra = []
for i in range(0,len(e_spectra)):
    spectra.append(e_spectra[i] + p_spectra[i])

standard_flux_list = []
for i in range(0,len(energy)):
    standard_flux = spectra[i] * energy[i]**p
    standard_flux_list.append(standard_flux)
ax.semilogx(energy,standard_flux_list,zorder=10,label='WebRun ($\mathregular{\phi}$ = 650 MV)',ls='--',lw=3,color='black')

# Plot GALPROP CR spectrum without solar modulation (WebRun)
energy = e_output_new[0]
e_spectra = e_output_new[1][0]
p_spectra = p_output_new[1][0]

spectra = []
for i in range(0,len(e_spectra)):
    spectra.append(e_spectra[i] + p_spectra[i])

standard_flux_list_e = []
standard_flux_list = []
for i in range(0,len(energy)):
    standard_flux = spectra[i] * energy[i]**p
    standard_flux_list.append(standard_flux)
    
    standard_flux_e = e_spectra[i] * energy[i]**p
    standard_flux_list_e.append(standard_flux_e)

ax.semilogx(energy,standard_flux_list,zorder=10,alpha=0.8,label='WebRun',lw=6,color='black',ls='-')

plt.xlabel('Energy [MeV]', fontsize=16)
plt.ylabel('E$\mathregular{^{3}}$ df/dE [MeV$\mathregular{^{2}}$ cm$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$ sr$\mathregular{^{-1}}$]',fontsize=16)
plt.ylim(1,1e6) 
plt.title('CR Electrons + Positrons LIS',fontsize=16)
plt.legend(loc=1,frameon=False,prop={'size':12})
plt.yscale("log")
plt.grid(ls=":",color="grey",alpha=0.5)
ax.tick_params(axis='both', labelsize=14)
plt.savefig('CR_electron_and_positron_flux_KE_semilogx_simple.pdf',bbox_inches='tight')
plt.show()
