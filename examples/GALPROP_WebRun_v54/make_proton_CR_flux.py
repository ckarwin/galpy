import matplotlib.pyplot as plt
from galpy.PlotGalprop import galpropDATA
import math
import pandas as pd
import numpy as np
import matplotlib

# Setup figure:
fig = plt.figure(figsize=(13,8))
ax = plt.gca()

m_p = 938.272 * (1.0 / 1e3) # proton mass [GeV]
p = 2.7 # power to mult. CR flux 

instance_new = galpropDATA("/project/ckarwin/astro/chris/annihilation_in_flight/GALPROP_Sims/WebRun/results_54_0780000h","54_0780000h.gz","spectra")
output_new  = instance_new.CRspectra(1,-1,joined=True,phi=0)
output_2  = instance_new.CRspectra(1,-1,joined=True,phi=650)

# Plot GALPROP CR flux with solar modulation (WebRun)
energy = output_2[0]
spectra = output_2[1][0]

flux_list = []
rigidity_list = []
standard_flux_list = []
for i in range(0,len(energy)):
    E = energy[i] * (1.0 / 1e3) #kinetic energy per nucleon in GeV
    F = (spectra[i] * 1e3 * 1e4 ) 
    R = ((E + m_p)**2 - m_p**2 )**0.5
    new_flux = (F*E) * (R**1.7)
    standard_flux = spectra[i] * energy[i]**p
    standard_flux_list.append(standard_flux)
    rigidity_list.append(R)
    flux_list.append(new_flux)
#ax.semilogx(rigidity_list,flux_list,label='WebRun (phi = 650 MeV)',lw=3,color='black')
ax.semilogx(energy,standard_flux_list,zorder=10,label='WebRun ($\mathregular{\phi}$ = 650 MV)',lw=3,color='cyan',ls='--')

# Plot GALPROP CR spectrum (WebRun) 
energy = output_new[0]
spectra = output_new[1][0]

flux_list = []
rigidity_list = []
standard_flux_list = []
for i in range(0,len(energy)):
    E = energy[i] * (1.0 / 1e3) #kinetic energy per nucleon in GeV
    F = (spectra[i] * 1e3 * 1e4 ) 
    R = ((E + m_p)**2 - m_p**2 )**0.5 #rigidity
    new_flux = (F*E) * (R**1.7)
    standard_flux = spectra[i] * energy[i]**p
    standard_flux_list.append(standard_flux)
    rigidity_list.append(R)
    flux_list.append(new_flux)
#plt.semilogx(rigidity_list,flux_list,label='WebRun (phi = 0 MeV)',lw=3,ls='--',color='blue')
ax.semilogx(energy,standard_flux_list,alpha=0.8,label='WebRun',lw=6,color='black',ls='-')

# Plot parameters rigidity
#plt.xlabel('Rigidity [GV]', fontsize=25, fontweight='bold')
#plt.ylabel('R$\mathregular{^{2.7}}$ df/dE [GV$\mathregular{^{1.7}}$ m$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$ sr$\mathregular{^{-1}}$]',fontsize=25,fontweight='bold')
#plt.xlim(0.5,1e4)
#plt.ylim(0,20000)

# Plot parameters KE
plt.xlabel('Energy [MeV/n]', fontsize=16)
plt.ylabel('E$\mathregular{^{2.7}}$ df/dE [(MeV/n)$\mathregular{^{1.7}}$ cm$\mathregular{^{-2}}$ s$\mathregular{^{-1}}$ sr$\mathregular{^{-1}}$]', fontsize=16)
plt.xlim(100,1e8)#100
plt.ylim(5e-1,210e3) #for protons E^2.7
plt.title('CR Protons LIS',fontsize=16)
plt.legend(loc=4,frameon=False,prop={'size':12})
ax.tick_params(axis='both', labelsize=14)
plt.grid(ls=":",color="grey",alpha=0.5)
plt.savefig('CR_proton_flux_KE_2.7.pdf',bbox_inches='tight')
plt.show()
