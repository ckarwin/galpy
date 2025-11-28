import matplotlib.pyplot as plt
import math
import numpy as np

# Setup Figure:
fig = plt.figure(figsize=(11,8))
ax = plt.gca()

def CR_Density(r,r_1,a,b):
    if r <= 30: 
        return ((r+r_1)/(8.5+r_1))**a*math.exp(-1*b*(r - 8.5)/(8.5+r_1))
    if 30 <= r <= 35:
        return (30/8.5)**a*math.exp(-1*b*(30 - 8.5)/8.5)
    if r > 30:
        return 0
            
radius = np.arange(0,40,0.2)

GC_density_list = []
M31_density_list = []
for each in radius:
    GC_value = CR_Density(each,0.55,1.64,4.01)
    M31_value = CR_Density(each,0,1.5,3.5)
    GC_density_list.append(GC_value)
    M31_density_list.append(M31_value)

plt.plot(radius,M31_density_list,label='M31 IEM (Karwin+19)',lw=6,ls='-',color='darkorange')
plt.plot(radius,GC_density_list,label='IG IEM (Ajello+16)',lw=6,ls='--')

plt.xlabel('Radius [kpc]',fontsize=16)
plt.ylabel('CR Source Density [arb. units]',fontsize=16)
plt.xlim((0,30))
plt.ylim((0,2.7))
plt.legend(loc=1, frameon=False,ncol=1)
ax.tick_params(axis='both', labelsize=14)
plt.savefig('CR_source_density.png',bbox_inches='tight')
plt.show()
