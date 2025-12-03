# Inputs:
from galpy.GalMaps import *
import numpy as np
import math
from scipy import stats, interpolate
from scipy.stats import gmean

# These examples extract events from a 47.5 degree box around the Galactic center.

# Running healpix:
skymap_file = "full/path/to/GALPROP/healpix/output"
emissivity_file = "full/path/to/GALPROP/fits/emissivity/output"
nuclei_file = "full/path/to/GALPROP/fits/nuclei/output"
instance = GalMapsHeal()
instance.read_healpix_file(skymap_file,verbose=False)
instance.make_healmap(7)
pixs=instance.get_polygon_region([42.5,137.5,137.5,42.5],[47.5,47.5,312.5,312.5]) 
instance.mask_region(pixs)
pixs=instance.get_disk_region(90,0,30)
instance.plot_healmap("skymap.png",fig_kwargs={'title':"example"},plot_kwargs={'norm':'log'})
instance.make_spectrum(pixs=pixs) 
instance.write_spectrum("output_spectrum.dat")
instance.plot_spectrum("spectra.pdf")
instance.get_emissivity_3d(emissivity_file,"prefix_name")
energy, spectra_1 = instance.CR_spectra_3d(nuclei_file, 'species')

# Running fits: 
skymap_file = "full/path/to/GALPROP/fits/mapcube/output"
emissivity_file = "full/path/to/GALPROP/fits/emissivity/output"
instance = GalMapsFITS()
instance.read_fits_file(skymap_file) 
pixs=instance.get_fits_region([-47.5,47.5],[0,47.5],lon2=[312.5,360])
instance.make_spectrum(pixs)
instance.write_spectrum("output_spectrum.dat")
instance.plot_galprop_skymap(skymap_file, "prefix_name")
instance.get_emissivity(emissivity_file, "prefix_name") 

# For general all-sky FITS map:
hdu = fits.open("your_file.fits")
data = hdu[1].data
header = hdu[1].header
wcs = WCS(header)
energy = [0.510,0.511,0.512]
instance = GalMapsFITS()
instance.read_fits_objects(energy,data,wcs)

# Synchrotron Emission example (FITS):
this_file = "full/path/to/GALPROP/fits/synchrotron/mapcube/output"
instance = GalMapsFITS()
instance.read_fits_file(this_file, sync=True)
pixs=instance.get_fits_region([-47.5,47.5],[0,47.5], lon2=[312.5,360])
instance.make_spectrum(pixs, sync=True)
instance.write_spectrum(f"synchrotron_spectrum.dat", data_type="sync")
instance.plot_spectrum("synchrotron_spectrum.pdf",\
        fig_kwargs={'title':'Inner Galaxy ($|l| < 47.5^\circ; |b|<47.5^\circ$)',\
        "xlabel":"Frequency [MHz]","ylabel":r"$I_\nu \ [\mathrm{erg \ cm^{-2} \ s^{-1} \ sr^{-1} \ Hz^{-1}}]$","ylim":(1e-20,1e-15)})

# Synchrotron Emission example (HEALPIX):
this_file = "../results/synchrotron_healpix_57_SA100_F98_example.gz"
instance = GalMapsHeal()
instance.read_healpix_file(this_file,verbose=False,sync=True)
instance.make_healmap(11)
pixs=instance.get_polygon_region([42.5,137.5,137.5,42.5],[47.5,47.5,312.5,312.5])
instance.plot_healmap("sync", plot_type="sync", plot_kwargs={'norm':'log'})
instance.make_spectrum(pixs=pixs,sync=True)
instance.write_spectrum(f"sync_spectrum.dat",data_type="sync")
instance.plot_spectrum("synchrotron_spectrum.pdf",\
        fig_kwargs={'title':'Inner Galaxy ($|l| < 47.5^\circ; |b|<47.5^\circ$)',\
        "xlabel":"Frequency [MHz]","ylabel":r"$I_\nu \ [\mathrm{erg \ cm^{-2} \ s^{-1} \ sr^{-1} \ Hz^{-1}}]$","ylim":(1e-20,1e-15)})

# Running Utils:
instance = Utils()

# Sum origianl spectra:
input_files = ["bremss_spectrum.dat","pion_decay_spectrum.dat","ic_spectrum.dat"]
instance.sum_spectra("total_spectrum.dat",input_files)

# Plot (mult) spectra:
inputs = ["output_spectrum_1.dat","output_spectrum_2.dat"]
labels = ["model1","model2"]
ls = ['-','--']
color = ['blue','orange']
savefile="all_spectra.pdf"
instance.plot_mult_spectra(savefile,inputs,labels,\
        fig_kwargs={'title':'Inner Galaxy ($|l| < 47.5^\circ; |b|<47.5^\circ$)',\
        'ylim':(1e-4,1e0),'xlim':(1e-2,1e4)},\
        plot_kwargs={'ls':ls,'color':color},show_plot=True)

# Converting to MEGAlib input format:
instance.gal2mega("heal","output_file",use_2d=False) # 3d Healpix input, GALPROP v57 only
instance.gal2mega("fits","output_file",use_2d=False) # 3d FITS input
instance.gal2mega("fits","output_file",use_2d=True) # 2d FITS input
