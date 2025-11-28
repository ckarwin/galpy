# Imports:
from astropy.io import fits
import healpy as hp
import numpy as np
from mhealpy import HealpixMap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import pandas as pd
from astropy.wcs import WCS
import astropy.wcs.utils as utils
from astropy.coordinates import SkyCoord
import math
import aplpy
import matplotlib.ticker

class GalMapsHeal:

    def read_healpix_file(self,input_file,verbose=False):
        
        """
        Read GALPROP map given in healpix format.
        
        Parameters
        ----------
        input_file : str 
            Input GALPROP map in healpix format.
        verbose : boolean, optional 
            If True print more info about data. Default is False.
        
        Note
        ----
        Currently only compatible with GALPROP v57. 
        """

        print()
        print("**********************")
        print("GALPROP HEALPIX READER")
        print()
        
        # Read in galprop map:
        input_file = input_file
        hdu = fits.open(input_file)
        header = hdu[1].header

        # Get map info:
        self.ordering = header["ORDERING"]
        self.nside = header["NSIDE"]
        self.NPIX = hp.nside2npix(self.nside) # number of pixels
        self.resolution = hp.nside2resol(self.nside, arcmin=True) / 60.0 # spatial resolution
        self.resolution = format(self.resolution,'.3f')
        
        # Print info:
        print()
        print("Input file: " + str(input_file))
        print("Ordering: " + str(self.ordering))
        print("NSIDE: " + str(self.nside))
        print("Approximate resolution [deg]: " + str(self.resolution))
        print("Number of pixels: " + str(self.NPIX))
        print()

        # Get data:
        # Note: energy and data are stored as numpy records:
        self.energy = hdu[2].data # MeV
        self.energy = self.energy['energy'] 
        self.num_ebins = self.energy.size
        self.data = hdu[1].data
    
        # Print info verbose:
        if verbose == True:
            print()
            print("Data record:")
            print("Size: " + str(self.data.size))
            print("units: ph/cm^2/s/MeV/sr")
            print(self.data)
            print()
            print("Number of Energy bins: " + str(self.num_ebins))
            print("Energy array [MeV]:")
            print(self.energy)
            print()

        return

    def make_healmap(self,energy_bin):

        """
        Define healpix object for given energy bin.
        
        Parameters
        -----------
        energy_bin : int 
            Energy slice to use for healpix object. 
        """

        # Get data slice for given energy bin:
        this_ebin = "Bin" + str(energy_bin)
        ebin_data = self.data[this_ebin]
        data_indices=np.arange(len(ebin_data))

        # Define Healpix object:
        self.galmap = HealpixMap(ebin_data, nside = self.nside, scheme = self.ordering)
        
        return

    def plot_healmap(self,savefile,plot_kwargs={},fig_kwargs={}):

        """
        Plot healpix map.

        Parameters
        -----------
        savefile : str 
            Name of output image file.
        plot_kwargs : dict, optional 
            Pass any kwargs to plt.imshow().
        fig_kwargs : dict, optional 
            Pass any kwargs to plt.gca().set().
        """
        
        # Make plot:
        plot,ax = self.galmap.plot(cmap="inferno",cbar=True,**plot_kwargs)#,norm=colors.LogNorm(),cbar=True)
        ax.get_figure().set_figwidth(7)
        ax.get_figure().set_figheight(6)
        plot.colorbar.set_label("$\mathrm{ph \ cm^{-2} \ s^{-1} \ sr^{-1}} \ MeV^{-1}$")
        ax.set(**fig_kwargs)
        plt.savefig(savefile,bbox_inches='tight')
        plt.show()
        plt.close()

        return

    def get_disk_region(self,theta,phi,rad):
    
        """
        Get pixels corresonding to disk region.
        
        Parameters
        ----------
        theta : float 
            Colatitude (zenith angle) in degrees, ranging from 0 - 180 degrees.
        phi : float
            Longitude (azimuthal angle) in degrees, ranging from 0 - 360 degrees..
        rad : float 
            Radius of region in degrees.
        """
       
        # Convert degree to radian:
        theta = np.radians(theta)
        phi = np.radians(phi)
        rad = np.radians(rad)
        
        # Get pixels:
        center_pix = self.galmap.ang2pix(theta,phi,lonlat=False)
        center_vec = self.galmap.pix2vec(center_pix)
        pixs = self.galmap.query_disc(center_vec,rad) 

        return pixs

    def get_polygon_region(self,theta,phi):

        """
        Get pixels corresponding to polygon region. 
        
        Parameters
        ----------
        theta : list
            List of theta values for vertices, in degrees. 
            Note that theta is the colatitude (zenith angle), 
            ranging from 0 - 180 degrees.
        phi : list   
            List of phi values for vertices, in degrees. 
            Note that phi is the longitude, ranging from 0 - 360 degrees.

        Note
        ----
        The order of the vertices matters. The polygon can't be self-intersecting,
        otherwise an "unknown exception" will be thrown.
        """
        
        # Convert lists in degrees to radians:
        # Note: healpy requires radians as input. 
        theta = np.array(theta)
        theta = np.radians(theta)
        phi = np.array(phi)
        phi = np.radians(phi) 
       
        # Make vertex array:
        vertices = []
        for i in range(0,len(theta)):
            center_pix = self.galmap.ang2pix(theta[i],phi[i],lonlat=False)
            center_vec = self.galmap.pix2vec(center_pix)
            vertices.append(list(center_vec))
        vertices = np.array(vertices)
        
        print()
        print("vertex array:")
        print("shape: " + str(vertices.shape))
        print(vertices)
        print()
        
        # Get pixels:
        pixs = self.galmap.query_polygon(vertices,inclusive=True)

        return pixs 
    
    def mask_region(self,pixs):
        
        """
        Mask region given by pixs arguement.
        
        Parameters
        ----------
        pixs : list or array
            healpy pixs to be masked.
        """

        self.galmap[pixs] = 0

        return

    def make_spectrum(self,pixs=None):

        """
        Make average spectrum over specified region.
        
        Parameters
        ----------
        pixs : array, optional 
            Healpix pixels to use. Default is None, which uses all-sky.
        """
        
        # Make spectrum:
        spectra_list = []
        for E in range(0,len(self.energy)):
            this_bin = "Bin%s" %str(E)
            
            # If averaging over all-sky:
            if pixs is None:
                spectra_list.append(np.mean(self.data[this_bin]))
            
            # If averaging over limited region:
            else:
                spectra_list.append(np.mean(self.data[this_bin][pixs]))
        
        # Scale of energy:
        self.spectra_list = (self.energy**2)*spectra_list
        
        return

    def write_spectrum(self, savefile, data_type="gamma"):

        """
        Write spectrum to file.
        
        Parameters
        ----------
        savefile : str 
            Name of output data file (tab delimited).
        data_type : str, optional
            Data type being saved. Options are gamma (default), 
            sync, and emiss.
        """
        
        # Need to reformat energy data to be commpatible with pandas:
        self.energy = np.array(self.energy).astype("float")

        # Write to file:
        if data_type == "sync":
            d = {"frequency[MHz]":self.energy,"flux[erg/cm^2/s/sr/Hz]":self.spectra_list}
            df = pd.DataFrame(data=d)
            df.to_csv(savefile,float_format='%10.5e',index=False,\
                    sep="\t",columns=["frequency[MHz]", "flux[erg/cm^2/s/sr/Hz]"])
        if data_type == "gamma":
            d = {"energy[MeV]":self.energy,"flux[MeV/cm^2/s/sr]":self.spectra_list}
            df = pd.DataFrame(data=d)
            df.to_csv(savefile,float_format='%10.5e',index=False,\
                    sep="\t",columns=["energy[MeV]", "flux[MeV/cm^2/s/sr]"])
        if data_type == "emiss":
            d = {"energy[MeV]":self.energy,"emissivity[MeV/s/sr]":self.emissivity}
            df = pd.DataFrame(data=d)
            df.to_csv(savefile,float_format='%10.5e', index=False,\
                    sep="\t",columns=["energy[MeV]", "emissivity[MeV/s/sr]"])
        
        return

    def plot_spectrum(self, savefile, fig_kwargs={}):

        """
        Plot map spectrum.
        
        Parameters
        ----------
        savefile : str 
            Name of saved image file.
        fig_kwargs : dict, optional
            Pass any kwargs to plt.gca().set()
        """

        # Setup figure:
        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()
            
        # Plot:
        plt.loglog(self.energy,self.spectra_list)
        
        plt.ylim(1e-5,1e-1)
        plt.xlabel("Energy [MeV]", fontsize=14)
        plt.ylabel("$\mathrm{E^2 \ dN/dE \ [\ MeV \ cm^{-2} \ s^{-1} \ sr^{-1}]}$",fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.tick_params(axis='both',which='major',length=9)
        ax.tick_params(axis='both',which='minor',length=5)
        ax.set(**fig_kwargs)
        plt.grid(ls=":",color="grey",alpha=0.3,lw=1)
        plt.savefig(savefile,bbox_inches='tight')
        plt.show()
        plt.close()

        return

    def plot_emissivity(self, emiss_array, emiss_energy, save_prefix, fig_kwargs={}):

        """
        Plot emissivity in units of MeV/s/sr.
        
        Parameters
        ----------
        emiss_array : array
            Array object from GALPROP emissivity file. This has shape
            (1, E, z, r), with z = scale height and r = radial distance.  
        emiss_energy : array
            Energy array. 
        save_prefix : str 
            Prefix of saved image file.
        fig_kwargs : dict, optional
            Pass any kwargs to plt.gca().set() 
        """
    
        # Setup figure:
        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()
            
        # Plot:
        plt.loglog(emiss_energy, emiss_array)
        
        plt.xlabel("Energy [MeV]", fontsize=14)
        plt.ylabel("$\mathrm{E^2 \ Emissivity \ [\ MeV \ s^{-1} \ sr^{-1}]}$",fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.tick_params(axis='both',which='major',length=9)
        ax.tick_params(axis='both',which='minor',length=5)
        ax.set(**fig_kwargs)
        plt.grid(ls=":",color="grey",alpha=0.3,lw=1)
        plt.savefig(f"{save_prefix}_emissivity.pdf",bbox_inches='tight')
        plt.show()
        plt.close()

        return
    

    def gal2mega(self, file_type, output_file, use_2d=False):

        """
        Convert GALPROP map to MEGAlib cosima input.
        
        Parameters
        ----------
        file_type : str
            'fits' for mapcupe fits file or 'heal' for healpix file.
        output_file : str
            Name of output file (do not include .dat). 
        use_2d : boolean, optional
            option to use 2d FITS file. Default is False (corresponding to 3d).
        
        Note
        ----
        The code supports non-all-sky input images. The MEGAlib input will still be all-sky,
        but the flux will be set to 0 for pixels outside of the input image. This is
        currently only supported in 2d mode. 
        """
        
        # Make energy array:
        energy_list = []
        for each in self.energy[:21]: 
            this_energy = float('{:.1f}'.format(each*1000.0)) # convert to keV and format
            energy_list.append(this_energy)
        print() 
        print("energy list [keV]:")
        print(energy_list)
        print()

        # Define phi (PA), theta (TA), and energy (EA) points:
        PA = np.arange(-179.5,181.5,1)
        TA = np.arange(0,180.5,0.5)
        EA = np.array(energy_list)

        # Convert PA to file input:
        PA_line = "PA"
        for i in range(0,len(PA)):
            PA_line += " " + str(PA[i])

        # Convert TA to file input:
        TA_line = "TA"
        for i in range(0,len(TA)):
            TA_line += " " + str(TA[i])

        # Convert EA to file input:
        EA_line = "EA"
        for i in range(0,len(EA)):
            EA_line += " " + str(EA[i])

        # Write file:
        f = open(output_file + ".dat","w")
        f.write("IP LIN\n")
        f.write(PA_line + "\n")
        f.write(TA_line + "\n")
        f.write(EA_line + "\n")
            
        image_pix=0
        # Make main:
        for E in range(0,len(energy_list)):
    
            this_E_list = []
            for i in range(0,len(PA)):
                
                if PA[i] > 0:
                    this_l = PA[i]
                if PA[i] < 0:
                    this_l = 360 + PA[i]

                for j in range(0,len(TA)):
        
                    this_b = 90-TA[j]
                    
                    # to get flux from healpix map:
                    if file_type == "heal":
                        
                        theta = np.radians(TA[j])
                        phi = np.radians(this_l)
                        this_ebin = "Bin%s" %str(E)
                        this_pix = self.galmap.ang2pix(theta,phi,lonlat=False)
                        this_flux = self.data[this_ebin][this_pix] / 1000.0 # ph/cm^2/s/keV/sr

                    # to get flux from mapcube:
                    if file_type == "fits":
                       
                        if use_2d == True:
                            pixs = self.wcs.all_world2pix(np.array([[this_l,this_b]]),0)
                         
                            # Check that pixel is in image:
                            if (0 <= pixs[0][0] < self.wcs.pixel_shape[0]) and (0 <= pixs[0][1] < self.wcs.pixel_shape[1]):
                                this_l_pix = int(math.floor(pixs[0][0]))
                                this_b_pix = int(math.floor(pixs[0][1]))
                                this_flux = self.data[this_b_pix,this_l_pix] / 1000.0 # ph/cm^2/s/keV/sr
                                image_pix += 1
                            # If not, set flux to zero:
                            else:
                                print("WARNING: Pixel out of input map! Setting flux to zero.")
                                this_flux = 0.0
                        else:
                            pixs = self.wcs.all_world2pix(np.array([[this_l,this_b,0]]),0)
                            
                            # Check that pixel is in image:
                            if (0 <= pixs[0][0] < self.wcs.pixel_shape[0]) and (0 <= pixs[0][1] < self.wcs.pixel_shape[1]):
                                this_l_pix = int(math.floor(pixs[0][0]))
                                this_b_pix = int(math.floor(pixs[0][1]))
                                this_flux = self.data[E,this_b_pix,this_l_pix] / 1000.0 # ph/cm^2/s/keV/sr
                                image_pix += 1
                            # If not, set flux to zero:
                            else:
                                print("WARNING: Pixel out of input map! Setting flux to zero.")
                                this_flux = 0.0

                    # Format:
                    this_flux = float('{:.5e}'.format(this_flux))

                    # Write line:
                    this_line = "AP " + str(i) + " " + str(j) + " " + str(E) + " " + str(this_flux) + "\n"
                    f.write(this_line)
                   
        # Close file:
        f.write("EN")
        f.close()
       
        if file_type == "fits":
            print("total image pixels used (for all E bins): " + str(image_pix))
        
        return
    
class GalMapsFITS(GalMapsHeal):

    def read_fits_file(self, input_file, sync=False):

        """Read GALPROP map given in fits format.

        Parameters
        ----------
        input_file : str
            Name of input GALPROP  FITS file.
        sync : bool, optional
            Option to read syncrtron skymap (default is False).
        """
        print()
        print("**********************")
        print("GALPROP Fits READER")
        print()

        hdu = fits.open(input_file)
        self.data = hdu[0].data
        self.energy = hdu[1].data
        if sync == True:
            self.energy = self.energy['Frequency']
        else:
            self.energy = self.energy['Energy']
        header = hdu[0].header
        self.wcs = WCS(header)

        return

    def read_fits_objects(self, energy, data, wcs):

        """Read input objects defining FITS file.

        Parameters
        ----------
        energy : list
            List of energies in MeV.
        data : array
            FITS data array in ph/cm2/s/MeV/sr. 
            Can be 2d or 3d. 
        wcs : WCS object
            Specifies world coordinate system from FITS file. 
            Use WCS(header)
        """
        
        self.data = data
        self.energy = energy
        self.wcs = wcs

        return

    def get_fits_region(self, lat, lon, lon2=[], use_2d=False):

        """
        Get pixels for specified spatial region.
        
        Parameters
        ----------
        lat : list 
            Min and max Galactic latitude of region, inclusive. 
        lon : list 
            Min and max Galactic longitude of region, inclusive.
        lon2 : list, optional
            Min and max Galactic longtitude of region (inclusive, exclusive). 
            This arguement compliments 'lon' for regions about the 
            Galactic center, since l ranges from 0 - 360 degrees. 
        """

        # Make ra and dec lists:
        if use_2d == True:
            index_array = np.argwhere(self.data[:,:] < 1e50) #arbitrary condition to ensure all pixels are extracted
        else:
            index_array = np.argwhere(self.data[0,:,:] < 1e50) #arbitrary condition to ensure all pixels are extracted
        ralist = []
        declist = []
        for each in index_array:
            ra = each[1]
            dec = each[0]
            ralist.append(ra)
            declist.append(dec)

        # Transfer pixels in wcs to sky coordinates:
        coords = utils.pixel_to_skycoord(ralist,declist,self.wcs)
    
        # Define keep index:
        
        # longitude main:
        i1 = (coords.l.deg>=lon[0]) & (coords.l.deg<=lon[1])
        
        # A second condition is needed for regions centered on the Galactic center, 
        # since l ranges from 0 - 360 degrees.
        if len(lon2) != 0:
            i2 = (coords.l.deg>=lon2[0]) & (coords.l.deg<lon2[1])
            l_cond = i1 | i2
        if len(lon2) == 0:
            l_cond = i1 
        
        # latitude condition:
        b_cond = (coords.b.deg>=lat[0]) & (coords.b.deg<=lat[1])
        
        # define keep index:
        keep_index = b_cond & l_cond

        # Convert sky coordinates in wcs back to pixels:
        pixs_SR =  utils.skycoord_to_pixel(coords[keep_index],self.wcs,mode="wcs")

        # Need to convert float to int
        # Note: need to round fist, since int rounds towards zero
        SR0 = np.round(pixs_SR[0]).astype(int)
        SR1 = np.round(pixs_SR[1]).astype(int)
     
        return [SR0,SR1]
 
    def make_spectrum(self, pixs=None, use_2d=False, sync=False):

        """
        Make spectrum by averaging over specified region.
        
        Parameters
        ----------
        pixs : list, optional
            Pixels of region to be used for calculation.
            Should be a list containing SRO and SR1, 
            returned from self.gets_fits_region.
        use_2d : boolean, optional
            Option to use 2d FITS file. Default is False, 
            corresponding to 3d. 
        sync : boolean, optional
            Option to calculate synchrotron spectrum. Default is False.
            Note: Synchrotron skymaps have units of specific intensity,
            i.e., [erg/cm2/s/sr/Hz]. 
        """
 
        spectra_list = []
        for E in range(0,len(self.energy)):
         
            if use_2d == False:
                # Get average over all-sky:
                if pixs is None:
                    spectra_list.append(np.mean(self.data[E,:,:]))

                # Get average over specified region:
                else:
                    spectra_list.append(np.mean(self.data[E,pixs[1],pixs[0]]))
        
            if use_2d == True:
                # Get average over all-sky:
                if pixs is None:
                    spectra_list.append(np.mean(self.data[:,:]))

                # Get average over specified region:
                else:
                    spectra_list.append(np.mean(self.data[pixs[1],pixs[0]]))

        spectra_list = np.array(spectra_list)
        
        if sync == True:
            self.energy = self.energy/1e6 # MHz
            self.spectra_list = spectra_list
        
        else:
            self.spectra_list = (self.energy**2)*spectra_list

        return

    def get_emissivity(self, emiss_file, save_prefix, r=8, z=None,
            make_plot=True, fig_kwargs={}, write=True):

        """
        Get emissivity in units of MeV/s/sr per H atom.
        
        Parameters
        ----------
        emiss_file : str
            GALPROP emissivity file. The array in this file has shape
            (1, E, z, r), with z = scale height and r = radial distance.  
        save_prefix : str 
            Prefix for saved outputs.
        r : int, optional
            Radial bin index. Corresponds to radial binning in 
            galdef file. Default is 8, which is nominally the local 
            bin (i.e. between 8 - 9 kpc).
        z : int, optional
            Scale height bin index. Corresponds to z binning in 
            galadef file. Default is None, for which the midpoint is
            used, corresponding to the Galactic plane.
        make_plot : bool, optional
            Option to plot emissivity.
        fig_kwargs : dict, optional
            Pass any kwargs to plt.gca().set()
        write : bool, optional
            Option to write emissivity to dat file. Defaul is True.
        """
       
        emiss_hdu = fits.open(emiss_file)
        emiss_array = emiss_hdu[0].data

        print()
        print("data shape (1, E, z, r): " + str(emiss_array.shape))
        print()

        if z != None:
            # Get emissivity at specified height:
            self.emissivity = np.array(emiss_array[0,:,z,r].tolist())
        else:
            # Get emissivity in Galactic plane:
            z = math.floor(emiss_array.shape[2]/2.0)
            self.emissivity = np.array(emiss_array[0,:,z,r].tolist())
        
        # Get energy array:
        energy_file = emiss_file.replace("emiss","mapcube")
        self.read_fits_file(energy_file)

        # Plot:
        if make_plot == True:
            self.plot_emissivity(self.emissivity, self.energy, save_prefix, fig_kwargs=fig_kwargs)
        
        # Save to dat file:
        if write == True:
            self.write_spectrum(f"{save_prefix}_emiss.dat", data_type="emiss")
        
        return 

    def plot_galprop_skymap(self, fits_file, prefix_name, 
            energy_index=0, vmin=None, vmax=None, plot_type="gamma"):
        
        """
        Plot a GALPROP skymap.
        
        Parameters
        ----------
        fits_file : str
            Input GALRPOP mapcube.
        prefix_name : str
            Prefix of saved image.
        energy_index : int, optional
            Energy index to plot (default is 0).
        vmin : float, optional
            Min value for plotting (default is None).
        vmax : float, optional
            Max value for plotting (default is None).
        plot_type : str, optional
            Type of data being plotted. Options are 
            gamma (default) or sync. 
        """

        # Load FITS file:
        with fits.open(fits_file) as hdul:
            data = hdul[0].data    # (E, lat, lon)
            header = hdul[0].header
            energy = hdul[1].data
            ne, ny, nx = data.shape

        # Extract correct 2D MAP for this energy bin:
        m = data[energy_index, :, :]
        
        # Rotate data to proper Galactic coords:
        m = np.roll(m, -m.shape[1] // 2, axis=1)
        m = np.fliplr(m)

        # Update header accordingly:
        header['CRPIX1'] = nx/2 + 0.5
        header['CRVAL1'] = 0.0
        header['CDELT1'] = -header['CDELT1']

        # Mask values <= 0 for log plot:
        m = np.where(m > 0, m, np.nan)

        # Extract ONLY the celestial part of the WCS:
        w = WCS(header).celestial

        # Mollweide projection from WCS:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection=w)

        # Log scaling
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

        # Plot image with WCS transform
        im = ax.imshow(m, origin='lower', cmap='inferno', norm=norm)

        # Colorbar on side
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.05, shrink=0.95)
        if plot_type == "gamma":
            label = "Intensity [$\mathrm{ph \ cm^{-2} \ s^{-1} \ sr^{-1} \ MeV^{-1}}$]"
        if plot_type == "sync":
            label = "Intensity [$\mathrm{erg \ cm^{-2} \ s^{-1} \ sr^{-1} \ Hz^{-1}}$]"
        cbar.set_label(label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        # Axis labels and grid
        ax.set_xlabel("Galactic Longitude",fontsize=16)
        ax.set_ylabel("Galactic Latitude",fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(color='grey', ls=':', alpha=0.6)

        energy_value = energy[energy_index][0].item()
        if plot_type == "gamma":
            formatted = f"{energy_value:.2f}"
            title = f"{prefix_name} ({formatted} MeV)"
        if plot_type == "sync":
            formatted = f"{energy_value/1e6:.2f}"
            title = f"{prefix_name} ({formatted} MHz)"
        ax.set_title(title, fontsize=16)

        plt.tight_layout()
        plt.savefig(f"{prefix_name}.png")
        plt.show()

        return

class Utils(GalMapsFITS):

    def sum_spectra(self,savefile,input_files):

        """
        Sum multiple spectra. Added files must have the same binning.
        
        Parameters
        ----------
        savefile : str 
            Name of saved .dat file.
        input_files : list 
            List of spectra to add, given as .dat files.
        """

        # Add spectra from list:
        for i in range(len(input_files)):
            df = pd.read_csv(input_files[i], delim_whitespace=True)
            this_energy = df["energy[MeV]"]
            this_spectra = df["flux[MeV/cm^2/s/sr]"]
            if i == 0:
                summed_spectra = this_spectra
            if i > 0:
                summed_spectra += this_spectra

        # Write to file:
            d = {"energy[MeV]":this_energy,"flux[MeV/cm^2/s/sr]":summed_spectra}
            df = pd.DataFrame(data=d)
            df.to_csv(savefile,float_format='%10.5e',index=False,sep="\t",columns=["energy[MeV]", "flux[MeV/cm^2/s/sr]"])
    
        return

    def plot_mult_spectra(self,savefile,input_files,labels,fig_kwargs={},\
            plot_kwargs={},show_plot=True):

        """
        Plot multiple map spectra.
        
        Parameters
        ----------
        savefile : str 
            Name of saved image file.
        input_files : list
            List of input file(s) to plot.
        labels : list 
            List of legend labels corresponding to input files.
            Must have a label for each input file.
        fig_kwargs : dict, optional 
            Pass any kwargs to plt.gca().set()
        plot_kwargs : dict, optional 
            Pass any kwargs to plt.plot().
            Each key must define a list corresponding to input_files. 
        show_plot : boolean, optional 
            Set to False to not show plot. 
        """

        # Setup figure:
        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()
             
        # Check for proper data type:
        if isinstance(input_files,list) == True:
            pass
        else:
            raise TypeError("'input_files' must be a list.")
        
        # Check labels:
        if len(labels) != len(input_files):
            raise ValueError("Must have a label for each input file.")    

        # Plot spectra:
        for i in range(len(input_files)):

            # Check for plot kwargs:
            this_kwargs = {}
            for key in plot_kwargs:
                this_kwargs[key] = plot_kwargs[key][i]
                    
            df = pd.read_csv(input_files[i], delim_whitespace=True)
            this_energy = df["energy[MeV]"]
            this_spectra = df["flux[MeV/cm^2/s/sr]"]
            plt.loglog(this_energy,this_spectra,label=labels[i],**this_kwargs)

        # Option to make dark plot:
        color1 = "white"
        color2 = "black"
        dark_plot = False
        if dark_plot == True:
            color1 = "black"
            color2 = "tan"
        
        plt.ylim(1e-5,1e-1)
        plt.xlabel("Energy [MeV]", fontsize=14, color=color2)
        plt.ylabel("$\mathrm{E^2 \ dN/dE \ [\ MeV \ cm^{-2} \ s^{-1} \ sr^{-1}]}$",fontsize=14, color=color2)
        plt.legend(frameon=True,ncol=2,loc=2,handlelength=2,prop={'size': 9.5})
        plt.xticks(fontsize=14,color=color2)
        plt.yticks(fontsize=14,color=color2)
        ax.tick_params(axis='both',which='major',length=9,color=color2)
        ax.tick_params(axis='both',which='minor',length=5,color=color2)
        ax.set(**fig_kwargs)
        ax.set_facecolor(color1)
        plt.setp(ax.spines.values(), color=color2)
        plt.grid(ls=":",color="grey",alpha=0.3,lw=1)
        plt.savefig(savefile,bbox_inches='tight')
        if show_plot == True:
            plt.show()
            plt.close()

        return
