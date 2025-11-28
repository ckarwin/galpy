#!/usr/bin/env python3

# Python-3 updated version of Plot Galprop v1.03
# Converted from original Python-2 code (Karwin 2025)

import os
import sys
import numpy as np
import math
from astropy.io import fits as pyfits


class galpropDATA:
    def __init__(self, fitsdir, galdefid, parser, Rsun=8.5):

        self.Rsun = Rsun

        # --------------------------------------------------------
        # ----------- SPECTRA / ABUNDANCES / RATIOS --------------
        # --------------------------------------------------------
        if parser in ("spectra", "abundances", "ratios"):
            self.density = {}
            hdu = pyfits.open(os.path.join(fitsdir, "nuclei_" + galdefid))
            data = hdu[0].data

            R = (np.arange(int(hdu[0].header["NAXIS1"])) *
                 hdu[0].header["CDELT1"] + hdu[0].header["CRVAL1"])

            inds = []
            weights = []

            if (R[0] > self.Rsun):
                inds.append(0)
                weights.append(1)
            elif (R[-1] <= self.Rsun):
                inds.append(-1)
                weights.append(1)
            else:
                for i in range(len(R) - 1):
                    if R[i] <= self.Rsun < R[i + 1]:
                        inds.extend([i, i + 1])
                        weights.extend([
                            (R[i + 1] - self.Rsun) / (R[i + 1] - R[i]),
                            (self.Rsun - R[i]) / (R[i + 1] - R[i])
                        ])
                        break

            self.energy = 10 ** (float(hdu[0].header["CRVAL3"]) +
                                 np.arange(int(hdu[0].header["NAXIS3"])) *
                                 float(hdu[0].header["CDELT3"]))

            Nnuclei = hdu[0].header["NAXIS4"]

            for i in range(1, Nnuclei + 1):
                id_str = f"{i:03d}"
                Z = int(hdu[0].header["NUCZ" + id_str])
                A = int(hdu[0].header["NUCA" + id_str])
                K = int(hdu[0].header["NUCK" + id_str])

                self.density.setdefault(Z, {})
                self.density[Z].setdefault(A, {})
                self.density[Z][A].setdefault(K, [])

                d = (data[i - 1, :, 0, inds].swapaxes(0, 1) *
                     np.array(weights)).sum(axis=1)

                self.density[Z][A][K].append(d / self.energy**2)

        # --------------------------------------------------------
        # -------------------- SYNCHROTRON ------------------------
        # --------------------------------------------------------
        if parser == "synchrotron":
            self.synchrotron = {}

            # Standard skymap
            if os.path.exists(os.path.join(fitsdir, "synchrotron_skymap_" + galdefid)):
                sf = pyfits.open(os.path.join(fitsdir, "synchrotron_skymap_" + galdefid))
                self.synchrotron["total"] = (
                    sf[0].data.sum(axis=3).sum(axis=2)[0] /
                    (sf[0].data.shape[2] * sf[0].data.shape[3])
                )
                self.synchrotron["nu"] = 10 ** (
                    float(sf[0].header["CRVAL3"]) +
                    np.arange(int(sf[0].header["NAXIS3"])) *
                    float(sf[0].header["CDELT3"])
                )

                # Q, U components
                for pol in ("Q", "U"):
                    fname = f"synchrotron_{pol}_skymap_{galdefid}"
                    if os.path.exists(os.path.join(fitsdir, fname)):
                        sf = pyfits.open(os.path.join(fitsdir, fname))
                        self.synchrotron[pol] = (
                            sf[0].data.sum(axis=3).sum(axis=2)[0] /
                            (sf[0].data.shape[2] * sf[0].data.shape[3])
                        )

            # Mapcube
            elif os.path.exists(os.path.join(fitsdir, "synchrotron_mapcube_" + galdefid)):
                sf = pyfits.open(os.path.join(fitsdir, "synchrotron_mapcube_" + galdefid))
                self.synchrotron["total"] = (
                    sf[0].data.sum(axis=2).sum(axis=1) /
                    (sf[0].data.shape[1] * sf[0].data.shape[2])
                )
                self.synchrotron["nu"] = sf[1].data.field(0)

                for pol in ("Q", "U"):
                    fname = f"synchrotron_{pol}_mapcube_{galdefid}"
                    if os.path.exists(os.path.join(fitsdir, fname)):
                        sf = pyfits.open(os.path.join(fitsdir, fname))
                        self.synchrotron[pol] = (
                            sf[0].data.sum(axis=2).sum(axis=1) /
                            (sf[0].data.shape[1] * sf[0].data.shape[2])
                        )

            # Healpix
            elif os.path.exists(os.path.join(fitsdir, "synchrotron_healpix_" + galdefid)):
                sf = pyfits.open(os.path.join(fitsdir, "synchrotron_healpix_" + galdefid))
                self.synchrotron["total"] = (
                    sf[1].data.field(0).sum(axis=0) / float(sf[1].header["NAXIS2"])
                )
                self.synchrotron["nu"] = sf[2].data.field(0)

                for pol in ("Q", "U"):
                    fname = f"synchrotron_{pol}_healpix_{galdefid}"
                    if os.path.exists(os.path.join(fitsdir, fname)):
                        sf = pyfits.open(os.path.join(fitsdir, fname))
                        self.synchrotron[pol] = (
                            sf[1].data.field(0).sum(axis=0) /
                            float(sf[1].header["NAXIS2"])
                        )

        # --------------------------------------------------------
        # --------------------- GAMMA RAYS ------------------------
        # --------------------------------------------------------
        if parser == "gamma":
            self.gamma_rays = {}

            # Standard skymaps
            file_sets = [
                ("ics_isotropic", "IC"),
                ("bremss", "bremss"),
                ("pion_decay", "pion_decay")
            ]

            for fname, key in file_sets:
                full = os.path.join(fitsdir, f"{fname}_skymap_{galdefid}")
                if os.path.exists(full):
                    sf = pyfits.open(full)
                    energy = 10 ** (
                        float(sf[0].header["CRVAL3"]) +
                        np.arange(int(sf[0].header["NAXIS3"])) *
                        float(sf[0].header["CDELT3"])
                    )
                    b = float(sf[0].header["CRVAL2"]) + np.arange(
                        int(sf[0].header["NAXIS2"])
                    ) * float(sf[0].header["CDELT2"])

                    self.gamma_rays["energy"] = energy
                    self.gamma_rays[key] = (
                        (sf[0].data.sum(axis=3) *
                         np.sin(np.pi/2 - np.radians(b))).sum(axis=2)[0] /
                        sf[0].data.shape[3] /
                        np.sin(np.pi/2 - np.radians(b)).sum() /
                        energy**2
                    )

            # Mapcubes
            for fname, key in file_sets:
                full = os.path.join(fitsdir, f"{fname}_mapcube_{galdefid}")
                if os.path.exists(full):
                    sf = pyfits.open(full)
                    b = (float(sf[0].header["CRVAL2"]) +
                         (np.arange(int(sf[0].header["NAXIS2"])) -
                          int(sf[0].header["CRPIX2"])) *
                         float(sf[0].header["CDELT2"]))

                    self.gamma_rays[key] = (
                        (sf[0].data.sum(axis=2) *
                         np.sin(np.pi/2 - np.radians(b))).sum(axis=1) /
                        sf[0].data.shape[2] /
                        np.sin(np.pi/2 - np.radians(b)).sum()
                    )
                    self.gamma_rays["energy"] = sf[1].data.field(0)

            # Healpix
            for fname, key in file_sets:
                full = os.path.join(fitsdir, f"{fname}_healpix_{galdefid}")
                if os.path.exists(full):
                    sf = pyfits.open(full)
                    self.gamma_rays[key] = (
                        sf[1].data.field(0).sum(axis=0) /
                        float(sf[1].header["NAXIS2"])
                    )
                    self.gamma_rays["energy"] = sf[2].data.field(0)

            # Total
            if "energy" in self.gamma_rays:
                energy = self.gamma_rays["energy"]
                total = np.zeros_like(energy)
                for key in ("IC", "bremss", "pion_decay"):
                    if key in self.gamma_rays:
                        total += self.gamma_rays[key]
                self.gamma_rays["total"] = total

    # --------------------------------------------------------------------
    # ------------------------ CR SPECTRA FUNCTION ------------------------
    # --------------------------------------------------------------------
    def CRspectra(self, Z, A, K=-1, joined=True, phi=0, out_energy=None):

        pmass = 939.0
        emass = 0.5109990615

        if out_energy is None:
            out_energy = self.energy
        else:
            out_energy = np.array(out_energy)

        if Z not in self.density:
            return out_energy, -np.ones((1, len(out_energy))), [(Z, -1, -1)]

        if joined:
            spectra = np.zeros((1, len(self.energy)))
            type_list = [(Z, A, K)]
        else:
            spectra = [np.zeros(len(self.energy))]
            type_list = [(Z, A, K)]

        spectra_brdn = [np.zeros(len(self.energy))]
        type_brdn = [(Z, A, K)]

        if A == -1:
            tmpA = self.density[Z]
        else:
            if A not in self.density[Z]:
                return out_energy, -np.ones((1, len(out_energy))), [(Z, A, -1)]
            tmpA = {A: self.density[Z][A]}

        for aA, vA in tmpA.items():
            if K == -1:
                tmpK = vA
            else:
                if K not in vA:
                    return out_energy, -np.ones((1, len(out_energy))), [(Z, A, K)]
                tmpK = {K: vA[K]}

            for kK, vK in tmpK.items():
                for sp in vK:
                    spectra[0] += sp

                    if not joined:
                        spectra.append(sp)
                        type_list.append((Z, aA, kK))

                    spectra_brdn.append(sp)
                    type_brdn.append((Z, aA, kK))

        out_brdn = np.zeros((len(spectra_brdn), len(out_energy)))

        for index in range(1, len(spectra_brdn)):
            aA = type_brdn[index][1]

            if aA > 0:
                energy_mod = out_energy + abs(Z)*phi/float(aA)
                mass = pmass
            elif aA == 0:
                energy_mod = out_energy + abs(Z)*phi
                mass = emass
            else:
                return out_energy, -np.ones((1, len(out_energy))), [(Z, A, K)]

            for i, en in enumerate(energy_mod):
                if self.energy[0] <= en <= self.energy[-1]:
                    j = np.searchsorted(self.energy, en)
                    j = max(1, j)

                    e0 = self.energy[j - 1]
                    e1 = self.energy[j]

                    y0 = spectra_brdn[index][j - 1]
                    y1 = spectra_brdn[index][j]

                    if y0 > 0 and y1 > 0:
                        sl = math.log(y0 / y1) / math.log(e0 / e1)
                        out_brdn[index, i] = y0 * (en / e0) ** sl
                    elif y0 > 0:
                        out_brdn[index, i] = (e1 - en) * y0 / (e1 - e0)
                    elif y1 > 0:
                        out_brdn[index, i] = (en - e0) * y1 / (e1 - e0)

                    out_brdn[index, i] *= (out_energy[i] * (out_energy[i] + 2 * mass)) / \
                                         (en * (en + 2 * mass))

                    out_brdn[0, i] += out_brdn[index, i]
                else:
                    out_brdn[index, i] = -1
                    out_brdn[0, i] = -1

        # Return only the requested components
        out = np.zeros((len(spectra), len(out_energy)))
        for k in range(len(spectra)):
            out[k] = out_brdn[k]

        return out_energy, out, type_list

    # --------------------------------------------------------------------
    # --------------------------- ABUNDANCES ------------------------------
    # --------------------------------------------------------------------
    def CRIsotopes(self, energy, phi=0):
        en = self.CRspectra(1, 1, joined=True, out_energy=[energy], phi=phi)[0][0]
        pr = self.CRspectra(1, 1, joined=True, out_energy=[energy], phi=phi)[1][0, 0]

        out = []
        for Z, vZ in self.density.items():
            for A, vA in vZ.items():
                fr = self.CRspectra(Z, A, joined=True, out_energy=[energy], phi=phi)[1][0, 0]
                fr /= pr
                out.append((Z, A, fr))

        return out


def require(condition):
    if not condition:
        print("Plot Galprop version 1.03 (Python-3 updated)")
        print("\nUsage: plotgalprop.py fitsdir galdefid parser [...]")
        sys.exit(1)


# ======================================================================
#                               MAIN
# ======================================================================
if __name__ == "__main__":
    argv = sys.argv

    require(len(argv) > 3)

    fitsdir = argv[1]
    galdefid = argv[2]
    parser = argv[3]

    gdsp = galpropDATA(fitsdir, galdefid, parser)

    # --------------------------------------------------------
    # ----------------------- SPECTRA -------------------------
    # --------------------------------------------------------
    if parser == "spectra":
        require(len(argv) == 8)

        phi = float(argv[4])
        Z = int(argv[5])
        A = int(argv[6])
        alpha = float(argv[7])

        energy, spectra, _ = gdsp.CRspectra(Z, A, joined=True, phi=phi)
        spectra *= energy ** alpha

        print("# Energy spectrum (GALPROP)")
        print("# Energy   Flux*E^alpha")
        for k in range(len(energy)):
            print(f"{energy[k]:12.4e} {spectra[0][k]:12.4e}")

    # --------------------------------------------------------
    # ------------------------- RATIOS ------------------------
    # --------------------------------------------------------
    elif parser == "ratios":
        require(len(argv) == 9)

        phi = float(argv[4])
        Z1 = int(argv[5])
        A1list = argv[6].split(",")
        Z2 = int(argv[7])
        A2list = argv[8].split(",")

        spectra1 = None
        spectra2 = None

        for A1 in A1list:
            _, s, _ = gdsp.CRspectra(Z1, int(A1), joined=True, phi=phi)
            if s[0][0] > 0:
                spectra1 = s if spectra1 is None else spectra1 + s

        for A2 in A2list:
            _, s, _ = gdsp.CRspectra(Z2, int(A2), joined=True, phi=phi)
            if s[0][0] > 0:
                spectra2 = s if spectra2 is None else spectra2 + s

        if spectra1 is None or spectra2 is None:
            sys.exit(0)

        energy = gdsp.energy

        print("# Ratio spectra")
        print("# Energy   Ratio")
        for k in range(len(energy)):
            if spectra1[0][k] > 0 and spectra2[0][k] > 0:
                print(f"{energy[k]:12.4e} {spectra1[0][k] / spectra2[0][k]:12.4e}")

    # --------------------------------------------------------
    # ---------------------- ABUNDANCES -----------------------
    # --------------------------------------------------------
    elif parser == "abundances":
        require(len(argv) == 6)

        phi = float(argv[4])
        E = float(argv[5])

        isotopes = gdsp.CRIsotopes(E, phi)
        pr = gdsp.CRspectra(1, 1, joined=True, out_energy=[E], phi=phi)[1][0][0]

        print("# Isotopic abundances (normalized to proton flux)")
        print("# Z   A   frac")
        for iso in isotopes:
            print(f"{iso[0]:6d} {iso[1]:6d} {iso[2]:12.4e}")

    # --------------------------------------------------------
    # ----------------------- GAMMA RAYS ----------------------
    # --------------------------------------------------------
    elif parser == "gamma":
        require(len(argv) == 5)

        alpha = float(argv[4])

        if "energy" in gdsp.gamma_rays:
            E = gdsp.gamma_rays["energy"]
            print("# Gamma-ray spectra")

            for j in range(len(E)):
                line = f"{E[j]:12.4e} {gdsp.gamma_rays['total'][j]*E[j]**alpha:12.4e}"
                if "pion_decay" in gdsp.gamma_rays:
                    line += f" {gdsp.gamma_rays['pion_decay'][j]*E[j]**alpha:12.4e}"
                if "IC" in gdsp.gamma_rays:
                    line += f" {gdsp.gamma_rays['IC'][j]*E[j]**alpha:12.4e}"
                if "bremss" in gdsp.gamma_rays:
                    line += f" {gdsp.gamma_rays['bremss'][j]*E[j]**alpha:12.4e}"
                print(line)

    # --------------------------------------------------------
    # ---------------------- SYNCHROTRON ----------------------
    # --------------------------------------------------------
    elif parser == "synchrotron":
        require(len(argv) == 5)

        alpha = float(argv[4])

        if "total" in gdsp.synchrotron:
            nu = gdsp.synchrotron["nu"]
            total = gdsp.synchrotron["total"]

            print("# Synchrotron spectra")
            print("# Frequency   f^alpha * flux")

            for j in range(len(nu)):
                print(f"{nu[j]:12.4e} {total[j] * nu[j] ** alpha:12.4e}")

