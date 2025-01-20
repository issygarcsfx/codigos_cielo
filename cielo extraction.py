# cielo extraction 

import numpy as np
import matplotlib.pyplot as plt 
import astropy.units as u
from astropy.constants import c, h, k_B, m_p
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import pandas as pd
import os
from astropy.table import Table, vstack
import glob
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
cosmo = FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc, Om0=0.3)

# function

def extraction(home, galaxy, snapshot, simulation, out, SFR_age, fig):

    # Select snapshot and galaxy to extract
    #Get gas particles
    gas = pd.read_parquet(f'{home}{simulation}/data_{galaxy}/info_gas/{snapshot}_infogas')

    #Get galaxy particles
    stars = pd.read_parquet(f'{home}{simulation}/data_{galaxy}/info_galaxy/{snapshot}_infogalaxy')

    #Get stars 
    info = pd.read_parquet(f'{home}{simulation}/data_{galaxy}/info_galaxy/{snapshot}_infogalaxy')

    Re = info.rhm_star.values[0]
    try:
        z = z_at_value(cosmo.lookback_time, info.lookback_time.values[0] * u.Gyr).value
    except:
        z = 0

    print("Redshift of the snapshot is {:.2f}".format(z))
    print(f"Extracting particles from galaxy 147062_2627:")
    print(f"There are {stars.shape[0]} stellar particles")
    print(f"There are {gas.shape[0]} gas particles")
    print(r"The galaxy has a stellar mass of logM={:.2f} Msol".format(np.log10(info.mgal_star_rvir.values[0])))
    print(r"The galaxy has a size of Reff = {} kpc".format(Re))

    # Create ages column 

    stars.loc[:, "ages"] = stars.age - info.lookback_time[0]

    # Calculation of metallicity (mass elements > He / total mass)
    params = list(stars)
    abundances = [s for s in params if "abund_" in s]
    metals = [a for a in abundances if "_H" not in a]  # Select all abundances but H and He
    # Mass of all metals in Msun
    stars.loc[:,"MZ"] = stars[metals].sum(axis=1).values * 1e10 / 0.6710997341149791   # abundances are in 10^10 h^-1 Msun, must be Msun 
    # Metallicity
    stars.loc[:,"Z"] = stars.MZ.values / stars.mass.values


    # MASKS

    ap = 25*np.sqrt(2)
    st_mask = stars.x**2 + stars.y**2 + stars.z**2 <= ap**2
    gas_mask = (gas.x**2 <= ap**2) & (gas.y**2 <= ap**2) & (gas.z**2 <= ap**2)
    gas_mask = gas_mask.values

    # Look at eq of state for gas particles 

    # Compute temperature from 
    gamma = 5./3.
    XH = 0.76
    mu = 4 * m_p / (1 + 3 * XH * 4 * XH * gas['ne'].values)
    T = (gamma - 1) * gas.internal_E.values * (u.km/u.s)**2 / k_B * mu
    T = T.to('K').value
    gas['T'] = T

    # Compute density in  
    rho = (gas.density.values * u.gram / u.cm**3).to("Msun/kpc^3").value / 1e10 / 0.7**2  # rho in units of 10^10h^2Msun kpc-3
    gas['rho_per_kpc'] = rho
    rho_min = rho[gas.SFR > 0].min()

    rho_new = (7e-26 * u.gram / u.cm**3).to("Msun/kpc^3").value / 1e10 #/ 0.7**2


    # Test both recipes 

    #recCIELO = 
    rec8000 = (T < 8000) | (rho / rho_new > 1)
    recT12 = np.log10(T) < 6 + 0.25 * np.log10(rho)
    
    # Separate ISM gas 
    # We use the second recipe (which includes more dust)

    # Select SF gas, only a fraction of this gas carries dust (cold)
    is_SF = gas.SFR > 0

    factor = np.zeros_like(T)
    factor[is_SF] = (1e6 - T[is_SF]) / (1e6 - 1e3)   # colf gas fraction
    # Select non SF gas
    factor[recT12 * ~is_SF] = 1.

    # Dust density (1 = T12 (non SF), 0 = out T12, frac = T12 (SF)) - Msun/pc^3 (SKIRT)
    dusty_density = (gas.density.values * u.gram / u.cm**3).to("Msun/pc^3").value * factor
    gas['rho_dust'] = dusty_density


    # Compute metallicty

    dusty_mass = gas.mass * factor
    gas['M_dust'] = dusty_mass

    dusty_gas = gas[gas_mask & recT12]

    

    # Calculation of metallicity (mass elements > He / total mass)
    params = list(gas)
    abundances = [s for s in params if "abund_" in s]
    metals = [a for a in abundances if "_H" not in a]  # Select all abundances but H and He

    gas["MZ"] = gas[metals].sum(axis=1).values * 1e10 / 0.6710997341149791
    # Metallicity
    gas["Z"] = gas.MZ / gas.mass 
    
    # Calculate metallicity
    # Mass of all metals in Msun
    dusty_gas.loc[:,"MZ"] = dusty_gas[metals].sum(axis=1).values * 1e10 / 0.6710997341149791
    # Metallicity
    dusty_gas.loc[:,"Z"] = dusty_gas.MZ.values / dusty_gas.mass.values 


    # Separate stars and SF regions by age 

    sfr_mask = stars.ages < 1e-2  #(10 Myrs)


    # Softening lenght
    h = 0.5  # 500 pc, units are in kpc
    stars.loc[:,"h"] = np.ones_like(stars.mass) * h

    # Write stars file
    
    stars_filename =f"{out}{simulation}_{galaxy}_stars_age{str(int(1e-2*1e3))}.dat"
    print ("Writing {:,} star particles to {}".format(len(stars), stars_filename))
    starsfile = open(stars_filename, 'w')
    starsfile.write('# SKIRT 9 import format for a particle source with the Bruzual Charlot SED family \n')
    starsfile.write('# \n')
    starsfile.write('# Column 1: x-coordinate (kpc) \n')
    starsfile.write('# Column 2: y-coordinate (kpc) \n')
    starsfile.write('# Column 3: z-coordinate (kpc) \n')
    starsfile.write('# Column 4: smoothing length (kpc) \n')
    starsfile.write('# Column 5: Initial mass (Msun) \n')
    starsfile.write('# Column 6: metallicity (1) \n')
    starsfile.write('# Column 7: age (Gyr) \n')
    starsfile.write('# \n')
    np.savetxt(starsfile, np.column_stack([stars.x.values, 
    stars.y.values, 
    stars.z.values,
    stars.h.values, 
    stars.mass.values, 
    stars.Z.values, 
    stars.ages.values]), 
    fmt=['%12.9g']*7)

    starsfile.close()

    # Write gas file
    gas_filename = f"{out}{simulation}_{galaxy}_gas{str(int(1e-2*1e3))}.dat"

    print ("Writing {:,} gas particles to {}".format(len(dusty_mass), gas_filename))
    gasfile = open(gas_filename, 'w')
    gasfile.write('# Gas particles for a simulated galaxy \n')
    gasfile.write('# SKIRT9 import format for a medium source using M_dust = f_dust x Z x M_gas \n')
    gasfile.write('# \n')
    gasfile.write('# Column 1: x-coordinate (kpc) \n')
    gasfile.write('# Column 2: y-coordinate (kpc) \n')
    gasfile.write('# Column 3: z-coordinate (kpc) \n')
    gasfile.write('# Column 4: smoothing length (kpc) \n')
    gasfile.write('# Column 5: mass (Msun) \n')
    gasfile.write('# Column 6: metallicity (1) \n')
    gasfile.write('# \n')
    np.savetxt(gasfile, np.column_stack([dusty_gas.x.values, 
    dusty_gas.y.values, 
    dusty_gas.z.values, 
    dusty_gas.SmoothingLength.values,
    dusty_gas.M_dust.values,
    dusty_gas.Z.values]), fmt=['%12.9g']*6)

    gasfile.close()

    # SF REGIONS

    # Select aperture and sf regions
    sf_regions = stars[st_mask* sfr_mask]

    # Smoothing lenght
    sf_regions.loc[:,"h"] = np.ones_like(sf_regions.mass) * h

    # SFR
    sf_regions.loc[:,"SFR"] = sf_regions.mass.values / 1e7 ## Msun / 10 Myrs, assuming contsant SFR over last 10 Myrs

    # Metallicity
    # Mass of all metals in Msun
    sf_regions.loc[:,"MZ"] = sf_regions[metals].sum(axis=1).values * 1e10 / 0.6710997341149791
    # Metallicity
    sf_regions.loc[:,"Z"] = sf_regions.MZ.values / sf_regions.mass.values 

    # Compactness
    logC = 5
    sf_regions.loc[:,"logC"] = np.ones_like(sf_regions.mass) * logC

    # ISM
    ism_P = (1e5 * k_B / u.cm**3 * u.K).to("Pa").value   # log10(P/k_B/cm^-3/K) = 5 or 10^11 K/m^3 
    sf_regions.loc[:,"P"] = np.ones_like(sf_regions.mass) * ism_P

    # Covering factor
    fpdr = 0.2
    sf_regions.loc[:,"fpdr"] = np.ones_like(sf_regions.mass) * fpdr

    # Write sf file 
    SFR_filename= f"{out}{simulation}_{galaxy}_SFR_age{str(int(1e-2*1e3))}_MAPPINGS.dat"

    print ("Writing {:,} star forming particles to {}".format(len(sf_regions), SFR_filename))
    SFRfile = open(SFR_filename, 'w')
    SFRfile.write('# SKIRT 9 import format for SFR with MAPPINGS-III: The SF rate and fpdr are not correlated in this file  \n')
    SFRfile.write('# \n')
    SFRfile.write('# Column 1: x-coordinate (kpc) \n')
    SFRfile.write('# Column 2: y-coordinate (kpc) \n')
    SFRfile.write('# Column 3: z-coordinate (kpc) \n')
    SFRfile.write('# Column 4: size h (kpc) \n')
    SFRfile.write('# Column 5: sfr (Msun/yr) \n')
    SFRfile.write('# Column 6: metallicity (1) \n')
    SFRfile.write('# Column 7: compactness (1) \n')
    SFRfile.write('# Column 8: pressure (Pa) \n')
    SFRfile.write('# Column 9: covering fraction (1) \n')
    np.savetxt(SFRfile, np.column_stack([sf_regions.x.values, 
    sf_regions.y.values, 
    sf_regions.z.values, 
    sf_regions.h.values, 
    sf_regions.SFR.values, 
    sf_regions.Z.values, 
    sf_regions.logC.values, 
    sf_regions.P.values, 
    sf_regions.fpdr]), fmt=['%12.9g']*9)

    SFRfile.close()

    # MAKE FIGURE
    Nstars = len(stars)
    Ngas = len(dusty_gas)
    Nsfr = 0
    if SFR_age > 0:
        Nsfr = len(sf_regions)

    fig, ax = plt.subplots(figsize=(6,4), ncols=3, nrows=2, sharex=True, sharey=True)

    ax[0,0].hexbin(stars.x, stars.y, 
                   stars.mass, reduce_C_function=np.sum, norm=LogNorm())

    ax[1,0].hexbin(stars.x, stars.z, 
                   stars.mass, reduce_C_function=np.sum, norm=LogNorm())

    if SFR_age > 0:
        ax[0,1].scatter(sf_regions.x, sf_regions.y, 
                    c=sf_regions.mass, s=5)

        ax[1,1].scatter(sf_regions.x, sf_regions.z, 
                    c=sf_regions.mass, s=5)


    ax[0,2].hexbin(dusty_gas.x, dusty_gas.y, 
                   dusty_gas.M_dust*0.14*dusty_gas.Z, reduce_C_function=np.sum, norm=LogNorm(), cmap='magma')

    ax[1,2].hexbin(dusty_gas.x, dusty_gas.z, 
                   dusty_gas.M_dust*0.14*dusty_gas.Z, reduce_C_function=np.sum, norm=LogNorm(), cmap='magma')

    ax[0,0].set_xlim(-25, 25)
    ax[0,0].set_ylim(-25, 25)
    for i in range(2):
        ax[i,0].set_ylabel("Y [kpc]")
        for j in range(3):
            ax[i,j].tick_params(axis="both", direction="in")
            ax[1,j].set_xlabel("X [kpc]")
            ax[i,j].set_aspect("equal")

    ## Text
    bbox = dict(fc="w", ec="b", alpha=0.8)
    ax[1,0].text(0, 20, "N = {}".format(Nstars), bbox=bbox)
    ax[1,1].text(0, 20, "N = {}".format(Nsfr), bbox=bbox)
    ax[1,2].text(0, 20, "N = {}".format(Ngas), bbox=bbox)

    plt.suptitle(f"Galaxy {simulation}-{galaxy}")
    plt.tight_layout()
    plt.savefig(f'{fig}{simulation}_{galaxy}_age{str(int(SFR_age*1e3))}_maps.png')

    return None

    
# run

home= '/disk3/ptissera/TABLAS_CIELO_JG/TABLAS_CIELO_Corrected/'

snapshot = int(input())

out = f'/home/igarcia/extracted_data_{snapshot}/'

simulations = ['229533_L11', '206641_L11', 'LG1', '147062', 'LG2_CompLG1', '124429_L11']

figs = f'/home/igarcia/images_{snapshot}/'


for simulation in simulations:
    #simulation = 'LG1'   # '229533_L11', '206641_L11', 'LG1', '147062', 'LG2_CompLG1', '124429_L11'
    # Get galaxy list
        haloes = [i.split('/')[-1].split('_')[-1] for i in glob.glob(f'{home}{simulation}/data_*')]
        print(haloes)
        for i in range(len(haloes)):
            galaxy= haloes[i]
            _ = extraction(home, galaxy, snapshot, simulation, out, 1e-2, figs)



