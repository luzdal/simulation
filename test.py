from objects_optics import OPT, oven, beam_laser, beam_atoms
# % %
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display, Latex
from scipy.optimize import curve_fit
from scipy.constants import u, convert_temperature, c, h, hbar, k, g

from scipy.special import erf
from scipy.interpolate import interp1d as interp
# % %
kB ,     pi =        k, np.pi
amu, cm2_m2 = 1.66e-27, 1e4
#-----------------------------------------------------------------
""" define atomic and laser beams, create oven object """
atoms_test = beam_atoms(iso=162, T_hl=1200)
laser_test = beam_laser(diam=1e-3, powr=1e-3, detun=1)
oven_test  = oven(light=laser_test, matter=atoms_test, res='b')
#-----------------------------------------------------------------
Di      = [2.5e-3, 5e-3, 10e-3]
Tem_dat = np.arange(1050, 1350, 50) 
II0_dat = np.array([2, 2.5, 3, 4.15, 6.15, 9.15])/100
color_j = ['r', 'g', 'b']
#-----------------------------------------------------------------
plt.figure(figsize=(5, 5))
plt.title(r'Differential photodetection between '+'\n'+' incident ($I_0$) and transmitted ($I$) intensities', size=14)
plt.plot(Tem_dat, II0_dat, '+', color='k', label='data', lw=2)
plt.xlabel(r'Temperature ($\degree$C)', size=14)
plt.ylabel(r'Relative absorption ($I/I_0$)', size=14)
plt.legend(), plt.grid(), plt.show()
#-----------------------------------------------------------------
plt.figure(figsize=(12, 4))
for j in range(len(Di)):
    laser_j = beam_laser(diam=Di[j], powr=0, detun=0)
    
    Flx_dat, dns_dat = [], []
    
    for i in range(len(Tem_dat)):
        Ti, Ii = Tem_dat[i], II0_dat[i]

        atoms_i = beam_atoms(iso=162, T_hl=Ti)
        oven_i  = oven(light=laser_j, matter=atoms_i, res='b')
        title_i = oven_i.__str__('matter')
        dens_i, flux_i  = oven_i.get_n(Ii), oven_i.get_Phi(Ii)
        
        Flx_dat.append(flux_i), dns_dat.append(dens_i)
    
    #print(Flx_dat, '\n', dns_dat)
    
    legnd_j = oven_i.__str__('light') 
    title_j = "Laser beam (P, det) = ("+legnd_j[23:]
    plt.plot(Tem_dat, Flx_dat, '+', color=color_j[j], label="Laser beam D = "+legnd_j[15:22], lw=2)
    plt.xlabel(r'Temperature ($\degree$C)', size=14)
    plt.ylabel(r'Flux (atoms/sec)', size=14) #plt.ylabel(r'$\Phi  $', size=14)
    plt.legend(), plt.grid()
# plt.title(title_i+"\n"+title_j, size=14)
plt.title(title_i, size=14)
plt.show()

