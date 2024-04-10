import math
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('classic')

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

plt.rcParams.update({
                    'text.usetex':True,
                    'font.family':'Helvetica',
                    'font.size':11, ## title (?)
                    'xtick.direction':"in",
                    'ytick.direction':"in",
                    'savefig.transparent':True,
                    'savefig.facecolor':'0.8',
#                    'figure.figsize':(4, 4),
                    'figure.max_open_warning':50, 
                    'figure.constrained_layout.use':True,
                    })



import numpy as np
pi = np.pi

import scipy.constants as spc
pi       = np.pi
c, h, kB = spc.c, spc.h , spc.k
eps0, Z0 = spc.epsilon_0, spc.physical_constants['characteristic impedance of vacuum'][0]
hbar, u  = h/(2*pi), spc.u
 
##################################### global params (immer in SI units [!] )
pi       = np.pi
c, h, kB = spc.c, spc.h , spc.k
eps0, Z0 = spc.epsilon_0, spc.physical_constants['characteristic impedance of vacuum'][0]
hbar, u  = h/(2*pi), spc.u

alpha_Yb =-1.8191526973423665e-39 
mass_Yb  = 174*u
wvlen_Yb = 759e-9

alpha_Dy =-136/2/eps0/c * 1.6488e-41
wvlen_Dy = 1064e-9
mass_Dy  = 162*u

kwargs_Yb = {'alpha':alpha_Yb, 'wvlen':wvlen_Yb, 'mass':mass_Yb}
kwargs_Dy = {'alpha':alpha_Dy, 'wvlen':wvlen_Dy, 'mass':mass_Dy}

##################################### make beam plot

from sim_ODT import beam_profile

z_arr   = np.linspace(-10, 10, 901)*1e-6
waist_i = [12e-6, 24e-6, 48e-6]
power_i = [45, 47, 50]

def U_ofP(*argv, waistx, waisty, plt_pts):
    for index in range(len(power_i)):
        P_i = power_i[index]
        
        params_i = {'waistx':waistx, 'waisty':waisty, 'power':P_i}
        
        atom = argv[0]
        
        if atom == 'Yb': params_i.update(kwargs_Yb)
        if atom == 'Dy': params_i.update(kwargs_Dy)
        
        beam_i = beam_profile(pts_xyz=[z_arr, 0, 0], **params_i)
    
        pot_i   = beam_i.get_U(*argv)
        depth_i = beam_i.get_U0(*argv)
        pl_tlab = r'$w_{0x}$ = %g um, $w_{0y}$ = %g um'%(waistx*1e6, waisty*1e6)
        pl_titl = atom+r': $\alpha$ = %g, $\lambda$ = %g nm'%(params_i['alpha'],params_i['wvlen']*1e9)
        pl_labl = r'$(P, U_0$) = (%g W, %d uK)'%(P_i, depth_i*1e6)

        plt.plot(plt_pts, pot_i*1e6, label=pl_labl)
    plt.ylabel(r'U/$k_B$ (uK)')
    plt.title(pl_titl)
    plt.legend(title=pl_tlab)
    plt.show()

    
U_ofP('Dy', 'kB', waistx=30e-6, waisty=30e-6, plt_pts=z_arr)
U_ofP('Yb', 'kB', waistx=30e-6, waisty=30e-6, plt_pts=z_arr)








