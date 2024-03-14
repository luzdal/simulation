import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display, Latex
from scipy.optimize import curve_fit
from scipy.constants import u, convert_temperature, c, h, hbar, k, atm, bar, torr, mmHg

from scipy.special import erf
from scipy.interpolate import interp1d as interp
# % %
import plotly.graph_objects as go
# % %
from matplotlib.cm import get_cmap
name = "Dark2"#"Pastel1"
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list

#############################################
################### plotz ###################
# ***************************************** # 0
def plts_0lin(*argv, **kwargs):
    if 'sos' in argv: print("plts_0(**{'xp':T_pts,'yp':n_pts,'xl':'x-axis','yl':'y-axis','tl':'title'})")
    xl, yl = 'x-axis', 'y-axis'
    if 'xl' in kwargs: xl = kwargs['xl']
    if 'yl' in kwargs: yl = kwargs['yl']
    if 'tl' in kwargs: tl = kwargs['tl']
    if 'xp' and 'yp' in kwargs: 
        xp, yp, fig = kwargs['xp'], kwargs['yp'], plt.figure(figsize=(10, 10))
        ax1  = fig.add_subplot(221)
        ax1.plot(xp, yp, '.', lw=2), ax1.set_xlabel(xl, fontsize=15), ax1.set_ylabel(yl, fontsize=15)
        ax1.tick_params(axis='both', which='major', labelsize=13)
        ax1.set_title(tl, horizontalalignment='center', verticalalignment='baseline', fontsize=16)
        fig.tight_layout(), display(fig)
    else: print('\n ** missing \'xp\' AND/OR \'yp\' **')

def plts_0log(*argv, **kwargs):
    if 'sos' in argv: print("plts_0(**{'xp':T_pts,'yp':n_pts,'xl':'x-axis','yl':'y-axis','tl':'title'})")
    xl, yl = 'x-axis', 'y-axis'
    if 'xl' in kwargs: xl = kwargs['xl']
    if 'yl' in kwargs: yl = kwargs['yl']
    if 'tl' in kwargs: tl = kwargs['tl']
    if 'xp' and 'yp' in kwargs: 
        xp, yp, fig = kwargs['xp'], kwargs['yp'], plt.figure(figsize=(10, 10))
        ax1  = fig.add_subplot(221)
        ax1.plot(xp, yp, '.', lw=2), ax1.set_xlabel(xl, fontsize=15), ax1.set_ylabel(yl, fontsize=15)
        ax1.tick_params(axis='both', which='major', labelsize=13)
        ax1.set_title(tl, horizontalalignment='center', verticalalignment='baseline', fontsize=16)
        ax1.set_yscale('log')
        fig.tight_layout(), display(fig)
    else: print('\n ** missing \'xp\' AND/OR \'yp\' **')

def plts_1stats(*argv, **kwargs):
    if argv:
        if argv[0]=='sos':
            print("plts_1stats(, **{'iso': , 'T_arr': , 'v_arr': })")
        else:
            T_arr, v_arr, iso = kwargs['T_arr'], kwargs['v_arr'], kwargs['iso']

            fig1 = go.Figure()
            fig1 = plt.figure(figsize=(10, 5))
            ax1, ax2  = fig1.add_subplot(221), fig1.add_subplot(222)
            for ii in range(len(T_arr)):
                stats_i = stats(matter=beam_atoms(iso=iso, T_hl=T_arr[ii]))
                lab_i   = "T (%d K, %d $\degree$C)"%(stats_i.T, stats_i.Tc)

                fv, cv  = stats_i.MB_PDF(v=v_arr), stats_i.MB_CDF(v=v_arr)

                ax1.plot(v_arr, fv, label=lab_i,lw=2, color=colors[ii]), ax1.legend(loc='center right')
                ax2.plot(v_arr, cv, label=lab_i,lw=2, color=colors[ii]), ax2.legend(loc='center right')

            ax1.title.set_text('Probability Density Function')
            ax1.set_xlabel('v [m/s]'), ax1.set_ylabel(r'$\rho(v)$ [s/m]')

            ax2.title.set_text('Cummulative Distribution Function')
            ax2.set_xlabel('v [m/s]'), ax2.set_ylabel(r'$\int\rho(v)dv$')
            fig1.tight_layout(), display(fig1)
    
def plts_2stats(*argv, **kwargs):
    if argv:
        if argv[0]=='sos':
            print("plts_2stats(, **{'iso': , 'T_pt': , 'v_arr': })")
        else:
            T_pt, v_arr, iso = kwargs['T_pt'], kwargs['v_arr'], kwargs['iso']

            fig = go.Figure()
            fig = plt.figure(figsize=(10, 5))
            ax1, ax2  = fig.add_subplot(221), fig.add_subplot(222)

            stats_2 = stats(matter=beam_atoms(iso=iso, T_hl=T_pt))
            titl_2  = ("Dy-"+str(stats_2.iso)
                       +" at T (%d K, %d $\degree$C)"%(stats_2.T, stats_2.Tc))
            lab_2   = r"||v|| "
            fit_2   = 'fit\Gauss'

            """ histograms """
            vs_rand, vz_rand = stats_2.get_vrand('vs', v=v_arr), stats_2.get_vrand('vz', v=v_arr)
            vx_rand, vy_rand = stats_2.get_vrand('vx', v=v_arr), stats_2.get_vrand('vy', v=v_arr)
            ## note: vxy_rand has random angle
            ax1.hist(vs_rand, bins=50, normed=1, label=lab_2   , fc='r', alpha=0.3, lw=0.2)
            ax2.hist(vz_rand, bins=50, normed=1, label=r'v$_z$', fc='b', alpha=0.3, lw=0.2)

            """ fits """
            vs, vz      = stats_2.get_vsz()[0], stats_2.get_vsz()[1]
            f_vs,  f_vz = stats_2.MB_PDF(v=vs), stats_2.MB_z(v=vz)
            ## note: vxy = vz
            ax1.plot(vs, f_vs, 'k--', lw=2, label=fit_2), ax2.plot(vz, f_vz, 'k--', lw=2, label=fit_2)

            ##
            loc1   = 'upper right'
            loc3 = loc1
            ax1.legend(loc=loc1), ax2.legend(loc=loc1)

            ax1.set_xlabel('Speed (m/s)') , ax1.set_ylabel('PDF')
            ax2.set_xlim(int(min(vz)), int(max(vz))) 
            ax2.set_xlabel(r'1d velocity (m/s)'), ax2.set_ylabel('Probability density')

            fig.suptitle(titl_2, horizontalalignment='center', verticalalignment='baseline')
            fig.tight_layout(), display(fig)  
            
def plts_3stats(*argv, **kwargs):
    if argv:
        if argv[0]=='sos':
            print("plts_2stats(, **{'iso': , 'T_pt': , 'v_arr': })")
        else:
            T_pt, v_arr, iso = kwargs['T_pt'], kwargs['v_arr'], kwargs['iso']

            fig = go.Figure()
            fig = plt.figure(figsize=(8, 8))
            ax1, ax2  = fig.add_subplot(321), fig.add_subplot(322)
            ax3, ax4  = fig.add_subplot(326), fig.add_subplot(324)

            stats_2 = stats(matter=beam_atoms(iso=iso, T_hl=T_pt))
            titl_2  = ("Dy-"+str(stats_2.iso)
                       +" at T (%d K, %d $\degree$C)"%(stats_2.T, stats_2.Tc))
            lab_2   = r"||v|| "
            fit_2   = 'fit\Gauss'

            """ histograms """
            vs_rand, vz_rand = stats_2.get_vrand('vs', v=v_arr), stats_2.get_vrand('vz', v=v_arr)
            vx_rand, vy_rand = stats_2.get_vrand('vx', v=v_arr), stats_2.get_vrand('vy', v=v_arr)
            ## note: vxy_rand has random angle
            ax1.hist(vs_rand, bins=50, normed=1, label=lab_2   , fc='r', alpha=0.3, lw=0.2)
            ax2.hist(vz_rand, bins=50, normed=1, label=r'v$_z$', fc='b', alpha=0.3, lw=0.2)
            ax3.hist(vx_rand, bins=50, normed=1, label=r'v$_x$', fc='b', alpha=0.3, lw=0.2)
            ax4.hist(vy_rand, bins=50, normed=1, label=r'v$_y$', fc='b', alpha=0.3, lw=0.2)

            """ fits """
            vs, vz      = stats_2.get_vsz()[0], stats_2.get_vsz()[1]
            f_vs,  f_vz = stats_2.MB_PDF(v=vs), stats_2.MB_z(v=vz)
            ## note: vxy = vz
            ax1.plot(vs, f_vs, 'k--', lw=2, label=fit_2), ax2.plot(vz, f_vz, 'k--', lw=2, label=fit_2)
            ax3.plot(vz, f_vz, 'k--', lw=2, label=fit_2), ax4.plot(vz, f_vz, 'k--', lw=2, label=fit_2)

            ##
            loc1   = 'upper right'
            loc3 = loc1
            ax1.legend(loc=loc1), ax2.legend(loc=loc3)
            ax3.legend(loc=loc3), ax4.legend(loc=loc3)

            ax1.set_xlabel('Speed (m/s)') , ax1.set_ylabel('PDF')
            ax2.set_xlim(int(min(vz)), int(max(vz))) 
            ax3.set_xlabel(r'1d velocity (m/s)'), ax2.set_ylabel('Probability density')

            fig.suptitle(titl_2, horizontalalignment='center', verticalalignment='baseline')
            fig.tight_layout(), display(fig)