import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import u, convert_temperature, c, h, hbar, k

from scipy.special import erf
from scipy.interpolate import interp1d as interp

import pylab as pl
from matplotlib.path import Path
import matplotlib.path as mpath 
import matplotlib.patches as patches

from matplotlib.cm import get_cmap

pi = np.pi

# name = "Accent"
name = "Pastel1"
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list


def get_setups(lab, apertures, *argv):
    if argv[0]=='lab' : print(lab)
    if argv[0]=='aps' : print(apertures)
    if argv[0]=='sos' : print('[argument options]: '
                             +'\n[1]: ovenX(\'lab\')'
                             +'\n[2]: ovenX(\'aps\') '
                             +'\n[3]: ovenX(\'\', {\'test_zpts\':#, \'zax\':[min#, max#, steps#]}) ') 
    elif argv[0]=='': return get_dimensions([apertures, lab], **argv[1])

def ovenA(*argv):
    lab, apertures = 'Dy/MQM: ', ({'tag':'oven\\ecell','z': -50,'y': 3},
                                  {'tag':'oven\\out'  ,'z': 0    ,'y': 3},
                                  {'tag':'TC\\in'     ,'z': 50   ,'y': 10,},
                                  {'tag':'TC\\out'    ,'z': 30   ,'y': 10,},
                                  {'tag':'test'       ,'z': 50   ,'y': 10},
                                  {'tag':'test'       ,'z': 50   ,'y': 10})
    return get_setups(lab, apertures, *argv)


def ovenB(*argv):
    lab, apertures = 'Er/Innsb: ', ({'tag':'oven\\EC'     ,'z': -49.6,'y': 3},
                                    {'tag':'oven'         ,'z': 0    ,'y': 3},
                                    {'tag':'TC chamber'   ,'z': 66   ,'y': 18,},
                                    {'tag':'test aperture','z': 100   ,'y': 10})
    return get_setups(lab, apertures, *argv)


class aperture():
    def __init__(self, **kwargs):
        self.pos, self.hig, self.name = kwargs['z'], kwargs['y'], kwargs['tag']

def get_dimensions(*argv, **kwargs):
    
    ap_array, Title, xpt_i, zax = argv[0][0], argv[0][1], kwargs['test_zpt'], kwargs['zax']
    
    A_n, X_n, Y_n   = [0]*len(ap_array), [0]*len(ap_array), [0]*len(ap_array)

    for i in range(len(ap_array)): X_n[i], Y_n[i], A_n[i] = (aperture(**ap_array[i]).pos, 
                                                             aperture(**ap_array[i]).hig/2,
                                                             aperture(**ap_array[i]).name)
    dic_A = {}
    for j in range(0, len(A_n)):  dic_A.update({r'A'+str(j):A_n[j]})
    
    tag_A, nam_A = [0]*len(dic_A), [0]*len(dic_A)
    for k in range(1, len(dic_A)):tag_A[k] = (str(list(dic_A.keys())[k])+' = '+str(list(dic_A.values())[k]))
        
    Tag_n = str(tag_A[1:])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x0, x1, y0, y1 =  X_n[0], X_n[1], -Y_n[0], Y_n[1]
    
    ## atomic beam params
    slp   ,  ang = (y1-y0)/(x1-x0)       , np.arctan((y1-y0)/(x1-x0))*180/pi
    dpt_i ,  ypt_i = np.arange(x0, xpt_i+1), np.arange(x0, xpt_i+1)*slp + y1
    
    d_thry = 2*max(ypt_i)

    ## atomic beam region coordinates
    x_l, x_L, x_R = x0, x1, max(dpt_i)
    y_d, y_D, y_U = y0, y1, max(ypt_i)
    ## shade atomic beam region
    ov_shade, ov_opty = 'lightgray', 0.2
    pp1 = plt.Polygon([[x_L, -y_D], [x_R, -y_D], [x_R, -y_U]], color=ov_shade, alpha=ov_opty, lw=0) 
    pp2 = plt.Polygon([[x_L,  y_D], [x_R,  y_D], [x_R,  y_U]], color=ov_shade, alpha=ov_opty, lw=0)
    pp3 = plt.Polygon([[x_l,  y_d], [x_l, -y_d], [x_L,  y_D]], color=ov_shade, alpha=ov_opty, lw=0) 
    pp4 = plt.Polygon([[x_l,  y_d], [x_L,  y_d], [x_L,  y_D]], color=ov_shade, alpha=ov_opty, lw=0) 
    pp0 = plt.Rectangle((x_l, y0), xpt_i-x_l, 2*y1, color=ov_shade, alpha=ov_opty*2, lw=0) 
    ax.add_patch(pp0), ax.add_patch(pp1), ax.add_patch(pp2), ax.add_patch(pp3), ax.add_patch(pp4)
    
    ## plot oven dashed lines
    ov_lin_c = 'k'
    ov_lin_s = 'dotted'
    ax_label0 =  r'$D_i$ = {0:0.1f}'.format(d_thry)+r'mm at $z_i$ = {0:0.0f}'.format(xpt_i)+"mm"
    ax.plot([xpt_i, xpt_i],  [max(ypt_i), -1*max(ypt_i)], color='k', linestyle='dashed', label=ax_label0) 
    
    ## plot oven vertical lines 
    ax_label1 = (r'($A_'+str(1)+r'$, $2\theta_'+str(1)+r'$) = ({0:0.0f}'.format(2*y1)
                 +'mm, {0:0.1f}'.format(2*ang)  +r'$\degree$) '+tag_A[1][5:])
    
    ax.plot([x0, x0], [y0, y1], color=ov_lin_c, label=ax_label1) 
    
    lenX_tot = [-x0/2]
    for ii in range(len(X_n)):
        ## apertures after oven
        if ii>1:
            lenX_tot.append(X_n[ii])
            lenX_i = np.sum(lenX_tot)
            
            X_fr, X_to, Y_fr, Y_to = x0/2, np.sum(lenX_tot)-lenX_tot[0], 0, Y_n[ii]
            
            ang_n = np.arctan( (Y_to-Y_fr)/(X_to-X_fr) )*180/pi
            c_i   = colors[ii-2]
            ax_labeln = (r'($A_'+str(ii)+r'$, $2\theta_'+str(ii)+r')$ = ({0:0.0f}'.format(2*Y_to)
                        +'mm, {0:0.1f}'.format(2*ang_n)  +r'$\degree$) '+tag_A[ii][5:])
            ## plot dashed lines
            ax.plot([X_fr, X_to], [Y_fr, Y_to],  color=c_i, linestyle='dotted')
            ax.plot([X_fr, X_to], [Y_fr, -Y_to], color=c_i, linestyle='dotted')
            ## plot vertical lines
            ax.plot([X_to, X_to],  [-Y_to, Y_to], color=c_i, label=ax_labeln, lw=4) 
            
    ## plot oven dotted lines
    ax.plot(dpt_i,  ypt_i, color=ov_lin_c, linestyle=ov_lin_s) 
    ax.plot(dpt_i, -ypt_i, color=ov_lin_c, linestyle=ov_lin_s) 
    
    ax.plot([x1, x1], [y0, y1], color=ov_lin_c)
    ax.set_ylabel(r'diameter (mm)', fontsize=14)
    ax.set_xlabel(r'axial position $z$ (mm)', fontsize=14)
    ax.set_xticks(np.arange(zax[0], zax[1], zax[2]))
    plt.title('\n'+'Atomic beam collimation setup:'+Title, fontsize=15)
    plt.legend(fontsize=13, loc='upper left'),  plt.grid(), plt.show()