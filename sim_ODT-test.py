
import numpy as np
import scipy.constants as spc
pi       = np.pi
c, h, kB = spc.c, spc.h , spc.k
eps0, Z0 = spc.epsilon_0, spc.physical_constants['characteristic impedance of vacuum'][0]
hbar, u  = h/(2*pi), spc.u
 

import math
import matplotlib
import matplotlib.pyplot as plt


import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
colors  = get_cmap("Dark2").colors  ## type: matplotlib.colors.ListedColormap
markers = [',', '+', '^', '.', 'o', '*', 'X']
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter
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

import datetime
stamp = (datetime.datetime.today())
from IPython.display import display, Latex


##################################### global params (bro: immer in SI units [!] )
pi       = np.pi
c, h, kB = spc.c, spc.h , spc.k
eps0, Z0 = spc.epsilon_0, spc.physical_constants['characteristic impedance of vacuum'][0]
hbar, u  = h/(2*pi), spc.u

alpha_Yb =-1.8191526973423665e-39 
mass_Yb  = 174*u
wvLen_Yb = 759e-9

alpha_Dy =-136/2/eps0/c * 1.6488e-41
wvLen_Dy = 1064e-9
mass_Dy  = 162*u

kwargs_Yb = {'alpha':alpha_Yb, 'wvLen':wvLen_Yb, 'mass':mass_Yb}
kwargs_Dy = {'alpha':alpha_Dy, 'wvLen':wvLen_Dy, 'mass':mass_Dy}

##################################### gaussian beam functions
def get_wvNum(**kwargs):
    wvLen = kwargs['wvLen']
    return 2*pi/wvLen

def get_rRange(*argv, **kwargs):
    wvLen = kwargs['wvLen']
    if argv:
        if argv[0]=='x': w0 = kwargs['waistx']
        if argv[0]=='y': w0 = kwargs['waisty']
    return pi*w0**2/wvLen

def fnc_invRad(*argv, z_pts, **kwargs):
    z, z0 = z_pts, get_rRange(*argv, **kwargs)
    return z/(z**2 + z0**2) 

def fnc_bWidth(*argv, z_pts, **kwargs):
    z, z0 = z_pts, get_rRange(*argv, **kwargs)
    if argv:
        if argv[0]=='x': w0 = kwargs['waistx']
        if argv[0]=='y': w0 = kwargs['waisty']
    return w0*np.sqrt(1 + (z/z0)**2 )

def fnc_Gphase(*argv, z_pts, **kwargs):
    z, z0 = z_pts, get_rRange(*argv, **kwargs)
    return np.arctan(z/z0)

def get_E0(**kwargs):
    P, w0_x, w0_y = kwargs['power'], kwargs['waistx'], kwargs['waisty']
    return np.sqrt(2*P/pi/w0_x/w0_y)

def get_I0(**kwargs):
    return ( get_E0(**kwargs) )**2

def get_Erec(**kwargs):
    m, k = kwargs['mass'], get_wvNum(**kwargs)
    return ( hbar*k )**2 / (kB*m)

def get_d2j_Ek(*argv, **kwargs):
    if argv[0]=='x' or argv[0]=='y': return 0
    if argv[0]=='z':
        z0x, z0y, E0 = get_rRange('x', **kwargs), get_rRange('y', **kwargs), get_E0(**kwargs)
        return - E0 * ( z0x**-2 + z0y**-2 ) / 2

def get_d2j_expRp(*argv, **kwargs):
    if argv[0]=='z': return 0
    if argv[0]=='x' or argv[0]=='y': 
        w0x, w0y = kwargs['waistx'], kwargs['waisty']
        return -2 * ( w0x**-2 + w0y**-2 )

##################################### beam object
class beam_profile():
    """ kwargs = {'waistx':, 'waisty':, 'power':, 'wvLen':, 'alpha':, 'mass':} """
    def __init__(self, pts_xyz, **kwargs):
        self.kwargs  = kwargs 

        self.xpts, self.xyz_x = pts_xyz[0], (pts_xyz[0], 0, 0)
        self.ypts, self.xyz_y = pts_xyz[1], (0, pts_xyz[1], 0)
        self.zpts, self.xyz_z = pts_xyz[2], (0, 0, pts_xyz[2])

        ## initialize beam characteristics
        self.mass, self.E_rec = kwargs['mass']  , get_Erec(**kwargs)
        self.alpha, self.k    = kwargs['alpha'] , get_wvNum(**kwargs)
        self.w0_x , self.z0_x = kwargs['waistx'], get_rRange('x', **kwargs)
        self.w0_y , self.z0_y = kwargs['waisty'], get_rRange('y', **kwargs) 
        self.wvLen, self.E0   = kwargs['wvLen'] , get_E0(**kwargs)
        self.P    , self.I0   = kwargs['power'] , get_I0(**kwargs)
        
    def invRad_xy(self, *argv):
        z = self.zpts
        if 'x' in argv: x, y, z = self.xyz_x
        if 'y' in argv: x, y, z = self.xyz_y
        if 'z' in argv: x, y, z = self.xyz_z
        return fnc_invRad(*argv[0], z_pts = z, **self.kwargs)

    def bWidth_xy(self, *argv):
        z = self.zpts
        if 'x' in argv: x, y, z = self.xyz_x
        if 'y' in argv: x, y, z = self.xyz_y
        if 'z' in argv: x, y, z = self.xyz_z
        return fnc_bWidth(*argv[0], z_pts = z, **self.kwargs)

    def Gphase_xy(self, *argv):
        z = self.zpts
        if 'x' in argv: x, y, z = self.xyz_x
        if 'y' in argv: x, y, z = self.xyz_y
        if 'z' in argv: x, y, z = self.xyz_z
        return fnc_Gphase(*argv[0], z_pts = z, **self.kwargs)

    def Rphase(self, *argv):
        x, y = self.xpts, self.ypts 
        if 'x' in argv: x, y, z = self.xyz_x
        if 'y' in argv: x, y, z = self.xyz_y
        if 'z' in argv: x, y, z = self.xyz_z
        return (x/self.bWidth_xy('x', *argv))**2 + (y/self.bWidth_xy('y', *argv))**2 

    def Iphase(self, *argv):
        x, y, z, k = self.xpts , self.ypts, self.zpts, self.k 
        if 'x' in argv: x, y, z = self.xyz_x
        if 'y' in argv: x, y, z = self.xyz_y
        if 'z' in argv: x, y, z = self.xyz_z
        ########## define components
        Ip_1, Ip_2, Ip_3 = (k*z, k*( (self.invRad_xy('x', *argv) *x**2) + (self.invRad_xy('y', *argv) *y**2) )/2,
                           ( self.Gphase_xy('x', *argv) + self.Gphase_xy('y', *argv) )/2 )
        return Ip_1+Ip_2-Ip_3
    
    def gauss_Ek(self, *argv):
        return np.sqrt( 2*self.P/pi/self.bWidth_xy('x', *argv)/self.bWidth_xy('y', *argv) )

    def gauss_E(self, *argv):
        E_k, phase_Re, phase_Im = self.gauss_Ek(*argv), self.Rphase(*argv)  , self.Iphase(*argv)
        return E_k*np.exp(-phase_Re-1j*phase_Im)
    
    def gauss_I(self, *argv):
        E = self.gauss_E(*argv)
        return (np.abs(E**2))

    def get_U(self, *argv):
        if argv:
            I, alpha = self.gauss_I(*argv), self.alpha
            U  = alpha*I/(2*eps0*c)
            if 'Erec' in argv: return U/self.E_rec 
            elif 'kB' in argv: return U/kB
            else: return U
    
    def get_U0(self, *argv):
        #print(argv)
        Umax, Umin = np.amax(self.get_U(*argv)), np.amin(self.get_U(*argv))
        return (Umax-Umin) 

    def get_d2jI(self, *argv):
        E0, d2jEk, d2jexp = self.E0, get_d2j_Ek(*argv, **self.kwargs), get_d2j_expRp(*argv, **self.kwargs)
        return 2*E0 * ( E0*d2jexp + d2jEk )

    def get_Uf(self, *argv):
        d2jI, alpha, m = self.get_d2jI(*argv), self.alpha, self.mass  
        return  np.sqrt( alpha*d2jI / 2 / m / eps0 / c )

############################################# 
xmax, xstp = 300, 500 
x_arr   = np.linspace(-xmax, xmax, xstp)*1e-6
y_arr   = np.linspace(-xmax, xmax, xstp)*1e-6
z_arr   = np.linspace(-2*xmax, 2*xmax, 2*xstp)*1e-6

def U_ofP(*argv, power, waistx, waisty, pts_xyz):
    pow_i, w0x_j, w0y_j = power, waistx, waisty
    x_pts, y_pts, z_pts = pts_xyz
    atom                = argv[0]
    fig, rowx, colx, lc = plt.figure(), 3, 4, 'center left'
    fig.set_figheight(6), fig.set_figwidth(6)
    ax0 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 0), colspan=2, rowspan=1) 
    ax1 = plt.subplot2grid(shape=(rowx, colx), loc=(1, 0), colspan=2, rowspan=1) 
    ax2 = plt.subplot2grid(shape=(rowx, colx), loc=(2, 0), colspan=2, rowspan=1)
    box0, box1, box2 = ax0.get_position(), ax1.get_position(), ax2.get_position()
    # ax10, ax11, ax12 = ax0.twinx(), ax1.twinx(), ax2.twinx()
    # ax10.set_axis_off(), ax11.set_axis_off(), ax12.set_axis_off()
    for i in range(len(pow_i)):
        for j in range(len(w0x_j)):
            ls_j, ls_i = ['-', '--'], ['-', '-']

            power_i, waistx_j, waisty_j, params_ji = pow_i[i], w0x_j[j], w0y_j[j], {'waistx':w0x_j[j], 'waisty':w0y_j[j], 'power':pow_i[i]}

            if atom == 'Yb': params_ji.update(kwargs_Yb)
            if atom == 'Dy': params_ji.update(kwargs_Dy)

            beam_ji = beam_profile(pts_xyz=pts_xyz, **params_ji)

            U_x, U0_x, Uf_x = beam_ji.get_U('x', 'kB'), beam_ji.get_U0('x', 'kB'), beam_ji.get_Uf('x', 'kB')
            U_y, U0_y, Uf_y = beam_ji.get_U('y', 'kB'), beam_ji.get_U0('y', 'kB'), beam_ji.get_Uf('y', 'kB')
            U_z, U0_z, Uf_z = beam_ji.get_U('z', 'kB'), beam_ji.get_U0('z', 'kB'), beam_ji.get_Uf('z', 'kB')

            kx,          = np.where(U_x == np.amin(U_x))
            xmin, U_xmin = x_pts[kx]*1e6, [np.amin(U_x[kx])*1e6]
            if len(kx)>1 : xmin = xmin[0]

            sl, tl = r'[$P, w_{0y}$] ; ($\Delta U, f_0$)', r'\bf ' + atom + r': \bf $\alpha$ = %g, $\lambda$ = %g nm, $w_{0x}$ = %g um'%(params_ji['alpha'],params_ji['wvLen']*1e9, waistx_j*1e6)

            xl_x, yl_x, sl_x = r'$x$ (um)', r'$U_x$ (uK/$k_B$)', r'[%g W, %g um] ; (%d uK,  %0.2f kHz)'%(power_i, waisty_j*1e6, U0_x*1e6, Uf_x*1e-3)
            xl_y, yl_y, sl_y = r'$y$ (um)', r'$U_y$ (uK/$k_B$)', r'[%g W, %g um] ; (%d uK,  %0.2f kHz)'%(power_i, waisty_j*1e6, U0_y*1e6, Uf_y*1e-3)
            xl_z, yl_z, sl_z = r'$z$ (um)', r'$U_z$ (uK/$k_B$)', r'[%g W, %g um] ; (%d uK,  %0.2f kHz)'%(power_i, waisty_j*1e6, U0_z*1e6, Uf_z*1e-3) 
            
            ax0.plot(x_pts*1e6, U_x*1e6, ls=ls_j[j], color=colors[i], label=sl_x), ax0.set_xlabel(xl_x), ax0.set_ylabel(yl_x)
            ax1.plot(y_pts*1e6, U_y*1e6, ls=ls_j[j], color=colors[i], label=sl_y), ax1.set_xlabel(xl_y), ax1.set_ylabel(yl_y)
            ax2.plot(z_pts*1e6, U_z*1e6, ls=ls_j[j], color=colors[i], label=sl_z), ax2.set_xlabel(xl_z), ax2.set_ylabel(yl_z)

    if 'print' in argv:  
        print(r'fx:%g, U0x:%g, Ux:%g ; '%( Uf_x, U0_x, U_x[0] ))
        print(r'fy:%g, U0y:%g, Uy:%g ; '%( Uf_y, U0_y, U_y[0] ))
        print(r'fz:%g, U0z:%g, Uz:%g ; '%( Uf_z, U0_z, U_z[0] ))
        return
    else: 
        ax0.legend(loc=lc, bbox_to_anchor=(1.05, 0.5), title=sl), ax0.set_xlim([- 50, 50]), ax0.set_yticks([0, U_xmin[0]/2, U_xmin[0]])
        ax1.legend(loc=lc, bbox_to_anchor=(1.05, 0.5), title=sl), ax1.set_xlim([- 50, 50]), ax1.set_yticks([0, U_xmin[0]/2, U_xmin[0]])
        ax2.legend(loc=lc, bbox_to_anchor=(1.05, 0.5), title=sl), ax2.set_xlim([-500,500]), ax2.set_yticks([0, U_xmin[0]/2, U_xmin[0]])
        plt.subplots_adjust( hspace=0.45 ), plt.suptitle(tl, x=0.5, y=0.925), fig.tight_layout()
        if 'save' in argv: return fig.savefig("Plt_"+str(stamp)+".pdf", bbox_inches='tight', transparent=True)
        else: plt.show()

U_ofP('Dy', 'save', power=[30, 40], waistx=[30e-6, 30e-6], waisty=[30e-6, 15e-6], pts_xyz=[x_arr, y_arr, z_arr])
# U_ofP('Yb', 'save', power=[3, 4], waistx=[30e-6, 30e-6], waisty=[30e-6, 15e-6], pts_xyz=[x_arr, y_arr, z_arr])
