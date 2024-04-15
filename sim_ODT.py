
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

##################################### global params (immer in SI units [!] )
pi       = np.pi
c, h, kB = spc.c, spc.h , spc.k
eps0, Z0 = spc.epsilon_0, spc.physical_constants['characteristic impedance of vacuum'][0]
hbar, u  = h/(2*pi), spc.u

alpha_Yb =-1.8191526973423665e-39 
mass_Yb  = 174*u
wvLen_Yb = 759e-9

 

alpha_Dy =-199 * 1.6488e-41
# alpha_Dy =-136/2/eps0/c * 1.6488e-41
wvLen_Dy = 1064e-9
mass_Dy  = 162*u

kwargs_Yb = {'alpha':alpha_Yb, 'wvLen':wvLen_Yb, 'mass':mass_Yb}
kwargs_Dy = {'alpha':alpha_Dy, 'wvLen':wvLen_Dy, 'mass':mass_Dy}

##################################### gaussian beam functions
def fnc_wvNum(**kwargs): return 2*pi/kwargs['wvLen']

def fnc_rRange(*argv, **kwargs):
    if argv: 
        if 'x' in argv: w0 = kwargs['waistx']
        if 'y' in argv: w0 = kwargs['waisty']
        ##print('        |--> fnc_rRange <--| ', argv,', (w0=%g)]'%(w0) ) ############################### print 
        return pi*w0**2/kwargs['wvLen']

def fnc_invRad(*argv, z_pts, **kwargs):
    z, z0 = z_pts, fnc_rRange(*argv, **kwargs)
    ##print('        |--> fnc_invRad <--| ', argv,', (z0=%g)]'%(z0) ) ############################### print 
    return z/(z**2 + z0**2) 

def fnc_bWidth(*argv, z_pts, **kwargs):
    z, z0 = z_pts, fnc_rRange(*argv, **kwargs)
    if argv:
        if 'x' in argv: w0 = kwargs['waistx']
        if 'y' in argv: w0 = kwargs['waisty']
        ##print('        |--> fnc_bWidth <--| ', argv,', (w0=%g, z0=%g)]'%(w0,z0) )
        return w0*np.sqrt(1 + (z/z0)**2 )

def fnc_Gphase(*argv, z_pts, **kwargs):
    z, z0 = z_pts, fnc_rRange(*argv, **kwargs)
    ##print('        |--> fnc_Gphase <--| ', argv,', (z0=%g)]'%(z0) ) ############################### print 
    return np.arctan(z/z0)

def fnc_E0(**kwargs):
    P, w0_x, w0_y = kwargs['power'], kwargs['waistx'], kwargs['waisty']
    return np.sqrt(2*P/pi/w0_x/w0_y)

def fnc_I0(**kwargs):
    return ( fnc_E0(**kwargs) )**2

def fnc_Erc(**kwargs):
    m, k = kwargs['mass'], fnc_wvNum(**kwargs)
    return ( hbar*k )**2 / (kB*m)

def fnc_d2j_Ek(*argv, **kwargs):
    ##print('        |--> fnc_d2j_Ek <--| ', argv) ############################### print 
    if argv[0]=='x' or argv[0]=='y': return 0
    if argv[0]=='z':
        z0x, z0y, E0 = fnc_rRange('x', **kwargs), fnc_rRange('y', **kwargs), fnc_E0(**kwargs)
        return - E0 * ( z0x**-2 + z0y**-2 ) / 2

def fnc_d2j_expRp(*argv, **kwargs):
    ##print('        |--> fnc_d2j_expRp <--| ', argv) ############################### print
    if argv[0]=='z': return 0
    if argv[0]=='x' or argv[0]=='y': 
        w0x, w0y = kwargs['waistx'], kwargs['waisty']
        return -2 * ( w0x**-2 + w0y**-2 )

##################################### beam object
class beam_profile():
    """ beam_profile(**{'waistx':, 'waisty':, 'power':, 'wvLen':, 'alpha':, 'mass':}) """
    def __init__(self, pts_xyz, **kwargs):
        self.kwargs  = kwargs 
        #print('\n', '-------------- init --------------')
        self.xpts, self.xyz_x = pts_xyz[0], (pts_xyz[0], 0, 0)
        self.ypts, self.xyz_y = pts_xyz[1], (0, pts_xyz[1], 0)
        self.zpts, self.xyz_z = pts_xyz[2], (0, 0, pts_xyz[2])
        #print('\n', '-------------- init --------------')
        # # pts_0 = np.zeros(len(pts_xyz[0])) 
        self.xx =  (pts_xyz[0], 0)
        self.yy =  (0, pts_xyz[1])

        ## initialize beam characteristics
        self.w0_x , self.w0_y  = kwargs['waistx'], kwargs['waisty']
        self.mass , self.alpha = kwargs['mass']  , kwargs['alpha']  
        self.wvLen, self.P     = kwargs['wvLen'] , kwargs['power'] 
        print('\n', ' ---------------------------- init ----------------------------', '\n')

    def get_E0(self): return fnc_E0(**self.kwargs)
    def get_I0(self): return fnc_I0(**self.kwargs)
    def get_Erc(self): return fnc_Erc(**self.kwargs)
    def get_wvNum(self): return fnc_wvNum(**self.kwargs)
    def get_rRange(self, *argv): return fnc_rRange(*argv, **self.kwargs)
    
    def get_invRad(self, *argv):
        ##print('        ---> get_invRad <---', argv) ################### print 
        z = self.zpts
        if 'x' in argv: x, y, z = self.xyz_x
        if 'y' in argv: x, y, z = self.xyz_y
        if 'z' in argv: x, y, z = self.xyz_z
        return fnc_invRad(*argv[0], z_pts = z, **self.kwargs)

    def get_bWidth(self, *argv):
        ##print('        ---> get_bWidth <---', argv) ################### print 
        z = self.zpts
        if 'x' in argv: x, y, z = self.xyz_x
        if 'y' in argv: x, y, z = self.xyz_y
        if 'z' in argv: x, y, z = self.xyz_z
        return fnc_bWidth(*argv[0], z_pts = z, **self.kwargs)

    def get_Gphase(self, *argv):
        ##print('        ---> get_Gphase <---', argv) ################### print 
        z = self.zpts
        if 'x' in argv: x, y, z = self.xyz_x
        if 'y' in argv: x, y, z = self.xyz_y
        if 'z' in argv: x, y, z = self.xyz_z
        return fnc_Gphase(*argv[0], z_pts = z, **self.kwargs)

    def fnc_Rphase(self, *argv):
        ##print('    * * fnc_Rphase', argv)
        x, W_x = self.xpts, self.get_bWidth('Wx', *argv)
        y, W_y = self.ypts, self.get_bWidth('Wy', *argv)

        if 'x' in argv: x, y, z = self.xyz_x
        if 'y' in argv: x, y, z = self.xyz_y
        if 'z' in argv: x, y, z = self.xyz_z
        return (x/W_x)**2 + (y/W_y)**2 

    def fnc_Iphase(self, *argv):
        ##print('    * * fnc_Iphase', argv)
        x, invRad_x, Gp_x, z = self.xpts, self.get_invRad('Rx', *argv), self.get_Gphase('Gx', *argv), self.zpts
        y, invRad_y, Gp_y, k = self.ypts, self.get_invRad('Ry', *argv), self.get_Gphase('Gy', *argv), self.get_wvNum()

        if 'x' in argv: x, y, z = self.xyz_x
        if 'y' in argv: x, y, z = self.xyz_y
        if 'z' in argv: x, y, z = self.xyz_z
        ########## define components
        Ip_1, Ip_2, Ip_3 = (k*z, k*( (invRad_x *x**2) + (invRad_y *y**2) )/2,
                           ( Gp_x + Gp_y )/2 )
        return Ip_1+Ip_2-Ip_3
    
    def fnc_Ek(self, *argv):
        ##print('    * * fnc_Ek', argv)
        W_x, W_y = self.get_bWidth('Wx', *argv), self.get_bWidth('Wy', *argv)
        return np.sqrt( 2*self.P / pi / W_x / W_y )

    def fnc_E(self, *argv):
        ##print('  * * * fnc_E', argv,' * * * ')
        E_k, phase_Re, phase_Im = self.fnc_Ek(*argv), self.fnc_Rphase(*argv)  , self.fnc_Iphase(*argv)
        return E_k*np.exp(-phase_Re-1j*phase_Im)
    
    def fnc_I(self, *argv):
        ##print('  * * * fnc_I', argv,' * * * ')
        E = self.fnc_E(*argv)
        return (np.abs(E**2))

    def fnc_U(self, *argv):
        ##print('  * * * fnc_U', argv,' * * * ')
        if argv:
            I, alpha = self.fnc_I(*argv), self.alpha
            U  = alpha*I/(2*eps0*c)
            if 'Erc' in argv: return U/self.get_Erc()
            elif 'kB' in argv: return U/kB
            else: return U

    
    def fnc_U0(self, *argv):
        Umax, Umin = np.amax(self.fnc_U(*argv)), np.amin(self.fnc_U(*argv))
        return (Umax-Umin) 

    def fnc_d2jI(self, *argv):
        ##print('    * * fnc_d2jI', argv,' * * * ')
        E0, d2jEk, d2jexp = self.get_E0(), fnc_d2j_Ek(*argv, **self.kwargs), fnc_d2j_expRp(*argv, **self.kwargs)
        return 2*E0 * ( E0*d2jexp + d2jEk )

    def fnc_Uf(self, *argv):
        ##print('  * * * fnc_Uf', argv,' * * * ')
        d2jI, alpha, m = self.fnc_d2jI(*argv), self.alpha, self.mass  
        return  np.sqrt( alpha*d2jI / 2 / m / eps0 / c )


#################################################################
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
    for i in range(len(pow_i)):
        for j in range(len(w0x_j)):
            ls_j, ls_i = ['-', '--'], ['-', '-']

            power_i, waistx_j, waisty_j, params_ji = pow_i[i], w0x_j[j], w0y_j[j], {'waistx':w0x_j[j], 'waisty':w0y_j[j], 'power':pow_i[i]}

            if atom == 'Yb': params_ji.update(kwargs_Yb)
            if atom == 'Dy': params_ji.update(kwargs_Dy)

            beam_ji = beam_profile(pts_xyz=pts_xyz, **params_ji)

            U_x, U0_x, Uf_x = beam_ji.fnc_U('x', 'kB'), beam_ji.fnc_U0('x', 'kB'), beam_ji.fnc_Uf('x', 'kB')
            U_y, U0_y, Uf_y = beam_ji.fnc_U('y', 'kB'), beam_ji.fnc_U0('y', 'kB'), beam_ji.fnc_Uf('y', 'kB')
            U_z, U0_z, Uf_z = beam_ji.fnc_U('z', 'kB'), beam_ji.fnc_U0('z', 'kB'), beam_ji.fnc_Uf('z', 'kB')

            kx,          = np.where(U_x == np.amin(U_x))
            xmin, U_xmin = x_pts[kx], [np.amin(U_x[kx])]
            if len(kx)>1 : xmin = xmin[0]

            sl, tl = r'[$P, w_{0y}$] ; ($\Delta U, f_0$)', r'\bf ' + atom + r': \bf $\alpha$ = %g, $\lambda$ = %g nm, $w_{0x}$ = %g um'%(params_ji['alpha'],params_ji['wvLen']*1e9, waistx_j*1e6)

            xl_x, yl_x, sl_x = r'$x$ (um)', r'$U_x$ (uK/$k_B$)', r'[%g W, %g um] ; (%d uK,  %0.2f kHz)'%(power_i, waisty_j*1e6, U0_x*1e6, Uf_x*1e-3)
            xl_y, yl_y, sl_y = r'$y$ (um)', r'$U_y$ (uK/$k_B$)', r'[%g W, %g um] ; (%d uK,  %0.2f kHz)'%(power_i, waisty_j*1e6, U0_y*1e6, Uf_y*1e-3)
            xl_z, yl_z, sl_z = r'$z$ (um)', r'$U_z$ (uK/$k_B$)', r'[%g W, %g um] ; (%d uK,  %0.2f kHz)'%(power_i, waisty_j*1e6, U0_z*1e6, Uf_z*1e-3) 
            
            ax0.plot(x_pts*1e6, U_x*1e6, ls=ls_j[j], color=colors[i], label=sl_x), ax0.set_xlabel(xl_x), ax0.set_ylabel(yl_x)
            ax1.plot(y_pts*1e6, U_y*1e6, ls=ls_j[j], color=colors[i], label=sl_y), ax1.set_xlabel(xl_y), ax1.set_ylabel(yl_y)
            ax2.plot(z_pts*1e6, U_z*1e6, ls=ls_j[j], color=colors[i], label=sl_z), ax2.set_xlabel(xl_z), ax2.set_ylabel(yl_z)

    if 'print' in argv:  
        #print(r'fx:%g, U0x:%g, Ux:%g ; '%( Uf_x, U0_x, U_x[100] ))
        #print(r'fy:%g, U0y:%g, Uy:%g ; '%( Uf_y, U0_y, U_y[100] ))
        #print(r'fz:%g, U0z:%g, Uz:%g ; '%( Uf_z, U0_z, U_z[100] ))
        print('\n')
        return
    else: 
        ax0.legend(loc=lc, bbox_to_anchor=(1.05, 0.5), title=sl), ax0.set_xlim([- 50, 50]), ax0.set_yticks([0, U_xmin[0]*1e6/2, U_xmin[0]*1e6])
        ax1.legend(loc=lc, bbox_to_anchor=(1.05, 0.5), title=sl), ax1.set_xlim([- 50, 50]), ax1.set_yticks([0, U_xmin[0]*1e6/2, U_xmin[0]*1e6])
        ax2.legend(loc=lc, bbox_to_anchor=(1.05, 0.5), title=sl), ax2.set_xlim([-500,500]), ax2.set_yticks([0, U_xmin[0]*1e6/2, U_xmin[0]*1e6])
        plt.subplots_adjust( hspace=0.45 ), plt.suptitle(tl, x=0.5, y=0.925), fig.tight_layout()
        if 'save' in argv: return fig.savefig("Plt_"+str(stamp)+".pdf", bbox_inches='tight', transparent=True)
        else: plt.show()

U_ofP('Dy', 'plot', power=[2.5], waistx=[30e-6], waisty=[30e-6], pts_xyz=[x_arr, y_arr, z_arr])
# U_ofP('Dy', 'plot', power=[2, 4], waistx=[30e-6, 30e-6], waisty=[30e-6, 15e-6], pts_xyz=[x_arr, y_arr, z_arr])
# U_ofP('Yb', 'save', power=[3, 4], waistx=[30e-6, 30e-6], waisty=[30e-6, 15e-6], pts_xyz=[x_arr, y_arr, z_arr])







# args_1 = {'waistx':30e-6, 'waisty':10e-6, 'power':40, 'wvLen':1060e-9}

# args_1.update(kwargs_Dy)

# beam_1 = beam_profile(**args_1, pts_xyz=[x_arr, y_arr, z_arr])
# beam_1.fnc_Uf('x'), beam_1.fnc_Uf('y'), beam_1.fnc_Uf('z')
# print(Uf_x)
# beam_1.get_rRange('y')
# beam_1.fnc_Iphase('z')
# beam_1.fnc_Rphase('z')
# beam_1.fnc_E('x')
# beam_1.fnc_Uf('x')
# beam_1.fnc_U('x', 'Erc')

# def plot_test():
#     E_x, E_y, E_z  = beam_1.fnc_E('x'), beam_1.fnc_E('y'), beam_1.fnc_E('z')
#     I_x, I_y, I_z  = beam_1.fnc_I('x'), beam_1.fnc_I('y'), beam_1.fnc_I('z')
#     U_x, U_y, U_z  = beam_1.fnc_U('x'), beam_1.fnc_U('y'), beam_1.fnc_U('z')

#     fig, rowx, colx, lc = plt.figure(), 3, 4, 'center left'
#     fig.set_figheight(6), fig.set_figwidth(6)
#     ax0 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 0), colspan=2, rowspan=1) 
#     ax1 = plt.subplot2grid(shape=(rowx, colx), loc=(1, 0), colspan=2, rowspan=1) 
#     ax2 = plt.subplot2grid(shape=(rowx, colx), loc=(2, 0), colspan=2, rowspan=1)
#     box0, box1, box2 = ax0.get_position(), ax1.get_position(), ax2.get_position() 


#     ax0.plot( x_arr, E_x )
#     ax1.plot( x_arr, I_x)
#     ax2.plot( x_arr, U_x)
#     plt.show()

