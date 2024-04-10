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


def get_wvNum(**kwargs):
    wvLen = kwargs['wvlen']
    return 2*pi/wvLen

def get_rRange(*argv, **kwargs):
    lamb = kwargs['wvlen']
    if argv:
        if argv[0]=='x': w0 = kwargs['waistx']
        if argv[0]=='y': w0 = kwargs['waisty']
    return pi*w0**2/lamb

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
    #return ( hbar*k )**2 / (2*m)
    return ( hbar*k )**2 / (kB*m)
    

class beam_profile():
    """ kwargs = {'waistx':, 'waisty':, 'power':, 'wvlen':, 'alpha':, 'mass':} """
    def __init__(self, pts_xyz, **kwargs):
        self.kwargs, (self.xpts, self.ypts, self.zpts) = kwargs, pts_xyz
        ## initialize beam characteristics
        self.alpha, self.k    = kwargs['alpha'] , get_wvNum(**kwargs), 
        self.w0_x , self.z0_x = kwargs['waistx'], get_rRange('x', **kwargs)
        self.w0_y , self.z0_y = kwargs['waisty'], get_rRange('y', **kwargs) 
        self.lamb , self.P    = kwargs['wvlen'] , kwargs['power']
        self.E0   , self.I0   = get_E0(**kwargs), get_I0(**kwargs) 
        self.E_rec            = get_Erec(**kwargs)
        
    def invRad_xy(self, *argv):
        if argv:
            if 'x' or 'y' in argv: 
                return fnc_invRad(*argv, z_pts = self.zpts, **self.kwargs)
    def bWidth_xy(self, *argv):
        if argv:
            if 'x' or 'y' in argv: 
                return fnc_bWidth(*argv, z_pts = self.zpts, **self.kwargs)
    def Gphase_xy(self, *argv):
        if argv:
            if 'x' or 'y' in argv: 
                return fnc_Gphase(*argv, z_pts = self.zpts, **self.kwargs)
    def Rphase(self):
        x, W_x = self.xpts, self.bWidth_xy('x')
        y, W_y = self.ypts, self.bWidth_xy('y') 
        return (x/W_x)**2 + (y/W_y)**2 

    def Iphase(self):
        z, k            = self.zpts, self.k
        x, invR_x, Gp_x = self.xpts, self.invRad_xy('x'), self.Gphase_xy('x')
        y, invR_y, Gp_y = self.ypts, self.invRad_xy('y'), self.Gphase_xy('y')
        ########## define components
        Iph_1 = k*z
        Iph_2 = ( (invR_x * x**2) + (invR_y * y**2) )*k/2 
        Iph_3 = (Gp_x + Gp_y)/2
        return Iph_1+Iph_2-Iph_3
    
    def gauss_Ek(self):
        P, W_x, W_y = self.P, self.bWidth_xy('x'), self.bWidth_xy('y')
        return np.sqrt(2*P/pi/W_x/W_y)

    def gauss_E(self):
        E_k, W_x, W_y      = self.gauss_Ek(), self.bWidth_xy('x'), self.bWidth_xy('y')
        phase_Re, phase_Im = self.Rphase()  , self.Iphase()
        
        return E_k*np.exp(-phase_Re-1j*phase_Im)
    
    def gauss_I(self):
        E = self.gauss_E()
        return (np.abs(E**2))
    
    def get_U(self, *argv):
        I, alpha = self.gauss_I(), self.alpha
        U        = alpha*I/(2*eps0*c)
        if argv:
            if 'kB'   in argv: return U/kB
            if 'Erec' in argv: return U/self.E_rec
        else: return U
    
    def get_U0(self, *argv):
        U = self.get_U(*argv)
        return (np.amax(U)-np.amin(U)) 

