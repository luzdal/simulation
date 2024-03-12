import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display, Latex
from scipy.optimize import curve_fit
from scipy.constants import u, convert_temperature, c, h, hbar, k, g

from scipy.special import erf
from scipy.interpolate import interp1d as interp

kB ,     pi =        k, np.pi
amu, cm2_m2 = 1.66e-27, 1e4

lamb_b, Gamma_b, Isat_b = 421.290e-9, 2*pi*32.2e6, cm2_m2*50e-3
lamb_r, Gamma_r, Isat_r = 626.082e-9, 2*pi*136e3 , cm2_m2*72e-6

#############################################################################################################
################################################### beams ###################################################
#############################################################################################################

# ********************************************************************************************************* # laser beam
class beam_laser():
    def __init__(self, diam, powr, detun):        
        def Imax(self): return self.P / pi / self.D**2   
        self.D    , self.P     = diam , powr
        self.detun, self.Imax  = detun, Imax(self)
    def get_freq(self, f_0):
        w_0 = self.get_wvlen(f_0)
        if w_0 == lamb_b: return Gamma_b*self.detun
        if w_0 == lamb_r: return Gamma_r*self.detun
    def get_wvlen(self, f):return c / f  ####check
    def get_kvec(self, f_L):return 2 * pi * f_L / c
    def get_Rrange(self, f_L):return pi * (self.D/2)**2 / self.get_wvlen(f_L)
    def get_s_param(self, Isat_0):return self.Imax/Isat_0
    
    def __str__(self, *argv):
        if argv:
            if argv[0] == 'b' or argv[0] == 'r':
                return (r'(D, P, det) = (%s mm, %s mW, %s*Gamma_'%(self.D*1e3, self.P*1e3, self.detun)
                    +argv[0]+")")
            
        else: return (r'(D, P, det) = (%s mm, %s mW, %s)'%(self.D*1e3, self.P*1e3, self.detun))
    

# ********************************************************************************************************* # atomic beam
class beam_atoms():
    """returns transition wavelength, frequency, natural linewitdh, and Isat"""
    def __init__(self, iso, T_hl):
        def mass(self): return iso*amu
        def v_mp(self): return np.sqrt( 3 * kB * convert_temperature(T_hl, 'Celsius', 'Kelvin') / mass(self) ) 
        self.iso, self.mass, self.v_mp, self.T_hl = iso, mass(self), v_mp(self), T_hl
    def get_Gamma(self, res):
        if self.iso==164 or self.iso==162:
            if res == 'b': return Gamma_b
            if res == 'r': return Gamma_r 
    def get_Isat(self, res):
        if self.iso==164 or self.iso==162:
            if res == 'b': return Isat_b
            if res == 'r': return Isat_r
    def get_wvlen(self, res):
        if self.iso==164 or self.iso==162:
            if res == 'b': return lamb_b
            if res == 'r': return lamb_r
    def get_kvec(self, res): return 2 * pi / self.get_wvlen(res)
    def get_freq(self, res): return c / self.get_wvlen(res)
    def get_sigma0(self, res): return 3 * self.get_wvlen(res)**2 / 2 / pi
    
    def __str__(self, *argv):
        if argv:
            br, common = argv[0], " Dy-"+str(self.iso)+"//"+str(argv[0])+"-trans :"
            
            if br =='b': BR = (' Gamma = 2pi * %g MHz = %g MHz ; Isat = %g mW/cm^2'
                  %(self.get_Gamma(br)/2/pi*1e-6, self.get_Gamma(br)*1e-6, self.get_Isat(br)*1e-1))
                
            elif br =='r': BR = (' Gamma = 2pi * %d kHz = %g kHz ; Isat = %g uW/cm^2'
                      %(self.get_Gamma(br)/2/pi*1e-3, self.get_Gamma(br)*1e-3, self.get_Isat(br)*1e2))

            return common + BR
    

    
#############################################################################################################
################################################# apparatus #################################################
#############################################################################################################

# ********************************************************************************************************* # OPT
class OPT():
    def __init__(self, **kwargs):
        def Tdpplr(self):return hbar * self.Gamma / kB / 2      ##  Doppler temperature
        def eta( self):return self.s / (1 + self.s)             ## "safety param"
        def Fmax(self):return hbar * self.kvec * self.Gamma / 2 ## opt_F/eta) at detun=0
        def amax(self):return self.F_max/self.mass
        def ares(self):return self.a_max*self.eta
        self.mass  , self.kvec  = kwargs['mass'] , kwargs['kvec']
        self.Gamma , self.detun = kwargs['Gamma'], kwargs['detun']
        self.s     , self.F_max = kwargs['s']    , Fmax(self)
        self.eta   , self.a_max = eta(self)      , amax(self)
        self.T_d   , self.a_res = Tdpplr(self)   , ares(self)
    def Fopt(self, vpts, k):
        if k=='+': k = -self.kvec
        if k=='-': k =  self.kvec
        return -((hbar*k*self.Gamma*self.s/2) * (1 +self.s+(2* (self.detun -k*vpts) /self.Gamma)**2)**-1)
    def get_F_vpts(self, vpts, get, k, leg):
        if k=='+': F = self.Fopt(vpts, k)
        if k=='-': F = self.Fopt(vpts, k)
        if k=='+-' or k=='-+': F = self.Fopt(vpts, k='+')+self.Fopt(vpts, k='-')
        if get=='vals': return F
        if get=='plt': self.plot_Fopt(vpts, F/hbar/self.kvec/self.Gamma, leg)
        elif get!='vals' and get!='plt': print('Actung!: input(get) EITHER = \'vals\' OR = \'plt\'') 
    def plot_Fopt(self, vpts, Fpts, leg):
        plt.title('Counter-propagating optical force', size=13)
        plt.xlabel(r'$v_z$ (m/s)', size=13), plt.ylabel(r'$F_z$ ($\hbar$$k$$\Gamma$)', size=13)
        return plt.plot(vpts, Fpts, label=leg)

    
# ********************************************************************************************************* # MOT
class MOT():
    def __init__(self, light, matter, res):
        def detun(self):return light.detun * self.Gamma
        def freqy(self):return self.detun + self.freq0
        def eta(self): return hbar * self.kvecL * self.Gamma / 2 / matter.mass / g
        def v_cap(self): return np.sqrt(self.a_max * self.D ) 
        
        self.Isat , self.wvlen, self.kvec0 = matter.get_Isat(res), matter.get_wvlen(res), matter.get_kvec(res)
        self.Gamma, self.freq0, self.v_mp  = matter.get_Gamma(res), matter.get_freq(res), matter.v_mp
        
        self.D    , self.P, self.detun = light.D, light.P, detun(self)
        self.freqL, self.s, self.kvecL = freqy(self), light.get_s_param(self.Isat), light.get_kvec(freqy(self))
        
        self.opt_params = {'mass':matter.mass,'detun':self.detun, 'kvec':self.kvecL, 'Gamma':self.Gamma, 's':self.s}
        self.opt        = OPT(**self.opt_params)
        
        self.a_max, self.eta = self.opt.a_max, eta(self)
        self.F_max, self.T_d = self.opt.F_max, self.opt.T_d
        self.v_cap           = v_cap(self)
        
# ********************************************************************************************************* # ZS
class ZS():
    def __init__(self, light, matter, res):
        def detun(self):return light.detun * self.Gamma
        def freqy(self):return self.detun + self.freq0
        self.Isat , self.wvlen, self.kvec0 = matter.get_Isat(res), matter.get_wvlen(res), matter.get_kvec(res)
        self.Gamma, self.freq0, self.v_mp  = matter.get_Gamma(res), matter.get_freq(res), matter.v_mp
        
        self.D    , self.P, self.detun = light.D, light.P, detun(self)
        self.freqL, self.s, self.kvecL = freqy(self), light.get_s_param(self.Isat), light.get_kvec(freqy(self))
        
        self.opt_params = {'mass':matter.mass,'detun':self.detun, 'kvec':self.kvecL, 'Gamma':self.Gamma, 's':self.s}
        self.opt        = OPT(**self.opt_params)
        
        self.T_d, self.a_res, self.F_max, self.mass = self.opt.F_max, self.opt.a_res, self.opt.T_d, matter.mass 
    def length(self, v_out):
        l1 = ( self.v_mp**2 - v_out**2 ) / 2 / self.a_res
        ## check: l2 = ( self.v_mp - v_out )**2 / 2 / self.a_res
        return ( self.v_mp**2 - v_out**2 ) / 2 / self.a_res
    #def t_of_z(self,  zpts):return (self.v_mp + np.sqrt( self.v_mp**2 - (2*self.a_res*zpts) ))/self.a_res 
    
    def v_cap(self, l0):return(np.sqrt(2*self.a_res/l0))
    
    def v_of_z(self,  zpts):return self.v_mp * np.sqrt( 1-(2*self.a_res*zpts/self.v_mp**2) )
    def sc_eventsNr(self, v_out):return self.mass*( self.v_mp - v_out )/hbar/self.kvecL
    def get_vpts(self, v_out, zpts, get):
        z_len, vpts = self.length(v_out), self.v_of_z(zpts)
        legd = ('$(v_i$, $v_f$) = ( {0:0.1f}'.format(self.v_mp)+', {0:0.1f}'.format(v_out)+')m/s, '
                +'$l$ = {0:0.2f}'.format(z_len)+'m')
        if get=='plt': return self.plot_v(vpts, zpts, legd=legd)
        if get=='val': return vpts
        elif get!='val' and get!='plt': print('Actung!: input(get) EITHER = \'val\' OR = \'plt\'')
    def plot_v(self, v_in, z_in, legd):
        plt.title('Zeeman slower', size=13), plt.xlabel(r'$z$ (m)', size=13), plt.ylabel(r'$v_z$ (m/s)', size=13)
        return plt.plot(z_in, v_in, '.', label=legd), plt.legend(fontsize=13)

# ********************************************************************************************************* # Oven
class oven():
    """res:{'b' or 'r'}"""
    def __init__(self, light, matter, res):
        
        self.strLight, self.strMatter = light.__str__(res), matter.__str__(res)
        
        def detun(self):return light.detun * self.Gamma
        def freqy(self):return self.detun + self.freq0
        def eta(self): return hbar * self.kvecL * self.Gamma / 2 / matter.mass / g
        
        self.sigma0, self.T_hl = matter.get_sigma0(res), matter.T_hl
        
        self.Isat , self.wvlen, self.kvec0 = matter.get_Isat(res), matter.get_wvlen(res), matter.get_kvec(res)
        self.Gamma, self.freq0, self.v_mp  = matter.get_Gamma(res), matter.get_freq(res), matter.v_mp
        
        self.D    , self.P, self.detun = light.D, light.P, detun(self)
        self.freqL, self.s, self.kvecL = freqy(self), light.get_s_param(self.Isat), light.get_kvec(freqy(self))
        
        self.opt_params = {'mass':matter.mass,'detun':self.detun, 'kvec':self.kvecL, 'Gamma':self.Gamma, 's':self.s}
        self.opt        = OPT(**self.opt_params)
        
        self.a_max, self.eta = self.opt.a_max, eta(self)
        self.F_max, self.T_d = self.opt.F_max, self.opt.T_d
        
    def __str__(self, *argv):
        if argv:
            if argv[0]   ==  'light': return(self.strLight)
            elif argv[0] == 'matter': return(self.strMatter)

    def get_Phi(self, I_I0):
        j = self.D ## note
        return self.get_n(I_I0) * pi * j**2 * self.v_mp
    
    def get_n(self, I_I0):
        j = self.D ## note
        return -1 * np.log(I_I0) / self.sigma0 / j



#############################################################################################################
################################################### plots ###################################################
#############################################################################################################


