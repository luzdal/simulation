from objects_oven import aperture, get_dimensions, ovenA, ovenB
from objects_plots import plts_0log, plts_0lin, plts_1stats, plts_2stats
# %% 
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display, Latex
from scipy.optimize import curve_fit
from scipy.constants import u, convert_temperature, c, h, hbar, k, atm, bar, torr, mmHg
# %%
from scipy.special import erf
from scipy.interpolate import interp1d as interp
# %%
kB ,     pi =        k, np.pi
amu, cm2_m2 = 1.66e-27, 1e4
# %%
lamb_b, Gamma_b, Isat_b = 421.290e-9, 2*pi*32.2e6, cm2_m2*50e-3
lamb_r, Gamma_r, Isat_r = 626.082e-9, 2*pi*136e3 , cm2_m2*72e-6
# %%
#############################################
################### beams ###################
# ***************************************** # laser
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
# **************************************** # atomic
class beam_atoms():
    """returns transition wavelength, frequency, natural linewitdh, and Isat"""
    def __init__(self, iso, T_hl):
        self.iso = iso
        if iso ==162 or iso ==164: self.elem = 'Dy'
        #self.D_vdw = 281e-12
        self.T_hlC, self.T_hlK = T_hl   , convert_temperature(T_hl, 'Celsius', 'Kelvin')
        self.mass , self.beta = iso*amu,  1/kB/self.T_hlK
        def v_mp(self): return np.sqrt( 3/self.beta/self.mass) 
        self.v_mp = v_mp(self)
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
            if br =='sos':
                return print('  class: x = beam_atoms(iso=a, T_hl=b) ; a = 162 or 164, b = temp in C \n'
                            +'objects:'+' x.iso, x.mass, x.v_mp, x.T_hl \n'
                            +'methods: res = \'b\' or \'r\'\n'
                            +'         x.get_Gamma(res); x.get_Isat(res); x.get_wvlen(res) \n'
                            +'         x.get_kvec(res) ; x.get_freq(res); x.get_sigma0(res)\n')
            # %%%%%%%%%%%%%%%    
            if br =='b': BR = (' Gamma = 2pi * %g MHz = %g MHz ; Isat = %g mW/cm^2'
                  %(self.get_Gamma(br)/2/pi*1e-6, self.get_Gamma(br)*1e-6, self.get_Isat(br)*1e-1))
            elif br =='r': BR = (' Gamma = 2pi * %d kHz = %g kHz ; Isat = %g uW/cm^2'
                      %(self.get_Gamma(br)/2/pi*1e-3, self.get_Gamma(br)*1e-3, self.get_Isat(br)*1e2))
            return common + BR
#############################################
################## physics ##################
# ***************************************** # OPT
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
# **************************************** # thermodyn
class thermodyn():
    """returns transition wavelength, frequency, natural linewitdh, and Isat"""
    def __init__(self, matter):
        #if iso ==162 or iso ==164: self.isoN = 'Dy'
        #self.D_vdw = 281e-12
        self.T_hlC, self.T_hlK, self.iso  = matter.T_hlC, matter.T_hlK, matter.iso
        self.mass , self.beta , self.elem = matter.mass , matter.beta , matter.elem
        self.v_mp = matter.v_mp 
        ##
    def get_P(self, *argv, **kwargs):
        T = self.T_hlK
        #print(bar, atm,  bar/atm)
        #print(bar, mmHg, bar/mmHg)
        #pm = np.array([+1, -1])
        #a, b = np.array(35170+pm*160), np.array(20.56+pm*0.12)
        #A, B = a.sum()/2, b.sum()/2
        A, B = np.array(35170), np.array(20.56)
        ln_P = - A/T + B
        P    = np.exp(ln_P)
        if argv:
            tl_n = str(self.elem)+' vapor pressure as a function of temperature'
            xl_n = r'${10^4/T}$ (K${^{-1}}$)'
            yl_n = r'P (mmHg)'
            if argv[0]=='sos': print("optional args:\nget_n('plt')\\\OR\\\get_n('plt', **{'tl':''})")
            if argv[0]=='plt': 
                d = {'xp':1e4/self.T_hlK, 'yp':P/mmHg, 'xl':xl_n, 'yl':yl_n, 'tl':tl_n}
                if kwargs:
                    d.update(kwargs)
                plts_0log(**d)
        return P
    def get_n(self, *argv, **kwargs):
        P = self.get_P()
        n = P*self.beta
        if argv:
            tl_n = str(self.elem)+' density as a function of temperature'
            xl_n = r'${T}$ ($^\circ$C)'
            if argv[0]=='sos': print("optional args:\nget_n('plt')\\\OR\\\get_n('plt', **{'tl':''})")
            if argv[0]=='plt': 
                d = {'xp':self.T_hlK, 'yp':n, 'xl':xl_n, 'tl':tl_n}
                if kwargs:
                    d.update(kwargs)
                plts_0lin(**d)
        return n
#############################################
################# apparatus #################
# ***************************************** # MOT
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
# ***************************************** # ZS
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
# ***************************************** # Oven
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
    def get_geometry(self, *argv, **kwargs):
        ## input: ('', **{'test_zpt':100, 'zax':[0, 200, 20]})
        if argv:
            if argv[0]=='sos': ovenA('sos')
            if argv[0]=='MQM'  : ovenA('', kwargs)
            if argv[0]=='Innsb': ovenB('', kwargs)

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
# ***************************************** # stats
class stats():
    def __init__(self, matter, **kwargs):
        
        self.m, self.iso = matter.mass,matter.iso
        self.T, self.Tc  = matter.T_hlK, matter.T_hlC
        
        self.beta = 1/kB/self.T 
        self.a0 = 1/np.sqrt(self.m*self.beta)
        self.a1 = np.sqrt(2) * self.a0
        
    def get_vrand(self, *argv, npts=100000, v):
        cdf, n = self.MB_CDF(v), npts
        """create interpolation function to CDF '' essentially x = g(y) from y = f(x)"""
        inv_cdf = interp(cdf, v)
        """ generate a set of velocity vectors in 3D from the MB inverse CDF function """
        speeds, theta, phi = ( inv_cdf(np.random.random(n)), 
                               np.arccos(np.random.uniform(-1, 1, n)), np.random.uniform(0, 2*np.pi, n) )
        vx, vy, vz         = ( speeds * np.sin(theta) * np.cos(phi), speeds * np.sin(theta) * np.sin(phi),
                               speeds * np.cos(theta) )
        if argv:
            if argv[0]=='vx': return vx
            if argv[0]=='vy': return vy
            if argv[0]=='vz': return vz
            if argv[0]=='vs': return speeds
        else: return speeds, vx, vy, vz
    def get_vsz(self, vs_max=1500):
        vs, vz = np.arange(0 , vs_max), np.arange(-4*self.a1, 4*self.a1, self.a1/50)
        return [vs, vz]
    def MB_z(self, v): 
        f_vz = np.exp(-v**2/self.a1**2)/np.sqrt(self.a1 * pi)
        return f_vz/f_vz.sum()/(v[1]-v[0])
    def MB_CDF(self, v):
        """create CDF '' essentially y = f(x)"""
        cdf = erf( v/self.a1 ) - np.sqrt(4/pi) * v * np.exp( -(v/self.a1)**2 )/self.a1
        return cdf 
    def MB_PDF(self, v):
        pdf = 4*pi*( 1/(2*pi*self.a0**2) )**1.5 *  v**2 * np.exp(-v**2 /self.a0**2 / 2)
        return pdf 
    def __str__(self, *argv):
        if argv:
            if argv[0] == 'matter': return(self.strMatter)
            elif argv[0]   ==  'light': return(self.strLight)
