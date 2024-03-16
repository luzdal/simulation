from objects_oven import aperture, get_dimensions, ovenA, ovenB
#from objects_plots import plts_0log, plts_0lin, plts_1stats, plts_2stats
# %% 
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display, Latex
# %%
SMALL_SIZE  = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams.update({ "text.usetex": True,
                     "font.family": "Helvetica"
                    })
# %%                  
from scipy.optimize import curve_fit
from scipy.constants import u, convert_temperature, c, h, hbar, k, atm, bar, torr, mmHg, N_A
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
        self.D_vdw = 281e-12
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

# **************************************** # thermodyn
class thermodyn():
    """returns transition wavelength, frequency, natural linewitdh, and Isat"""
    def __init__(self, matter):
        self.T_hlC, self.T_hlK, self.iso  = matter.T_hlC, matter.T_hlK, matter.iso
        self.mass , self.beta , self.elem = matter.mass , matter.beta , matter.elem
        self.v_mp , self.D_vdw = matter.v_mp, matter.D_vdw
        self.tag  = str(self.elem)+'-'+str(self.iso)
        ##
        def get_P(self):
            T = self.T_hlK
            #print(bar, atm,  bar/atm)
            #print(bar, mmHg, bar/mmHg)
            #pm = np.array([+1, -1])
            #a, b = np.array(35170+pm*160), np.array(20.56+pm*0.12)
            #A, B = a.sum()/2, b.sum()/2
            A, B = np.array(35170), np.array(20.56)
            ln_P = - A/T + B
            return np.exp(ln_P)
        def get_n(self): return get_P(self)*self.beta
        ##
        self.n, self.P = get_n(self), get_P(self)

        def get_mfp(self): return 1/(self.n*pi*np.sqrt(2)*self.D_vdw**2)
        self.mfp = get_mfp(self)

    def get_plots(self, *argv):
        xp_P, yp_P = 1e4/self.T_hlK, self.P/mmHg
        xp_N, yp_N = self.T_hlK    , self.n*N_A
        xl_P, yl_P = r'${10^4/T}$ (K${^{-1}}$)', r'P (mmHg)'
        xl_N, yl_N = r'${T}$ ($^\circ$C)'      , r'n (P${/k_BT}$)'#r'${N}$ (${n{\cdot}N_A}$)'

        tl_P       = r'\bf Vapor pressure for ' + self.tag
        tl_N       = r'\bf Atomic density for ' + self.tag
        
        xp1, yp1, xl1, yl1, tl1 = xp_P, yp_P, xl_P, yl_P, tl_P
        xp2, yp2, xl2, yl2, tl2 = xp_N, yp_N, xl_N, yl_N, tl_N
        if argv:
            if argv[0]=='sos': print('plts_test(\'arg\') --> \'arg\':\'P\', \'n\', \'Pn\'')
            else: 
                fig = plt.figure(figsize=(10, 5))
                if argv[0]=='P' or argv[0]=='all':
                    ax = fig.add_subplot(121)
                    ax.plot(xp1, yp1, '.', lw=2), ax.set_yscale('log'), ax.set_xlabel(xl1), ax.set_ylabel(yl1), ax.set_title(tl1)
                if argv[0]=='n' or argv[0]=='all':
                    ax = fig.add_subplot(122) 
                    ax.plot(xp2, yp2, '.', lw=2), plt.grid(), ax.set_xlabel(xl2), ax.set_ylabel(yl2), ax.set_title(tl2)
                
                # if argv[0]=='mfp' or argv[0]=='all':
                #     ax = fig.add_subplot(223) 
                #     yp3 = self.mfp
                #     ax.plot(xp2, yp3, '.', lw=2), plt.grid(), ax.set_xlabel(xl2), ax.set_ylabel(yl2), ax.set_title(tl2)
                
                fig.tight_layout(), plt.show()
    def get_beta(self, T): return 1/kB/T
    def get_a0(self, T): return 1/np.sqrt(self.mass * self.get_beta(T))
    def get_a1(self, T): return np.sqrt(2) * self.get_a0(T)


# ***************************************** # stats
class stats():
    def __init__(self, thermo, v):
        
        self.thermo = thermo

        self.m   , self.iso          = thermo.mass,  thermo.iso
        self.T_i, self.Tc_i, self.v  = thermo.T_hlK, thermo.T_hlC, v
        
        def beta(self, T): return 1/kB/T
        def a0(self, T): return 1/np.sqrt(self.m * beta(self, T))
        def a1(self, T): return np.sqrt(2) * a0(self, T)

        self.beta, self.a0, self.a1 = beta(self, T=self.T_i), a0(self, T=self.T_i), a1(self, T=self.T_i) #np.sqrt(2) * self.a0

    def PDF(self, v, T):
        pdf, a0 = [0]*len(self.T_i), self.thermo.get_a0(T) ### self.T_i --> T
        for ii in range(len(self.T_i)): pdf[ii] = 4*pi*( 1/(2*pi*a0[ii]**2) )**1.5 *  v**2 * np.exp(-v**2 /a0[ii]**2 / 2)
        return pdf

    def CDF(self, v, T):
        cdf, a1 = [0]*len(T), self.thermo.get_a1(T) 
        for ii in range(len(T)): cdf[ii] = erf( v/a1[ii] ) - np.sqrt(4/pi) * v * np.exp( -(v/a1[ii])**2 )/a1[ii]
        return cdf

    def CDFinv(self, v, T): 
        cdf_inv = [0]*len(T)
        for ii in range(len(T)): cdf_inv[ii] = interp(self.CDF(v, T)[ii], v) 
        return cdf_inv

    def v_rand(self, n=10):
        m, T   = 4, 1200
        v, T_m = self.v, np.ones(m)*T
        theta_n, vs_n, phi_n  = [0]*len(T_m), [0]*len(T_m), [0]*len(T_m)
        ## """ generate a set of velocity vectors in 3D from the MB inverse CDF function """
        for m in range(len(T_m)):
            vs_n[m]    = self.CDFinv(v, T_m)[m](np.random.random(n)) 
            theta_n[m] = np.arccos(np.random.uniform(-1, 1, n))
            phi_n[m]   = np.random.uniform(0, 2*np.pi, n)
        
        vz_n = vs_n * np.cos(theta_n)
        vx_n = vs_n * np.sin(theta_n) * np.cos(phi_n)
        vy_n = vs_n * np.sin(theta_n) * np.sin(phi_n)
        
        for i in range(len(vs_n)): 
            print( '(i:%d,T:%d,vs:%d,theta:%d,phi:%d,vz:%d,vx:%d,vy:%d):'
                %(i,T_m[i],vs_n[i][0],theta_n[i][0],phi_n[i][0],vz_n[i][0],vx_n[i][0],vy_n[i][0] ))

    def get_plots(self, *argv):
        loc1 = 'center right'
        if argv:
            if argv[0]=='sos': print('jkdfhkjaf')
            else:
                xp1, xl1, sl1 = self.v     , 'v (m/s)'         , [0]*len(self.T_i)
                yp1, yl1, tl1 = [0]*len(sl1),  r'$\rho_v$ (s/m)', r'\bf Probability Density Function'
                yp2, yl2, tl2 = [0]*len(sl1),  r'$\int\rho_vdv$', r'\bf Cummulative Distribution Function'
                #print(sl1, self.T_i)
                for ii in range(len(sl1)): 
                    Tk_ii, Tc_ii  = self.T_i[ii], self.Tc_i[ii]
                    yp1[ii], yp2[ii], sl1[ii] = (self.PDF(v=xp1, T=self.T_i)[ii], self.CDF(v=xp1, T=self.T_i)[ii], 
                                                r'T(K, ${^\circ}$C) = %d, %d'%(Tk_ii, Tc_ii))

                fig = plt.figure(figsize=(10, 5))
                # % %
                if argv[0]=='PDF' or argv[0]=='all':
                    ax = fig.add_subplot(121)
                    for ii in range(len(sl1)): ax.plot(xp1, yp1[ii], label=sl1[ii], lw=0.6)
                    ax.set_xlabel(xl1), ax.set_ylabel(yl1), ax.set_title(tl1), ax.legend(loc=loc1)
                # % %
                if argv[0]=='CDF' or argv[0]=='all':
                    ax = fig.add_subplot(122)
                    for ii in range(len(sl1)): ax.plot(xp1, yp2[ii], label=sl1[ii], lw=0.6)
                    ax.set_xlabel(xl1), ax.set_ylabel(yl2), ax.set_title(tl2), ax.legend(loc=loc1)
                # % %
                fig.tight_layout(), plt.show()
                
                # if argv[0]=='CDF' or argv[0]=='all':
                #     a=1


    # def get_vsz(self, vs_max=1500):
    #     vs, vz = np.arange(0 , vs_max), np.arange(-4*self.a1, 4*self.a1, self.a1/50)
    #     return [vs, vz]
    # def MB_z(self, v): 
    #     f_vz = np.exp(-v**2/self.a1**2)/np.sqrt(self.a1 * pi)
    #     return f_vz/f_vz.sum()/(v[1]-v[0])
    # # def MB_CDF(self, v):

    # def MB_CDF(self):
    #     v = self.v
    #     """create CDF '' essentially y = f(x)"""
    #     cdf = erf( v/self.a1 ) - np.sqrt(4/pi) * v * np.exp( -(v/self.a1)**2 )/self.a1
    #     return cdf 

    # # def MB_PDF(self, v):
    # def __str__(self, *argv):
    #     if argv:
    #         if argv[0] == 'matter': return(self.strMatter)
    #         elif argv[0]   ==  'light': return(self.strLight)

    # def get_plts(self, *argv):
    #     if argv:
    #         if argv[0]=='PDF_CFD':
    #             a = 1
    #     return


    # def MB_PDF(self):
    #     v = self.v
    #     pdf = 4*pi*( 1/(2*pi*self.a0**2) )**1.5 *  v**2 * np.exp(-v**2 /self.a0**2 / 2)
    #     return pdf 




