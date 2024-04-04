from objects_oven import aperture, get_dimensions, ovenA, ovenB
# from test_commands import plot_pts, plot_axs

# from test_commands import sample_plot
#from objects_plots import plts_0log, plts_0lin, plts_1stats, plts_2stats
# %% 
import numpy as np
import math
# %% 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({'font.size': 11}, ## title
                    )

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'
#plt.rcParams['savefig.facecolor'] = "1"
plt.rcParams['savefig.transparent'] = True
plt.rcParams['savefig.facecolor'] = "0.8"
plt.rcParams['figure.figsize'] = 4, 4.
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['figure.constrained_layout.use'] = True

import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
colors  = get_cmap("Dark2").colors  ## type: matplotlib.colors.ListedColormap
markers = [',', '+', '^', '.', 'o', '*', 'X']
# %% 
import datetime
stamp = (datetime.datetime.today())
from IPython.display import display, Latex
# %%                
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d as interp
from scipy.constants import u, convert_temperature, c, h, g, hbar, k, atm, bar, torr, mmHg, N_A
# %%
kB ,     pi =        k, np.pi
amu, cm2_m2 = 1.66e-27, 1e4
#############################################
################### beams ###################
# ***************************************** # laser
class beam_laser():
    def __init__(self, diam, powr, detun, **kwarg):   
        def get_Imax(self): return self.P / pi / self.D**2   
        self.D    , self.P     = diam , powr
        self.detun, self.Imax  = detun, get_Imax(self)
        if kwarg:
            self.atomic = kwarg['trans']
            def get_lamb(self):return c / self.freq
            def get_kvec(self):return 2 * pi * self.freq / c
            def get_zR(self  ):return pi * ( self.D/2 )**2 / self.lamb
            
            self.freq = self.atomic.freq + self.detun * self.atomic.Gamma
            self.lamb = get_lamb(self)
            self.kvec = get_kvec(self)
            self.zR   = get_zR(self)
            
    def __str__(self):
        if self.atomic.res=='b':
            return ( r' (D=%d cm, P=%d mW)'%(self.D*1e2, self.P*1e3)
                    +r' $f$ = %g MHz'%(self.freq*1e-9)
                    +r' + $\delta\times$%d MHz'%(self.atomic.Gamma*1e-6)
                    )
        if self.atomic.res=='r':
            return ( r' (D=%d cm, P=%d mW)'%(self.D*1e2, self.P*1e3)
                    +r' $f$ = %g GHz'%(self.freq*1e-9)
                    +r' + $\delta\times$%d kHz'%(self.atomic.Gamma*1e-3)
                    )
# **************************************** # atomic
class beam_atoms():
    """returns transition wavelength, frequency, natural linewitdh, and Isat"""
    def __init__(self, iso, res, T_hl):
        self.iso, self.res, self.Tk, self.Tc = iso, res, T_hl, convert_temperature(T_hl, 'Celsius', 'Kelvin')

        if iso==162 or iso==164:
            self.elem, self.D, self.mass = 'Dy', 281e-12, iso*amu 
            if res=='b':
                self.lamb, self.Gamma, self.Isat = 421.290e-9,  2*pi*32.2e6, cm2_m2*50e-3
            if res=='r':
                self.lamb, self.Gamma, self.Isat = 626.082e-9, 2*pi*136e3, cm2_m2*72e-6
                
            def get_kvec(self):return 2 * pi / self.lamb
            def get_freq(self):return c / self.lamb
            def get_xsec(self):return 3 * self.lamb**2 / 2 / pi
            self.kvec, self.freq, self.xsec = get_kvec(self), get_freq(self), get_xsec(self)   
    def __str__(self):
        if self.iso==162 or self.iso==164:
            if self.res =='b':return(' Gamma = 2pi * %g MHz = %g MHz ; Isat = %g mW/cm^2'
                  %(self.Gamma/2/pi*1e-6, self.Gamma*1e-6, self.Isat*1e-1))
            if self.res =='r':return(' Gamma = 2pi * %d kHz = %g kHz ; Isat = %g uW/cm^2'
                  %(self.Gamma/2/pi*1e-3, self.Gamma*1e-3, self.Isat*1e2))
#############################################
################## physics ##################
# ***************************************** # OPT
class OPT():
    def __init__(self, light, matter, **kwarg):
        def get_param_s(self, Imax, Isat):return  Imax/Isat
        def get_Tdpplr(self):return hbar * matter.Gamma / kB / 2 
        def get_eta(self, s):return s / (1 + s) ## "safety param"

        def get_Fmax(self):return hbar * light.kvec * matter.Gamma / 2       ## opt_F/eta) at detun=0
        def get_amax(self):return get_Fmax(self) / matter.mass
        def get_ares(self):return get_amax(self) * get_eta( self)
        
        self.s    , self.Fmax   = get_param_s(self, Imax=light.Imax, Isat=matter.Isat), get_Fmax(self)

        self.eta  , self.amax   = get_eta(self, s=self.s)    , get_amax(self)
        self.light, self.matter = light, matter

        self.tag                = '['+str(matter.elem)+'-'+str(matter.iso)+'] '

        if kwarg:
            if 'v' in kwarg: self.v =  kwarg['v']

    def get_param_s(self, Imax, Isat):return  Imax/Isat

    def get_Fopt(self, vpts, detun, k, s, Gamma):
        return -((hbar*k*Gamma* s/2) * (1 + s+(2* (detun-k*vpts) /Gamma)**2)**-1)

    def Fopt_tot(self, *argv, vpts, detun, **kwargs): 
        if kwargs:
            if 'res' in kwargs: res = kwargs['res']
            atoms_j = beam_atoms(iso=164, T_hl=1100, res=res)
            laser_j = beam_laser(diam=self.light.D, powr=self.light.P, detun=detun, **{'trans':atoms_j})
            
            Gamma_j, k_j, s_j = atoms_j.Gamma, laser_j.kvec, self.get_param_s(Imax=laser_j.Imax, Isat=atoms_j.Isat)
            detun_i, str_j    = detun*Gamma_j, laser_j.__str__()

            F_t = ( self.get_Fopt(vpts=vpts, detun=detun_i, k= k_j, s=s_j, Gamma=Gamma_j)
                  + self.get_Fopt(vpts=vpts, detun=detun_i, k=-k_j, s=s_j, Gamma=Gamma_j) )
            return [F_t/hbar/k_j/Gamma_j, str_j]

    def get_props(self, *argv):
        # Gamma, kvec, res   = self.matter.Gamma, self.light.kvec, self.matter.res
        detuns, vpts = np.arange(-1.5, 2, 0.5), self.v

        xl1, yl1, lc1 = r'$v_z$ (m/s)', r'$F_z$ ($\hbar$$k$$\Gamma$)', 'center left'
        yp1, yp2      = [0]*len(detuns)   , [0]*len(detuns)
        xp1, xp2      = [vpts]*len(detuns), [vpts*0.75e-2]*len(detuns)
    
        if argv:
            fig, rowx, colx = plt.figure(), 1, 3
            fig.set_figheight(2.25), fig.set_figwidth(9)
            ax1 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 0))
            ax2 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 2))
            for i in range(len(detuns)): 
                yp1[i], tl1 = self.Fopt_tot('+', vpts=xp1[i], detun=detuns[i], **{'res':'b'})
                yp2[i], tl2 = self.Fopt_tot('+', vpts=xp2[i], detun=detuns[i], **{'res':'r'})
                ax1.plot(xp1[i], yp1[i], label=r'$\delta=$'+str(detuns[i]), color=colors[i])
                ax2.plot(xp2[i], yp2[i], label=r'$\delta=$'+str(detuns[i]), color=colors[i])
                box1, box2 = ax1.get_position(), ax2.get_position()
                ax1.set_position([box1.x0, box1.y0, box1.width * 0.95, box1.height])
                ax2.set_position([box2.x0, box2.y0, box2.width * 0.95, box2.height])
                ax1.legend(loc=lc1, bbox_to_anchor=(1.5, 0.5), title=tl2[:19])
            ax1.set_title(tl1[19:]), ax2.set_title(tl2[19:])
            ax1.set_ylabel(yl1), ax1.set_xlabel(xl1)
            ax2.set_ylabel(yl1), ax2.set_xlabel(xl1)
            if argv[0]=='plot': return  plt.show()
            if argv[0]=='save': return  fig.savefig("Plt_"+str(stamp)+".pdf", bbox_inches='tight', transparent=True)  
            elif argv[0]=='print': 
                for i in range(len(detuns)): 
                    print('i:%d, detun:%g'%(i, detuns[i])+'\n'+'v_z:%s'%(yp1[i]))
#############################################
################## physics ##################
# **************************************** # thermodyn
class thermodyn():
    def __init__(self, matter):
        self.Tc, self.elem, self.iso, self.D = matter.Tc, matter.elem, matter.iso, matter.D
        self.Tk, self.mass, self.tag         = matter.Tk, matter.mass, '['+str(self.elem)+'-'+str(self.iso)+'] '
        ##
    def get_beta(self, T):return 1/kB/T
    def get_a0(self  , T):return 1/np.sqrt(self.mass * self.get_beta(T))
    def get_a1(self  , T):return np.sqrt(2) * self.get_a0(T)
    def get_vAvg(self, T):return np.sqrt(8/self.get_beta(T)/self.mass/pi)
    def get_pres(self, T):return 10**( 5.006 + 9.579 -15336/T -1.1114*np.log10(T) )
    def get_dens(self, T, P):return P * self.get_beta(T)
    def get_mfp(self, n): return 1 / pi / np.sqrt(2) / n / self.D**2
    def get_effFlx(self, n, dpts,**kwargs):
        if kwargs:
            if   'Tk_i' in kwargs:  T_i, Tc_i = kwargs['Tk_i'], convert_temperature(kwargs['Tk_i'], 'Kelvin', 'Celsius')
            elif 'Tc_i' in kwargs: Tc_i,  T_i = kwargs['Tc_i'], convert_temperature(kwargs['Tc_i'], 'Celsius', 'Kelvin')
        A, v_avg, Flx = pi*(dpts/2)**2, self.get_vAvg(T=T_i), [0]*len(T_i)
        for i in range(len(T_i)): Flx[i] = A*v_avg[i]*n[i]/4
        return [[dpts]*len(Flx), Flx, Tc_i]
    def get_L(self, *argv):
        if argv:
            if argv[0] == 'mqm': return 30e-3

    def get_Kn(self, *argv, n): 
        return self.get_mfp(n=n) / self.get_L(*argv)

    def get_PnFlx0(self, *argv, **kwargs):
        ###
        if kwargs:
            if   'Tk_i' in kwargs:  T_i, Tc_i = kwargs['Tk_i'], convert_temperature(kwargs['Tk_i'], 'Kelvin', 'Celsius')
            elif 'Tc_i' in kwargs: Tc_i,  T_i = kwargs['Tc_i'], convert_temperature(kwargs['Tc_i'], 'Celsius', 'Kelvin')
        else: T_i, Tc_i = self.Tk, self.Tc
        ###
        xl1, yl1, yl3 = r'$T$ ($^\circ$C)', r'$P$ (Pa)', r'$\Lambda$ (m)' #Mean free path Number density
        xl2, yl2      = r'$T$ ($^\circ$C)', r'$\rho_{_N}$ (Nm$^{-3}$)'
        xl4, yl4      = r'$d$ (mm)', r'$\Phi_0$ (Ns$^{-1}$)'
        lc2, lc3, lc1 = 'upper right', 'upper left', 'upper center'
        tl1, tl2      = r'\bf Vapor pressure $P$', r'\bf Number density $\rho_{_N}$ and '+'\n '+r'\bf mean free path $\Lambda$' 
        tl4           = r'\bf Atomic flow rate $\Phi_0$ at'+'\n '+r'\bf given oven aperture diameter'
        ###
        # xp_P, yp_P = 1e4/T_i, self.get_P(T=T_i)/mmHg
        yp1, yp2      = self.get_pres(T=T_i) , self.get_dens(T=T_i, P=self.get_pres(T=T_i))
        xp1, yp3      = Tc_i                 , self.get_mfp(n=yp2)
        xp4, yp4, sl4 = self.get_effFlx(n=yp2, dpts=np.arange(0, 20e-3, 1e-3), **{'Tc_i':Tc_i})

        yp5 = self.get_Kn('mqm', n=yp2)
        sl5 = r'$%g$'
        
        if argv:         
            fig, rowx, colx = plt.figure(), 2, 2
            fig.set_figheight(6), fig.set_figwidth(7)
            ax1 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 0))
            ax2 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 1), sharex=ax1)
            # ax2 = plt.subplot2grid(shape=(rowx, colx), loc=(1, 0), sharex=ax1)
            ax3 = ax2.twinx()
            ax4 = plt.subplot2grid(shape=(rowx, colx), loc=(1, 1)) 
            ax1.set_title(tl1), ax2.set_title(tl2), ax4.set_title(tl4)
            #####
            ax1.plot(xp1, yp1, '-', lw=2, color='k')
            ax2.plot(xp1, yp2, '-', lw=2, color='k') 
            ax3.plot(xp1, yp3, '-', lw=2, color='gray')
            for i in range(len(yp4)): 
                ax4.plot(xp4[i]*1e3, yp4[i], color=colors[i], label=str(sl4[i])+r' $^\circ$C')
                ax3.plot(xp1[i], yp3[i], '.', lw=2, color=colors[i], label=sl5%(yp5[i]))
                ax3.legend(loc='upper center', title=r'$K_n = \Lambda/L_{CC} $')
            ##### 
            ax1.set_ylabel(yl1), ax1.set_yscale('log', base=10), ax2.grid()
            ax2.set_ylabel(yl2), ax4.set_yscale('log', base=10), ax2.set_xlabel(xl1), ax2.tick_params(axis='y', labelcolor='k')
            ax3.set_ylabel(yl3), ax3.tick_params(axis='y', labelcolor='gray')
            ax4.set_ylabel(yl4), ax4.set_xlabel(xl4), ax4.legend(), ax4.set_xlim([2, 16]), ax4.set_xticks(np.arange(3, 16, 3)) 
            #####
            j, = np.where(Tc_i == 1100)
            if len(j) != 0:ax1.axvline(x=Tc_i[j[0]], ymin=min(yp1), ymax=max(yp1), ls='--', lw=1, color='k',label='(T, P) = (%g, %g)'%(Tc_i[j[0]], yp1[j[0] ])), ax1.legend(loc=lc1)
            k, = np.where(xp4[0] == 3e-3)
            if len(k) != 0:ax4.axvline(x=xp4[0][k]*1e3, ymin=min(yp4[0]), ymax=max(yp4[0]), ls='--', lw=1, color='k',label='Mqm'), ax4.legend()
            #####
            if argv[0]=='plot': return fig.tight_layout(), plt.show()
            if argv[0]=='save': return fig.tight_layout(), fig.savefig("Plt_"+str(stamp)+".pdf", bbox_inches='tight', transparent=True)  
            if argv[0]=='print':
                for i in range(len(T_i)): 
                    print('i:%d, T:[%d K, %d C], P:%g, n:%g, Lamb:%g'%(i, T_i[i], Tc_i[i], yp1[i], yp2[i], yp3[i]))

    def func_theta(self, **kwargs):
        if   'theta_deg' in kwargs: return kwargs['theta_deg']*pi/180
        elif 'theta_rad' in kwargs: return kwargs['theta_rad']
    def param_q(self, **kwargs):
        b0, q, ang_k = kwargs['beta0'], [0]*len(kwargs['beta0']), self.func_theta(**kwargs) 
        for i in range(len(b0)): q[i] = np.tan(ang_k) / b0[i] ## double check (!!)
        return q
    def param_alpha0(self, *argv, **kwargs):
        b0, a0 = kwargs['beta0'], [0]*len(kwargs['beta0'])
        for i in range(len(b0)): 
            top_i = 1 - 2*b0[i]**3 + ( (2*b0[i]**2 - 1)* np.sqrt(1+b0[i]**2) )
            bot_i = (3*b0[i]**2)*( np.sqrt(1+b0[i]**2) - (b0[i]**2) / np.sinh(1/b0[i]) ) 
            a0[i] = ( (1/2) - (top_i/bot_i) )
        if argv:
            if argv[0]=='check': return (b0, a0)
        else: return a0
    def func_R(self, *argv, **kwargs):
        R, q = [0]*(len(self.param_q(**kwargs))), self.param_q(**kwargs)
        for i in range(len(R)): 
            R_i, R_0 = [], np.zeros((len(q),), dtype=int)
            for q_i in q[i]:
                if q_i<1: R_i.append(np.arccos(q_i) - q_i * (1-q_i**2)**(1/2))
                if q_i>1: R_i.append(R_0[i])
            R[i] = R_i
        if argv:
            if argv[0]=='check': return (self.func_theta(**kwargs)*180/pi, q, R)
        else: return R

    def param_zeta(self, *argv, **kwargs):
        zeta_0, zeta_1 = self.param_alpha0(**kwargs), [0]*len(self.param_alpha0(**kwargs))
        for i in range(len(zeta_1)):
            zeta_1[i] = 1-zeta_0[i]
        return zeta_0, zeta_1

    def get_W(self, *argv, **kwargs):
        b0, a_i, W = kwargs['beta0'], self.param_alpha0(**kwargs), [0]*len(kwargs['beta0'])
        for i in range(len(W)):
            w2_i, w3_i = (1-2*a_i[i])*(b0[i]-np.sqrt(1+b0[i]**2)), (1+a_i[i])*(1-np.sqrt(1+b0[i]**2)) / b0[i]**2
            W[i] = 1+(2*(w2_i+w3_i)/3)
        return W

    def func_J_qLT(self, *argv, a, angs, q, R):
        j1, j2, j3 = a , (1-a) * R * 2 / pi , (2/3/q) * (1-2*a) * ( 1 - (1-q**2)**(3/2) ) * 2 / pi
        return (np.cos(angs)*(j1+j2+j3))
    def func_J_qGT(self, *argv, a, angs, q, R):
        j1, j2 = a, (1-a) * 4 / 3 / pi / q
        return (np.cos(angs)*(j1+j2))
    
    def get_J(self, *argv, **kwargs):
        b0, a_i, q_ik  = kwargs['beta0'], self.param_alpha0(**kwargs), self.param_q(**kwargs)
        J, R_ik, ang_k = [0]*len(a_i)   , self.func_R(**kwargs)      , self.func_theta(**kwargs)
        for i in range(len(R_ik)):
            J_i, J_0 = [], np.ones((len(R_ik),), dtype=int)
            for theta_k, k in zip (ang_k, range(len(ang_k))):
                Rik, qik = R_ik[i][k], q_ik[i][k]
                if theta_k == 0:  J_i.append(J_0[i])
                else: 
                    if Rik!=0: J_i.append(self.func_J_qLT(a=a_i[i], angs=ang_k[k], q=qik, R=Rik))
                    if Rik==0: J_i.append(self.func_J_qGT(a=a_i[i], angs=ang_k[k], q=qik, R=Rik))
            J[i] = J_i
        if argv:
            if argv[0]=='check':
                fig, rowx, colx, loc1 = plt.figure(), 2, 2, 'upper right'
                fig.set_figheight(5.5), fig.set_figwidth(6)
                ax0 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 0), colspan=colx-1)
                ax1 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 1), colspan=colx-1)
                ax2 = plt.subplot2grid(shape=(rowx, colx), loc=(1, 0), colspan=colx-1)
                ax3 = plt.subplot2grid(shape=(rowx, colx), loc=(1, 1), colspan=colx-1, sharex=ax1)

                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.85, box.height*0.7])
                
                (xp0, yp0), (xp1, yp1, yp2) = self.param_alpha0(argv[0], **kwargs), self.func_R(argv[0], **kwargs) 
                tl1, tl2 = r'$q=$ tan$(\theta)\beta_0^{-1}$', r'$R=$ arccos$(q)-q\sqrt{1-q^2}$'
                xl0, yl1, yl2, yl3  = r'$\beta_0$'          , r'$q(\theta)$'          , r'$R(q)$', r'$R(\theta)$'
                yl0, xl1, xl2, sli  = r'$\alpha(\beta_0)$'  , r'$\theta$ ($^{\circ}$)', r'$q$'   , r'$%g$' 
                ax0.set_xlabel(xl0), ax1.set_xlabel(xl1), ax2.set_xlabel(xl2), ax3.set_xlabel(xl1), ax1.set_title(tl1)
                ax0.set_ylabel(yl0), ax1.set_ylabel(yl1), ax2.set_ylabel(yl2), ax3.set_ylabel(yl3), ax2.set_title(tl2)
                
                ax1.set_xticks([0, 45, 90])  #ax2.set_ylim([0, 1.5]), ax2.set_xlim([0, 5]), ax1.set_ylim([0, 90])

                for ii in range(len(yp1)):
                    ax0.plot(xp0[ii], yp0[ii], '.', color=colors[ii]), ax2.plot(yp1[ii], yp2[ii], '-', color=colors[ii], label=sli%(xp0[ii]))
                    ax1.plot(xp1    , yp1[ii], '-', color=colors[ii]), ax3.plot(xp1    , yp2[ii], '-', color=colors[ii])
                ax2.legend(loc='center left', bbox_to_anchor=(1.15, 1.35), title=r'$\beta_0$')

                fig.tight_layout() 
                # plt.show()
                
                if argv[1]:
                    if argv[1]=='save': return fig.savefig("Plt_"+str(stamp)+".pdf", bbox_inches='tight', transparent=True)
                else: return plt.show()

            else:
                fig, rowx, colx, loc1 = plt.figure(), 1, 1, 'upper right'
                fig.set_figheight(4), fig.set_figwidth(4)
                ####
                ax0 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 0), colspan=colx) 
                xl0, yl0 = r'$\theta$ ($^{\circ}$)', r'j($\theta$)'
                for ii in range(len(J)):
                    # ax0.plot(ang_k*180/pi, J[ii], '.', color=colors[ii], label=r'$\beta$ = %g'%(b0[ii]))
                    if b0[ii] == 0.1: 
                        ax0.plot(ang_k*180/pi, J[ii], '--', color=colors[ii], label=r'%g'%(b0[ii])+' (Mqm)')
                    else: 
                        ax0.plot(ang_k*180/pi, J[ii], '--', color=colors[ii], label=r'%g'%(b0[ii]))

                ax0.legend(title=r'$\beta_0$')

                ax0.set_ylabel(yl0), ax0.set_xlabel(xl0), fig.tight_layout()#, plt.legend()
                if argv[0]=='plot': plt.show()
                if argv[0]=='save': fig.savefig("Plt_"+str(stamp)+".pdf", bbox_inches='tight', transparent=True)
        else: return J
            
# ***************************************** # stats
class stats():
    def __init__(self, thermo, v):
        self.tag, self.thermo, self.v  = self.thermo.tag , thermo, v
        self.m  , self.iso             = self.thermo.mass, self.thermo.iso
        self.Tk , self.Tc              = self.thermo.Tk  , self.thermo.Tc

    def MB_z(self, v, T): 
        fv, a1   = [0]*len(T), self.thermo.get_a1(T)
        for ii in range(len(T)): 
            fv_i   = np.exp(-v**2/a1[ii]**2)/np.sqrt(a1[ii] * pi)
            fv[ii] = fv_i / fv_i.sum() / ( v[1]-v[0] )
        return fv
    def PDF(self, v, T):
        pdf, a0 = [0]*len(T), self.thermo.get_a0(T)
        for ii in range(len(T)): pdf[ii] = 4*pi*( 1/(2*pi*a0[ii]**2) )**1.5 *  v**2 * np.exp(-v**2 /a0[ii]**2 / 2)
        return pdf
    def CDF(self, v, T):
        cdf, a1 = [0]*len(T), self.thermo.get_a1(T) 
        for ii in range(len(T)): cdf[ii] = erf( v/a1[ii] ) - np.sqrt(4/pi) * v * np.exp( -(v/a1[ii])**2 )/a1[ii]
        return cdf
    def CDFinv(self, v, T): 
        cdf_inv = [0]*len(T)
        for ii in range(len(T)): cdf_inv[ii] = interp(self.CDF(v, T)[ii], v) 
        return cdf_inv

    def get_velocities(self, *argv, n=10000, m=1, T=1200, bins=50, **kwargs):

        v, T_m = self.v, np.ones(m)*T
        def get_theta(): return np.arccos(np.random.uniform(-1, 1, n))
        def get_phi():   return np.random.uniform(0, 2*pi, n)
        def get_vz( vs, theta     ): return vs * np.cos(theta)
        def get_vxy(vs, theta, phi): return vs * np.sin(theta) * np.cos(phi), vs * np.sin(theta) * np.sin(phi)
        #def get_vy( vs, theta, phi): return vs * np.sin(theta) * np.sin(phi)
        ######################### """ initialize """
        vs_n, vz_n, theta_n, phi_n = [0]*len(T_m), [0]*len(T_m), [0]*len(T_m), [0]*len(T_m)
        ######################### """" fit """"
        a1_m = self.thermo.get_a1(T)
        vs   = np.arange(0 , self.v.max()+1)
        vz   = np.arange(-4 * a1_m, 4 * a1_m, a1_m/50)
        ######################### """ plot stuff """
        tl        = (r'T = %d K, %d bins, %d pseudorandom samples '%(T_m, bins, n))
        sl1, sl2  = r'$\rho_v$', r'${C_v}^{-1}$'
        xl1, xl2  = r'speed (m/s)', r'axial velocity (m/s)'
        yl1, yl2  = r'$\|v\|$ (m/s)', r'$v_x$ (m/s)'
        """ generate a set of velocity vectors in 3D from the MB inverse CDF function """
        for m in range(len(T_m)):
            vs_n[m]              = self.CDFinv(v, T_m)[m](np.random.random(n)) 
            theta_n[m], phi_n[m] = np.arccos(np.random.uniform(-1, 1, n)), np.random.uniform(0, 2*np.pi, n)
            ##########
            vz_n, vx_n, vy_n = get_vz( vs=vs_n[m], theta=theta_n[m]), get_vxy(vs=vs_n[m], theta=theta_n[m], phi=phi_n[m])
            vs_fit, vz_fit   = self.PDF(v=vs, T=T_m)[m], self.MB_z(v=vz, T=T_m)[m]
            ##########
            if argv:
                if argv[0]=='histo':
                    fig, rowx, colx, loc1 = plt.figure(), 2, 2, 'upper right'
                    fig.set_figheight(4.5), fig.set_figwidth(4.5)
                    ####
                    ax0 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 0), colspan=colx)
                    ax1 = plt.subplot2grid(shape=(rowx, colx), loc=(1, 0), colspan=colx)
                    ######################### """ (vs, vj) histogram & fit """
                    ax0.hist(vs_n[m], bins=bins, density=True, fc='r', alpha=0.3, lw=0.2, label=r'$v_s$')
                    ax0.plot(vs, vs_fit, 'k--', label=sl1), ax0.legend(loc=loc1), ax0.set_xlabel(xl1)
                    ax1.hist(vz_n, bins=bins, density=True, fc='b', alpha=0.3, lw=0.2, label=r'$v_z$')
                    ax1.plot(vz, vz_fit, 'k--', label=sl2), ax1.legend(loc=loc1), ax1.set_xlabel(xl2)
                    ####
                    ax0.set_xlim([-150, 1100]), ax1.set_xlim([-1100, 1100])
                    ax0.set_title(tl)
                    if argv[1]=='plot': return fig.tight_layout(), plt.show()
                    if argv[1]=='save': return fig.tight_layout(), fig.savefig("Plt_"+str(stamp)+".pdf", bbox_inches='tight', transparent=True)
                    if argv[1]=='vz':   return [vz, vz_fit]
                    elif argv[1]=='print':
                        for i in range(len(vs_n)): 
                           print( '(i:%d,T:%d,vs:%d,theta:%d,phi:%d,vz:%d,vx:%d,vy:%d):'
                               %(i,T_m[i],vs_n[i][0],theta_n[i][0],phi_n[i][0],vz_n[0],vx_n[0],vy_n[0] ))
                ###################
                if argv[0]=='dist':
                    tl2 = (self.tag+r'\bf PDF and CDF')
                    if kwargs:
                        if   'Tk_i' in kwargs: T_i , Tc_i =  kwargs['Tk_i'], convert_temperature(kwargs['Tk_i'], 'Celsius', 'Kelvin')
                        elif 'Tc_i' in kwargs: Tc_i, T_i  =  kwargs['Tc_i'], convert_temperature(kwargs['Tc_i'], 'Kelvin', 'Celsius')
                    else: T_i, T_c  = self.Tk, self.Tc
                    fig, rowx, colx = plt.figure(), 2, 1 
                    fig.set_figheight(4.5), fig.set_figwidth(4.5)
                    xp1, xl1, sl1 = self.v      , r'$v$ (m/s)'         , [0]*len(T_i)
                    yp1, yl1      = [0]*len(sl1),  r'PDF, $\rho_v$ (s/m)'
                    yp2, yl2      = [0]*len(sl1),  r'CDF, $\int\rho_vdv$'
                    ax0 = plt.subplot2grid(shape=(rowx, colx), loc=(0, 0))
                    ax1 = plt.subplot2grid(shape=(rowx, colx), loc=(1, 0), sharex=ax0)
                    for ii in range(len(T_i)): 
                        sl1          = r'  T(%d K, %d ${^\circ}$C)'%(T_i[ii], Tc_i[ii])
                        yp1, yp2 = self.PDF(v=xp1, T=T_i)[ii], self.CDF(v=xp1, T=T_i)[ii]
                        ax0.plot(xp1, yp1, color=colors[ii], label=sl1), ax0.legend(), 
                        ax0.set_ylabel(yl1)
                        ax1.plot(xp1, yp2, color=colors[ii], label=sl1)#, ax1.legend(), 
                        ax1.set_ylabel(yl2), ax1.set_xlabel(xl1)
                    # Shrink current axis by 20%
                    box = ax0.get_position()
                    ax0.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # Put a legend to the right of the current axis
                    ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)
                    plt.setp(ax0.get_xticklabels(), visible=False)
                    plt.tight_layout()
                    if argv[1]=='plot': return plt.show()
                    if argv[1]=='save': return plt.savefig("Plt_"+str(stamp)+".pdf", bbox_inches='tight', transparent=True)
                    #if argv[1]=='print': return print(max(vz_n)) 
