import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy import optimize
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

plt.rcParams.update({'font.size': 12} )

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'
#plt.rcParams['savefig.facecolor'] = "1"
plt.rcParams['savefig.transparent'] = True
plt.rcParams['savefig.facecolor'] = "0.8"
plt.rcParams['figure.figsize'] = 4, 4.
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['figure.constrained_layout.use'] = True

import datetime
stamp = (datetime.datetime.today())


x_array = np.linspace(1,100,50)

def _1gaussian(x, amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2)))


# Lorentzian Peak Fitting


x_array = np.linspace(1,300,250)

params_1 = {'amp':50 , 'cent':100, 'wid':5 }
params_2 = {'amp':100, 'cent':150, 'wid':10}
params_3 = {'amp':50 , 'cent':200, 'wid':5 }


def get_lorenz(x, amp, cent, wid):
    return (amp*wid**2/((x-cent)**2+wid**2))

def get_yarr_lorenz(xpts, params):
    x_array, y_array = xpts, [0]*len(params)

    for i in range(len(params)):
        dic_i = params[i]
        amp, cent, wid = dic_i['amp'], dic_i['cent'], dic_i['wid']
        y_array[i]     = get_lorenz(xpts, amp, cent, wid)

    return y_array[0]+y_array[1]+y_array[2]



y_array_3lorentz = get_yarr_lorenz(x_array, params=[params_1, params_2, params_3])

# creating some noise to add the the y-axis data
y_noise_3lorentz = (((np.random.ranf(250))))*5
y_array_3lorentz += y_noise_3lorentz

def plot_1():

    fig = plt.figure(figsize=(4,3))
    gs = gridspec.GridSpec(1,1)
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(x_array, y_array_3lorentz, "ro")

    #ax1.set_xlim(-5,105)
    #ax1.set_ylim(-0.5,5)

    ax1.set_xlabel("x array")
    ax1.set_ylabel("y array")

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.tick_params(axis='both',which='major', direction="in", top="on", right="on", bottom="on")
    ax1.tick_params(axis='both',which='minor', direction="in", top="on", right="on", bottom="on")

    plt.show()


def _1Lorentzian(x, amp, cen, wid):
    return amp*wid**2/((x-cen)**2+wid**2)

def _3Lorentzian(x, amp1, cen1, wid1, amp2,cen2,wid2, amp3,cen3,wid3):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) + (amp2*wid2**2/((x-cen2)**2+wid2**2)) + (amp3*wid3**2/((x-cen3)**2+wid3**2))



def plot_2(*argv):

    amp1, cen1, wid1 = 50 , 100 , 5 
    amp2, cen2, wid2 = 100, 150 , 10
    amp3, cen3, wid3 = 50 , 200 , 5 


    popt_3lorentz, pcov_3lorentz = scipy.optimize.curve_fit(_3Lorentzian, x_array, y_array_3lorentz, p0=[amp1, cen1, wid1,                                                                                     amp2, cen2, wid2, amp3, cen3, wid3])

    perr_3lorentz = np.sqrt(np.diag(pcov_3lorentz))

    pars_1 = popt_3lorentz[0:3]
    pars_2 = popt_3lorentz[3:6]
    pars_3 = popt_3lorentz[6:9]
    lorentz_peak_1 = _1Lorentzian(x_array, *pars_1)
    lorentz_peak_2 = _1Lorentzian(x_array, *pars_2)
    lorentz_peak_3 = _1Lorentzian(x_array, *pars_3)


    residual_3lorentz = y_array_3lorentz - (_3Lorentzian(x_array, *popt_3lorentz))

    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(2,1, height_ratios=[1, 0.25])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    gs.update(hspace=0) 

    ax1.plot(x_array, y_array_3lorentz, "ro")
    ax1.plot(x_array, _3Lorentzian(x_array, *popt_3lorentz), 'k--')#,\
             #label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))

    # peak 1
    ax1.plot(x_array, lorentz_peak_1, "g")
    ax1.fill_between(x_array, lorentz_peak_1.min(), lorentz_peak_1, facecolor="green", alpha=0.5)
      
    # peak 2
    ax1.plot(x_array, lorentz_peak_2, "y")
    ax1.fill_between(x_array, lorentz_peak_2.min(), lorentz_peak_2, facecolor="yellow", alpha=0.5)  

    # peak 3
    ax1.plot(x_array, lorentz_peak_3, "c")
    ax1.fill_between(x_array, lorentz_peak_3.min(), lorentz_peak_3, facecolor="cyan", alpha=0.5) 

    # residual
    ax2.plot(x_array, residual_3lorentz, "bo")
        

    ax2.set_xlabel("x array")
    ax1.set_ylabel("y array")
    ax2.set_ylabel("Res.")

    ax1.legend(loc="best")

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    ax1.tick_params(axis='both', which='both', direction="in", top="on", right="on", bottom="off")
    ax2.tick_params(axis='both', which='both', direction="in", top="off", right="on", bottom="on")

    if argv:
        if argv[0]=='plot': return plt.show()
        if argv[0]=='save': return fig.savefig("foo_"+str(stamp)+"_"+"raw_3Lorentz.png", format="png",dpi=1000)
        if argv[0]=='print':   
            # this cell prints the fitting parameters with their errors
            print ("-------------Peak 1-------------")
            print ("amplitude = %0.2f (+/-) %0.2f" % (pars_1[0], perr_3lorentz[0]))
            print ("center = %0.2f (+/-) %0.2f" % (pars_1[1], perr_3lorentz[1]))
            print ("width = %0.2f (+/-) %0.2f" % (pars_1[2], perr_3lorentz[2]))
            print ("area = %0.2f" % np.trapz(lorentz_peak_1))
            print ("--------------------------------")
            print ("-------------Peak 2-------------")
            print ("amplitude = %0.2f (+/-) %0.2f" % (pars_2[0], perr_3lorentz[3]))
            print ("center = %0.2f (+/-) %0.2f" % (pars_2[1], perr_3lorentz[4]))
            print ("width = %0.2f (+/-) %0.2f" % (pars_2[2], perr_3lorentz[5]))
            print ("area = %0.2f" % np.trapz(lorentz_peak_2))
            print ("--------------------------------")
            print ("-------------Peak 3-------------")
            print ("amplitude = %0.2f (+/-) %0.2f" % (pars_3[0], perr_3lorentz[6]))
            print ("center = %0.2f (+/-) %0.2f" % (pars_3[1], perr_3lorentz[7]))
            print ("width = %0.2f (+/-) %0.2f" % (pars_3[2], perr_3lorentz[8]))
            print ("area = %0.2f" % np.trapz(lorentz_peak_3))
            print ("--------------------------------")

plot_1()
plot_2('plot')
# plot_3('plot')


# if argv:
#     if argv[0]=='plot': return plt.show()
#     if argv[0]=='save': return fig.savefig("foo_"+str(stamp)+"_"+"fit3Lorentzian_peaks_resid.png", format="png",dpi=1000)


# # # Voigt Peak Fitting

# # In[24]:


# x_array = np.linspace(1,100,50)

# ampG1 = 20
# cenG1 = 50
# sigmaG1 = 5
# ampL1 = 80
# cenL1 = 50
# widL1 = 5

# y_array_voigt = (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x_array-cenG1)**2)/((2*sigmaG1)**2)))) +                ((ampL1*widL1**2/((x_array-cenL1)**2+widL1**2)) )

# # creating some noise to add the the y-axis data
# y_noise_voigt = (((np.random.ranf(50))))*5
# y_array_voigt += y_noise_voigt


# # In[25]:


# fig = plt.figure(figsize=(4,3))
# gs = gridspec.GridSpec(1,1)
# ax1 = fig.add_subplot(gs[0])

# ax1.plot(x_array, y_array_voigt, "ro")

# #ax1.set_xlim(-5,105)
# #ax1.set_ylim(-0.5,5)

# ax1.set_xlabel("x array")
# ax1.set_ylabel("y array")

# ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
# #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

# ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

# ax1.tick_params(axis='both',which='major', direction="in", top="on", right="on", bottom="on")
# ax1.tick_params(axis='both',which='minor', direction="in", top="on", right="on", bottom="on")

# if argv:
#     if argv[0]=='plot': return plt.show()
#     if argv[0]=='save': return fig.savefig("foo_"+str(stamp)+"_"+"raw_voigt.png", format="png",dpi=1000)


# # In[26]:


# def _1Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
#     return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/((2*sigmaG1)**2)))) +              ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)) )


# # In[27]:


# popt_1voigt, pcov_1voigt = scipy.optimize.curve_fit(_1Voigt, x_array, y_array_voigt, p0=[ampG1, cenG1, sigmaG1,                                                                                          ampL1, cenL1, widL1])

# perr_1voigt = np.sqrt(np.diag(pcov_1voigt))

# pars_1 = popt_1voigt
# voigt_peak_1 = _1Voigt(x_array, *pars_1)


# # In[28]:


# residual_1voigt = y_array_voigt - (_1Voigt(x_array, *popt_1voigt))


# # In[29]:


# fig = plt.figure(figsize=(4,4))
# gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
# ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1])
# gs.update(hspace=0) 

# ax1.plot(x_array, y_array_voigt, "ro")
# ax1.plot(x_array, _1Voigt(x_array, *popt_1voigt), 'k--')#,\
#          #label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))

# # peak 1
# ax1.plot(x_array, voigt_peak_1, "g")
# ax1.fill_between(x_array, voigt_peak_1.min(), voigt_peak_1, facecolor="green", alpha=0.5)

# # residual
# ax2.plot(x_array, residual_1voigt, "bo")
    
# #ax1.set_xlim(-5,105)
# #ax1.set_ylim(-0.5,8)

# #ax2.set_xlim(-5,105)
# #ax2.set_ylim(-0.5,0.75)

# ax2.set_xlabel("x array")
# ax1.set_ylabel("y array")
# ax2.set_ylabel("Res.")

# ax1.legend(loc="best")

# ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
# #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

# ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

# ax1.xaxis.set_major_formatter(plt.NullFormatter())

# ax1.tick_params(axis='both', which='both', direction="in", top="on", right="on", bottom="off")
# ax2.tick_params(axis='both', which='both', direction="in", top="off", right="on", bottom="on")


# if argv:
#     if argv[0]=='plot': return plt.show()
#     if argv[0]=='save': return fig.savefig("foo_"+str(stamp)+"_"+"fit1Voigt_peaks_resid.png", format="png",dpi=1000)


# # In[30]:


# ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1


# # In[31]:


# print ("gauss. weight = %0.2f" % ((pars_1[0]/(pars_1[0]+pars_1[3]))*100))
# print ("lorentz. weight = %0.2f" % ((pars_1[3]/(pars_1[0]+pars_1[3]))*100))

