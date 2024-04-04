
# coding: utf-8

# In[1]:


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


def _2gaussian(x, amp1,cen1,sigma1, amp2,cen2,sigma2):
    return (amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))) + 
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen2)/sigma2)**2))))





def plot_1(*argv):
    """ Single Gaussian Peak Fitting """

    # linearly spaced x-axis of 10 values between 1 and 10
    amp1, sigma1, cen1 = 100, 10, 50
    y_array_gauss = amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2)))

    # creating some noise to add the the y-axis data
    y_noise_gauss = (np.exp((np.random.ranf(50))))/5
    y_array_gauss += y_noise_gauss

    ## [fit] _1dgaussian()
    popt_gauss, pcov_gauss = scipy.optimize.curve_fit(_1gaussian, x_array, y_array_gauss, p0=[amp1, cen1, sigma1])
    perr_gauss             = np.sqrt(np.diag(pcov_gauss))

    # this cell prints the fitting parameters with their errors
    print ("amplitude = %0.2f (+/-) %0.2f" % (popt_gauss[0], perr_gauss[0]))
    print ("center = %0.2f (+/-) %0.2f" % (popt_gauss[1], perr_gauss[1]))
    print ("sigma = %0.2f (+/-) %0.2f" % (popt_gauss[2], perr_gauss[2]))


    fig = plt.figure(figsize=(4,3))
    gs = gridspec.GridSpec(1,1)
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(x_array, y_array_gauss, "ro")
    ax1.plot(x_array, _1gaussian(x_array, *popt_gauss), 'k--')#,\
             #label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))

    ax1.set_xlim(-5,105)
    ax1.set_ylim(-0.5,5)

    ax1.set_xlabel("x array")
    ax1.set_ylabel("y array")

    ax1.legend(loc="best")

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.tick_params(axis='both',which='major', direction="in", top="on", right="on", bottom="on")
    ax1.tick_params(axis='both',which='minor', direction="in", top="on", right="on", bottom="on")
    if argv:    
        if argv[0]=='plot': return plt.show()
        if argv[0]=='save': return fig.savefig("foo_"+str(stamp)+"_"+"fitGaussian.png", format="png",dpi=1000)


# plot_1('plot')



def plot_2(*argv):
    """ Multiple Gaussian Peak Fitting """
    amp1, sigma1, cen1 = 100, 10, 40
    amp2, sigma2, cen2 = 75, 5, 65
    y_array_2gauss = (amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))) + 
                      amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen2)/sigma2)**2))))

    # creating some noise to add the the y-axis data
    y_noise_2gauss = (np.exp((np.random.ranf(50))))/5
    y_array_2gauss += y_noise_2gauss
    ## fit 
    popt_2gauss, pcov_2gauss = scipy.optimize.curve_fit(_2gaussian, x_array, y_array_2gauss, p0=[amp1, cen1, sigma1,amp2, cen2, sigma2])
    perr_2gauss = np.sqrt(np.diag(pcov_2gauss))
    ## fit params
    pars_1 = popt_2gauss[0:3]
    pars_2 = popt_2gauss[3:6]
    gauss_peak_1 = _1gaussian(x_array, *pars_1)
    gauss_peak_2 = _1gaussian(x_array, *pars_2)
    ## residual
    residual_2gauss = y_array_2gauss - (_2gaussian(x_array, *popt_2gauss))

    fig, gs = plt.figure(figsize=(4,4)), gridspec.GridSpec(2,1, height_ratios=[1,0.25])
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    gs.update(hspace=0) 

    # data
    ax1.plot(x_array, y_array_2gauss, "ro", label='data')
    # fit 
    ax1.plot(x_array, _2gaussian(x_array, *popt_2gauss), 'k--', label='fit') 

    # peak 1
    ax1.plot(x_array, gauss_peak_1, "g", label='peak 1')
    ax1.fill_between(x_array, gauss_peak_1.min(), gauss_peak_1, facecolor="green", alpha=0.5)
      
    # peak 2
    ax1.plot(x_array, gauss_peak_2, "y", label='peak 2')
    ax1.fill_between(x_array, gauss_peak_2.min(), gauss_peak_2, facecolor="yellow", alpha=0.5)  

    # residual
    ax2.plot(x_array, residual_2gauss, "bo")
        
    ax1.set_xlim(-5,105), ax1.set_ylim(-0.5,8)

    ax2.set_xlim(-5,105), ax2.set_ylim(-0.5,0.75)

    ax2.set_xlabel("x array"), ax1.set_ylabel("y array")
    ax2.set_ylabel("Res.")   , ax1.legend(loc="best")

    
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    ax1.tick_params(axis='both',which='both', direction="in", top="on", right="on", bottom="off" )
    ax1.tick_params(axis='both',which='both', direction="in", top="on", right="on", bottom="off" )

    ax2.tick_params(axis='both',which='both', direction="in", top="off", right="on", bottom="on" )
    ax2.tick_params(axis='both',which='both', direction="in", top="off", right="on", bottom="on" )

    if argv:
            if argv[0]=='plot': return plt.show()
            if argv[0]=='save': return fig.savefig("foo_"+str(stamp)+"_"+"fit2Gaussian.png", format="png",dpi=1000)
            if argv[0]=='print': 

                # this cell prints the fitting parameters with their errors
                # print ("----------------------------- Peak 1 -------------------------- ")
                print ("-------------------------------------------------------")
                print (" ********* Peak 1 ********* ")
                print ("amplitude = %0.2f (+/-) %0.2f" % (pars_1[0], perr_2gauss[0]))
                print ("center = %0.2f (+/-) %0.2f" % (pars_1[1], perr_2gauss[1]))
                print ("sigma = %0.2f (+/-) %0.2f" % (pars_1[2], perr_2gauss[2]))
                print ("area = %0.2f" % np.trapz(gauss_peak_1))
                # print ("----------------------------- Peak 2 -------------------------- ")
                print (" ********* Peak 2 ********* ")
                print ("amplitude = %0.2f (+/-) %0.2f" % (pars_2[0], perr_2gauss[3]))
                print ("center = %0.2f (+/-) %0.2f" % (pars_2[1], perr_2gauss[4]))
                print ("sigma = %0.2f (+/-) %0.2f" % (pars_2[2], perr_2gauss[5]))
                print ("area = %0.2f" % np.trapz(gauss_peak_2))
                print ("-------------------------------------------------------")


plot_1('plot')

plot_2('plot')
