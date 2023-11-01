# This is a program implementing numerical routines for
# modeling Electron-Beam Collection Current Efficiency (EBIC) 
# with fitting using metal thickness (tm), screen-charged region (W), 
# diffusion length (L) and portion of absorbed electrons (G0) 
# as fitting parameters. 
# 
# The low-energy threshold of collection efficiency corresponds to 
# metal thickness; the peak of collection efficiency corresponds to e-h pair
# absorption inside the screen-charged region; high energy corresponds
# to e-h pair creation far away from the screen-charge region deep inside
# the studied sample, and the collection efficiency is defined by the diffusion length 
#
# Data should be stored as a .svg file with the first column as electron beam 
# energy in [keV] and the second column as Ic/(Ib*Eb) in [keV].
# 
# The fit_data() function returns initial data, fitted curve, and optimal parameters,
# plot it with estimated errors and confidence intervals.


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy.integrate import quad
from scipy.optimize import curve_fit

def Ic(Eb, tm, W, L, G0):
    """
    Returns Electron Beam Collection Efficiency 
    (EBIC) value as a function of electron beam energyepsabs = 1e-2, 
    for given parameters
    E.B. Yakimov et al 2020 J. Phys. D: Appl. Phys. 53 495108
    
    Parameters
    ----------
    Eb : float, array 
        Electron beam energy in meV
    tm : float
        Metal thickness in nm
    W : float
        Screen charge region thickness in nm
    L : float 
        Diffusion length in nm
    G0 : float 
        η/Ei portion of absorbed electrons (η)
        with e-h pair formation energy (Ei)
    
    Returns
    -------
    I : float, array
        Electron Beam Collection Efficiency in meV-1 
    """    
    
    def A(z, Eb):
        R = 7.34*Eb**(1.75) #[nm] for Eb in [keV]
        return np.piecewise(z, 
                            [     z < 0.22*R,    z >= 0.22*R],
                            [lambda z: 12.84, lambda z: 3.97])

    def h(z, Eb):
        R = 7.34*Eb**(1.75) #[nm] for Eb in [keV]
        return 1.603/R*np.exp(-A(z, Eb)*(z/R - 0.22)**2)
    
    def hexp(z, Eb):
        R = 7.34*Eb**(1.75) #[nm] for Eb in [keV]
        return 1.603/R*np.exp(-A(z, Eb)*(z/R - 0.22)**2)*np.exp(-(z-W)/L)

    
    I1 = np.asarray([quad(   h, tm,      W, args=(_Eb), 
                          epsabs = 1e-2, epsrel = 1e-2)[0] for _Eb in Eb])
    I2 = np.asarray([quad(hexp, tm, np.inf, args=(_Eb), 
                          epsabs = 1e-2, epsrel = 1e-2)[0] for _Eb in Eb])
    I  = (I1 + I2)
    I = I*G0
    return I


def fit_data(path, tm, W, speed = True, plot = True, error = False):
    """
    Computes optimal parameters for EBIC data 
    and returns fitted data
    
    
    Parameters
    ----------
    path : string
        Path to data file
    tm : float
      Initial guess value of metal thickness
    W : float
        Initial guess value of SCR
    
    Returns
    -------
    Eb : array
        Electron beam energies from data file in meV
    I : array
        Collection efficiencies for E from data file in meV-1
    Ic : array
        Fitted collection efficiencies for E and popt
    popt : [tm, W, L, G0]
        Optimal fitting parameters for Ic() func
        
    Other Parameters
    ----------------
    speed : bollean
        True for Elapsed time display
    plot : boolean
        True for plot fitted data
    error : boolean
        True for plot confidence intefval
    """
    
            
    def plot_data(Eb, I, popt, perr):
        """
        Plots experimental and fitted data
        """
        
        I_fit = Ic(Eb, *popt)
        
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]})
        gs = axs[0].get_gridspec()
        
        # main axes
        axs[0].set_xlabel(r'Beam energy $E_b$ (keV)')
        axs[0].set_ylabel(r'Collection Efficiency ${I_c}/{I_b E_b}$ (keV$^{-1}$)')
        axs[0].plot(Eb, I,     'sk', label = r'data')
        axs[0].plot(Eb, I_fit, 'r-', label = r'fit, $L_D = %.0f \,nm$'%popt[2])

        # error axes
        axs[1].set_ylabel(r'Error (%)')
        axs[1].plot(Eb, (I - I_fit)/I*100, '-ro')
        
        if error:
            popt_up, popt_dw = popt+0., popt+0.
            
            popt_dw[2] = popt[2] + perr[2] # approx for lognormal
            popt_up[2] = popt[2] - perr[2]/3 # approx for lognormal

            
            axs[0].fill_between(Eb, Ic(Eb, *popt_up), Ic(Eb, *popt_dw), 
                                facecolor = (1, 0, 0, 0.1), label = 'Confidence interval')
            
        axs[0].legend()
        plt.tight_layout()
    
    
    if speed:
        import time 
        start = time.time()
    
    csv = np.loadtxt(path, delimiter = ',')
    
    Eb = csv[:,0] #[keV]
    Imax = max(csv[:,1])
    I = csv[:,1] #Ic/(Ib*Eb) in [keV-1]
    
    X = 7.34*Eb**(1.75) # [nm] for Eb in [keV]
    
    guess = [tm, W, 51, 1.]
    bounds = ([1.0, 100.0, 5.0, 1e-4], [250.0, 5.0E4, 850, 10.])
    
    popt, pcov = curve_fit(Ic, Eb, I, p0 = guess, bounds = bounds) 
    perr = np.sqrt(np.diag(pcov))
    
    print('Metal thickness : %.2f nm ± %.2f nm\n'%(popt[0], perr[0]),
          'SCR width : %.2f nm ± %.2f nm\n'%(popt[1], perr[1]),
          'Diffusion length : %.2f ± %.2f nm\n'%(popt[2], perr[2]),
          'η/Ei : %.2f ± %.2f eV-1'%(popt[3], perr[3]))

    if speed:
        print('-----------------------------------------')
        end = time.time()
        print('Elapsed time: %.2f s'%(end - start))
        
    if plot:
        plot_data(Eb, I, popt, perr)

    return Eb, I, Ic(Eb, *popt), popt

