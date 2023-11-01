# This is listing of a program implementing numerical routines for
# inverse Laplace transform of experimental transients with following 
# interpretation for Deep-Level Transient Spectroscopy (DLTS)
# 
# All functions present below are stored as individual .py module files
# separated with comment lines and stored in the /modules directory. 
# Results stored with hp() in /processed directory with .LDLTS extension.
#
# Data stored in /data directory as .DLTS and .PERS extension files
# and can be read with read_file(). 
# 
# interface() and demo() functions are part of the interface 
# and all computational routines can be controlled from there. 
#
# Contin(), reSpect(), L1L2() and L2() are 4 algorithms to perform
# the inverse Laplace transform routines. All 4 of them are needed to 
# increase certainty of obtained results. 

########## interface.py module ############

def interface(path):
    """Initiates and displays widgets from iPyWigets
    and sends data to demo() with interactive_output()
    """

    import numpy as np
    import ipywidgets as widgets

    from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, Label
    
    from read_file import read_file
    from demo import demo
    
    import warnings
    warnings.filterwarnings('ignore')
    
    interface.path = path


    t, C, T = read_file(path)
    
    if len(T.shape) != 0:
        cut = len(T) - 1
    else:
        cut = 1
    Index = widgets.IntSlider(
        value=0,
        min=0, # max exponent of base
        max=cut, # min exponent of base
        step=1, # exponent step
        description='')

    Methods = widgets.SelectMultiple(
        options = ['L2', 'L1+L2', 'Contin', 'reSpect'],
        value   = ['Contin'],
        #rows    = 10,
        description = 'Methods:',
        disabled = False)

    Nz = widgets.IntText(
        value=100,
        description=r'$N_f=$',
        disabled=False)

    Reg_L1 = widgets.FloatLogSlider(
        value=1e-8,
        base=10,
        min=-10, # max exponent of base
        max=1, # min exponent of base
        step=0.2, # exponent step
        description=r'L1: $\lambda_1$')

    Reg_L2 = widgets.FloatLogSlider(
        value=1e-8,
        base=10,
        min=-10, # max exponent of base
        max=1, # min exponent of base
        step=0.2, # exponent step
        description=r'L2: $\lambda_2$')
    
    Reg_C = widgets.FloatLogSlider(
        value=1E-1,
        base=10,
        min=-8, # max exponent of base
        max=2, # min exponent of base
        step=0.2, # exponent step
        description=r'Contin: $\lambda_{\text{C}}$')
    
    Reg_S = widgets.FloatLogSlider(
        value=1E-2,
        base=10,
        min=-12, # max exponent of base
        max=2, # min exponent of base
        step=0.2, # exponent step
        description=r'reSpect: $\lambda_{\text{S}}$')

    Bounds = widgets.IntRangeSlider(
        value=[-2, 2],
        min=-5,
        max=5,
        step=1,
        description=r'$10^{a}\div 10^{b}$:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    
    dt = widgets.BoundedFloatText(
        value=150,
        min=0,
        max=1000,
        step=1,
        description='Time step, ms',
        disabled=False)

    Plot = widgets.ToggleButton(
        value=True,
        description='Hide graphics?',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Hides graphics',
        icon='eye-slash')
    
    Residuals = widgets.ToggleButton(
        value=False,
        description='Compute L-curve?',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Plots L-curve to choose best value of regularization parameter of L2 reg. method',
        icon='calculator')
    
    LCurve = widgets.Checkbox(
        value = False,
        description = 'Use L-Curve optimal?',
        disabled = False)
    
    Arrhenius = widgets.Checkbox(
        value = False,
        description = 'Draw Arrhenius instead DLTS?',
        disabled = False)

    Heatplot = widgets.ToggleButton(
        value=False,
        description='Plot heatmap?',
        disabled=False,
        button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Plots heatmap of data from chosen file',
        icon='braille')


    left_box = VBox([Methods, Nz, dt])
    centre_box = VBox([Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds])
    right_box = VBox([LCurve, Arrhenius, Plot, Residuals, Heatplot])
    ui = widgets.HBox([left_box, centre_box, right_box])
    Slider = widgets.HBox([Label('Transient №'),Index])
    out = widgets.interactive_output(demo, {'Index':Index,   'Nz':Nz, 
                                            'Reg_L1':Reg_L1, 'Reg_L2':Reg_L2, 'Reg_C':Reg_C, 'Reg_S':Reg_S, 
                                            'Bounds':Bounds, 'dt':dt,         'Methods':Methods,
                                            'Plot':Plot,     'LCurve':LCurve, 'Arrhenius':Arrhenius,
                                            'Residuals':Residuals, 'Heatplot': Heatplot})
    display(ui, Slider, out)

########## demo.py module ############

def demo(Index, Nz, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, dt, Methods, Plot, Residuals, LCurve, Arrhenius, Heatplot):
    """Gets data from widgets initialized with interface(), 
    calls laplace() to process data and calls plot_data(), hp()
    to plot results

    Parameters:
    -------------
    Index : int
        Index of transient in dataset
    Nz : int
        Value which is length of calculated vector
    Reg_L1, Reg_L2 : floats
        Reg. parameters for L1 and L2 regularization
    Reg_C, Reg_S : floats
        Reg. parameters for CONTIN and reSpect algorithms
    Bounds : list
        [lowerbound, upperbound] bounds of emission rates domain points
    dt : int
        Time step of transient data points in ms
    Methods : list 
        Methods to process data
    Plot : boolean
        Calls plot_data() if True
    Residuals : boolean
        Calls regopt() and plots L-curve to control 
        regularization if True
    LCurve : boolean, 
        Automatically picks optimal reg. parameter from 
        L-curve if True
    Heatplot : boolean
        Plots heatplot for all dataset and saves data if True
        in .LDLTS if True
    """

    import numpy as np

    from read_file import read_file
    from laplace import laplace
    from plot_data import plot_data
    from hp import hp
    from read_file import read_file
    from regopt import regopt
    from interface import interface

    Bounds = 10.0**np.asarray(Bounds)

    t, C, T = read_file(interface.path, dt, proc = True)# time, transients, temperatures 
    
    data = laplace(t, C[Index], Nz, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Methods)
    #print(data)
    
    if Plot:
        ay, aph = plot_data(t, C[Index], data, T, Index)
        
        if Heatplot:
            hp(t, C, T, aph, Methods, Index, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz, LCurve, Arrhenius)
        
    if Residuals:
        regopt(t, C[Index], ay, Methods, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz)

########## laplace.py module ############

def laplace(t, F, Nz, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Methods):

    """ Initiates routines for chosen method

    Parameters:
    -------------
    t : array 
        Time domain data from experiment
    F : array,
        Transient data from experiment F(t)
    Nz : int
        Number of points z to compute, must be smaller than len(F)
    Reg_L1, Reg_L2 : floats
        Reg. parameters for L1 and L2 regularization
    Reg_C, Reg_S : floats
        Reg. parameters for CONTIN and reSpect algorithms
    Bounds : list
        [lowerbound, upperbound] of s domain points
    Methods : list 
        Names of processing methods

    Returns:
    -------------
    data : list of [[s, f, F_restored, Method1], ...]
        Data list for processed data

    """
    import numpy as np

    from L2 import L2
    from L1L2 import L1L2
    from contin import Contin
    from reSpect import reSpect

    data = []

    for i in Methods:
        if i == 'L2':
            s, f, F_hat = L2(t, F, Bounds, Nz, Reg_L2)
            data.append([s, f, F_hat, 'L2'])

        elif i == 'L1+L2':
            s, f, F_hat = L1L2(t, F, Bounds, Nz, Reg_L1, Reg_L2)
            data.append([s, f, F_hat, 'L1+L2'])

        elif i == 'Contin':
            s, f, F_hat = Contin(t, F, Bounds, Nz, Reg_C)
            data.append([s, f, F_hat, 'Contin'])

        elif i == 'reSpect':
            s, f, F_hat = reSpect(t, F, Bounds, Nz, Reg_S)
            data.append([s, f, F_hat, 'reSpect'])

    data = np.asarray(data)
    return data

########## plot_data.py module ############

def plot_data(t, F, data, T, Index):
    """Gets data from demo() and plots it:

    Parameters:
    -------------
    t : array 
        Time domain data from experiment
    F : array,
        Transient data from experiment F(t)
    data : list of [[s, f, F_restored, Method1], ...]    
        Data list for processed data
    T : float 
        Temperature value of certain transient
    Index : int 
        Index of transient in initial dataset (not data)

    Returns:
    -------------
    ay : matplotlib axes 
        Axes for L-Curve plotting 
    [ahp1, ahp2] : list of matplotlib axes 
        Axes for its Arrhenuis or DLTS plots in hp()
    """

    import numpy as np
    import matplotlib.pyplot as plt

    ## Plotting main plot f(s)
    fig = plt.figure(constrained_layout=True, figsize = (9.5,11))
    widths  = [0.5, 0.5]
    heights = [0.3, 0.3, 0.4]
    spec = fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths,
                              height_ratios=heights)


    ax  = fig.add_subplot(spec[0,:])
    ax.set_title(r'Temperature %.2f K'%T[Index])
    ax.set_ylabel(r'Amplitude, arb. units')
    ax.set_xlabel(r'Emission rate, $s^{-1}$')
    ax.set_xscale('log')
    ax.grid(True, which = "both", ls = "-")
    #print(data[:,2])
    for i, e in enumerate(data[:,-1]):
        if e == 'L2':
            ax.plot(data[i][0], data[i][1], 'b-', label = e)
        elif e == 'L1+L2':
            ax.plot(data[i][0], data[i][1], 'm-', label = e)
        elif e == 'Contin':
            ax.plot(data[i][0],  data[i][1]*data[i][0], 'c-', label = e)
        elif e == 'reSpect':
            ax.plot(data[i][0],  data[i][1], 'y-', label = e)
    ax.legend()

    # Axes for L-Curve
    ay = fig.add_subplot(spec[1, 0])

    # Plotting transients F(t)
    az = fig.add_subplot(spec[1, 1])
    az.set_ylabel(r'Transient , arb. units')
    az.set_xlabel(r'Time $t$, $s$')
    az.grid(True, which = "both", ls = "-")
    az.plot(t, F, 'ks-', label = 'Original')
    az.set_xscale('log')
    for i, e in enumerate(data[:,-1]):
        if e == 'L2':
            d = data[i][2][:-1] # last point sucks
            az.plot(t[:-1], d, 'b>-', label = e)
        elif e == 'L1+L2':
            d = data[i][2]
            az.plot(t, d, 'm*-', label = e)
        elif e == 'Contin':
            d = data[i][2]
            az.plot(t, d, 'cx-', label = e)
        elif e == 'reSpect':
            d = data[i][2]
            az.plot(t, d - d[-1] + F[-1], 'y+-', label = e)
    az.legend()


    plt.tight_layout()

    ahp1, ahp2 = fig.add_subplot(spec[2, 0]), fig.add_subplot(spec[2, 1])

    return ay, [ahp1, ahp2]

########## hp.py module ############

def hp(t, C, T, ahp, Methods, Index, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz, LCurve = False, Arrhenius = False):
    """Plots heatmap in ahp = [ahp1, ahp2] axes
    ahp1 for T vs Emission rates and
    ahp2 for Arrhenius or DLTS plots

    Parameters:
    -------------
    t : array 
        Time domain data from experiment
    C : 2D array (len(t), len(T))
        Contains transients for temperatures 1, 2, ...
        [F1(t), F2(t),...] from experiment
    T : array 
        Temperature from experiment
    ahp : list of matplotlib axes [ahp1, ahp2] 
        Axes to plot heatplot and Arrhenius
    Methods : list
        Method names used for plotting
    Index : int 
        Index to plot specific slice of heatplot
    Reg_L1, Reg_L2 : floats
        Reg. parameters for L1 and L2 regularization
    Reg_C, Reg_S : floats
        Reg. parameters for CONTIN and reSpect algorithms
    Bounds : list
        [lowerbound, upperbound] of s domain points
    Nz : int
        Value which is length of calculated vector f(s)
    LCurve : boolean 
        Plot using L-curve criteria if True
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib import cm
    from matplotlib import gridspec

    from L2 import L2
    from L1L2 import L1L2
    from contin import Contin
    from reSpect import reSpect
    from regopt import regopt

    import sys

    def progressbar(i, iterations):
        """Prints simple progress bar"""
        i = i + 1
        sys.stdout.write("[%-20s] %d%%  Building Heatmap" % ('#'*np.ceil(i*100/iterations*0.2).astype('int'), 
            np.ceil(i*100/iterations))+'\r')
        sys.stdout.flush()

    cut = len(T)
    cus = Nz

    if len(Methods) > 1:
        print('Choose only one Method')
        Methods = Methods[0]

    XZ = []
    YZ = []
    ZZ = []

    for M in Methods:
        if M == 'L2':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L2(t, C[i], Bounds, Nz, Reg_L2)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)

                progressbar(i, cut)

        elif M == 'L1+L2':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L1L2(t, C[i], Bounds, Nz, Reg_L1, Reg_L2)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)

                progressbar(i, cut)

        elif M == 'Contin':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                if LCurve:
                    ay = 0
                    Reg_C = regopt(t, C[i], ay, Methods, Reg_L1, Reg_L2, 
                                    Reg_C, Reg_S, Bounds, Nz, LCurve)
                TEMPE, TEMPX, a = Contin(t, C[i], Bounds, Nz, Reg_C)
                #print(YZ[-1][0], 'K; a = ', Reg_C)
                XZ.append(TEMPE)
                ZZ.append(TEMPX*TEMPE)

                progressbar(i, cut)

        elif M == 'reSpect':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                if LCurve:
                    ay = 0
                    Reg_S = regopt(t, C[i], ay, Methods, Reg_L1, Reg_L2, 
                                    Reg_C, Reg_S, Bounds, Nz, LCurve)
                TEMPE, TEMPX, a = reSpect(t, C[i], Bounds, Nz, Reg_S)
                #print(YZ[-1][0], 'K; a = ', Reg_C)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)

                progressbar(i, cut)


    XZ = np.asarray(XZ)
    YZ = np.asarray(YZ)
    ZZ = np.asarray(ZZ)

    ahp1, ahp2 = ahp[0], ahp[1]

    if Methods[0] == 'reSpect':
        v = np.abs(np.average(ZZ[10:-10,5:-5]))*20
        vmin, vmax = v/1e1, v/2
        cmap = cm.jet
        levels = np.linspace(vmin, vmax, 20)
        #print(v)

    elif Methods[0] == 'Contin':
        v = np.abs(np.average(ZZ[10:-10,5:-5]))*10
        vmin, vmax = 0, v
        cmap = cm.gnuplot2
        levels = 20

    #extent = [np.log10(Bounds[0]), np.log10(Bounds[1]), (T[-1]), (T[0])]

    x, y = np.meshgrid(TEMPE, T)

    ahp1.set_xlabel(r'Emission rate $e_{n,p}$, s')
    ahp1.set_title(Methods[0])
    ahp1.set_ylabel('Temperature T, K')
    ahp1.grid(True)
    #normalize = plt.Normalize(vmin = -v, vmax = v)

    heatmap = ahp1.contourf(x, y, ZZ, levels = levels,   cmap=cmap, corner_mask = False,
                            vmin = vmin, vmax = vmax, extend = 'both')
    plt.colorbar(heatmap)
    ahp1.set_xscale('log')

    if Arrhenius:

        ahp2.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useOffset = False)

        arrh = ahp2.contourf(1/y, np.log(x*y**-2), ZZ, levels = levels, cmap=cmap,
                             vmin = vmin, vmax = vmax, extend = 'both')

        ahp2.set_xlabel('Temperature $1/T$, $K^-1$')
        ahp2.set_ylabel('$\ln(e\cdot T^-2)$')

    else:
        ahp2.set_xlabel('Temperature, K')
        ahp2.set_ylabel('LDLTS signal, arb. units')
        for i in range(int(len(TEMPE)*0.1), int(len(TEMPE)*0.8), 20):
        #    ad.plot(T, ZZ[:, i], label=r'$\tau = %.3f s$'%(1/TEMPE[i]))
            ahp2.plot(T, ZZ[:, i]/np.amax(ZZ[:,i]), label=r'$\tau = %.3f s$'%(1/TEMPE[i]))
        #ahp2.set_yscale('log')
        #ahp2.set_ylim(1E-4, 10)
        ahp2.grid()
        ahp2.legend()

    plt.show()
    plt.tight_layout()

    ##save file
    #Table = []
    #Table.append([0] + (1/TEMPE).tolist())
    #for i in range(cut):
    #    Table.append([T[i]] + (ZZ[i,:]).tolist())

    Table = []
    e = 1/TEMPE
    Table.append([0] + e.tolist())
    for i in range(cut):
        Table.append([T[i]] + (ZZ[i,:]).tolist())


    np.savetxt('processed/NEW-FILE'%((t[1]-t[0])*1000) +'_1'+'.LDLTS', 
                Table, delimiter='\t', fmt = '%4E')

########## regopt.py module ############

def regopt(t, F, ay, Methods, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz, LCurve = False):
    """ 
    Computes L-curve from residual and solution norm 
    and derives optimal regularization parameter from 
    its curvature plot. (max(k) -> a_opt)

    Parameters:
    -------------
    t : array 
        Time domain data from experiment
    F : array,
        Transient data from experiment F(t)
    ay : matplotlib axes
        Axes for L-curve plotting
    Methods : list 
        Names of processing methods
    Reg_L1, Reg_L2 : floats
        Reg. parameters for L1 and L2 regularization
    Reg_C, Reg_S : floats
        Reg. parameters for CONTIN and reSpect algorithms
    Bounds : list
        [lowerbound, upperbound] of emission rates domain points
    Nz : int
        Number of points in emissior rates domain to compute, 
        must be smaller than len(F)
    LCurve : boolean
        If True regopt() returns optimal regularization 
        paremeter for chosen method

    Returns:
    -------------
    alpha[i] : float 
        Optimal reg. parameter for chosen method
        using L-curve criteria
    """
    from laplace import laplace
    #from matplotlib.cm import jet
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import savgol_filter

    import sys

    def progressbar(i, iterations):
        """Prints simple progress bar"""
        i = i + 1
        sys.stdout.write("[%-20s] %d%%  Building L-Curve" % ('#'*np.ceil(i*100/iterations*0.2).astype('int'), 
                        np.ceil(i*100/iterations))+'\r')
        sys.stdout.flush()

    def curvature(x, y, a):
        """Returns curvature of line defined by:

        k = (x'*y''-x''*y')/((x')^2+(y')^2)^(3/2)

        Parameters:
        -------------
        x, y : arrays of x and y respectively
        a : array of reg parameters 

        Returns:
        -------------
        k : array of curvature values
        """
        x = savgol_filter(x, 13, 1)
        y = savgol_filter(y, 13, 1)
        da = np.gradient(a)
        f_x  = np.gradient(x)/da
        f_y  = np.gradient(y)/da
        f_xx = np.gradient(f_x)/da
        f_yy = np.gradient(f_y)/da

        k = (f_x*f_yy - f_xx*f_y)/(f_x**2 + f_y**2)**(3/2)
        return savgol_filter(k, 5, 1)
        #return k

    res = []  # residuals norm ||Cf - F||2
    sol = []  # solution norm ||f||2

    alpha_L2  = 10**np.linspace(np.log10(Reg_L2)  - 3, np.log10(Reg_L2)  + 3, 40)
    alpha_C   = 10**np.linspace(np.log10(Reg_C) - 3, np.log10(Reg_C) + 3, 40)
    alpha_S   = 10**np.linspace(np.log10(Reg_S) - 3, np.log10(Reg_S) + 3, 40)

    if LCurve:
        alpha_C = 10**np.linspace(np.log10(Reg_C) - 3, np.log10(Reg_C) + 3, 40)
    alpha = alpha_C

    data = []

    Fx = F

    for i in Methods:

        if len(Methods) > 1:
            print('!!!Choose only one Method!!!')
            break

        if i == 'L2':
            for j, v in enumerate(alpha_L2):
                data = laplace(t, F, Nz, Reg_L1, v, Reg_C, Reg_S, Bounds, Methods)
                e, f, F_restored = data[0][0], data[0][1], data[0][2]

                res.append(np.linalg.norm(np.abs(Fx) - np.abs(F_restored), ord = 2)**2)
                sol.append(np.linalg.norm(f, ord = 2)**2)
                progressbar(j, len(alpha_L2))
            alpha = alpha_L2
            break

        elif i == 'L1+L2':
            break

        elif i == 'Contin':
            for j, v in enumerate(alpha_C):
                data = laplace(t, F, Nz, Reg_L1, Reg_L2, v, Reg_S, Bounds, Methods)
                e, f, F_restored = data[0][0], data[0][1], data[0][2]

                res.append(np.linalg.norm(np.abs(Fx) - np.abs(F_restored), ord = 2)**2)
                #sol.append(np.linalg.norm(f, ord = 2)**2)
                sol.append(np.linalg.norm(f*e, ord = 2)**2)
                progressbar(j, len(alpha_C))
            alpha = alpha_C
            break

        elif i == 'reSpect':
            for j, v in enumerate(alpha_S):
                data = laplace(t, F, Nz, Reg_L1, Reg_L2, Reg_C, v, Bounds, Methods)
                e, f, F_restored = data[0][0], data[0][1], data[0][2]

                res.append(np.linalg.norm(np.abs(Fx - Fx[-1]) - np.abs(F_restored - F_restored[-1]), ord = 2)**2)
                sol.append(np.linalg.norm(f, ord = 2)**2)
                progressbar(j, len(alpha_S))
            alpha = alpha_S
            break

    # Plotting L-curve and its normalized curvature

    if len(data) == 0:
        ay.annotate(text = 'Choose only one method \n Contin, reSpect or L2', 
                    xy = (0.5,0.5), ha="center", size = 16)
        plt.tight_layout()
    elif LCurve:
        k = curvature(np.log10(res), np.log10(sol), alpha)
        k_max = np.amax(k)
        if Methods[0] == 'reSpect':
            i = np.where(k == np.amax(k[1:-1]))
        else:
            i = np.where(k == np.amax(k[1:-1]))
        i = np.squeeze(i)
        return alpha[i]
    else:
        k = curvature(np.log10(res), np.log10(sol), alpha)
        k_max = np.amax(k)
        i = np.where(k == np.amax(k))
        if Methods[0] != 'reSpect':
            i = np.where(k == np.amax(k[1:-1]))
        i = np.squeeze(i)

        ay.plot(np.log10(res),    np.log10(sol),    'k-', )

        ay.set_ylabel(r'Solution norm $\lg||x||^2_2$', c='k')
        ay.set_xlabel(r'Residual norm $\lg||\eta-Cx||^2_2$', c='k')

        ay_k = ay.twinx()
        ay_k_t = ay_k.twiny()
        ay_k_t.set_xscale('log')
        ay_k_t.plot(alpha,    k/k[i],    'r-')
        ay_k.set_ylabel(r'Curvature, arb. units', c='r')
        ay_k.set_ylim(-0.1, 1.1)
        #ay_k.set_yscale('log')
        #ay_k.set_ylim(1e-3, 2.0)
        ay_k_t.set_xlabel(r'Reg. parameter $\lambda_{%.s}$'%(Methods[0]), c='r')

        ay_k_t.spines['top'].set_color('red')
        ay_k_t.spines['right'].set_color('red')
        ay_k_t.xaxis.label.set_color('red')
        ay_k_t.tick_params(axis='x', colors='red', which='both')
        ay_k.yaxis.label.set_color('red')
        ay_k.tick_params(axis='y', colors='red', which='both')

        # Draw maximal curvature point of L-curve
        ay_k_t.plot(alpha[i], k[i]/k[i], 'r*')
        ay.plot(np.log10(res[i]), np.log10(sol[i]), 'r*') #highlight optimal lambda
        ay.annotate(r"$\lambda_\mathrm{opt} =$ %.1e"%alpha[i], c = 'k',
            xy = (np.log10(res[i])*0.98, np.log10(sol[i])*0.98), ha = 'left')

        plt.tight_layout()

########## read_file.py module ############

def read_file(Path, dt=150, proc = True):
    """Returns data from file

    Parameters:
    --------------
    Path : str
        Path to file
    dt : float
        Step between time points in ms
    proc: boolean
        Process transient if True

    Returns:
    --------------
    time : array
        Time domain points in s
    C : array of [F1(time), F2(time), ...]
        Contains transients experimental data 
        for range of temperatures
    T : array
        Contains experimental temperature points
    """
    import numpy as np

    def process(C, proc):
        """Returns processed transient C_p if proc is True"""

        def get_Baseline(C):
            """Returns baseline of transient C"""
            l = len(C)
            c1, c2, c3 = C[0], C[int(l/2)-1], C[l-1]
            return (c1*c3 - c2**2)/(c1+c3 - 2*c2)

        if proc:
            C_p = C
            for i, _C in enumerate(C):
                F = _C
                F = F/np.average(F[-1])
                if F[0] > F[-1]:
                    F = F - min(F)
                else:
                    F = F - max(F)
                F = np.abs(F)
                F = F + np.average(F)*2
                F = F - get_Baseline(F)*0
                C_p[i] = F
            return np.asarray(C_p)

        else:
            C = C/C[-1]
            return C + np.average(C)*2

    Path = str(Path)

    txt  = np.genfromtxt(Path, delimiter='\t')
    if len(txt.shape) == 2:
        T    = txt[:,0]
        cut  = len(T)

        C    = []
        time = []
        for i in range(0,cut):
            C.append(txt[i][3:-2])

        for i in range(0, len(C[0])):
            time.append(dt/1000*(i+1))
    else:
        T    = txt[0]
        C    = txt[1:]
        time = np.arange(dt, dt*len(C), dt)*1e-3


    C    = np.asarray(C)
    C    = process(C, proc)
    time = np.asarray(time)
    T    = np.asarray(T)
    #print(time)

    return time, C, T

########## filebrowser.py module ############

import os
import ipywidgets as widgets

l = widgets.Layout(width='50%')

class FileBrowser(object):
    def __init__(self):
        self.path = os.getcwd()
        self._update_files()

    def _update_files(self):
        self.files = list()
        self.folders = list()
        if(os.path.isdir(self.path)):
            for f in os.listdir(self.path):
                ff = os.path.join(self.path, f)
                if os.path.isdir(ff):
                    self.folders.append(f)
                else:
                    self.files.append(f)

    def widget(self):
        box = widgets.VBox()
        self._update(box)
        return box

    def _update(self, box):

        def on_click(b):
            if b.description == '..':
                self.path = os.path.split(self.path)[0]
            else:
                self.path = os.path.join(self.path, b.description)
            self._update_files()
            self._update(box)

        buttons = []
        if self.files or self.folders:
            button = widgets.Button(layout = l, description='..', button_style='primary')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.folders:
            if f[0] != '.' and f[:2] != '__' and f != 'processed' and f != 'modules':
                button = widgets.Button(layout = l, description=f, button_style='info')
                button.on_click(on_click)
                buttons.append(button)
        for f in self.files:
            if f[0] != '.' and f[-5:] == '.DLTS' or f[-5:] == '.PERS':
                button = widgets.Button(layout = l, description=f, button_style='success')
                button.on_click(on_click)
                buttons.append(button)

        box.children = tuple([widgets.HTML("<h2>%s</h2>" % (self.path,))] + buttons)

########## contin.py module ############

from __future__ import division
import numpy as np

def Contin(t, F, bound, Nz, alpha):
    """
    Module to implement routines of numerical inverse 
    Laplace tranform using Contin  algorithm [1]
    
    F(t) = ∫f(z)*exp(-z*t)

    [1] Provencher, S. (1982)

    Parameters:
    --------------
    t : array 
        Time domain data from experiment
    F : array,
        Transient data from experiment F(t)
    bound : list
        [lowerbound, upperbound] of z domain points
    Nz : int
        Number of points z to compute, must be smaller than len(F)
    alpha : float
        Regularization parameter

    Returns:
    --------------
    z : array 
        Emission rates domain points (evenly spaced on log scale)
    f : array
        Inverse Laplace transform f(z)
    F_restored : array 
        Reconstructed transient from C@f(z)
    """

    # set up grid points (# = Nz)
    h = np.log(bound[1]/bound[0])/(Nz - 1)      # equally spaced on logscale
    z = bound[0]*np.exp(np.arange(Nz)*h)        # z (Nz by 1)

    # construct C matrix from [1]
    z_mesh, t_mesh = np.meshgrid(z, t)
    C = np.exp(-t_mesh*z_mesh)       
    C[:, 0] /= 2.
    C[:, -1] /= 2.
    C *= h

    # construct regularization matrix R to impose gaussian-like peaks in f(z)
    # R - tridiagonal matrix (1,-2,1)
    Nreg = Nz + 2
    R = np.zeros([Nreg, Nz])
    R[0, 0] = 1.
    R[-1, -1] = 1.
    R[1:-1, :] = -2*np.diag(np.ones(Nz)) + np.diag(np.ones(Nz-1), 1) \
        + np.diag(np.ones(Nz-1), -1)

    #R = U*H*Z^T 
    U, H, Z = np.linalg.svd(R, full_matrices=False)     # H diagonal
    Z = Z.T
    H = np.diag(H)
    
    #C*Z*inv(H) = Q*S*W^T
    Hinv = np.diag(1.0/np.diag(H))
    Q, S, W = np.linalg.svd(C.dot(Z).dot(Hinv), full_matrices=False)  # S diag
    W = W.T
    S = np.diag(S)

    # construct GammaTilde & Stilde
    # ||GammaTilde - Stilde*f5||^2 = ||Xi||^2
    Gamma = np.dot(Q.T, F)
    Sdiag = np.diag(S)
    Salpha = np.sqrt(Sdiag**2 + alpha**2)
    GammaTilde = Gamma*Sdiag/Salpha
    Stilde = np.diag(Salpha)

    # construct LDP matrices G = Z*inv(H)*W*inv(Stilde), B = -G*GammaTilde
    # LDP: ||Xi||^2 = min, with constraint G*Xi >= B
    Stilde_inv = np.diag(1.0/np.diag(Stilde))
    G = Z @ Hinv @ W @ Stilde_inv
    B = -G @ GammaTilde

    # call LDP solver
    Xi = ldp(G, B)

    # final solution
    zf = np.dot(G, Xi + GammaTilde)
    f = zf/z

    F_restored = C@zf

    return z, f, F_restored


def ldp(G, h):
    """
    Helper for Contin() for solving NNLS [1]
    
    [1] - Lawson and Hanson’s (1974)
    Parameters:
    -------------
    G : matrix
        Z*inv(H)*W*inv(Stilde)
    h : array
        -G*GammaTilde

    Returns:
    -------------
    x : array
        Solution of argmin_x || Ax - b ||_2 
    """

    from scipy.optimize import nnls

    m, n = G.shape
    A = np.concatenate((G.T, h.reshape(1, m)))
    b = np.zeros(n+1)
    b[n] = 1.

    # Solving for argmin_x || Ax - b ||_2 
    x, resnorm = nnls(A, b)

    r = A@x - b

    if np.linalg.norm(r) == 0:
        print('\n No solution found, try different input!')
    else:
        x = -r[0:-1]/r[-1]
    return x

########## L1L2.py module ############

def L1L2(t, F, bound, Nz, alpha1, alpha2, iterations = 10000):
    """
    Returns solution using mixed L1 and/or L2 regularization
    with simple gradient descent

    F(t) = ∫f(s)*exp(-s*t)ds

    or

    min = ||C*f - F||2 + alpha1*||I*f||1 + alpha2*||I*f||2

    Parameters:
    ------------
    t : array 
        Time domain data from experiment
    F : array,
        Transient data from experiment F(t)
    bound : list
        [lowerbound, upperbound] of s domain points
    Nz : int
        Number of points z to compute, must be smaller than len(F)
    alpha1, alpha 2 : floats
        Regularization parameters for L1 and L2 regularizers
    iterations : int 
        Maximum number of iterations. Optional


    Returns:
    ------------
    s : array
        Emission rates domain points (evenly spaced on log scale)
    f : array
        Inverse Laplace transform f(s)
    F_restored : array 
        Reconstructed transient from C@f + intercept
    """

    import numpy as np
    from scipy.sparse import diags
    from sklearn.linear_model import ElasticNet
    
    # set up grid points (# = Nz)
    h = np.log(bound[1]/bound[0])/(Nz - 1)      # equally spaced on logscale
    s = bound[0]*np.exp(np.arange(Nz)*h)        # z (Nz by 1)

    # construct C matrix from [1]
    s_mesh, t_mesh = np.meshgrid(s, t)
    C = np.exp(-t_mesh*s_mesh)       
    C[:, 0] /= 2.
    C[:, -1] /= 2.
    C *= h
    
    alpha = alpha1 + alpha2
    l1_ratio = alpha1/alpha

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol = 1e-12,
                       fit_intercept = True, max_iter = iterations)
    model.fit(C, F)
    
    f = model.coef_

    F_restored = C@f + model.intercept_
    return s, f, F_restored

########## L2.py module ############

def L2(t, F, bound, Nz, alpha):
    """
    Returns solution for problem imposing L2 regularization

    F(t) = ∫f(s)*exp(-s*t)ds

    or

    min = ||C*f - F||2 + alpha*||I*f||2


    Parameters:
    ------------
    t : array 
        Time domain data from experiment
    F : array,
        Transient data from experiment F(t)
    bound : list
        [lowerbound, upperbound] of s domain points
    Nz : int
        Number of points z to compute, must be smaller than len(F)
    alpha : float
        Regularization parameters for L2 regularizers
    iterations : int 
        Maximum number of iterations. Optional


    Returns:
    ------------
    s : array
        Emission rates domain points (evenly spaced on log scale)
    f : array
        Inverse Laplace transform f(s)
    F_restored : array 
        Reconstructed transient from C@f + intercept
    """

    import numpy as np
    from scipy.sparse import diags

    # set up grid points (# = Nz)
    h = np.log(bound[1]/bound[0])/(Nz - 1)      # equally spaced on logscale
    s = bound[0]*np.exp(np.arange(Nz)*h)        # z (Nz by 1)

    # construct C matrix from [1]
    s_mesh, t_mesh = np.meshgrid(s, t)
    C = np.exp(-t_mesh*s_mesh)       
    C[:, 0] /= 2.
    C[:, -1] /= 2.
    C *= h

    l2 = alpha

    data = [-2*np.ones(Nz), 1*np.ones(Nz), 1*np.ones(Nz)]
    positions = [-1, -2, 0]
    I = diags(data, positions, (Nz+2, Nz)).toarray()
    #I      = np.identity(Nz)

    f   = np.linalg.solve(l2*np.dot(I.T,I) + np.dot(C.T,C), np.dot(C.T,F))

    F_restored = C@f

    return s, f, F_restored#, res_norm, sol_norm

########## reSpect.py module ############

import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.optimize import nnls, minimize, least_squares

def Initialize_f(F, s, kernMat, *argv):
    """
    Computes initial guess for f and then call get_f()
    to return regularized solution:

    argmin_f = ||F - kernel(f)||2 +  lam * ||L f||2

    Parameters:
    ------------
    F : array
        Transient experimental data
    s : array
        Tau-domain points
    kernMat : matrix (len(s), len(F))
        Matrix of inverse Laplace transform
    F_b : float
        Baseline value, optional

    Returns:
    -----------
    flam : array
        Regularized inverse of Laplace transform
    F_b : float
        Baseline value
    """

    f    = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)  # initial guess for f
    lam  = 1e0

    if len(argv) > 0:
        F_b       = argv[0]
        flam, F_b = get_f(lam, F, f, kernMat, F_b)
        return flam, F_b
    else:
        flam     = get_f(lam, F, f, kernMat)
        return flam


def get_f(lam, F, f, kernMat, *argv):
    """
    Solves following equation for f. Uses jacobianLM() with 
    scipy.optimize.least_squares() solver:

    argmin_f = ||F - kernel(f)||2 +  lam * ||L f||2

    Parameters:
    -------------
    lam : float
        Regularization parameter
    F : array
        Transient experimental data
    f : array
        Guessed f(s) for emission rates
    kernMat : matrix (len(f), len(F))
        Matrix of inverse Laplace transform
    F_b : float
        Baseline value,optional

    Returns:
    -------------
    flam : array
        Regularized solution
    F_b : float
        Baseline for solution
    """

    # send fplus = [f, F_b], on return unpack f and F_b
    if len(argv) > 0:
        fplus= np.append(f, argv[0])
        res_lsq = least_squares(residualLM, fplus, jac=jacobianLM, args=(lam, F, kernMat))
        return res_lsq.x[:-1], res_lsq.x[-1]

    # send normal f, and collect optimized f back
    else:
        res_lsq = least_squares(residualLM, f, jac=jacobianLM, args=(lam, F, kernMat))
        return res_lsq.x



def residualLM(f, lam, F, kernMat):
    """
    Computes residuals for below equation 
    and used with scipy.optimize.least_squares():

    argmin_f = ||F - kernel(f)||2 +  lam * ||L f||2

    Parameters:
    -------------
    f : array
        Solution for above equation f(s)
    lam : float
        Regularization parameter
    F : array
        Experimental transient data
    kernMat : matrix (len(f), len(F))
        Matrix of inverse Laplace transform
    F_b : float
        Plateau value for F experimental data

    Returns:
    -----------
    r : array 
        Residuals (||F - kernel(f)||2)''
    """

    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];
    nl  = ns - 2;

    r   = np.zeros(n + nl);

    # if plateau then unfurl F_b
    if len(f) > ns:
        F_b    = f[-1]
        f      = f[:-1]
        r[0:n] = (1. - kernel_prestore(f, kernMat, F_b)/F)  # same as |F - kernel(f)|
    else:
        r[0:n] = (1. - kernel_prestore(f, kernMat)/F) # same as |F - kernel(f)| w/o F_b

    r[n:n+nl] = np.sqrt(lam) * np.diff(f, n=2)  # second derivative

    return r


def jacobianLM(f, lam, F, kernMat):
    """
    Computes jacobian for scipy.optimize.least_squares()

    argmin_f = ||F - kernel(f)||2 +  lam * ||L f||2

    Parameters:
    -------------
    f : array
        Solution for above equation
    lam : float
        Regularization parameter
    F : array
        Experimental transient data
    kernMat : matrix (len(f), len(F))
        Matrix of inverse Laplace transform

    Returns:
    ------------
    Jr : matrix (len(f)*2 - 2, len(F) + 1)
        Contains Jr(i, j) = dr_i/df_j
    """

    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];
    nl  = ns - 2;

    # L is a ns*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
    L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))
    L  = L[1:nl+1,:]

    # Furnish the Jacobian Jr (n+ns)*ns matrix
    Kmatrix         = np.dot((1./F).reshape(n,1), np.ones((1,ns)));

    if len(f) > ns:

        F_b    = f[-1]
        f      = f[:-1]

        Jr  = np.zeros((n + nl, ns+1))

        Jr[0:n, 0:ns]   = -kernelD(f, kernMat) * Kmatrix;
        Jr[0:n, ns]     = -1./F                          # column for dr_i/dF_b

        Jr[n:n+nl,0:ns] = np.sqrt(lam) * L;
        Jr[n:n+nl, ns]  = np.zeros(nl)                      # column for dr_i/dF_b = 0

    else:

        Jr  = np.zeros((n + nl, ns))

        Jr[0:n, 0:ns]   = -kernelD(f, kernMat) * Kmatrix;
        Jr[n:n+nl,0:ns] = np.sqrt(lam) * L;

    return Jr


def kernelD(f, kernMat):
    """
    Helper for jacobianLM() approximates dK_i/df_j = K * e(f_j)

    argmin_f = ||F - kernel(f)||2 +  lam * ||L f||2

    Parameters:
    -------------
    f : array
        Solution for above equation
    kernMat : matrix (len(f), len(F))
        Matrix of inverse Laplace transform

    Returns:
    -------------
    DK : matrix (len(f), len(F)) 
        Jacobian
    """

    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];

    # A n*ns matrix with all the rows = f'
    fsuper  = np.dot(np.ones((n,1)), np.exp(f).reshape(1, ns))
    DK      = kernMat  * fsuper

    return DK


def getKernMat(s, t):
    """
    Mesh grid for s and t domain to construct 
    kernel matrix

    Parameters:
    -------------
    s: array
        Tau domain points
    t: array
        Time domain points from experiment

    Returns:
    -------------
    np.exp(-T/S) * hsv : matrix (len(s), len(t))
        Matrix of inverse Laplace transform, where hsv 
        trapezoidal coefficients
    """
    ns          = len(s)
    hsv         = np.zeros(ns);
    hsv[0]      = 0.5 * np.log(s[1]/s[0])
    hsv[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
    hsv[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))
    S, T        = np.meshgrid(s, t);

    return np.exp(-T/S) * hsv;


def kernel_prestore(f, kernMat, *argv):
    """
    Function for prestoring kernel

    argmin_f = ||F - kernel(f)||2 +  lam * ||L f||2

    Parameters:
    -------------
    f : array
        Solution of above equation
    kernMat : matrix (len(f), len(F))
        Matrix of inverse Laplace transform

    Returns:
    ------------
    np.dot(kernMat, np.exp(f)) + F_b : 
        Stores kernMat*(f)+ F_b 
    """

    if len(argv) > 0:
        F_b = argv[0]
    else:
        F_b = 0.

    return np.dot(kernMat, np.exp(f)) + F_b


def reSpect(t, F, bound, Nz, alpha):
    """
    Main routine to implement reSpect algorithm from [1].

    [1] Shanbhag, S. (2019)

    Parameters:
    --------------
    t : array
        Time domain points from experiment
    F : array
        Experimental transient F(t)
    bound : list 
        [lowerbound, upperbound] of bounds for tau-domain points
    Nz : int
        Length of tau-domain array
    alpha : float
        Regularization parameter

    Returns:
    --------------
    1/s[::-1] : array
        Tau-domain points
    np.exp(f)[::-1] : array
        Inverse Laplace transform of F(t)
    kernMat@np.exp(f)[::] : array
        Restored transient F_restored(t)
    """

    n    = len(t)
    ns   = Nz    # discretization of 'tau'

    tmin = t[0];
    tmax = t[n-1];

    smin, smax = 1/bound[1], 1/bound[0]  # s is tau domain points!

    hs   = (smax/smin)**(1./(ns-1))
    s    = smin * hs**np.arange(ns)  # s here is tau domain points

    kernMat = getKernMat(s, t)

    fgs, F_b  = Initialize_f(F, s, kernMat, np.min(F))

    alpha = alpha

    f, F_b  = get_f(alpha, F, fgs, kernMat, F_b);

    K   = kernel_prestore(f, kernMat, F_b);

    return 1/s[::-1], np.exp(f)[::-1], kernMat@np.exp(f)[::]
