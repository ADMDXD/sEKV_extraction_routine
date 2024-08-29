# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:47:55 2023

@author: andordam
"""

import numpy as np
import matplotlib.pyplot as plt 
from extract_data import extract_data
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy import optimize
from scipy.stats import linregress

mpl.rcdefaults()
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


my_colors = {
  "dark_blue": "#0096a0",
  "light_blue": "#5bbeb6",
  "blue_green": "#80C075",
  "green": "#a4c134",
  "green_orange": "#CEB71B",
  "light_orange": "#f8ad02",
  "orange": "#EE650F",
  "dark_red" : "#CB1815",
  "magenta": "#AD254D",
  "purple": "#8E3285"
}

color_dict=["dark_blue","light_blue","green","green_orange","light_orange","orange","dark_red","magenta","purple"]



def EKV_extractor(IDVG, VG, VSB, IDVD_meas, VD_meas, plots, echo):
    
    # index=np.asarray(np.where(VG<0))
    
    IDVG_meas=IDVG[10:]
    VG_meas=VG[10:]
    
    
    global n_slope, lambda_c, ispec, VTH, VG_EKV, gmUTid_EKV, gmUTid, IC, ID_EKV, Gds, Gm, gm, gds
    #%%Extraction of slope factor (n) from measurements
    
    #On this block of code we extract n_slope from the IV measurements
    #Following ekv the when the transistor is biased in weak inversion
    #n=ID/gm Ut and n is the plateau of this function
    
    gm_over_id = np.abs(np.gradient(np.log(IDVG_meas),VG_meas)) #Calculation of transconductance as derivative of ID with respect to VG
    
    n =1/(gm_over_id*0.026) #Calculation of slope values 
    
    
    n_slope= min(n) #Calculation of the slope factor as the minimum value of the n values (plateau)
    
    #This value will be a constant during the whole fitting
    
    #%%Extraction of a first guess of the specific current (ispec)
    
    gmUTid = gm_over_id*n_slope*0.026 #With the previously extracted n_slope we can extract gm Ut n/id values
    
    #To extract a first guess of ispec we will assume no velocity of saturation effects (lambda = 0) and IC = 1, therefore IC = ID/ispec = 1 => ID=ispec
    
    IC=1
    lambda_c0=0
    
    #Making these assumptions makes gm Ut n/id to be equal to the golden ratio!
    
    numerator = ((IC * lambda_c0 + 1.0) ** 2 + 4.0 * IC) ** (1 / 2) - 1.0
    denominator = IC * (lambda_c0 * (lambda_c0 * IC + 1.0) + 2.0)
    golden_ratio = numerator / denominator
    
    #We extract ispec as the drain current value where gm Ut n/id = golden ratio (where ID=ispec and IC=0)
    
    f = interp1d(gmUTid, IDVG_meas, kind="cubic") #Interpolation to extract the function ID(gmUTid)
    
    ispec0 = f(golden_ratio) 
    


    
    #%%Optimization of ispec and lambda_c
    
    #For the optimization we calculate the square differences between the measured gmoverid and the gmoverid from the EKV
    #We let a genetic algorithim from optimize toolbox to perform the optimization
    
    scale=1e6 #scaling values to set the bounds on the same range (around 0...1)
    
    #This is the function to obtain the gmoverid as described by the EKV model
    def gmoverid_f(ID, lambda_c0, ispec0 ):
        IC = ID/ispec0
        numerator = ((IC * lambda_c0 + 1.0) ** 2 + 4.0 * IC) ** (1 / 2) - 1.0
        denominator = IC * (lambda_c0 * (lambda_c0 * IC + 1.0) + 2.0)
        return numerator / denominator
    
    #This is the cost function to obtain the square differences between both functions in logarithmic scale for proper weigthing
    def cost(x):
        lambda_c = x[0]
        ispec = x[1]/scale
        
        gm_over_id = np.abs(np.gradient(np.log(IDVG_meas),VG_meas))
        gmUTid_meas = gm_over_id*n_slope*0.026
        gmUtid_EKV = gmoverid_f(IDVG_meas, lambda_c, ispec)
        weights = np.diff(np.log10(IDVG_meas))
        weights = (np.r_[weights[0],weights])**2
        sqr = np.sum(((np.log10(gmUTid_meas)-np.log10(gmUtid_EKV))**2)*weights**0.5)
        return sqr
    
    bounds = [(0, 1), (1e-9*scale, 1e-5*scale)] #Bounds for lambda_c and ispec
    
    resultsA = optimize.differential_evolution(cost, bounds)  #Optimizer
    
    lambda_c = resultsA.x[0]
    ispec = resultsA.x[1]/scale
    
    #%% Optimization of VTH
    
    UT=0.026  #Thermal voltage
    
    
    VTH=0 #Initial guess for threshold voltage
    
    #For the optimization we calculate the square differences between the measured VG and the VG from the EKV
    #We let a genetic algorithim from optimize toolbox to perform the optimization
    
    #This is the function to obtain the VG as described by the EKV model
    def VGB_f(ID, VTH):
        IC = ID/ispec
        fIC = -1 + np.sqrt((1 + lambda_c*IC)**2 + 4*IC)
        VGB  = (fIC + np.log(fIC/2))*(n_slope*UT) + n_slope*VSB + VTH
        return VGB
    
    #This is the cost function to obtain the square differences between both functions
    def cost(x):
        VTH = x
        VGB_math = VGB_f(IDVG_meas, VTH)
        sqr = np.sum((VGB_math-VG_meas)**2)
        return sqr
    
    bounds = [(0.2, 0.9)] #Bound for VTH
    
    results = optimize.differential_evolution(cost,bounds) #Optimizer
    
    VTH = results.x[0]
    
    VG_EKV = VGB_f(IDVG_meas, VTH)
    
    ID_EKV = IDVG_meas
    
    IC = ID_EKV/ispec
    
    numerator = np.sqrt((lambda_c * IC + 1)**2 + 4 * IC) - 1
    denominator = IC * (lambda_c * (lambda_c * IC + 1) + 2)
    gmUTid_EKV = numerator / denominator

    #%% Extraction of GDS
        
    if (IDVD_meas == 'None'):
        GDS=0
        Gm=0
        gm=0
        gds=0
        intercept=0
        IC_val=0
        ID_val=0
    else:        
        IDs=IDVD_meas
        VDs=VD_meas
        Gspec=ispec/0.026
        
        f1=interp1d(VG_meas, IDVG_meas, fill_value="extrapolate")

        VG=np.linspace(0,1.2,np.size(IDs,1)-0)
        GDS=np.empty(np.size(VG))
        Gm = np.empty(np.size(VG))
        gds=np.empty(np.size(VG))
        intercept=np.empty(np.size(VG))
        IC_val=np.empty(np.size(VG))
        ID_val=np.empty(np.size(VG))
        
        for vg in range(np.size(VG)):
            # ID_measVD=data[:,vg]
            # VD_meas=data[:,0]
            ID_measVD=IDs[:,vg]
            VD_meas=VDs
            gradient=np.gradient(ID_measVD,VD_meas)
            index = np.argmin(np.abs(gradient))
            GDS[vg]=gradient[index]
            ID_index=ID_measVD[index]
            VD_index=VD_meas[index]
            intercept[vg] = GDS[vg]*VD_index - ID_index
            gds[vg]=GDS[vg]/Gspec
            ID_val[vg]=f1(np.round(VG[vg],2))
            IC_val[vg]=ID_val[vg]/ispec
          
      #%%Plots
     
    if plots == 1:
    
        labels = [r'Measured', r'EKV']
        
       
        fig, ax = plt.subplots(1,1, sharex=True)
        
        ax.plot(IC,gmUTid_EKV, '-', color=my_colors[color_dict[0]], linewidth=2, 
                      markerfacecolor="None", markeredgewidth=1,
                      markersize=10, markevery=1, label= labels[1])
        ax.plot( IC,gmUTid , 'o', color=my_colors[color_dict[1]], linewidth=2,
                      markerfacecolor="None", markeredgewidth=1,
                      markersize=10, markevery=2, label= labels[0])
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        
        ax.set_ylabel(r'$(g_\text{m} \text{n} U_\text{T}) / I_\text{D} [-]$', fontsize = 20)
        ax.set_xlabel(r'$I_\text{C} [-]$', fontsize = 20)
        
        
        ax.legend(bbox_to_anchor = (1, 0.5), loc = 'center left', fontsize=12)
        ax.legend(loc = 'best')
        
        ax.set_xlim([1e-3, 1e3])
        ax.set_ylim([4e-2,2])
        
        ax.grid(True, which="both", ls="--", color='0.9')
        ax.tick_params(labelsize=15)
        
        plt.legend(loc = 'best')
        
        fig.set_size_inches([5,4])
        
        
        plt.tight_layout()  
        
        
        
        
        fig, ax = plt.subplots(1,1, sharex=True)
        ax1 = ax.twinx()
        
        ax.plot(VG_meas, IDVG_meas, 'o', color=my_colors[color_dict[1]], linewidth=2,
                      markerfacecolor="None", markeredgewidth=1,
                      markersize=10, markevery=2, label= labels[0])
        ax.plot(VG_EKV, ID_EKV, '-', color=my_colors[color_dict[0]], linewidth=2, 
                      markerfacecolor="None", markeredgewidth=1,
                      markersize=10, markevery=1, label= labels[1])
        ax.set_yscale('log')
        ax1.plot(VG_meas, IDVG_meas, 'o', color=my_colors[color_dict[1]], linewidth=2,
                      markerfacecolor="None", markeredgewidth=1,
                      markersize=10, markevery=2, label= labels[0])
        ax1.plot(VG_EKV, ID_EKV, '-', color=my_colors[color_dict[0]], linewidth=2, 
                      markerfacecolor="None", markeredgewidth=1,
                      markersize=10, markevery=1, label= labels[1])
        
        
        
        ax.set_ylabel(r'$I_\text{D} [A]$', fontsize = 20)
        ax.set_xlabel(r'$V_\text{G} [V]$', fontsize = 20)
        
        
        ax.legend(bbox_to_anchor = (1, 0.5), loc = 'center left', fontsize=12)
        ax.legend(loc = 'best')
        
        # ax.set_xlim([1e-3, 1e3])
        # ax.set_ylim([4e-2,2])
        
        ax.grid(True, which="both", ls="--", color='0.9')
        ax.tick_params(labelsize=15)
        
        
        fig.set_size_inches([5,4])
        
        
        plt.tight_layout() 
        
    if echo==1:
        print('***INITIAL GUESES***')
        print('n = '+str(round(n_slope,2)))
        print('lambda_c = '+str(lambda_c0))
        print('ispec0 = '+str(round(ispec0/1e-6,2))+' uA')
        print('***FINAL EXTRACTION***')
        print('n = '+str(round(n_slope,2)))
        print('lambda_c = '+str(round(lambda_c,2)))
        print('ispec = '+str(round(ispec/1e-6,2))+' uA')
        print('VTH = '+str(round(VTH/1e-3,2))+' mV')
    
    return n_slope, lambda_c, ispec, VTH, VG_EKV, gmUTid_EKV, gmUTid, IC, ID_EKV, GDS, Gm, gds, gm
