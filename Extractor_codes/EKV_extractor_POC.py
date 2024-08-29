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
import pandas as pd
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



#%%HOW TO EXTRACT PARAMETER N EXAMPLE

#%%FIRST WE LOAD ID VG CURVES FOLDERS

# section = 'ID_VG'
# quantity = 'ID'
# mainpath = r'C:\Users\andordam\cernbox\MisCosas\WorkInProgress\TTS_65nm_measurements'

#%% Extraction of data before irradiation


transistor = 'nMOS_1.00x0.10'
label = r'nMOS\,1.00\,x\,0.10'


tran=0

VSB=0 #Source to bulk voltage

plots=1
echo=1


# path = mainpath + r'\Results\WF13\TTS1\Chip05-100C-300Mrad\Left_Array\Irradiation\Irradiation_TJ65WF13TTS1LC00520210831_step' + str(0) + '.txt'
# data, lines = extract_data(path, transistor, section, quantity,1)

path = r'C:\Users\andordam\cernbox\MisCosas\WorkInProgress\EKVmodelParameters\data\IDVGshort.csv'
df = pd.read_csv(path, comment='%')
src = df.values
ID_meas = src[10:,1]
VG_meas = src[10:,0]


# ID_meas=data[10:,2]
# VG_meas=data[10:,0]



# global n_slope, lambda_c, ispec, VTH, VGB, gmUTid_EKV, gmUTid, IC

mainpath = r'C:\Users\andordam\cernbox\MisCosas\WorkInProgress'
#%%Extraction of slope factor (n) from measurements

#On this block of code we extract n_slope from the IV measurements
#Following ekv the when the transistor is biased in weak inversion
#n=ID/gm Ut and n is the plateau of this function

gm_over_id = np.abs(np.gradient(np.log(ID_meas),VG_meas)) #Calculation of transconductance as derivative of ID with respect to VG

n =1/(gm_over_id*0.026) #Calculation of slope values 


n_slope= min(n) #Calculation of the slope factor as the minimum value of the n values (plateau)

#This value will be a constant during the whole fitting

#%%Plot

fig, ax = plt.subplots(1,2, sharex=True)

ax[0].plot(ID_meas,n, 'o', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'Measured')
ax[0].axhline(y=n_slope, linestyle='--', color='k', marker='None', linewidth = 1)
ax[0].set_yscale('log')
ax[0].set_xscale('log')


ax[0].set_ylabel(r'$n = I_\text{D} / (g_\text{m} U_\text{T})$', fontsize = 20)
ax[0].set_xlabel(r'$I_\text{D} [\text{A}]$', fontsize = 20)
ax[0].text(ID_meas[20], n[16], r'n = '+str(np.round(n_slope,2)),
        verticalalignment='top', horizontalalignment='left', fontsize=15)


# ax.legend(bbox_to_anchor = (1, 0.5), loc = 'center left', fontsize=12)
ax[0].legend(loc = 'best', fontsize=12)

# ax.set_xlim([1e-3, 1e3])
# ax.set_xlim([min(ID_meas),max(ID_meas)])

ax[0].grid(True, which="both", ls="--", color='0.9')
# ax.set_yticklabels([])
# ax.set_xticklabels([])

# plt.legend(loc = 'best')

# fig.set_size_inches([6,4])


# plt.tight_layout() 

# fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step01.jpg')
# fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step01.pdf', format="pdf")

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

f = interp1d(gmUTid[10:-1], ID_meas[10:-1], kind="cubic") #Interpolation to extract the function ID(gmUTid)

ispec0 = f(golden_ratio) 

#%%Plot

# fig, ax = plt.subplots(1,1, sharex=True)

ax[1].plot(ID_meas,gmUTid, 'o', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'Measured')
ax[1].axhline(y=golden_ratio, linestyle='--', color='k', marker='None', linewidth = 1)
ax[1].axvline(x=ispec0, linestyle='--', color='k', marker='None', linewidth = 1)
ax[1].set_yscale('log')
ax[1].set_xscale('log')


ax[1].set_ylabel(r'$(g_\text{m} \text{n} U_\text{T}) / I_\text{D}$', fontsize = 20)
ax[1].set_xlabel(r'$I_\text{D} [\text{A}]$', fontsize = 20)
ax[1].text(ID_meas[5], gmUTid[40], r'$I_\text{spec}\,=\,$'+str(np.round(ispec0*1e6,2))+r'$\mu A$',
        verticalalignment='top', horizontalalignment='left', fontsize=15)
ax[1].text(ID_meas[5], gmUTid[20], r'$\frac{g_\text{m} n U_\text{T}}{I_\text{D}}\,=\,$'+str(np.round(golden_ratio,3)),
        verticalalignment='top', horizontalalignment='left', fontsize=15)


# ax.legend(bbox_to_anchor = (1, 0.5), loc = 'center left', fontsize=12)
ax[1].legend(loc = 'best', fontsize=12)

# ax.set_xlim([min(ID_meas),max(ID_meas)])

ax[1].grid(True, which="both", ls="--", color='0.9')
# ax.set_yticklabels([])
# ax.set_xticklabels([])

# plt.legend(loc = 'best')

fig.set_size_inches([8,4])


plt.tight_layout() 

fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step0102.jpg')
fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step0102.pdf', format="pdf")

# fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step02.jpg')
# fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step02.pdf', format="pdf")

#%%Plot

IC = ID_meas/ispec0
numerator = ((IC * lambda_c0 + 1.0) ** 2 + 4.0 * IC) ** (1 / 2) - 1.0
denominator = IC * (lambda_c0 * (lambda_c0 * IC + 1.0) + 2.0)
gmUtid_EKV = numerator/denominator
gm_over_id = np.abs(np.gradient(np.log(ID_meas),VG_meas))
gmUTid_meas = gm_over_id*n_slope*0.026

fig, ax = plt.subplots(1,2, sharex=False, sharey=True)

ax[0].plot(IC,gmUtid_EKV, '-', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'EKV')
ax[0].plot(IC,gmUTid_meas, 'o', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'Measured')
ax[0].text(IC[5], gmUtid_EKV[20], 
        r'$\newline I_\text{spec}\,=\,$'+str(np.round(ispec0*1e6,2))+r'$\,\mu A \newline$'+r'$n\,=\,$'+str(np.round(n_slope,2))+r'\newline$\lambda_\text{c}\,=\,$'+str(np.round(lambda_c0,2)),
        verticalalignment='top', horizontalalignment='left', fontsize=15)
ax[0].set_yscale('log')
ax[0].set_xscale('log')


ax[0].set_ylabel(r'$(g_\text{m} n U_\text{T}) / I_\text{D}$', fontsize = 20)
ax[0].set_xlabel(r'$\text{IC}$', fontsize = 20)


# ax.legend(bbox_to_anchor = (1, 0.5), loc = 'center left', fontsize=12)
ax[0].legend(loc = 'best', fontsize=12)

# ax[0].set_xlim([min(IC),max(IC)])

ax[0].grid(True, which="both", ls="--", color='0.9')
# ax[0].set_yticklabels([])
# ax[0].set_xticklabels([])

# plt.legend(loc = 'best')

fig.set_size_inches([8,4])


plt.tight_layout() 



#%%Optimization of ispec and lambda_c

#For the optimization we calculate the square differences between the measured 
# gmoverid and the gmoverid from the EKV
#We let a genetic algorithim from optimize toolbox to perform the optimization

scale=1e6 #scaling values to set the bounds on the same range (around 0...1)

#This is the function to obtain the gmoverid as described by the EKV model
def gmoverid_f(ID, lambda_c0, ispec0 ):
    IC = ID/ispec0
    numerator = ((IC * lambda_c0 + 1.0) ** 2 + 4.0 * IC) ** (1 / 2) - 1.0
    denominator = IC * (lambda_c0 * (lambda_c0 * IC + 1.0) + 2.0)
    return numerator / denominator

#This is the cost function to obtain the square differences between both 
#functions in logarithmic scale for proper weigthing
def cost(x):
    lambda_c = x[0]
    ispec = x[1]/scale
    
    gm_over_id = np.abs(np.gradient(np.log(ID_meas),VG_meas))
    gmUTid_meas = gm_over_id*n_slope*0.026
    gmUtid_EKV = gmoverid_f(ID_meas, lambda_c, ispec)
    weights = np.diff(np.log10(ID_meas))
    weights = (np.r_[weights[0],weights])**2
    sqr = np.sum(((np.log10(gmUTid_meas)-np.log10(gmUtid_EKV))**2)*weights**0.5)
    return sqr

bounds = [(0, 1), (1e-9*scale, 1e-5*scale)] #Bounds for lambda_c and ispec

resultsA = optimize.differential_evolution(cost, bounds)  #Optimizer

lambda_c = resultsA.x[0]
ispec = resultsA.x[1]/scale

#%%Plot

IC = ID_meas/ispec
numerator = np.sqrt((lambda_c * IC + 1)**2 + 4 * IC) - 1
denominator = IC * (lambda_c * (lambda_c * IC + 1) + 2)
gmUtid_EKV = numerator/denominator
gm_over_id = np.abs(np.gradient(np.log(ID_meas),VG_meas))
gmUTid_meas = gm_over_id*n_slope*0.026


ax[1].plot(IC,gmUtid_EKV, '-', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'EKV')
ax[1].plot(IC,gmUTid_meas, 'o', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'Measured')
ax[1].text(IC[5], gmUtid_EKV[20], 
        r'$\newline I_\text{spec}\,=\,$'+str(np.round(ispec*1e6,2))+r'$\,\mu A \newline$'+r'$n\,=\,$'+str(np.round(n_slope,2))+r'\newline$\lambda_\text{c}\,=\,$'+str(np.round(lambda_c,2)),
        verticalalignment='top', horizontalalignment='left', fontsize=15)
ax[1].set_yscale('log')
ax[1].set_xscale('log')


ax[1].set_xlabel(r'$\text{IC}$', fontsize = 20)


# ax.legend(bbox_to_anchor = (1, 0.5), loc = 'center left', fontsize=12)
ax[1].legend(loc = 'best', fontsize=12)

# ax[1].set_xlim([min(IC),max(IC)])

ax[1].grid(True, which="both", ls="--", color='0.9')
# ax[1].set_yticklabels([])
# ax[1].set_xticklabels([])

# plt.legend(loc = 'best')

fig.set_size_inches([8,4])

fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step03.jpg')
fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step03.pdf', format="pdf")

#%%Plot

UT=0.026  #Thermal voltage


VTH=0.3 #Initial guess for threshold voltage

IC = ID_meas/ispec
fIC = -1 + np.sqrt((1 + lambda_c*IC)**2 + 4*IC)
VGB  = (fIC + np.log(fIC/2))*(n_slope*UT) + n_slope*VSB + VTH

fig, ax = plt.subplots(1,2, sharex=False, sharey=True)
ax0 = ax[0].twinx()


ax[0].plot(VGB,ID_meas, '-', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'EKV')
ax[0].plot(VG_meas,ID_meas, 'o', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'Measured')
ax[0].text(VGB[35], ID_meas[5], 
        r'$\newline I_\text{spec}\,=\,$'+str(np.round(ispec*1e6,2))+r'$\,\mu A \newline$'+r'$n\,=\,$'+str(np.round(n_slope,2))+r'\newline $\lambda_\text{c}\,=\,$'+str(np.round(lambda_c,2))+ r'$\newline V_\text{TH}\,=\,$'+str(np.round(VTH*1e3,2))+r'$\,mV \newline$',
        verticalalignment='top', horizontalalignment='left', fontsize=13)
ax[0].set_yscale('log')
# ax[0].set_xscale('log')
ax0.plot(VGB,ID_meas, '-', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'EKV')
ax0.plot(VG_meas,ID_meas, 'o', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'Measured')


ax[0].set_ylabel(r'$I_\text{D}\,[A]$', fontsize = 20)
ax[0].set_xlabel(r'$V_\text{G}\,[V]$', fontsize = 20)


# ax.legend(bbox_to_anchor = (1, 0.5), loc = 'center left', fontsize=12)
ax[0].legend(loc = 'upper left', fontsize=12)

# ax[0].set_xlim([-0, 1.2])
# ax[0].set_ylim([min(ID_meas),max(ID_meas)])

ax[0].grid(True, which="both", ls="--", color='0.9')
# ax[0].set_yticklabels([])
# ax[0].set_xticklabels([])
# ax0.set_yticklabels([])
# ax0.set_xticklabels([])

# plt.legend(loc = 'best')

# fig.set_size_inches([8,4])


ax0.get_yaxis().set_ticks([]) 

#%% Optimization of VTH

UT=0.026  #Thermal voltage


VTH=0.3 #Initial guess for threshold voltage

#For the optimization we calculate the square differences between the measured 
#VG and the VG from the EKV
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
    VGB_math = VGB_f(ID_meas, VTH)
    sqr = np.sum((VG_meas-VGB_math)**2)
    return sqr

bounds = [(0.1, 1.2)] #Bound for VTH

results = optimize.differential_evolution(cost,bounds) #Optimizer

VTH = results.x[0]


VGB=VGB_f(ID_meas, VTH)

IC = ID_meas/ispec

numerator = np.sqrt((lambda_c * IC + 1)**2 + 4 * IC) - 1
denominator = IC * (lambda_c * (lambda_c * IC + 1) + 2)
gmUTid_EKV = numerator / denominator



#%%Plot
    
ax1 = ax[1].twinx()

ax[1].plot(VGB,ID_meas, '-', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'EKV')
ax[1].plot(VG_meas,ID_meas, 'o', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'Measured')
ax[1].text(VGB[33], ID_meas[5], 
        r'$\newline I_\text{spec}\,=\,$'+str(np.round(ispec*1e6,2))+r'$\,\mu A \newline$'+r'$n\,=\,$'+str(np.round(n_slope,2))+r'\newline $\lambda_\text{c}\,=\,$'+str(np.round(lambda_c,2))+ r'$\newline V_\text{TH}\,=\,$'+str(np.round(VTH*1e3,2))+r'$\,mV \newline$',
        verticalalignment='top', horizontalalignment='left', fontsize=13)
ax[1].set_yscale('log')
# ax[0].set_xscale('log')
ax1.plot(VGB,ID_meas, '-', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'EKV')
ax1.plot(VG_meas,ID_meas, 'o', color='k', linewidth=2, 
              markerfacecolor="None", markeredgewidth=1,
              markersize=5, markevery=2, label = 'Measured')

ax[1].set_xlabel(r'$\text{V}_\text{G}\,[V]$', fontsize = 20)
   
# ax.legend(bbox_to_anchor = (1, 0.5), loc = 'center left', fontsize=12)
ax[1].legend(loc = 'upper left', fontsize=12)

# ax.set_xlim([1e-3, 1e3])
# ax[0].set_ylim([min(ID_meas),max(ID_meas)])

ax[1].grid(True, which="both", ls="--", color='0.9')
# ax[1].set_yticklabels([])
# ax[1].set_xticklabels([])
# ax1.set_yticklabels([])
# ax1.set_xticklabels([])

# plt.legend(loc = 'best')

fig.set_size_inches([8,4])


plt.tight_layout()   

fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step04.jpg')
fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step04.pdf', format="pdf")

  #%%Plots
 
    
if echo==1:
    # print('***INITIAL GUESES***')
    # print('n = '+str(round(n_slope,2)))
    # print('lambda_c = '+str(lambda_c0))
    # print('ispec0 = '+str(round(ispec0/1e-6,2))+' uA')
    print('***FINAL EXTRACTION***')
    print('n = '+str(round(n_slope,2)))
    print('lambda_c = '+str(round(lambda_c,2)))
    print('ispec = '+str(round(ispec/1e-6,2))+' uA')
    print('VTH = '+str(round(VTH/1e-3,2))+' mV')
    

#%%FIRST WE LOAD ID VD CURVES FOLDERS
# section = 'ID_VD'
# quantity = 'ID'
# mainpath = r'C:\Users\andordam\cernbox\MisCosas\WorkInProgress\TTS_65nm_measurements'  

path = r'C:\Users\andordam\cernbox\MisCosas\WorkInProgress\EKVmodelParameters\data\IDVD_VG_1X0P1.csv'
df = pd.read_csv(path, comment='%')
src = df.values
IDs = src[:,1::2]
VDs = src[:,0]

VSB=0 #Source to bulk voltage

UT=0.026  

plots=1
echo=1

Gspec=ispec/0.026

f1=interp1d(VG_meas, ID_meas, fill_value="extrapolate")

# path = mainpath + r'\Results\WF13\TTS1\Chip05-100C-300Mrad\Left_Array\Irradiation\Irradiation_TJ65WF13TTS1LC00520210831_step' + str(0) + '.txt'
# data, lines = extract_data(path, transistor, section, quantity,1)

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




#%%Plot

fig, ax = plt.subplots(1,1, sharex=True)


for vd in range(9,np.size(VG),2):
    ID_measVD=IDs[:,vd]
    ax.plot(VD_meas,ID_measVD, 'o', color='k', linewidth=2, 
                  markerfacecolor="None", markeredgewidth=1,
                  markersize=5, markevery=2)
    y = GDS[vd] * (VD_meas-1.2) + ID_measVD[np.where(abs(VD_meas-1.2) == min(abs(VD_meas-1.2)))][0]
    ax.plot(VD_meas,y, '-', color='k', linewidth=2, 
                  markerfacecolor="None", markeredgewidth=1,
                  markersize=5, markevery=2)


ax.set_ylabel(r'$I_\text{D} [\text{A}]$', fontsize = 20)
ax.set_xlabel(r'$V_\text{D} [\text{V}]$', fontsize = 20)


ax.grid(True, which="both", ls="--", color='0.9')


plt.legend(['Measured', r'Calculated $\text{G}_{ds}$'],loc = 'best', fontsize=12)

fig.set_size_inches([6,4])


plt.tight_layout() 

fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step05.jpg')
fig.savefig(mainpath+r'\EKVmodelParameters\plots\extraction_step05.pdf', format="pdf")

