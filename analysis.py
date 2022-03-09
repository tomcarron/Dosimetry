'''
Scripts for analysis of Dosimetry experiment data
'''
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
from scipy import optimize
from scipy.optimize import curve_fit
import astropy.constants as const

distance_measurements = pd.read_csv('data/Distance_measurements.csv')[:-2]
distance=(distance_measurements.to_numpy()[:,0]).astype(np.float64)
dx=0.1
Cs_x_counts=(distance_measurements.to_numpy()[:,1]).astype(np.float64)
Co_x_counts=(distance_measurements.to_numpy()[:,2]).astype(np.float64)
Cs2_x_counts=(distance_measurements.to_numpy()[:,3]).astype(np.float64)
Na_x_counts=(distance_measurements.to_numpy()[:,4]).astype(np.float64)

neutron_activity = pd.read_csv('data/Neutron_Activity.csv')[:-3]
shielding_Al = pd.read_csv('data/Shielding.csv')[:-7]
shielding_Pb = pd.read_csv('data/Shielding.csv')[8:]
background_time=60*60 + 60*14 +7 #1hr 14 mins and 7 seconds in seconds. Time the background radiation is measured for
background_N=3311
background_per_minute=(background_N/background_time)*60
Neutron_offset_t=103  #offset for source to detector time for neutron activity experiment

'''
shielding measurements to array
'''
al_shielding=(shielding_Al.to_numpy()[1:,0]).astype(np.float64)
al_shield_counts=(shielding_Al.to_numpy()[1:,1]).astype(np.float64)
pb_shielding=(shielding_Pb.to_numpy()[1:,0]).astype(np.float64)
pb_shield_counts=(shielding_Pb.to_numpy()[1:,1]).astype(np.float64)

'''
Neutron activity measurements to arrays
'''
neutron_N=(neutron_activity.to_numpy()[:,0]).astype(np.float64)
neutron_tf=(neutron_activity.to_numpy()[:,1]).astype(np.float64)
neutron_dt=(neutron_activity.to_numpy()[:,2]).astype(np.float64)

'''
Dose constants in units of J m^2 kg^-1
'''
Cs_DC=2.359e-12
Cs2_DC=2.359e-12
Co_DC=8.928e-12
Na_DC=4.544e-12

#Function to take N counts (background included), measured for t seconds and return Ndot ^-1/2 and error, and Ndot and error.

def calculate(N,t):
    B=background_N                                  #measured background counts
    #print('B',B)
    bt=background_time                              #time measuring background in s
    #print('bt',bt)
    N_s=N/t                                         #count rate
    #print('N_S',N_s)
    dt=1/t                                          #error of 1s in measured time.
    #print('dt',dt)
    N_s_err=np.sqrt(N_s+dt**2)                   #err in count rate
    #print('N_s_err',N_s_err)
    B_s=B/bt                                        #background count rate
    #print('B_s',B_s)
    dbt=1/bt                                        # error of 1s in measured time
    #print('dbt',dbt)
    B_s_err=np.sqrt(B_s**2+dbt**2)                  #err in background count rate
    #print('B_s_err',B_s_err)
    N_mod=N_s-B_s                                   #count rate with background subtracted
    #print('N_mod',N_mod)
    N_mod_err=np.sqrt((N_s_err)**2 + (B_s_err)**2)  #error in count rate with background subtracted
    #print('N_mod_err',N_mod_err)
    N_dot_exp=N_mod**(-0.5)                  #raising count rate with subtracted background to power of -1/2
    #print('N_dot_exp',N_dot_exp)
    err_Ndot_exp=0.5*(N_mod**(-3/2))*N_mod_err      #error in Ndot ^-1/2 --> error in counts, time, background counts, background time and propagation via formulas all accounted for.
    #print('err_Ndot_exp',err_Ndot_exp)
    return N_dot_exp, err_Ndot_exp, N_mod, N_mod_err

#print(calculate(Cs_x_counts[0],60))

def cal_factor(D_dot,err_D_dot,N_dot,err_N_dot):
    epsilon=D_dot/N_dot
    sqrt=np.sqrt((err_N_dot/N_dot)**2 + (err_D_dot/D_dot)**2)
    err_epsilon=epsilon*np.sqrt((err_N_dot/N_dot)**2 + (err_D_dot/D_dot)**2)
    return epsilon, err_epsilon, sqrt

def dose_rate(Dose_const,A,dA,x,x0,dx,dx0):
    d=x-x0
    dd=np.sqrt((dx**2)+(dx0**2))
    y=(Dose_const*A)/(d**2)
    y_err=np.sqrt( (((Dose_const*dA)/(d**2))**2) +(((-2*Dose_const*A*dd)/(d**3))**2) )
    return y, y_err

def linear(x,a,c):
    y=c+a*x
    return y

def activity(epsilon,a,Dose_const):
    activity=epsilon/(Dose_const*a**2)
    return activity

'''
#calculating x0
'''
distance_2=np.linspace(-2,16,300)
popt_Cs, pcov_Cs = curve_fit(linear, distance, calculate(Cs_x_counts,60)[0])
Cs_perr = np.sqrt(np.diag(pcov_Cs))
print('Cs_popt',popt_Cs)
print('Cs_perr',Cs_perr)
popt_Co, pcov_Co = curve_fit(linear, distance, calculate(Co_x_counts,60)[0])
Co_perr = np.sqrt(np.diag(pcov_Co))
print('Co_popt',popt_Co)
print('Co_perr',Co_perr)
popt_Cs2, pcov_Cs2 = curve_fit(linear, distance, calculate(Cs2_x_counts,60)[0])
Cs2_perr = np.sqrt(np.diag(pcov_Cs2))
print('Cs2_popt',popt_Cs2)
print('Cs2_perr',Cs2_perr)
popt_Na, pcov_Na = curve_fit(linear, distance, calculate(Na_x_counts,60)[0])
Na_perr = np.sqrt(np.diag(pcov_Na))
print('Na_popt',popt_Na)
print('Na_perr',Na_perr)

Cs_fit_y=popt_Cs[0]*distance_2 + popt_Cs[1]
i=0
while i < len(distance_2):
    if Cs_fit_y[i] < 1e-3 and Cs_fit_y[i] > -1e-3:
        Cs_x0=distance_2[i]
        Cs_y_x0=Cs_fit_y[i]
        i=i+1
    else:
        i=i+1

Co_fit_y=popt_Co[0]*distance_2 + popt_Co[1]

i=0
while i < len(distance_2):
    if Co_fit_y[i] < 1e-2 and Co_fit_y[i] > -1e-2:
        Co_x0=distance_2[i]
        Co_y_x0=Co_fit_y[i]
        i=i+1
    else:
        i=i+1

Cs2_fit_y=popt_Cs2[0]*distance_2 + popt_Cs2[1]

i=0
while i < len(distance_2):
    if Cs2_fit_y[i] < 1e-2 and Cs2_fit_y[i] > -1e-2:
        Cs2_x0=distance_2[i]
        Cs2_y_x0=Cs2_fit_y[i]
        i=i+1
    else:
        i=i+1

Na_fit_y=popt_Na[0]*distance_2 + popt_Na[1]

i=0
while i < len(distance_2):
    if Na_fit_y[i] < 1e-3 and Na_fit_y[i] > -1e-3:
        Na_x0=distance_2[i]
        Na_y_x0=Na_fit_y[i]
        i=i+1
    else:
        i=i+1

print('Cs_x0',Cs_x0)
print('Co_x0',Co_x0)
print('Na_x0',Na_x0)
print('Cs2_x0',Cs2_x0)
#print(calculate(Cs_x_counts[0],60))
'''
Calculating dose rates. use largest distance-x0 and count rate as activity.
'''
#Cs (first source)
D_dot_Cs=dose_rate(Cs_DC,calculate(Cs_x_counts[0],60)[2],calculate(Cs_x_counts[0],60)[3],distance[0],Cs_x0,0.1,0.1)
print(D_dot_Cs[0],'p/m',D_dot_Cs[1],'Dose rate of Cs-137 in W/kg')

#Calibration factor with error calculated using strongest Cs source
cal_fac_Cs=cal_factor(D_dot_Cs[0],D_dot_Cs[1],calculate(Cs_x_counts[0],60)[2],calculate(Cs_x_counts[0],60)[3])
print("epsilon=",cal_fac_Cs[0],"p/m",cal_fac_Cs[1],"sqrt:",cal_fac_Cs[2])

D_dot_Co=dose_rate(Co_DC,calculate(Co_x_counts[0],60)[2],calculate(Co_x_counts[0],60)[3],distance[0],Co_x0,0.1,0.1)
cal_fac_Co=cal_factor(D_dot_Co[0],D_dot_Co[1],calculate(Co_x_counts[0],60)[2],calculate(Co_x_counts[0],60)[3])

D_dot_Cs2=dose_rate(Cs2_DC,calculate(Cs2_x_counts[0],60)[2],calculate(Cs2_x_counts[0],60)[3],distance[0],Cs2_x0,0.1,0.1)
cal_fac_Cs2=cal_factor(D_dot_Cs2[0],D_dot_Cs2[1],calculate(Cs2_x_counts[0],60)[2],calculate(Cs2_x_counts[0],60)[3])

D_dot_Na=dose_rate(Na_DC,calculate(Na_x_counts[0],60)[2],calculate(Na_x_counts[0],60)[3],distance[0],Na_x0,0.1,0.1)
cal_fac_Na=cal_factor(D_dot_Na[0],D_dot_Na[1],calculate(Na_x_counts[0],60)[2],calculate(Na_x_counts[0],60)[3])
'''
#plots
'''
'''
#Counts for 1 second vs x
'''
plt.figure(0)
#plt.scatter(distance,Cs_x_counts)
plt.errorbar(distance,calculate(Cs_x_counts,60)[2],calculate(Cs_x_counts,60)[3],dx,'.',label='Cs-137')
plt.xlabel('distance (cm)')
plt.ylabel('$\dot N$',rotation='horizontal',fontsize='large')
plt.legend()
plt.savefig('plots/Cs_dist_1s.png',dpi=400,bbox_inches='tight')

plt.figure(1)
#plt.scatter(distance,Co_x_counts)
plt.errorbar(distance,calculate(Co_x_counts,60)[2],calculate(Co_x_counts,60)[3],dx,'.',label='Co')
plt.xlabel('distance (cm)')
plt.ylabel('$\dot N$',rotation='horizontal',fontsize='large')
plt.legend()
plt.savefig('plots/Co_dist_1s.png',dpi=400,bbox_inches='tight')

plt.figure(2)
#plt.scatter(distance,Cs2_x_counts)
plt.errorbar(distance,calculate(Cs2_x_counts,60)[2],calculate(Cs2_x_counts,60)[3],dx,'.',label='Cs2')
plt.xlabel('distance (cm)')
plt.ylabel('$\dot N$',rotation='horizontal',fontsize='large')
plt.legend()
plt.savefig('plots/Cs2_dist_1s.png',dpi=400,bbox_inches='tight')

plt.figure(3)
#plt.scatter(distance,Na_x_counts)
plt.errorbar(distance,calculate(Na_x_counts,60)[2],calculate(Na_x_counts,60)[3],dx,'.',label='Na')
plt.xlabel('distance (cm)')
plt.ylabel('$\dot N$',rotation='horizontal',fontsize='large')
plt.legend()
plt.savefig('plots/Na_dist_1s.png',dpi=400,bbox_inches='tight')


plt.figure(4)
plt.errorbar(distance,calculate(Cs_x_counts,60)[0],calculate(Cs_x_counts,60)[1],dx,'.',label='Cs-137')
plt.plot(distance_2,Cs_fit_y)
plt.xlabel('x (cm)')
plt.ylabel(r'$\dot N^{-\frac{1}{2}}$',fontsize='large',rotation='horizontal')
plt.legend(loc='upper left')
plt.ylim(0)
plt.xlim(-1.5)
plt.savefig('plots/Cs_dist_mod.png',dpi=400,bbox_inches='tight')

plt.figure(5)
plt.errorbar(distance,calculate(Co_x_counts,60)[0],calculate(Co_x_counts,60)[1],dx,'.',label='Co-60')
plt.plot(distance_2,Co_fit_y)
plt.xlabel('x (cm)')
plt.ylabel(r'$\dot N^{-\frac{1}{2}}$',fontsize='large',rotation='horizontal')
plt.ylim(0)
plt.xlim(-2)
plt.legend(loc='upper left')
plt.savefig('plots/Co_dist_mod.png',dpi=400,bbox_inches='tight')

plt.figure(6)
plt.errorbar(distance,calculate(Cs2_x_counts,60)[0],calculate(Cs2_x_counts,60)[1],dx,'.',label='Cs-137 (source 2)')
plt.plot(distance_2,Cs2_fit_y)
plt.xlabel('x (cm)')
plt.ylabel(r'$\dot N^{-\frac{1}{2}}$',fontsize='large',rotation='horizontal')
plt.legend(loc='upper left')
plt.ylim(0)
plt.xlim(0)
plt.savefig('plots/Cs2_dist_mod.png',dpi=400,bbox_inches='tight')

plt.figure(7)
plt.errorbar(distance,calculate(Na_x_counts,60)[0],calculate(Na_x_counts,60)[1],dx,'.',label='Na-22')
plt.plot(distance_2,Na_fit_y)
plt.xlabel('x (cm)')
plt.ylabel(r'$\dot N^{-\frac{1}{2}}$',fontsize='large',rotation='horizontal')
plt.legend(loc='upper left')
plt.ylim(0)
plt.xlim(0)
plt.savefig('plots/Na_dist_mod.png',dpi=400,bbox_inches='tight')




'''
Calculate the activities using the fit parameters
'''
activities=[]
activities.append(activity(cal_fac_Cs[0],popt_Cs[0],Cs_DC))
activities.append(activity(cal_fac_Co[0],popt_Co[0],Co_DC))
activities.append(activity(cal_fac_Cs2[0],popt_Cs2[0],Cs2_DC))
activities.append(activity(cal_fac_Na[0],popt_Na[0],Na_DC))

print(activities,"Cs,Co,Cs2,Na")


"""
calculate the sum of all four dose rates at a distance of 1m. How long til 1.5mSv?
"""

def sum_dose_rates(x):
    y=0
    y=y+dose_rate(Cs_DC,calculate(Cs_x_counts[0],60)[2],calculate(Cs_x_counts[0],60)[3],x,Cs_x0,0.1,0.1)[0]
    y=y+dose_rate(Co_DC,calculate(Co_x_counts[0],60)[2],calculate(Co_x_counts[0],60)[3],x,Co_x0,0.1,0.1)[0]
    y=y+dose_rate(Cs2_DC,calculate(Cs2_x_counts[0],60)[2],calculate(Cs2_x_counts[0],60)[3],x,Cs2_x0,0.1,0.1)[0]
    y=y+dose_rate(Na_DC,calculate(Na_x_counts[0],60)[2],calculate(Na_x_counts[0],60)[3],x,Na_x0,0.1,0.1)[0]
    #takes x in cm and returns Ddot in W/kg
    return y

total_dose_rate=sum_dose_rates(100)
print("total dose rate",total_dose_rate)

#Quality factor=1
#total dose rate is Sv per second, t x total_dose_rate = 1.5mSv

time=1.5e-3 / total_dose_rate
print("time to 1.5mSv in seconds",time)

time2=((time/3600)/24)/365
print("time to 1.5mSv in years",time2)


#Calculate the measured dose rate of background radiation









plt.show()
