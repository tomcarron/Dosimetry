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
neutron_activity = pd.read_csv('data/Neutron_Activity.csv')[:-3]
shielding_Al = pd.read_csv('data/Shielding.csv')[:-7]
shielding_Pb = pd.read_csv('data/Shielding.csv')[8:]
background_time=60*60 + 60*14 +7 #1hr 14 mins and 7 seconds in seconds. Time the background radiation is measured for
background_N=3311
background_per_minute=(background_N/background_time)*60
Neutron_offset_t=103  #offset for source to detector time for neutron activity experiment

'''
Distance measurements to arrays. counts measured for 60s
'''
distance=(distance_measurements.to_numpy()[:,0]).astype(np.float64)
Cs_x_counts=(distance_measurements.to_numpy()[:,1]).astype(np.float64)
Cs_x_N=((Cs_x_counts-np.full(len(Cs_x_counts),background_per_minute) ) /60 ) #Background subtracted and converted to per second
Co_x_counts=(distance_measurements.to_numpy()[:,2]).astype(np.float64)
Co_x_N=((Co_x_counts-np.full(len(Co_x_counts),background_per_minute) ) /60 )  #Background subtracted and converted to per second
Cs2_x_counts=(distance_measurements.to_numpy()[:,3]).astype(np.float64)
Cs2_x_N=((Cs2_x_counts-np.full(len(Cs2_x_counts),background_per_minute) ) /60 )  #Background subtracted and converted to per second
Na_x_counts=(distance_measurements.to_numpy()[:,4]).astype(np.float64)
Na_x_N=((Na_x_counts-np.full(len(Na_x_counts),background_per_minute) ) /60 ) #Background subtracted and converted to per second

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
Functions for errors etc
'''
#Ndot ^-0.5, takes N per second
def Ndot_exponent(N):
    y=N**(-0.5)
    return y

#error in Ndot ^-1/2, takes N, t and error in t
def err_Ndot_exponent(N,t,dt):
    Ndot_exp=Ndot_exponent(N)
    dN=np.sqrt(N)
    y=Ndot_exp*np.sqrt(((-0.5*dN)/N)**2+((-0.5*dt)/t)**2)
    return y
'''
Plotting distance measurements
'''
t=1
dt=1/60
dx=0.1

'''
Counts for 1 second vs x
'''
plt.figure(0)
#plt.scatter(distance,Cs_x_counts)
plt.errorbar(distance,Cs_x_N,np.sqrt(Cs_x_N),0.1,'.')
plt.xlabel('distance (cm)')
plt.ylabel('$\dot N$',rotation='horizontal',fontsize='large')
plt.savefig('plots/Cs_dist_1s.png',dpi=400,bbox_inches='tight')

plt.figure(1)
#plt.scatter(distance,Co_x_counts)
plt.errorbar(distance,Co_x_N,np.sqrt(Co_x_N),0.1,'.')
plt.xlabel('distance (cm)')
plt.ylabel('$\dot N$',rotation='horizontal',fontsize='large')
plt.savefig('plots/Co_dist_1s.png',dpi=400,bbox_inches='tight')

plt.figure(2)
#plt.scatter(distance,Cs2_x_counts)
plt.errorbar(distance,Cs2_x_N,np.sqrt(Cs2_x_N),0.1,'.')
plt.xlabel('distance (cm)')
plt.ylabel('$\dot N$',rotation='horizontal',fontsize='large')
plt.savefig('plots/Cs2_dist_1s.png',dpi=400,bbox_inches='tight')

plt.figure(3)
#plt.scatter(distance,Na_x_counts)
plt.errorbar(distance,Na_x_N,np.sqrt(Na_x_N),0.1,'.')
plt.xlabel('distance (cm)')
plt.ylabel('$\dot N$',rotation='horizontal',fontsize='large')
plt.savefig('plots/Na_dist_1s.png',dpi=400,bbox_inches='tight')

plt.figure(4)
plt.errorbar(distance,Ndot_exponent(Cs_x_N),err_Ndot_exponent(Cs_x_N,t,dt),dx,'.')
plt.xlabel('x (cm)')
plt.ylabel(r'$\dot N^{-\frac{1}{2}}$',fontsize='large',rotation='horizontal')
plt.savefig('plots/Cs_dist_mod.png',dpi=400,bbox_inches='tight')

plt.figure(5)
plt.errorbar(distance,Ndot_exponent(Co_x_N),err_Ndot_exponent(Co_x_N,t,dt),dx,'.')
plt.xlabel('x (cm)')
plt.ylabel(r'$\dot N^{-\frac{1}{2}}$',fontsize='large',rotation='horizontal')
plt.savefig('plots/Co_dist_mod.png',dpi=400,bbox_inches='tight')

plt.figure(6)
plt.errorbar(distance,Ndot_exponent(Cs2_x_N),err_Ndot_exponent(Cs2_x_N,t,dt),dx,'.')
plt.xlabel('x (cm)')
plt.ylabel(r'$\dot N^{-\frac{1}{2}}$',fontsize='large',rotation='horizontal')
plt.savefig('plots/Cs2_dist_mod.png',dpi=400,bbox_inches='tight')

plt.figure(7)
plt.errorbar(distance,Ndot_exponent(Na_x_N),err_Ndot_exponent(Na_x_N,t,dt),dx,'.')
plt.xlabel('x (cm)')
plt.ylabel(r'$\dot N^{-\frac{1}{2}}$',fontsize='large',rotation='horizontal')
plt.savefig('plots/Na_dist_mod.png',dpi=400,bbox_inches='tight')

'''

'''

plt.show()
