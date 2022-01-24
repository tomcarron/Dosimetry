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
Cs_x_N=Cs_x_counts-np.full(len(Cs_x_counts),background_per_minute)   #Background subtracted
Co_x_counts=(distance_measurements.to_numpy()[:,2]).astype(np.float64)
Co_x_N=Co_x_counts-np.full(len(Co_x_counts),background_per_minute)   #Background subtracted
Cs2_x_counts=(distance_measurements.to_numpy()[:,3]).astype(np.float64)
Cs2_x_N=Cs2_x_counts-np.full(len(Cs2_x_counts),background_per_minute)   #Background subtracted
Na_x_counts=(distance_measurements.to_numpy()[:,4]).astype(np.float64)
Na_x_N=Na_x_counts-np.full(len(Na_x_counts),background_per_minute)  #Background subtracted

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
Plotting distance measurements
'''
plt.figure(0)
plt.scatter(distance,Cs_x_N)
plt.xlabel('distance (cm)')
plt.ylabel('Counts')
plt.savefig('plots/Cs_dist.png',dpi=400,bbox_inches='tight')

plt.figure(1)
plt.scatter(distance,Co_x_N)
plt.xlabel('distance (cm)')
plt.ylabel('Counts')
plt.savefig('plots/Co_dist.png',dpi=400,bbox_inches='tight')

plt.figure(2)
plt.scatter(distance,Cs2_x_N)
plt.xlabel('distance (cm)')
plt.ylabel('Counts')
plt.savefig('plots/Cs2_dist.png',dpi=400,bbox_inches='tight')

plt.figure(3)
plt.scatter(distance,Na_x_N)
plt.xlabel('distance (cm)')
plt.ylabel('Counts')
plt.savefig('plots/Na_dist.png',dpi=400,bbox_inches='tight')

'''

'''

plt.show()
