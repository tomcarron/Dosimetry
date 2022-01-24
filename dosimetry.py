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
'''
print(distance)
print(neutron_activity)
print(shielding_Al)
print(shielding_Pb)
'''
distance=(distance_measurements.to_numpy()[:,0]).astype(np.float64)
Cs_x_counts=(distance_measurements.to_numpy()[:,1]).astype(np.float64)
Co_x_counts=(distance_measurements.to_numpy()[:,2]).astype(np.float64)
Cs2_x_counts=(distance_measurements.to_numpy()[:,3]).astype(np.float64)
Na_x_counts=(distance_measurements.to_numpy()[:,4]).astype(np.float64)

plt.figure(0)
plt.scatter(distance,Cs_x_counts)
plt.xlabel('distance (cm)')
plt.ylabel('Counts')
plt.savefig('plots/Cs_dist.png',dpi=400,bbox_inches='tight')

plt.figure(1)
plt.scatter(distance,Co_x_counts)
plt.xlabel('distance (cm)')
plt.ylabel('Counts')
plt.savefig('plots/Co_dist.png',dpi=400,bbox_inches='tight')

plt.figure(2)
plt.scatter(distance,Cs2_x_counts)
plt.xlabel('distance (cm)')
plt.ylabel('Counts')
plt.savefig('plots/Cs2_dist.png',dpi=400,bbox_inches='tight')

plt.figure(3)
plt.scatter(distance,Na_x_counts)
plt.xlabel('distance (cm)')
plt.ylabel('Counts')
plt.savefig('plots/Na_dist.png',dpi=400,bbox_inches='tight')


plt.show()
