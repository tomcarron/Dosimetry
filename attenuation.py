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
Co_DC=8.928e-12
Na_DC=4.544e-12
