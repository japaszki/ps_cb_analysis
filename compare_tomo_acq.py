# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:55:27 2022

@author: JohnG
"""
import os
import yaml
import pickle
import matplotlib.pyplot as plt
import numpy as np
import stats_lib as slib

working_dir = os.getcwd()
print("Working directory is " + working_dir)

#Create directory to store results:
output_dir = working_dir + '/tomo_vs_acq_outputs/'
try:
    print("Creating directory " + output_dir)
    os.makedirs(output_dir)
except:
    pass

#Load input parameters:
print("Loading input paramters from " + working_dir + '/analysis_params.yaml ...')
with open('analysis_params.yaml', 'r') as f:
    yaml_params = yaml.load(f)
    
file_indices = yaml_params['file_indices']
scan_param_names = yaml_params['scan_parameter_names']
scan_settings = yaml_params['scan_parameter_settings']

with open(working_dir + '/tomo_outputs/tomo_filt_results.pickle', 'rb') as f:
    tomo_filt_results = pickle.load(f)

with open(working_dir + '/acq_outputs/acq_results.pickle', 'rb') as f:
    acq_results = pickle.load(f)
    
with open(working_dir + '/phase_space_outputs/tomo_phase_space_results.pickle', 'rb') as f:
    ps_results = pickle.load(f)

acq_harmonics = yaml_params['acq_file']['channel_harmonics'] 
fs_harms = yaml_params['analysis']['fs_harms']

#Combine data for all harmonics used in internal acq:
filt_u_vs_h = [None for h in fs_harms]
filt_l_vs_h = [None for h in fs_harms]
ps_u_vs_h = [None for h in fs_harms]
ps_l_vs_h = [None for h in fs_harms]
acq_dipole_vs_h = [None for h in fs_harms]
acq_quad_vs_h = [None for h in fs_harms]

for h_index, h in enumerate(fs_harms):
    filt_u = np.array(0)
    filt_l = np.array(0)

    ps_u = np.array(0)
    ps_l = np.array(0)
    
    acq_dipole = np.array(0)
    acq_quad = np.array(0)
    
    for channel_index, mode in enumerate(acq_harmonics):
        for scan_index, scan_param in enumerate(scan_settings):           
            filt_u = np.append(filt_u, np.abs(tomo_filt_results['filter_method_modes'][scan_index][21-mode, :, h_index]))
            filt_l = np.append(filt_l, np.abs(tomo_filt_results['filter_method_modes'][scan_index][mode, :, h_index]))
            
            ps_u = np.append(ps_u, np.abs(ps_results['phase_space_method_modes'][scan_index][21-mode,:,h_index]))
            ps_l = np.append(ps_l, np.abs(ps_results['phase_space_method_modes'][scan_index][mode,:,h_index]))
            
            acq_dipole = np.append(acq_dipole, acq_results['tomo_runs']['Dipole'][scan_index][channel_index,:,h_index])
            acq_quad = np.append(acq_quad, acq_results['tomo_runs']['Quad'][scan_index][channel_index,:,h_index])
            
    filt_u_vs_h[h_index] = filt_u
    filt_l_vs_h[h_index] = filt_l
    
    ps_u_vs_h[h_index] = ps_u
    ps_l_vs_h[h_index] = ps_l
        
    acq_dipole_vs_h[h_index] = acq_dipole
    acq_quad_vs_h[h_index] = acq_quad


#Edge method only gives dipole and quadrupole mode spectra:
pos_u = np.array(0)
pos_l = np.array(0)
width_u = np.array(0)
width_l = np.array(0)

for channel_index, mode in enumerate(acq_harmonics):
    for scan_index, scan_param in enumerate(scan_settings):
        pos_u = np.append(pos_u, np.abs(tomo_filt_results['edge_method_pos_modes'][scan_index][21-mode, :]))
        pos_l = np.append(pos_l, np.abs(tomo_filt_results['edge_method_pos_modes'][scan_index][mode, :]))
        width_u = np.append(width_u, np.abs(tomo_filt_results['edge_method_width_modes'][scan_index][21-mode, :]))
        width_l = np.append(width_l, np.abs(tomo_filt_results['edge_method_width_modes'][scan_index][mode, :]))

#Compare filter method mode amplitudes from tomoscope with dipole channel internal acq:
filt_vs_acq_slope_error = []
ps_vs_acq_slope_error = []
for h_index, h in enumerate(fs_harms):    
    [_, slopes] = slib.gpr_plot([acq_dipole_vs_h[h_index], acq_dipole_vs_h[h_index]],\
              [filt_l_vs_h[h_index] + filt_u_vs_h[h_index], 3e9*(ps_l_vs_h[h_index] + ps_u_vs_h[h_index])],\
              str(h) + '*fs sideband amplitude from internal dipole acq',\
            'Tomogram mode amplitude, ' + str(h) + '*fs)',\
            plot_labels=['Filter method', 'Phase space method * 3e9'], 
            title=None, filename = output_dir + '/filter_phase_space_vs_acq_' + str(h) + 'fs.png')
    
    ps_vs_acq_slope_error.append((2 * slopes[1][0]) / (slopes[1][2] - slopes[1][1]))
    filt_vs_acq_slope_error.append((2 * slopes[0][0]) / (slopes[0][2] - slopes[0][1]))

edge_vs_acq_slope_error = []
[_, slopes] = slib.gpr_plot([acq_dipole_vs_h[0], acq_dipole_vs_h[1]],\
         [pos_u + pos_l, width_u + width_l],\
         'Sideband amplitude from internal dipole acq',\
         'Tomogram mode amplitude [s]',\
        plot_labels=['Bunch position vs. 1*fs sideband', 'Bunch width vs. 2*fs sideband'],\
        title=None,\
        filename = output_dir + '/edge_vs_acq.png')

edge_vs_acq_slope_error.append((2 * slopes[0][0]) / (slopes[0][2] - slopes[0][1]))
edge_vs_acq_slope_error.append((2 * slopes[1][0]) / (slopes[1][2] - slopes[1][1]))

edge_vs_acq_corr = [None for _ in range(2)]
edge_vs_acq_corr[0] = np.abs(np.corrcoef(pos_u + pos_l, acq_dipole_vs_h[0])[0,1])
edge_vs_acq_corr[1] = np.abs(np.corrcoef(width_u + width_l, acq_dipole_vs_h[1])[0,1])
    
filt_vs_acq_corr = [np.abs(np.corrcoef(filt_l_vs_h[h_index] + filt_u_vs_h[h_index], acq_dipole_vs_h[h_index])[0,1])\
                   for h_index, h in enumerate(fs_harms)]
    
ps_vs_acq_corr = [np.abs(np.corrcoef(ps_l_vs_h[h_index] + ps_u_vs_h[h_index], acq_dipole_vs_h[h_index])[0,1])\
                 for h_index, h in enumerate(fs_harms)]

plt.figure(figsize=(10,10))
plt.plot([1, 2], edge_vs_acq_corr, '-s', label='Edge method')
plt.plot(fs_harms, filt_vs_acq_corr, '-s', label='Filter method')
plt.plot(fs_harms, ps_vs_acq_corr, '-s', label='Phase space method')
plt.xlabel('fs harmonic')
plt.ylabel('Pearson correlation coefficient of tomoscope vs. internal acq data')
plt.ylim([0,1])
plt.legend(loc=0, fontsize='medium')
plt.rc('font', size=16)
plt.savefig(output_dir + '/tomo_acq_correlation_vs_fs_harm.png')
plt.show()

plt.figure(figsize=(10,10))
plt.plot([1, 2], edge_vs_acq_slope_error, '-s', label='Edge method')
plt.plot(fs_harms, filt_vs_acq_slope_error, '-s', label='Filter method')
plt.plot(fs_harms, ps_vs_acq_slope_error, '-s', label='Phase space method')
plt.xlabel('fs harmonic')
plt.ylabel('Slope / uncertainty of tomoscope vs. internal acq data')
# plt.ylim([0,1])
plt.ylim(bottom=0)
plt.legend(loc=0, fontsize='medium')
plt.rc('font', size=16)
plt.savefig(output_dir + '/tomo_acq_slope_error_vs_fs_harm.png')
plt.show()

# filt_median = [np.percentile(filt_l_vs_h[h_index] + filt_u_vs_h[h_index], 50) for h_index, h in enumerate(fs_harms)]
# ps_median = [np.percentile(ps_l_vs_h[h_index] + ps_u_vs_h[h_index], 50) for h_index, h in enumerate(fs_harms)]
# acq_median = [np.percentile(acq_dipole_vs_h[h_index], 50) for h_index, h in enumerate(fs_harms)]


# plt.figure(figsize=(10,10))
# plt.plot(fs_harms, filt_median, '-s', label='Filter method')
# plt.xlabel('fs harmonic')
# plt.ylabel('Median value')
# plt.legend(loc=0, fontsize='medium')
# plt.rc('font', size=16)
# plt.show()
 
# plt.figure(figsize=(10,10))
# plt.plot(fs_harms, ps_median, '-s', label='Phase space method')
# plt.xlabel('fs harmonic')
# plt.ylabel('Median value')
# plt.legend(loc=0, fontsize='medium')
# plt.rc('font', size=16)
# plt.show()

# plt.figure(figsize=(10,10))
# plt.plot(fs_harms, acq_median, '-s', label='Internal acquisition')
# plt.xlabel('fs harmonic')
# plt.ylabel('Median value')
# plt.legend(loc=0, fontsize='medium')
# plt.rc('font', size=16)
# plt.show()
    
#Compare dipole, sextupole and decapole modes from phase space method:
[_, slopes] = slib.gpr_plot([np.hstack([ps_l_vs_h[0], ps_u_vs_h[0]]), np.hstack([ps_l_vs_h[0], ps_u_vs_h[0]])],\
         [np.hstack([ps_l_vs_h[2], ps_u_vs_h[2]]), np.hstack([ps_l_vs_h[4], ps_u_vs_h[4]])],\
        'Tomogram mode amplitude \n(phase space method, 1*fs)',\
        'Tomogram mode amplitude \n(phase space method, N*fs)',\
        plot_labels=['N=3', 'N=5'], \
        title='Odd multipole modes in phase space',\
        filename = output_dir + '/phase_space_odd_harms.png')
        
#Compare dipole, sextupole and decapole modes in internal acq:
[_, slopes] = slib.gpr_plot([acq_dipole_vs_h[0], acq_dipole_vs_h[0]],\
         [acq_dipole_vs_h[2], acq_dipole_vs_h[4]],\
        '1*fs sideband amplitude from internal dipole acq',\
        'N*fs sideband amplitude from internal dipole acq',\
        plot_labels=['N=3', 'N=5'],\
        title='Odd harmonics in internal acq',\
        filename = output_dir + '/acq_odd_harms.png')
     
#Compare quadrupole, octupole, and 12-pole modes from phase space method:
slib.gpr_plot([np.hstack([ps_l_vs_h[1], ps_u_vs_h[1]]), np.hstack([ps_l_vs_h[1], ps_u_vs_h[1]])],\
              [np.hstack([ps_l_vs_h[3], ps_u_vs_h[3]]), np.hstack([ps_l_vs_h[5], ps_u_vs_h[5]])],\
     'Tomogram mode amplitude \n(phase space method, 2*fs)',\
     'Tomogram mode amplitude \n(phase space method, N*fs)',\
     plot_labels=['N=4', 'N=6'],\
     title='Even harmonics in phase space',\
     filename = output_dir + '/phase_space_even_harms.png')

#Compare quadrupole, octupole, and 12-pole modes in internal acq:
slib.gpr_plot([acq_dipole_vs_h[1], acq_dipole_vs_h[1]],\
              [acq_dipole_vs_h[3], acq_dipole_vs_h[5]],\
        '2*fs sideband amplitude from internal dipole acq',\
        'N*fs sideband amplitude from internal dipole acq',\
        plot_labels=['N=4', 'N=6'],\
        title='Even harmonics in internal acq',\
        filename = output_dir + '/acq_even_harms.png')
    
ps_harm_vs_h1_corr = [np.abs(np.corrcoef(np.hstack([ps_l_vs_h[0], ps_u_vs_h[0]]),\
                               np.hstack([ps_l_vs_h[h_index], ps_u_vs_h[h_index]]))[0,1])\
                   for h_index, h in enumerate(fs_harms)]

ps_harm_vs_h2_corr = [np.abs(np.corrcoef(np.hstack([ps_l_vs_h[1], ps_u_vs_h[1]]),\
                               np.hstack([ps_l_vs_h[h_index], ps_u_vs_h[h_index]]))[0,1])\
                   for h_index, h in enumerate(fs_harms)]

acq_harm_vs_h1_cov = [np.abs(np.corrcoef(acq_dipole_vs_h[0], acq_dipole_vs_h[h_index])[0,1])\
                      for h_index, h in enumerate(fs_harms)]
    
acq_harm_vs_h2_cov = [np.abs(np.corrcoef(acq_dipole_vs_h[1], acq_dipole_vs_h[h_index])[0,1])\
                      for h_index, h in enumerate(fs_harms)]
    
plt.figure(figsize=(10,10))
plt.plot(fs_harms, ps_harm_vs_h1_corr, '-s',  label='Phase space') #,
plt.plot(fs_harms, acq_harm_vs_h1_cov, '-s',  label='Internal acq')
plt.xlabel('N')
plt.ylabel('Pearson correlation coefficient of N*fs vs. 1*fs')
plt.ylim([0,1])
plt.legend(loc=0, fontsize='medium')
plt.rc('font', size=16)
plt.savefig(output_dir + '/correlation_1fs_vs_fs_harm.png')
plt.show()

plt.figure(figsize=(10,10))
plt.plot(fs_harms, ps_harm_vs_h2_corr, '-s', label='Phase space')
plt.plot(fs_harms, acq_harm_vs_h2_cov, '-s', label='Internal acq')
plt.xlabel('N')
plt.ylabel('Pearson correlation coefficient of N*fs vs. 2*fs')
plt.ylim([0,1])
plt.legend(loc=0, fontsize='medium')
plt.rc('font', size=16)
plt.savefig(output_dir + '/correlation_2fs_vs_fs_harm.png')
plt.show()