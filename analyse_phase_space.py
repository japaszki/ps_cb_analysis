# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:04:52 2022

@author: JohnG
"""
import numpy as np
import portable_fig as pfig
import pickle
import yaml
import os
import phase_space_analysis_lib as pslib
import scan

working_dir = os.getcwd()
print("Working directory is " + working_dir)
 
#Load input parameters:
print("Loading input paramters from " + working_dir + '/analysis_params.yaml ...')
with open('analysis_params.yaml', 'r') as f:
    yaml_params = yaml.load(f)

stab_harms = yaml_params['stab_harms']
scan_param_names = yaml_params['scan_parameter_names']
scan_settings = yaml_params['scan_parameter_settings']
file_indices = yaml_params['file_indices']
analysis = yaml_params['analysis']
file = yaml_params['tomo_file']
        
#Create directory to store results:
output_dir = working_dir + '/phase_space_outputs/'
try:
    print("Creating directory " + output_dir)
    os.makedirs(output_dir)
except:
    pass

#Specify output dir for plots:
file['output_dir'] = output_dir

percentiles = [25, 50, 75]

modes_runs = [np.zeros([analysis['N_buckets_fft'], len(file_indices[scan_index]), \
         len(analysis['fs_harms'])], dtype=complex) for scan_index, scan_param in enumerate(scan_settings)]

fallback_file = working_dir + '\\' + file['file_base'] + str(file_indices[0][0])

#Do phase-space mode analysis on all runs:
for scan_index, scan_param in enumerate(scan_settings):
    for file_index, file_suffix in enumerate(file_indices[scan_index]):       
        modes_runs[scan_index][:,file_index,:] = \
            pslib.read_and_analyse_phase_space(working_dir + '/' + file['file_base'] + str(file_suffix),\
                                               fallback_file, file['file_base'] + str(file_suffix), analysis, True)

total_osc_filt_percentiles = [[None for _ in enumerate(analysis['fs_harms'])] for pc in percentiles]
stab_osc_filt_percentiles = [[None for _ in enumerate(analysis['fs_harms'])] for pc in percentiles]

for fs_harm_index, fs_harm in enumerate(analysis['fs_harms']):
    
    spectrum_filt_percentiles = [[np.percentile(np.absolute(modes_runs[scan_index][:,:,fs_harm_index]), pc, axis=1) \
                              for scan_index, scan_param in enumerate(scan_settings)] for pc in percentiles]
    
    for pc_index, pc in enumerate(percentiles):
        total_osc_filt_percentiles[pc_index][fs_harm_index] = [np.sum(spectrum_filt_percentiles[pc_index][scan_index])\
                                                               for scan_index, scan_param in enumerate(scan_settings)]
        stab_osc_filt_percentiles[pc_index][fs_harm_index] = [np.sum(spectrum_filt_percentiles[pc_index][scan_index][stab_harms])\
                                                              for scan_index, scan_param in enumerate(scan_settings)]
    
    pf = pfig.portable_fig()
    pf.set_figsize((12,9))
    modes = np.arange(analysis['N_buckets_fft'])
    for scan_index, scan_param in enumerate(scan_settings):
        pf.plot(modes, spectrum_filt_percentiles[1][scan_index], '.-',\
                 label=scan.scan_point_label(scan_param_names, scan_param))
        pf.fill_between(modes, spectrum_filt_percentiles[0][scan_index], spectrum_filt_percentiles[2][scan_index],
                alpha=0.2, antialiased=True)
    pf.xlabel("Mode")
    pf.ylabel("Amplitude [s]")
    pf.title('Mode spectrum, phase space method, mode ' + str(fs_harm) +\
              ' component, \n averaged over shots')
    pf.legend(loc=0, fontsize='medium')
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    pf.set_filename(output_dir + '/mean_mode_spectrum_phase_space_vs_param_' + str(fs_harm) + 'fs')
    pf.set_fontsize(20)
    pf.gen_plot()
    pf.save_yaml()


[x_axes, y_indices, labels] = scan.gen_1d_scans(scan_settings, scan_param_names)

#Set of plots with each parameter as the x-axis:
for x_param_index, x_param_name in enumerate(scan_param_names):
    #list of remaining parameters not acting as x-axis:
    rem_param_indices = [param_index for param_index, param_name in enumerate(scan_param_names)\
                         if param_index != x_param_index]   
     
    #one plot for each value of remaining parameters:
    for plot_index, x_axis in enumerate(x_axes[x_param_index]):
        #list of scan indices to plot:
        y_indices_plot = y_indices[x_param_index][plot_index]

        pf = pfig.portable_fig()
        pf.set_figsize((12,9))
        for fs_harm_index, fs_harm in enumerate(analysis['fs_harms']):
            
            y_plot = [total_osc_filt_percentiles[1][fs_harm_index][i] for i in y_indices_plot]
            y_fill_l = [total_osc_filt_percentiles[0][fs_harm_index][i] for i in y_indices_plot]
            y_fill_u = [total_osc_filt_percentiles[2][fs_harm_index][i] for i in y_indices_plot]
            
            pf.plot(x_axis, y_plot, '.-', label='fs * ' + str(fs_harm))
            pf.fill_between(x_axis, y_fill_l, y_fill_u, alpha=0.2, antialiased=True)
        pf.xlabel(x_param_name)
        pf.ylabel('Total amplitude across all modes, phase space method')
        pf.legend(loc=0, fontsize='medium')
        pf.title(labels[x_param_index][plot_index])
        pf.set_filename(output_dir + '/phase_space_total_osc_vs_param_' + str(x_param_index) +\
                            '_plot_' + str(plot_index))
        pf.set_fontsize(20)
        pf.gen_plot()
        pf.save_yaml()
        
        pf = pfig.portable_fig()
        pf.set_figsize((12,9))
        for fs_harm_index, fs_harm in enumerate(analysis['fs_harms']):
            
            y_plot = [stab_osc_filt_percentiles[1][fs_harm_index][i] for i in y_indices_plot]
            y_fill_l = [stab_osc_filt_percentiles[0][fs_harm_index][i] for i in y_indices_plot]
            y_fill_u = [stab_osc_filt_percentiles[2][fs_harm_index][i] for i in y_indices_plot]
            
            pf.plot(x_axis, y_plot, '.-', label='fs * ' + str(fs_harm))
            pf.fill_between(x_axis, y_fill_l, y_fill_u, alpha=0.2, antialiased=True)
        pf.xlabel(x_param_name)
        pf.ylabel('Total amplitude across modes ' + str(stab_harms) + ', phase space method')
        pf.title(labels[x_param_index][plot_index])
        pf.legend(loc=0, fontsize='medium')
        pf.set_filename(output_dir + '/phase_space_stab_osc_vs_param_' + str(x_param_index) +\
                            '_plot_' + str(plot_index))
        pf.set_fontsize(20)
        pf.gen_plot()
        pf.save_yaml()

tomo_statistics_data = {'phase_space_method_modes' : modes_runs}

with open(output_dir + '/tomo_phase_space_results.pickle', 'wb') as f:
    pickle.dump(tomo_statistics_data, f)