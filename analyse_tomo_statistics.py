# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:28:49 2022

@author: JohnG
"""
import numpy as np
import pylab as plt
import pickle
import yaml
import os
import tomo_analysis_lib as tlib
import stats_lib as slib
import scan

working_dir = os.getcwd()
print("Working directory is " + working_dir)
 
#Load input parameters:
print("Loading input paramters from " + working_dir + '/analysis_params.yaml ...')
with open('analysis_params.yaml', 'r') as f:
    yaml_params = yaml.load(f)

stab_harms = yaml_params['stab_harms']
file_indices = yaml_params['file_indices']
scan_param_names = yaml_params['scan_parameter_names']
scan_settings = yaml_params['scan_parameter_settings']
analysis = yaml_params['analysis']
file = yaml_params['tomo_file']
        
#Create directory to store results:
output_dir = working_dir + '/tomo_outputs/'
try:
    print("Creating directory " + output_dir)
    os.makedirs(output_dir)
except:
    pass

#Specify output dir for plots:
file['output_dir'] = output_dir

percentiles = [25, 50, 75]

pos_modes_edge_window_runs = [np.zeros([analysis['N_buckets_fft'], len(file_indices[scan_index])], dtype=complex)\
                                   for scan_index, scan_param in enumerate(scan_settings)]
width_modes_edge_window_runs = [np.zeros([analysis['N_buckets_fft'], len(file_indices[scan_index])], dtype=complex)\
                                    for scan_index, scan_param in enumerate(scan_settings)]
modes_filt_window_runs = [np.zeros([analysis['N_buckets_fft'], len(file_indices[scan_index]), \
        len(analysis['fs_harms'])], dtype=complex) for scan_index, scan_param in enumerate(scan_settings)]

for scan_index, scan_param in enumerate(scan_settings):
    for file_index, file_suffix in enumerate(file_indices[scan_index]):
        file['filename'] = working_dir + '/' + file['file_base'] + str(file_suffix)
        file['display_name'] = 'shot_' + str(file_suffix)
        [pos_modes_edge_window_runs[scan_index][:, file_index],\
         width_modes_edge_window_runs[scan_index][:, file_index], modes_filt_curr] = \
            tlib.read_and_analyse_tomo(file, analysis, True, True)   
        modes_filt_window_runs[scan_index][:,file_index,:] = np.transpose(modes_filt_curr)

pos_spectrum_percentiles = [[np.percentile(np.absolute(pos_modes_edge_window_runs[scan_index]), pc, axis=1)\
                           for scan_index, scan_param in enumerate(scan_settings)] for pc in percentiles]
    
width_spectrum_percentiles = [[np.percentile(np.absolute(width_modes_edge_window_runs[scan_index]), pc, axis=1)\
                               for scan_index, scan_param in enumerate(scan_settings)] for pc in percentiles]

total_osc_edge_percentiles = [[None for _ in range(2)] for pc in percentiles]
stab_osc_edge_percentiles = [[None for _ in range(2)] for pc in percentiles]


for pc_index, pc in enumerate(percentiles):
    total_osc_edge_percentiles[pc_index][0] = \
        [np.sum(pos_spectrum_percentiles[pc_index][scan_index]) for scan_index, scan_param in enumerate(scan_settings)]
        
    total_osc_edge_percentiles[pc_index][1] = \
        [np.sum(width_spectrum_percentiles[pc_index][scan_index]) for scan_index, scan_param in enumerate(scan_settings)]
        
    stab_osc_edge_percentiles[pc_index][0] = \
        [np.sum(pos_spectrum_percentiles[pc_index][scan_index][stab_harms])\
         for scan_index, scan_param in enumerate(scan_settings)]
            
    stab_osc_edge_percentiles[pc_index][1] = \
        [np.sum(width_spectrum_percentiles[pc_index][scan_index][stab_harms])\
         for scan_index, scan_param in enumerate(scan_settings)]

total_osc_filt_percentiles = [[None for _ in enumerate(analysis['fs_harms'])] for pc in percentiles]
stab_osc_filt_percentiles = [[None for _ in enumerate(analysis['fs_harms'])] for pc in percentiles]

for fs_harm_index, fs_harm in enumerate(analysis['fs_harms']):
    spectrum_filt_percentiles =\
        [[np.percentile(np.absolute(modes_filt_window_runs[scan_index][:,:,fs_harm_index]), pc, axis=1) \
            for scan_index, scan_param in enumerate(scan_settings)] for pc in percentiles]
    
    for pc_index, pc in enumerate(percentiles):
        total_osc_filt_percentiles[pc_index][fs_harm_index] = \
            [np.sum(spectrum_filt_percentiles[pc_index][scan_index])\
             for scan_index, scan_param in enumerate(scan_settings)]
        stab_osc_filt_percentiles[pc_index][fs_harm_index] = \
            [np.sum(spectrum_filt_percentiles[pc_index][scan_index][stab_harms])\
             for scan_index, scan_param in enumerate(scan_settings)]
    
    plt.figure('mean_mode_spectrum_filt_vs_gain', figsize=(10,10))
    modes = np.arange(analysis['N_buckets_fft'])
    for scan_index, scan_param in enumerate(scan_settings):
        plt.plot(modes, spectrum_filt_percentiles[1][scan_index], '.-', \
                 label=scan.scan_point_label(scan_param_names, scan_param))
        plt.fill_between(modes, spectrum_filt_percentiles[0][scan_index], spectrum_filt_percentiles[2][scan_index],
                alpha=0.2, antialiased=True)
    plt.xlabel("Mode")
    plt.ylabel("Amplitude [s]")
    plt.title('Mode spectrum, filter method, fs * ' + str(fs_harm) +\
              ' component, \n averaged over shots')
    plt.legend(loc=0, fontsize='medium')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(output_dir + '/mean_mode_spectrum_filt_vs_gain_' + str(fs_harm) + 'fs.png')
    plt.rc('font', size=16)
    plt.show()
    plt.close()

modes = np.arange(analysis['N_buckets_fft'])

plt.figure('mean_pos_mode_spectrum_edge_vs_gain', figsize=(10,10))
for scan_index, scan_param in enumerate(scan_settings):
    plt.plot(modes, pos_spectrum_percentiles[1][scan_index], '.-', \
             label=scan.scan_point_label(scan_param_names, scan_param))
    plt.fill_between(modes, pos_spectrum_percentiles[0][scan_index], pos_spectrum_percentiles[2][scan_index],
            alpha=0.2, antialiased=True)
plt.xlabel("Mode")
plt.ylabel("Amplitude [s]")
plt.title('Bunch position oscillation, edge method,\naveraged over shots')
plt.legend(loc=0, fontsize='medium')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(output_dir + '/mean_pos_mode_spectrum_edge_vs_param.png')
plt.rc('font', size=16)
plt.show()
plt.close()

plt.figure('mean_width_mode_spectrum_edge_vs_gain', figsize=(10,10))
for scan_index, scan_param in enumerate(scan_settings):
    plt.plot(modes, width_spectrum_percentiles[1][scan_index], '.-',\
             label=scan.scan_point_label(scan_param_names, scan_param))
    plt.fill_between(modes, width_spectrum_percentiles[0][scan_index], width_spectrum_percentiles[2][scan_index],
            alpha=0.2, antialiased=True)
plt.xlabel("Mode")
plt.ylabel("Amplitude [s]")
plt.title('Bunch width oscillation, edge method,\naveraged over shots')
plt.legend(loc=0, fontsize='medium')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(output_dir + '/mean_width_mode_spectrum_edge_vs_param.png')
plt.rc('font', size=16)
plt.show()
plt.close()

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

        plt.figure('fs_harm_total_osc_vs_param', figsize=(10,10))
        for fs_harm_index, fs_harm in enumerate(analysis['fs_harms']):
            y_plot = [total_osc_filt_percentiles[1][fs_harm_index][i] for i in y_indices_plot]
            y_fill_l = [total_osc_filt_percentiles[0][fs_harm_index][i] for i in y_indices_plot]
            y_fill_u = [total_osc_filt_percentiles[2][fs_harm_index][i] for i in y_indices_plot]
            
            plt.plot(x_axis, y_plot, '.-', label='fs * ' + str(fs_harm))
            plt.fill_between(x_axis, y_fill_l, y_fill_u, alpha=0.2, antialiased=True)
        plt.xlabel(x_param_name)
        plt.ylabel('Total amplitude across all modes, filter method')
        plt.title(labels[x_param_index][plot_index])
        plt.legend(loc=0, fontsize='medium')
        plt.savefig(output_dir + '/fs_harm_total_osc_vs_param_' + str(x_param_index) +\
                            '_plot_' + str(plot_index) + '.png')
        plt.rc('font', size=16)
        plt.show()
        plt.close()

        plt.figure('fs_harm_stab_harms_osc_vs_param', figsize=(10,10))
        for fs_harm_index, fs_harm in enumerate(analysis['fs_harms']):
            y_plot = [stab_osc_filt_percentiles[1][fs_harm_index][i] for i in y_indices_plot]
            y_fill_l = [stab_osc_filt_percentiles[0][fs_harm_index][i] for i in y_indices_plot]
            y_fill_u = [stab_osc_filt_percentiles[2][fs_harm_index][i] for i in y_indices_plot]
            
            plt.plot(x_axis, y_plot, '.-', label='fs * ' + str(fs_harm))
            plt.fill_between(x_axis, y_fill_l, y_fill_u, alpha=0.2, antialiased=True)
        plt.xlabel(x_param_name)
        plt.ylabel('Total amplitude across modes ' + str(stab_harms) + ', filter method')
        plt.title(labels[x_param_index][plot_index])
        plt.legend(loc=0, fontsize='medium')
        plt.savefig(output_dir + '/fs_harm_stab_osc_vs_param_' + str(x_param_index) +\
                            '_plot_' + str(plot_index) + '.png')
        plt.rc('font', size=16)
        plt.show()
        plt.close()


                    
        plt.figure('edge_total_osc_vs_param', figsize=(10,10))
        
        y_plot = [total_osc_edge_percentiles[1][0][i] for i in y_indices_plot]
        y_fill_l = [total_osc_edge_percentiles[0][0][i] for i in y_indices_plot]
        y_fill_u = [total_osc_edge_percentiles[2][0][i] for i in y_indices_plot]
        
        plt.plot(x_axis, y_plot, '.-', label='Position')
        plt.fill_between(x_axis, y_fill_l, y_fill_u, alpha=0.2, antialiased=True)
        
        y_plot = [total_osc_edge_percentiles[1][1][i] for i in y_indices_plot]
        y_fill_l = [total_osc_edge_percentiles[0][1][i] for i in y_indices_plot]
        y_fill_u = [total_osc_edge_percentiles[2][1][i] for i in y_indices_plot]
        
        plt.plot(x_axis, y_plot, '.-', label='Width')
        plt.fill_between(x_axis, y_fill_l, y_fill_u, alpha=0.2, antialiased=True)
        plt.xlabel(x_param_name)
        plt.ylabel('Total amplitude across all modes, edge method')
        plt.title(labels[x_param_index][plot_index])
        plt.legend(loc=0, fontsize='medium')
        plt.savefig(output_dir + '/edge_total_osc_vs_param_' + str(x_param_index) +\
                            '_plot_' + str(plot_index) + '.png')
        plt.rc('font', size=16)
        plt.show()
        plt.close()


        plt.figure('edge_stab_harms_osc_vs_param', figsize=(10,10))
        
        y_plot = [stab_osc_edge_percentiles[1][0][i] for i in y_indices_plot]
        y_fill_l = [stab_osc_edge_percentiles[0][0][i] for i in y_indices_plot]
        y_fill_u = [stab_osc_edge_percentiles[2][0][i] for i in y_indices_plot]
        
        plt.plot(x_axis, y_plot, '.-', label='Position')
        plt.fill_between(x_axis, y_fill_l, y_fill_u, alpha=0.2, antialiased=True)
        
        y_plot = [stab_osc_edge_percentiles[1][1][i] for i in y_indices_plot]
        y_fill_l = [stab_osc_edge_percentiles[0][1][i] for i in y_indices_plot]
        y_fill_u = [stab_osc_edge_percentiles[2][1][i] for i in y_indices_plot]
        
        plt.plot(x_axis, y_plot, '.-', label='Width')
        plt.fill_between(x_axis, y_fill_l, y_fill_u, alpha=0.2, antialiased=True)
        plt.xlabel(x_param_name)
        plt.ylabel('Total amplitude across modes ' + str(stab_harms) + ', edge method')
        plt.title(labels[x_param_index][plot_index])
        plt.legend(loc=0, fontsize='medium')
        plt.savefig(output_dir + '/edge_stab_osc_vs_param_' + str(x_param_index) +\
                            '_plot_' + str(plot_index) + '.png')
        plt.rc('font', size=16)
        plt.show()
        plt.close()

#Scatter plot comparing oscillation modes obtained using edge and filter methods:
scatter_pos_edge = np.concatenate([np.ravel(np.absolute(\
       pos_modes_edge_window_runs[scan_index]))\
        for scan_index, scan_param in enumerate(scan_settings)])    
        
scatter_pos_filt = np.concatenate([np.ravel(np.absolute(\
       modes_filt_window_runs[scan_index][:,:,0]))\
        for scan_index, scan_param in enumerate(scan_settings)])

scatter_width_edge = np.concatenate([np.ravel(np.absolute(\
       width_modes_edge_window_runs[scan_index]))\
        for scan_index, scan_param in enumerate(scan_settings)])    
        
scatter_width_filt = np.concatenate([np.ravel(np.absolute(\
       modes_filt_window_runs[scan_index][:,:,1]))\
        for scan_index, scan_param in enumerate(scan_settings)])
   
slib.gpr_plot([scatter_pos_edge, scatter_width_edge],\
         [scatter_pos_filt, scatter_width_filt],\
         'Edge method amplitude [s]',\
         'Filter method amplitude',\
         plot_labels=['Bunch pos. vs. 1*fs', 'Bunch width vs. 2*fs'],\
         title='Estimated mode amplitudes',\
         filename=output_dir + '/filter_vs_edge.png')
    
tomo_statistics_data = {'edge_method_pos_modes' : pos_modes_edge_window_runs,
                        'edge_method_width_modes' : width_modes_edge_window_runs,
                        'filter_method_modes' : modes_filt_window_runs}

with open(output_dir + '/tomo_filt_results.pickle', 'wb') as f:
    pickle.dump(tomo_statistics_data, f)