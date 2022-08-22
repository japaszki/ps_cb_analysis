# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:10:03 2022

@author: Jan Cheeky Pea
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import os
import yaml
import pickle
import cbfb_analysis_lib as clib
import stats_lib as slib
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
file = yaml_params['acq_file']
fs_fit_params = yaml_params['fs_fit']
        
#Create directory to store results:
output_dir = working_dir + '/acq_outputs/'
try:
    print("Creating directory " + output_dir)
    os.makedirs(output_dir)
except:
    pass

#Specify output dir for plots:
file['output_dir'] = output_dir
    
spectrogram_f_lims = [-2000, 2000]
spectrogram_t_lims = [2000, 2810]
spectrogram_channel = 1

t_slice = 0.3
N_fft = 256
raw_plots = False

data_types_plot = ['Dipole', 'Quad']
percentiles = [25, 50, 75]
  
#Run one spectrogram with plots:
[dipole_cmplx, quad_cmplx] = clib.import_data(working_dir + '/' + file['file_base']\
                   + str(file_indices[0][0]), file['num_buffers'], len(file['channel_harmonics']))
[f, t, s_dipole, s_quad] = clib.get_spectrogram(dipole_cmplx, quad_cmplx, N_fft, file['acq_start_time'], file['f_samp'], True)

#Arrays for complex IQ data:
cmplx_runs = {}
cmplx_runs['Dipole'] = [[None for _ in enumerate(file_indices[scan_index])] for scan_index, scan_param in enumerate(scan_settings)]
cmplx_runs['Quad'] = [[None for _ in enumerate(file_indices[scan_index])] for scan_index, scan_param in enumerate(scan_settings)]

#Arrays for spectrograms:
s_dipole_runs = [[np.zeros([s_dipole[0].shape[0], s_dipole[0].shape[1], len(file_indices[scan_index])])\
                 for channel_index, channel_harmonic in enumerate(file['channel_harmonics'])]\
                 for scan_index, scan_param in enumerate(scan_settings)]
s_quad_runs = [[np.zeros([s_dipole[0].shape[0], s_dipole[0].shape[1], len(file_indices[scan_index])])\
                for channel_index, channel_harmonic in enumerate(file['channel_harmonics'])]\
                for scan_index, scan_param in enumerate(scan_settings)]

for scan_index, scan_param in enumerate(scan_settings):
    for file_index, file_suffix in enumerate(file_indices[scan_index]):
        filename = working_dir + '/' + file['file_base'] + str(file_suffix)
        for channel_index, channel_harmonic in enumerate(file['channel_harmonics']):
            [cmplx_runs['Dipole'][scan_index][file_index], cmplx_runs['Quad'][scan_index][file_index]] = \
                clib.import_data(filename, file['num_buffers'], len(file['channel_harmonics']))
                
            [_, _, s_dipole, s_quad] = clib.get_spectrogram(cmplx_runs['Dipole'][scan_index][file_index], \
                cmplx_runs['Quad'][scan_index][file_index], N_fft, file['acq_start_time'], file['f_samp'], False)
            
            s_dipole_runs[scan_index][channel_index][:,:,file_index] = s_dipole[channel_index]
            s_quad_runs[scan_index][channel_index][:,:,file_index] = s_quad[channel_index]

#Calculate mean of spectrograms over runs:
s_dipole_runs_avg = [[np.mean(s_dipole_runs[scan_index][channel], axis=2) for channel in range(len(file['channel_harmonics']))] \
                        for scan_index, scan_param in enumerate(scan_settings)]
s_quad_runs_avg = [[np.mean(s_quad_runs[scan_index][channel], axis=2) for channel in range(len(file['channel_harmonics']))] \
                      for scan_index, scan_param in enumerate(scan_settings)]

#Plot averaged spectrograms:
f_indices = (f > spectrogram_f_lims[0]) & (f <= spectrogram_f_lims[1])
t_indices = (t > spectrogram_t_lims[0]) & (t <= spectrogram_t_lims[1])

for channel_index, channel_harmonic in enumerate(file['channel_harmonics']):
    s1 = s_dipole_runs_avg[0][channel_index][f_indices, :]
    s2 = s1[:, t_indices]
    plt.figure(figsize=(10,10)) 
    plt.pcolormesh(t[t_indices], f[f_indices], 10*np.log10(s2), cmap='hot', shading='flat')
    plt.title('Dipole h' + str(channel_harmonic) + ' , averaged over shots')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time after acq start [ms]')
    plt.rc('font', size=16)
    plt.savefig(output_dir + '/dipole_spectrogram_mean_h' + str(channel_harmonic) + '.png')
    plt.show()
        
    s1 = s_quad_runs_avg[0][channel_index][f_indices, :]
    s2 = s1[:, t_indices]
    plt.figure(figsize=(10,10)) 
    plt.pcolormesh(t[t_indices], f[f_indices], 10*np.log10(s2), cmap='hot', shading='flat')
    plt.title('Quad h' + str(channel_harmonic) + ' , averaged over shots')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time after acq start [ms]')
    plt.rc('font', size=16)
    plt.savefig(output_dir + '/quad_spectrogram_mean_h' + str(channel_harmonic) + '.png')
    plt.show()
    
    #Plot on and off spectra for one time slice:
    #Find t-index closest to specified slice time:
    t_slice_index = np.argmin(abs(t - t_slice))
    plt.figure(figsize=(10,10))
        
    for scan_index, scan_param in enumerate(scan_settings):
        plt.plot(f[f_indices], 10*np.log10(s_dipole_runs_avg[scan_index][channel_index][f_indices, t_slice_index]), \
                 label=scan.scan_point_label(scan_param_names, scan_param))
    plt.xlabel('Baseband frequency [Hz]')
    plt.ylabel('PSD [dB]')
    plt.title('Dipole h ' + str(channel_harmonic))
    plt.legend(loc=0, fontsize='medium')
    plt.rc('font', size=16)
    plt.show()
    
    plt.figure(figsize=(10,10)) 
    for scan_index, scan_param in enumerate(scan_settings):
        plt.plot(f[f_indices], 10*np.log10(s_quad_runs_avg[scan_index][channel_index][f_indices, t_slice_index]), \
                 label=scan.scan_point_label(scan_param_names, scan_param))
    plt.xlabel('Baseband frequency [Hz]')
    plt.ylabel('PSD [dB]')
    plt.title('Quad h ' + str(channel_harmonic))
    plt.legend(loc=0, fontsize='medium')
    plt.rc('font', size=16)
    plt.show()
    
    
#Generate parameters to pass to fitting function:
fs_params = {'f_samp' : file['f_samp'],\
             'fmax' : fs_fit_params['f_lim'],\
             't_half_span' : fs_fit_params['t_half_span'],\
             't_centres' : np.linspace(fs_fit_params['t_start'], fs_fit_params['t_end'],\
                                       fs_fit_params['t_N_samp']),\
             'N_harmonics_fit' : fs_fit_params['N_harms_fit'],\
             'N_harmonics_samp' : 2*max(analysis['fs_harms'])+1, \
             'samp_half_span' : fs_fit_params['samp_f_half_span']}
    
fs_init = {'amps' : fs_fit_params['init_amp'] * np.ones(fs_params['N_harmonics_fit']),\
           'width' : fs_fit_params['init_width']}

fs_bounds = {'fs_max_func' : scipy.interpolate.interp1d(fs_fit_params['fs_max_times'], fs_fit_params['fs_max_vals']),\
             'fs_min_func' : scipy.interpolate.interp1d(fs_fit_params['fs_min_times'], fs_fit_params['fs_min_vals']),\
             'max_amp' : fs_fit_params['fs_max_amp'],\
             'min_amp' : fs_fit_params['fs_min_amp'],\
             'min_width' : fs_fit_params['fs_min_width']}    
    
#Perform fit of fs:
    
#Assemble chosen shots into array:
fs_fit_list = []
for data_type in fs_fit_params['data_types']:
    for shot_index in fs_fit_params['shot_indices']:
        for channel_index in fs_fit_params['channel_indices']:
            fs_fit_list.append(cmplx_runs[data_type][fs_fit_params['scan_index']][shot_index][channel_index])
        
fs_fit_data = np.array(fs_fit_list)
fs_fit = clib.fs_sidebands(fs_params, fs_init, fs_bounds, file['acq_start_time'], file['f_samp'], fs_fit_data, True)

#Find synchrotron frequency vs time:
plt.figure(figsize=(10,10)) 
plt.plot(fs_params['t_centres'], fs_fit)
plt.plot(fs_params['t_centres'], fs_bounds['fs_max_func'](fs_params['t_centres']), '--')
plt.plot(fs_params['t_centres'], fs_bounds['fs_min_func'](fs_params['t_centres']), '--')
plt.xlabel('Time [ms]')
plt.ylabel('f_s [Hz]')
plt.title('Estimated synchrotron frequency')
plt.rc('font', size=16)
plt.savefig(output_dir + '/fs_vs_time_fit.png')
plt.show()
    
fb_on_h_vs_gain = [{key : np.zeros([len(file['channel_harmonics']),\
                                    len(scan_settings),\
                                    len(analysis['fs_harms'])])\
                    for key in data_types_plot}\
                    for pc in percentiles]
    
tomo_h_runs = {key : [np.zeros([len(file['channel_harmonics']), \
                               len(file_indices[scan_index]), \
                               len(analysis['fs_harms'])], dtype=float) \
                     for scan_index, scan_param in enumerate(scan_settings)]\
                     for key in data_types_plot}

h_amps_all = {key : [[] for h_index, h in enumerate(analysis['fs_harms'])] for key in data_types_plot}

for data_type in data_types_plot:
    # channel_index = 1
    # fmax = 1250
    # t_centre = 2080
    # t_half_span = 40
    
    # gain_index = 0
    # run_index = 5
    # clib.plot_zoom(cmplx_runs[data_type][gain_index][run_index][channel_index], [2100, 2115], [-fmax, fmax],\
    #                file['acq_start_time'], file['f_samp'], title='')
    # clib.plot_zoom(cmplx_runs[data_type][gain_index][run_index][channel_index], [2700, 2715], [-fmax, fmax],\
    #                file['acq_start_time'], file['f_samp'], title='')
       
    # gain_index = 1
    # run_index = 5
    # clib.plot_zoom(cmplx_runs[data_type][gain_index][run_index][channel_index], [2100, 2115], [-fmax, fmax],\
    #                file['acq_start_time'], file['f_samp'], title='')
    # clib.plot_zoom(cmplx_runs[data_type][gain_index][run_index][channel_index], [2700, 2715], [-fmax, fmax],\
    #                file['acq_start_time'], file['f_samp'], title='')
        
    for channel_index, channel_harmonic in enumerate(file['channel_harmonics']):
        amps_samp_runs = [np.zeros([fs_params['N_harmonics_samp'],\
                                    len(fs_params['t_centres']),\
                                    len(cmplx_runs[data_type][scan_index])])\
                             for scan_index, scan_param in enumerate(scan_settings)]
            
        for scan_index, scan_param in enumerate(scan_settings):
            for run_index, run_data in enumerate(cmplx_runs[data_type][scan_index]):
                amps_samp_runs[scan_index][:,:,run_index] = \
                    clib.sample_fs_harmonics(fs_params, fs_fit, file['acq_start_time'],\
                                             file['f_samp'], run_data[channel_index])
                
                plt.figure(figsize=(10,10)) 
                for h_index, h in enumerate(analysis['fs_harms']):
                    plt.plot(fs_params['t_centres'], \
                        amps_samp_runs[scan_index][2*h-1,:,run_index] + amps_samp_runs[scan_index][2*h,:,run_index],\
                        label = str(h) + '*f_s')
        
                plt.axvline(file['fb_start_time'], color='b', linestyle='--', label='Feedback On')
                plt.axvline(file['fb_end_time'], color='k', linestyle='--', label='Feedback Off')
                plt.legend(loc=0, fontsize='medium')
                plt.xlabel('Time [ms]')
                plt.ylabel('Amplitude [counts]')
                plt.title(data_type + ' h' + str(file['channel_harmonics'][channel_index]) + ', ' +\
                          scan.scan_point_label(scan_param_names, scan_param) +\
                          ', shot ' + str(run_index))
                plt.rc('font', size=16)
                plt.show()
            
        #Plot statistics of fs harmonics vs time:
        for scan_index, scan_param in enumerate(scan_settings):
            
            plt.figure(figsize=(10,10)) 
            for h_index, h in enumerate(analysis['fs_harms']):
                data_ul = amps_samp_runs[scan_index][2*h-1,:,:] + amps_samp_runs[scan_index][2*h,:,:]
                plt.plot(fs_params['t_centres'], np.percentile(data_ul, 50, axis=1), label = str(h) + '*f_s')
                plt.fill_between(fs_params['t_centres'], np.percentile(data_ul, 25, axis=1),\
                                 np.percentile(data_ul, 75, axis=1), alpha=0.2, antialiased=True)
            plt.axvline(file['fb_start_time'], color='b', linestyle='--', label='Feedback On')
            plt.axvline(file['fb_end_time'], color='k', linestyle='--', label='Feedback Off')
            plt.legend(loc=0, fontsize='medium')
            plt.xlabel('Time [ms]')
            plt.ylabel('Amplitude [counts]')
            plt.title(data_type + ' h' + str(channel_harmonic) + ', ' + \
                      scan.scan_point_label(scan_param_names, scan_param) + ', statistics')
            plt.rc('font', size=16)
            plt.savefig(output_dir + '/' + data_type + '_fs_harmonics_stats_scan_index_' + str(scan_index) +\
                        '_h' + str(channel_harmonic) + '.png')
            plt.show()
            
        #Take mean of each fs harmonic between fb on and off times:
        fb_on_time_indices = (fs_params['t_centres'] >= file['fb_start_time']) &\
            (fs_params['t_centres'] < file['fb_end_time'])
        tomo_ref_time_indices = (fs_params['t_centres'] >= file['tomo_ref_start_time']) &\
            (fs_params['t_centres'] < file['tomo_ref_end_time'])
        
        
        for h_index, h in enumerate(analysis['fs_harms']):
            for scan_index, scan_param in enumerate(scan_settings):
                #Sum together the upper and lower sideband magnitudes of the specifiec fs harmonic:
                data_ul = amps_samp_runs[scan_index][2*h-1,:,:] + amps_samp_runs[scan_index][2*h,:,:]
                
                #Save all individual data points for plotting later:
                h_amps_all[data_type][h_index].append(data_ul)
                
                #Take the mean of each run over the specified time window:
                tomo_mean = np.mean(data_ul[tomo_ref_time_indices,:], axis=0)
                tomo_h_runs[data_type][scan_index][channel_index,:,h_index] = tomo_mean
                
                #Take the mean of each run over the feedback on time window:
                fb_on_mean = np.mean(data_ul[fb_on_time_indices,:], axis=0)
                        
                #Take quartiles of window mean over runs:
                for pc_index, pc in enumerate(percentiles):
                    fb_on_h_vs_gain[pc_index][data_type][channel_index,scan_index,h_index] =\
                        np.percentile(fb_on_mean, pc)
            
        [x_axes, y_indices, labels] = scan.gen_1d_scans(scan_settings, scan_param_names)
        
        #Set of plots with each parameter as the x-axis:
        for x_param_index, x_param_name in enumerate(scan_param_names):
            #list of remaining parameters not acting as x-axis:
            rem_param_indices = [param_index for param_index, param_name in enumerate(scan_param_names)\
                                 if param_index != x_param_index]   
             
            #one plot for each value of remaining parameters:
            for plot_index, x_axis in enumerate(x_axes[x_param_index]):
                
                plt.figure(figsize=(10,10)) 
                for h_index, h in enumerate(analysis['fs_harms']):                           
                    y_plot = fb_on_h_vs_gain[1][data_type][channel_index, y_indices[x_param_index][plot_index], h_index]
                    y_fill_l = fb_on_h_vs_gain[0][data_type][channel_index, y_indices[x_param_index][plot_index], h_index]
                    y_fill_u = fb_on_h_vs_gain[2][data_type][channel_index, y_indices[x_param_index][plot_index], h_index]
                        
                    plt.plot(x_axis, y_plot, '.-', label = str(h) + '*f_s')
                    plt.fill_between(x_axis, y_fill_l, y_fill_u, alpha=0.2, antialiased=True)
                    
                plt.xlabel(x_param_name)
                plt.ylabel('Mean amplitude between ' + str(file['fb_start_time']) + ' ms and ' +\
                           str(file['fb_end_time']) + ' ms')
                plt.legend(loc=0, fontsize='medium')
                plt.title(data_type + ' h' + str(channel_harmonic) + ' fs harmonics vs ' + x_param_name + ', ' +\
                          labels[x_param_index][plot_index])
                plt.rc('font', size=16)
                plt.savefig(output_dir + '/' + data_type + '_fs_harmonics_vs_param_' + str(x_param_index) +\
                            '_plot_' + str(plot_index) + '_h' + str(channel_harmonic) + '.png')
                plt.show()
        
h_amps_all_1d = {key : [np.concatenate([np.ravel(h_amps_all[key][h_index][index])\
                 for index, data in enumerate(h_amps_all[key][h_index])])
                 for h_index, h in enumerate(analysis['fs_harms'])]\
                 for key in data_types_plot}

slib.scatter_plot_with_linreg([h_amps_all_1d['Dipole'][h_index] for h_index, h in enumerate(analysis['fs_harms'])],\
                              [h_amps_all_1d['Quad'][h_index] for h_index, h in enumerate(analysis['fs_harms'])],\
                              'Dipole acq amplitude',\
                              'Quad acq amplitude',\
                               plot_labels=[str(h) + '*f_s' for h_index, h in enumerate(analysis['fs_harms'])],\
                               alpha=0.3,\
                               title='Instantaneous amplitudes',\
                               filename=output_dir + '/fs_harmonics_instantaneous_dipole_vs_quad.png')

slib.scatter_plot_with_linreg([h_amps_all_1d['Dipole'][analysis['fs_harms'].index(2)]],\
         [h_amps_all_1d['Dipole'][analysis['fs_harms'].index(4)]],\
         '2*f_s amplitude',\
         '4*f_s amplitude',\
         plot_labels=None,\
         alpha=0.3,\
         title='Instantaneous dipole acq amplitudes',\
         filename=output_dir + '/dipole_instantaneous_2fs_vs_4fs.png')

slib.scatter_plot_with_linreg([h_amps_all_1d['Quad'][analysis['fs_harms'].index(2)]],\
         [h_amps_all_1d['Quad'][analysis['fs_harms'].index(4)]],\
         '2*f_s amplitude',\
         '4*f_s amplitude',\
         plot_labels=None,\
         alpha=0.3,\
         title='Instantaneous quad acq amplitudes',\
         filename=output_dir + '/quad_instantaneous_2fs_vs_4fs.png')


acq_statistics_data = {'data_types' : data_types_plot,
                       'tomo_runs' : tomo_h_runs}
 
with open(output_dir + '/acq_results.pickle', 'wb') as f:
    pickle.dump(acq_statistics_data, f)