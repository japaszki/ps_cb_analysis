# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:00:26 2022

@author: JohnG
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def tomo_bucket(tomo_data, tomo_params, bucket_params, dipole_freq, fs_harm, plots, plot_name):
    #Plot tomoscope data of single bucket
    time_indices = (bucket_params['T_vec'] >= \
                    bucket_params['T_start'] + bucket_params['bucket']*bucket_params['T_bucket']) & \
                    (bucket_params['T_vec'] < bucket_params['T_start'] + (bucket_params['bucket']+1)*bucket_params['T_bucket'])
    N_frames = tomo_data.shape[0]
    N_bins = tomo_data.shape[1]
    bucket_tomo = tomo_data[tomo_params['start_frame']:tomo_params['end_frame'], time_indices]
    bucket_tomo_trans = np.transpose(bucket_tomo)
    
    bunch_peak_index = np.argmax(np.mean(bucket_tomo_trans, axis=1))
    
    turns = np.arange(0, N_frames*tomo_params['delta_turns'], tomo_params['delta_turns'])
    samples = np.arange(0, (N_bins-0.5)*tomo_params['dt_samp'], tomo_params['dt_samp'])
    samples_plot = samples[time_indices]
    
    if plots:
        plot_y, plot_x = np.meshgrid(turns[tomo_params['start_frame']:tomo_params['end_frame']], samples_plot)
        plt.figure(figsize=(10,10))
        plt.pcolormesh(plot_x, plot_y, bucket_tomo_trans, cmap='hot', shading='auto')
        plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
        plt.xlabel('Time [s]')
        plt.ylabel('Turn')
        plt.title(plot_name + ', bucket ' + str(bucket_params['bucket']))
        plt.colorbar()
        plt.rc('font', size=16)
        plt.savefig('tomogram_zoom.png')
        plt.show()
        plt.close()
    
    #Perform FFT on each tomogram sample:
    N_samples = bucket_tomo_trans.shape[0]
    N_data = bucket_tomo_trans.shape[1]
    N_fft = len(np.fft.fft(bucket_tomo_trans[0,:]))
    N_rfft = len(np.fft.rfft(bucket_tomo_trans[0,:]))
    
    fft_data_nowin = np.zeros([N_samples, N_fft], dtype='complex')
    fft_data_win = np.zeros([N_samples, N_fft], dtype='complex')
    rfft_data_win = np.zeros([N_samples, N_rfft], dtype='complex')
    for i in range(N_samples):
        fft_data_nowin[i,:] = np.fft.fft(bucket_tomo_trans[i,:])
        fft_data_win[i,:] = np.fft.fft(bucket_tomo_trans[i,:] * np.hanning(N_data))
        rfft_data_win[i,:] = np.fft.rfft(bucket_tomo_trans[i,:] * np.hanning(N_data))
    
    turn_span = (tomo_params['end_frame'] - tomo_params['start_frame']) * tomo_params['delta_turns']
    df = 1 / turn_span
    freqscale = np.arange(N_fft) * df
    freqscale_rfft = np.arange(N_rfft) * df
    
    if plots:
        plt.figure('sample_by_sample_fftzoom', figsize=(10,10))  
        plot_y, plot_x = np.meshgrid(freqscale, samples_plot)
        plt.pcolormesh(plot_x, plot_y, np.log10(np.abs(fft_data_win)), cmap='hot', shading='auto')
        plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [1/turn]')
        plt.title('Sample-by-sample FFT, bucket ' + str(bucket_params['bucket']) + ', dB')
        plt.colorbar()
        plt.rc('font', size=16)
        plt.savefig(plot_name + '_sample_by_sample_fft_zoom.png')
        plt.show()
        plt.close()

    #Gaussian filter specification:
    filt_width = dipole_freq/4
    filt_f0 = dipole_freq*fs_harm
    freq_scaling = np.exp(-2*np.power(freqscale-filt_f0, 2)/filt_width**2)\
        + np.exp(-2*np.power(np.flip(freqscale)-filt_f0, 2)/filt_width**2)
        
    freq_scaling_rfft = np.exp(-2*np.power(freqscale_rfft-filt_f0, 2)/filt_width**2)
            
    #Perform inverse FFT on modified spectrum:
    ifft_data = np.zeros([N_samples, N_data])
    for i in range(N_samples):
        ifft_data[i,:] = np.real(np.fft.ifft(fft_data_nowin[i,:] * freq_scaling, N_data))
    
    if plots:
        plot_y, plot_x = np.meshgrid(turns[tomo_params['start_frame']:tomo_params['end_frame']], samples_plot)
        plt.figure(figsize=(10,10)) 
        plt.pcolormesh(plot_x, plot_y, ifft_data, cmap='hot', shading='auto')
        plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
        plt.xlabel('Time [s]')
        plt.ylabel('Turn')
        plt.title(plot_name + ', bucket ' + str(bucket_params['bucket']) + ', fs * ' + str(fs_harm) + ' component')
        plt.colorbar()
        plt.rc('font', size=16)
        plt.savefig(plot_name + '_tomogram_bucket ' + str(bucket_params['bucket']) + '_fs_times_' + str(fs_harm) + '.png')
        plt.show()
        plt.close()
    
    peak_freq = np.zeros(N_samples)
    peak_cmplx = np.zeros(N_samples, dtype='complex')

    for i in range(N_samples):
        rfft_sample_scaled = rfft_data_win[i,:] * freq_scaling_rfft
        peak_index = np.argmax(np.abs(rfft_sample_scaled))
        peak_freq[i] = freqscale_rfft[peak_index]
        peak_cmplx[i] = rfft_sample_scaled[peak_index] / N_rfft
    
    if plots:
        plt.figure(figsize=(10,10)) 
        plt.plot(samples_plot, peak_freq)
        plt.xlabel('Time [s]')
        plt.ylabel('Fitted frequency [1/turn]')
        plt.title(plot_name + ', bucket ' + str(bucket_params['bucket']) + ', fs * ' + str(fs_harm) + ' component')
        plt.show()
        plt.close()
        
        plt.figure(figsize=(10,10)) 
        plt.plot(samples_plot, np.angle(peak_cmplx))
        plt.xlabel('Time [s]')
        plt.ylabel('Oscillation phase [rad]')
        plt.title(plot_name + ', bucket ' + str(bucket_params['bucket']) + ', fs * ' + str(fs_harm) + ' component')
        plt.show()
        plt.close()
        
        plt.figure(figsize=(10,10)) 
        plt.plot(samples_plot, np.abs(peak_cmplx))
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title(plot_name + ', bucket ' + str(bucket_params['bucket']) + ', fs * ' + str(fs_harm) + ' component')
        plt.show()
        plt.close()
    
    return [peak_freq, peak_cmplx, bunch_peak_index]

def fs_harm_mode_analysis(tomo_data, tomo_params, bucket_params, N_buckets, N_buckets_fft, dipole_freq, fs_harm, plots, plot_name):
    #Obtain FFT amplitude and phase at the specified fs harmonic, vs bucket and sample index.
    
    fs_harm_cmplx_vs_sample = [None for _ in range(N_buckets)]
    bunch_peak_sample = [None for _ in range(N_buckets)]
    
    for bucket in range(N_buckets):
        bucket_params['bucket'] = bucket
        
        [_, fs_harm_cmplx_vs_sample[bucket], bunch_peak_sample[bucket]] = \
            tomo_bucket(tomo_data, tomo_params, bucket_params, dipole_freq, fs_harm, False, '')
    
    #Scan sample offset relative to the bunch centre:
    fs_harm_cmplx = np.zeros(N_buckets, dtype='complex')
    offset_range = range(-15, 16)
    total_amp_vs_offset = np.zeros(len(offset_range))
    mode_spectrum_vs_offset = [None for _ in enumerate(offset_range)]
    
    for offset_index, offset in enumerate(offset_range):
        for bucket in range(N_buckets):          
            fs_harm_cmplx[bucket] = fs_harm_cmplx_vs_sample[bucket][bunch_peak_sample[bucket]+offset]
            
        #Pad complex amplitudes to number of buckets:
        fs_harm_cmplx_padded = np.zeros(N_buckets_fft, dtype='complex')
        fs_harm_cmplx_padded[0:N_buckets] = fs_harm_cmplx
        
        #Total oscillation amplitude over all bunches for this sample offset:
        total_amp_vs_offset[offset_index] = np.sum(np.abs(fs_harm_cmplx_padded))
        
        #Calculate mode spectrum using FFT of complex amplitude vs bucket:
        mode_spectrum_vs_offset[offset_index] = np.fft.fft(fs_harm_cmplx_padded)/N_buckets_fft
    
    #Pick sample offset which results in maximum total oscillation amplitude:
    optimum_offset = offset_range[np.argmax(total_amp_vs_offset)]
        
    if plots:
        
        # fs_harm_abs = np.abs(np.array(fs_harm_cmplx_vs_sample))
        # percentiles = [25, 50, 75]
        # fs_harm_abs_percentiles = [np.percentile(fs_harm_abs, pc, axis=1) for pc in percentiles]
        
        plt.figure(figsize=(10,10)) 
        for bucket in range(N_buckets):
            N_samples = fs_harm_cmplx_vs_sample[bucket].shape[0]
            plt.plot(np.arange(N_samples) - bunch_peak_sample[bucket], np.abs(fs_harm_cmplx_vs_sample[bucket]))
        plt.xlabel('Sample')
        plt.ylabel('Filtered amplitude [V]')
        plt.title(plot_name + ', all buckets, fs * ' + str(fs_harm) + ' component')
        plt.rc('font', size=16)
        plt.savefig(plot_name + '_amplitude_fs_times_' + str(fs_harm) + '.png')
        plt.show()
        plt.close()
        
        #Plot total mode oscillation amplitude vs sample offset:
        plt.figure(figsize=(10,10)) 
        plt.plot(offset_range, total_amp_vs_offset)
        plt.xlabel('Offset from bunch centre [samples]')
        plt.ylabel('Total amplitude [V]')
        plt.rc('font', size=16)
        plt.show()
        plt.close()
        
        #Plot mode spectrum at optimum offset:
        plt.figure(figsize=(10,10)) 
        plt.bar(np.arange(N_buckets_fft), np.abs(mode_spectrum_vs_offset[optimum_offset]))
        plt.xlabel('Mode')
        plt.ylabel('Mode amplitude [V]')
        plt.title('Mode spectrum at optimum offset = ' + str(optimum_offset))
        plt.rc('font', size=16)
        plt.show()
        plt.close()
        
    return mode_spectrum_vs_offset[optimum_offset]

def mode_analysis(bunch_pos, bunch_width, tomo_params, N_buckets, plots):
    #Calculate time span of FFT window and thus frequency resolution:
    turn_span = (tomo_params['end_frame'] - tomo_params['start_frame']) * tomo_params['delta_turns']
    df = 1 / turn_span
    
    N_bunches = bunch_pos.shape[0]
    
    #Subtract mean position from each bunch position:
    bunch_rel_pos = np.empty_like(bunch_pos[:,tomo_params['start_frame']:tomo_params['end_frame']])
    for bunch in range(N_bunches):
        bunch_rel_pos[bunch,:] = np.hanning(bunch_pos[bunch,tomo_params['start_frame']:tomo_params['end_frame']].shape[0]) * \
            (bunch_pos[bunch,tomo_params['start_frame']:tomo_params['end_frame']] - \
             np.mean(bunch_pos[bunch,tomo_params['start_frame']:tomo_params['end_frame']]))
    
    #Extend array to include empty buckets:
    bunch_rel_pos_padded = np.zeros([N_buckets, bunch_rel_pos.shape[1]])
    bunch_rel_pos_padded[0:bunch_rel_pos.shape[0], :] = bunch_rel_pos
    
    #Calculate normalised 2D FFT:
    fft_pos = np.fft.rfft2(bunch_rel_pos_padded) * 4 / np.size(bunch_rel_pos_padded)
    fft_freq = np.arange(fft_pos.shape[1]) * df  

    #Find frequency peak, estimate of synchrotron frequency:
    max_pos_freq_index = np.argmax(np.mean(np.abs(fft_pos), axis=0))
    pos_mode_spectrum = fft_pos[:, max_pos_freq_index]
    
    #Subtract mean width from each bunch width:
    bunch_rel_width = np.empty_like(bunch_width[:,tomo_params['start_frame']:tomo_params['end_frame']])
    for bunch in range(N_bunches):
        bunch_rel_width[bunch,:] = np.hanning(bunch_width[bunch,tomo_params['start_frame']:tomo_params['end_frame']].shape[0]) * \
            (bunch_width[bunch,tomo_params['start_frame']:tomo_params['end_frame']] - \
             np.mean(bunch_width[bunch,tomo_params['start_frame']:tomo_params['end_frame']]))
                
    #Extend array to include empty buckets:
    bunch_rel_width_padded = np.zeros([N_buckets, bunch_rel_width.shape[1]])
    bunch_rel_width_padded[0:bunch_rel_width.shape[0], :] = bunch_rel_width  
    
    #Calculate normalised 2D FFT:
    fft_width = np.fft.rfft2(bunch_rel_width_padded) * 4 / np.size(bunch_rel_width_padded)

    #Find frequency peak, estimate of synchrotron frequency:
    max_width_freq_index = np.argmax(np.mean(np.abs(fft_width), axis=0))
    width_mode_spectrum = fft_width[:, max_width_freq_index]
    
    if plots:
        plt.figure('bunch_pos_img')
        plot_y, plot_x = np.meshgrid(tomo_params['delta_turns']*np.arange(tomo_params['start_frame'],\
                                                                          tomo_params['end_frame']), np.arange(N_bunches+1))
        plt.pcolormesh(plot_x, plot_y, bunch_rel_pos, cmap='hot', shading='auto')
        plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
        plt.title('Bunch position offset')
        plt.xlabel('Bunch')
        plt.ylabel('Turn')
        plt.colorbar()
        plt.rc('font', size=16)
        plt.savefig('bunch_pos_img.png')
        plt.show()
        plt.close()
        
        plt.figure('bunch_width_img')
        # plot_y, plot_x = np.meshgrid(delta_turns*np.arange(start_frame, end_frame), np.arange(N_bunches+1))
        plt.pcolormesh(plot_x, plot_y, bunch_rel_width, cmap='hot', shading='auto')
        plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
        plt.title('Bunch width offset')
        plt.xlabel('Bunch')
        plt.ylabel('Turn')
        plt.colorbar()
        plt.rc('font', size=16)
        plt.savefig('bunch_width_img.png')
        plt.show()
        plt.close()
        
        plt.figure('bunch_pos_2dfft')  
        plot_y, plot_x = np.meshgrid(fft_freq, np.arange(N_buckets+1))
        plt.pcolormesh(plot_x, plot_y, np.abs(fft_pos), cmap='hot', shading='auto')
        plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
        plt.xlabel('Mode')
        plt.ylabel('Frequency [1/turn]')
        plt.title('Bunch position oscillation')
        plt.colorbar()
        plt.rc('font', size=16)
        plt.savefig('bunch_pos_2dfft.png')
        plt.show()
        plt.close()
        
        plt.figure('bunch_width_2dfft')  
        plot_y, plot_x = np.meshgrid(fft_freq, np.arange(N_buckets+1))
        plt.pcolormesh(plot_x, plot_y, np.abs(fft_width), cmap='hot', shading='auto')
        plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
        plt.xlabel('Mode')
        plt.ylabel('Frequency [1/turn]')
        plt.title('Bunch width oscillation')
        plt.colorbar()
        plt.rc('font', size=16)
        plt.savefig('bunch_width_2dfft.png')
        plt.show()
        plt.close()
                      
        plt.figure('bunch_pos_fft')
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        ax.bar([x for x in range(N_buckets)], np.absolute(pos_mode_spectrum))
        ax.set_xlabel("Mode")
        ax.set_ylabel("Amplitude [s]")
        plt.title('Bunch position oscillation')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.savefig('bunch_pos_modes.png')
        plt.show()
        plt.close()
        
        plt.figure('bunch_width_fft')
        plt.bar([x for x in range(N_buckets)], np.absolute(width_mode_spectrum))
        plt.xlabel("Mode")
        plt.ylabel("Amplitude [s]")
        plt.title('Bunch width oscillation')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.savefig('bunch_width_modes.png')
        plt.show()
        plt.close()
        
    return [pos_mode_spectrum, width_mode_spectrum, fft_freq[max_pos_freq_index], fft_freq[max_width_freq_index]]
     
def modes_vs_time(bunch_pos, bunch_width, tomo_params, window, resolution, N_buckets, N_modes_plt):
    #Define time windows
    window_centres = np.round(np.arange(tomo_params['start_frame']+window/2, tomo_params['end_frame']-window/2, resolution))
    N_windows = window_centres.shape[0]

    #Calculate modes for each window
    N_bunches = bunch_pos.shape[0]
    pos_modes = np.empty([N_buckets, N_windows])
    width_modes = np.empty([N_buckets, N_windows])
    
    for i in range(N_windows):
        tomo_params_window = {'start_frame' : int(window_centres[i]-window/2),\
                              'end_frame' : int(window_centres[i]+window/2), \
                               'delta_turns' : tomo_params['delta_turns']}
        
        [pos_modes_curr, width_modes_curr, _, _] = \
            mode_analysis(bunch_pos, bunch_width, tomo_params_window, N_buckets, False)
            
        pos_modes[:,i] = np.absolute(pos_modes_curr)
        width_modes[:,i] = np.absolute(width_modes_curr)
    
            
    #Find dominant oscillation modes:
    pos_dom_modes = np.argsort(np.amax(pos_modes, axis=1))
    width_dom_modes = np.argsort(np.amax(width_modes, axis=1))
    
    #Plot mode amplitudes vs time:
    if N_modes_plt > 0:
        plt.figure('pos_modes_vs_turn')
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        for i in range(1,N_modes_plt+1):
            ax.plot(window_centres*tomo_params['delta_turns'], pos_modes[pos_dom_modes[-i], :],\
                    label = 'Mode ' + str(pos_dom_modes[-i]))
        ax.plot(window_centres*tomo_params['delta_turns'], \
                np.sum(pos_modes[pos_dom_modes[0:(N_bunches-N_modes_plt)], :], axis=0), \
                label = 'Remaining modes')
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mode magnitude [s]")
        plt.title('Bunch position modes')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend(loc=0, fontsize='medium')
        plt.savefig('pos_modes_vs_turn.png')
        plt.show()
        plt.close()
        
        plt.figure('width_modes_vs_turn')
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        for i in range(1,N_modes_plt+1):
            ax.plot(window_centres*tomo_params['delta_turns'], width_modes[width_dom_modes[-i], :], \
                    label = 'Mode ' + str(width_dom_modes[-i]))
        ax.plot(window_centres*tomo_params['delta_turns'], \
                np.sum(width_modes[width_dom_modes[0:(N_bunches-N_modes_plt)], :], axis=0), \
                label = 'Remaining modes')
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mode magnitude [s]")
        plt.title('Bunch width modes')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend(loc=0, fontsize='medium')
        plt.savefig('width_modes_vs_turn.png')
        plt.show()
        plt.close()
        
def read_and_analyse_tomo(file, analysis, plots, plot_raw_waterfall):
    with open(file['filename'], 'r') as f:
        lines = f.readlines()
    
    N_bins = int(lines[file['bins_per_frame_line']-1])
    delta_turns = int(lines[file['delta_turns_line']-1])
    dt_samp = float(lines[file['bin_width_line']-1])
    N_data_lines = len(lines) - file['data_start_line'] + 1 - analysis['frames_discard_end']*N_bins
    
    data_1d = np.empty(N_data_lines)
    
    for i in range(N_data_lines):
        data_1d[i] = float(lines[i+file['data_start_line']-1])

    N_frames = int(N_data_lines / N_bins)
    data_2d = np.reshape(data_1d, [N_frames, N_bins])
    
    #Plot raw data from tomoscope:
    if plot_raw_waterfall:
        plot_y, plot_x = np.mgrid[0:N_frames*delta_turns:delta_turns, 0:((N_bins-0.5)*dt_samp):dt_samp]
        plt.pcolormesh(plot_x, plot_y, data_2d, cmap='hot', shading='flat')
        plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
        plt.xlabel('Time [s]')
        plt.ylabel('Turn')
        plt.title(file['display_name'])
        plt.colorbar()
        plt.rc('font', size=16)
        plt.savefig(file['output_dir'] + 'waterfall_' + file['display_name'] +'.png')
        plt.show()
        plt.close()
    
    T_bucket = (analysis['T_end'] - analysis['T_start']) / analysis['N_buckets']
    T_vec = np.arange(N_bins) * dt_samp
    
    bunch_pos = np.empty([analysis['N_buckets'], N_frames])
    bunch_width = np.empty([analysis['N_buckets'], N_frames])
    
    #Plots to help with alignment of limits:
    if plots:
        time_indices = (T_vec >= analysis['T_start']) & (T_vec < analysis['T_end'])
        plt.plot(T_vec[time_indices], data_2d[0, time_indices])
        plt.title('Turn 1, within time limits')
        plt.xlabel('Time [s]')
        plt.show()
    
        for bucket in range(analysis['N_buckets']):
            time_indices = (T_vec >= analysis['T_start'] + bucket*T_bucket) & (T_vec < analysis['T_start'] + (bucket+1)*T_bucket)
            bucket_profile = data_2d[analysis['alignment_frame'], time_indices]
            plt.plot(bucket_profile)
        plt.title('Turn ' + str(analysis['alignment_frame'] * delta_turns) + ', individual buckets')
        plt.xlabel('Time [samples]')
        plt.show()
    
    #Identify edges of bunch in each bucket to determine width and position:
    for frame in range(N_frames):
        for bucket in range(analysis['N_buckets']):
            time_indices = (T_vec >= analysis['T_start'] + bucket*T_bucket) & (T_vec < analysis['T_start'] + (bucket+1)*T_bucket)
            bucket_profile = data_2d[frame, time_indices]
            profile_minus_threshold = savgol_filter(bucket_profile - np.mean(bucket_profile), 11, 3)
            above_threshold_indices = np.argwhere(profile_minus_threshold > 0)
            
            rising_edge_index = np.min(above_threshold_indices)
            falling_edge_index = np.max(above_threshold_indices)
            
            rising_edge_frac_index = -profile_minus_threshold[rising_edge_index-1] /\
                (profile_minus_threshold[rising_edge_index] - profile_minus_threshold[rising_edge_index-1])
            
            falling_edge_frac_index = profile_minus_threshold[falling_edge_index] /\
                (profile_minus_threshold[falling_edge_index+1] - profile_minus_threshold[falling_edge_index])
                
            rising_edge_interp_index = rising_edge_index - rising_edge_frac_index
            falling_edge_interp_index = falling_edge_index + falling_edge_frac_index
            
            bunch_pos[bucket, frame] = 0.5*dt_samp*(rising_edge_interp_index + falling_edge_interp_index)
            bunch_width[bucket, frame] = dt_samp*(falling_edge_interp_index - rising_edge_interp_index)
    
    if plots:
        for bucket in range(analysis['N_buckets']):
            plt.plot(bunch_pos[bucket,:])
        plt.xlabel('Turn')
        plt.ylabel('Bunch position [s]')
        plt.show()
        
        for bucket in range(analysis['N_buckets']):
            plt.plot(bunch_width[bucket,:])
        plt.xlabel('Turn')
        plt.ylabel('Bunch width [s]')
        plt.show()   
    
    tomo_params = {'start_frame' :  analysis['mode_window_start'], \
                   'end_frame' : analysis['mode_window_end'], \
                   'delta_turns' : delta_turns, \
                   'dt_samp' : dt_samp}
            
    #Mode analysis using edge data:
    [pos_modes_edge, width_modes_edge, _, quad_freq] = mode_analysis(bunch_pos, bunch_width, tomo_params, analysis['N_buckets_fft'], plots)
    
    #Plot mode windows vs time:
    if plots:
        modes_vs_time(bunch_pos, bunch_width, tomo_params, \
                  analysis['mode_vs_time_window'], analysis['mode_vs_time_resolution'],\
                      analysis['N_buckets_fft'], analysis['mode_vs_time_N_modes_plt'])
    
    #Mode analysis using band filtering:
    bucket_params = {'T_vec' : T_vec, 'T_start' : analysis['T_start'], 'T_bucket' : T_bucket}
    
    fs_harm_modes = [None for fs_harm_index, fs_harm in enumerate(analysis['fs_harms'])]
    
    for fs_harm_index, fs_harm in enumerate(analysis['fs_harms']):
        fs_harm_modes[fs_harm_index] = fs_harm_mode_analysis(data_2d, tomo_params, bucket_params,\
                                    analysis['N_buckets'], analysis['N_buckets_fft'], quad_freq/2, fs_harm, False, file['filename'])   
    
    return [pos_modes_edge, width_modes_edge, fs_harm_modes]