# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:35:49 2022

@author: JohnG
"""
import csv
import numpy as np
import portable_fig as pfig
import scipy.signal

def signed(uint, Nbits):
    return (uint - (2 ** Nbits)) if uint >= (2 ** (Nbits-1)) else uint

def import_data(filename_base, num_buffers, num_channels):        
    data = [[] for _ in range(num_buffers)]
        
    for buffer in range(num_buffers):
        filename = str(filename_base) + "_ch" + str(buffer) + ".txt"
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            data[buffer] = list()

            for row in reader:
                if row: #check if row not empty
                    data[buffer].append(signed(int(row[0]), 16))
            
    #Combine real and imaginary data into complex arrays:
    dipole_cmplx = [np.zeros(len(data[0])) for _ in range(num_channels)]
    quad_cmplx = [np.zeros(len(data[0])) for _ in range(num_channels)]
    
    for channel in range(num_channels):
        dipole_cmplx[channel] = np.array(data[channel*4]) + 1j * np.array(data[channel*4+1])
        quad_cmplx[channel] = np.array(data[channel*4+2]) + 1j * np.array(data[channel*4+3])
        
    return [dipole_cmplx, quad_cmplx]

def plot_zoom(cmplx_data, t_plot, f_plot, acq_start_time, f_samp, title=''):
    ctime = acq_start_time + 1e3 * np.arange(cmplx_data.shape[0])/f_samp
    
    #Raw IQ plots:
    t_plot_indices = (ctime > t_plot[0]) & (ctime <= t_plot[1])
    pf = pfig.portable_fig()
    pf.set_figsize((12,9))
    pf.plot(ctime[t_plot_indices], np.real(cmplx_data[t_plot_indices]))
    pf.plot(ctime[t_plot_indices], np.imag(cmplx_data[t_plot_indices]))
    pf.ylabel('Value')
    pf.xlabel('Time [ms]')
    pf.title(title)
    pf.gen_plot()
        
    #Plot frequency-domain data:
    N_window = sum(t_plot_indices)
    fft_vec_unsort = np.fft.fft(cmplx_data[t_plot_indices] * np.hanning(N_window))
    freq_vec_unsort = np.fft.fftfreq(N_window, 1/f_samp)
    #Sort result so that points are ascending in frequency:
    sort_indices = np.argsort(freq_vec_unsort)
    freq_vec = freq_vec_unsort[sort_indices]    
    fft_vec = fft_vec_unsort[sort_indices]
    
    f_plot_indices = (freq_vec > f_plot[0]) & (freq_vec <= f_plot[1])
    pf = pfig.portable_fig()
    pf.set_figsize((12,9))
    pf.plot(freq_vec[f_plot_indices], 20*np.log10(np.abs(fft_vec[f_plot_indices])))
    pf.ylabel('PSD [dB]')
    pf.xlabel('Baseband frequency [Hz]')
    pf.title(title)
    pf.gen_plot()
    
    return [freq_vec, fft_vec]

def get_spectrogram(dipole_cmplx, quad_cmplx, N_fft, acq_start_time, f_samp, plots): 
    s_dipole = [None for _ in range(len(dipole_cmplx))]
    s_quad = [None for _ in range(len(dipole_cmplx))]
    
    for channel in range(len(dipole_cmplx)):    
        _, _, s_dipole_unsort = scipy.signal.spectrogram(dipole_cmplx[channel], nperseg=N_fft, noverlap=N_fft/2,\
                                                           fs=f_samp, window=('tukey', 1), return_onesided=False, mode='psd')
        f_unsort, t_since_acq_start, s_quad_unsort = scipy.signal.spectrogram(quad_cmplx[channel], nperseg=N_fft, noverlap=N_fft/2, \
                                                         fs=f_samp, window=('tukey', 1), return_onesided=False, mode='psd')
    
        t = acq_start_time + t_since_acq_start * 1e3
        #Sort frequency values:
        sort_indices = np.argsort(f_unsort)
        f = f_unsort[sort_indices]
        s_dipole[channel] = s_dipole_unsort[sort_indices,:]
        s_quad[channel] = s_quad_unsort[sort_indices,:]

        if plots:
            pf = pfig.portable_fig()
            pf.set_figsize((12,9))
            pf.pcolormesh(t, f, 10*np.log10(s_dipole[channel]), cmap='hot', shading='flat')
            pf.title('Dipole channel ' + str(channel))
            pf.ylabel('Frequency [Hz]')
            pf.xlabel('Time [ms]')
            pf.gen_plot()
            
            pf = pfig.portable_fig()
            pf.set_figsize((12,9))
            pf.pcolormesh(t, f, 10*np.log10(s_quad[channel]), cmap='hot', shading='flat')
            pf.title('Quad channel ' + str(channel))
            pf.ylabel('Frequency [Hz]')
            pf.xlabel('Time [ms]')
            pf.gen_plot() 

    return [f, t, s_dipole, s_quad]

def fs_peaks(freq_vec, spacing, width, amplitudes):
    #amplitude vector is ordered: 0, fs, -fs, 2fs, -2fs, etc...
    a_vec = np.zeros_like(freq_vec)
    # for i in range(len(amplitudes)):
    for i, amp in enumerate(amplitudes):
        if i == 0:
            h = 0
        else:
            h = np.floor((i+1)/2)
            #even indices are negative multiples:
            if (i % 2) == 0:
                h = -h  
        a_vec += amp * np.exp(-2*np.power(freq_vec-h*spacing, 2)/width**2)
    return a_vec

def mse(x, freq_vec, fft_vec):
    square_error = np.power(np.abs(fft_vec) - fs_peaks(freq_vec, x[0], x[1], x[2:]), 2)
    norm_square_error = np.mean(square_error) / np.mean(np.power(np.abs(fft_vec), 2))
    
    # weighted_error = square_error * np.power(np.abs(fft_vec), 1)
    # return np.mean(weighted_error)
    return norm_square_error

def fs_sidebands(params, init, bounds, acq_start_time, f_samp, cmplx_data, plots):
    N_times = params['t_centres'].shape[0]
    ctime = acq_start_time + 1e3 * np.arange(cmplx_data.shape[1])/f_samp

    #Initialise arrays to save data to:
    fs_fit = np.zeros(N_times)

    #Perform fit centered around each sample time:
    for i in range(N_times):
        #Obtain frequency spectrum centered around time sample of interest:
        t_centre = params['t_centres'][i]
        t_plot_indices = (ctime > t_centre - params['t_half_span']) & (ctime <= t_centre + params['t_half_span'])
        N_window = sum(t_plot_indices)
        freq_vec_unsort = np.fft.fftfreq(N_window, 1/params['f_samp'])
        fft_vec_unsort = np.fft.fft(cmplx_data[:, t_plot_indices] * np.hanning(N_window))
        
        #Sort result so that points are ascending in frequency:
        sort_indices = np.argsort(freq_vec_unsort)
        freq_vec = freq_vec_unsort[sort_indices]
    
        fft_vec = fft_vec_unsort[:, sort_indices]
        fft_abs = np.abs(fft_vec)
        
        f_plot_indices = (freq_vec > -params['fmax']) & (freq_vec <= params['fmax'])
        freq_vec_plot = freq_vec[f_plot_indices]
        fft_abs_plot = fft_abs[:, f_plot_indices]
        
        #Construct initial vector:
        init_spacing = (bounds['fs_min_func'](t_centre) + bounds['fs_max_func'](t_centre)) / 2
        x0 = np.concatenate((np.array([init_spacing, init['width']]), init['amps']))
        
        #Set search bounds
        x_min = np.concatenate((np.array([bounds['fs_min_func'](t_centre), bounds['min_width']]), \
                                bounds['min_amp'] * np.ones_like(init['amps'])))
        
        x_max = np.concatenate((np.array([bounds['fs_max_func'](t_centre), bounds['fs_max_func'](t_centre)/4]), \
                                bounds['max_amp'] * np.ones_like(init['amps'])))
        
        x_bounds = scipy.optimize.Bounds(x_min, x_max)
        
        #Perform optimisation:
        opt_res = scipy.optimize.minimize(mse, x0, args=(freq_vec_plot, fft_abs_plot), bounds=x_bounds, method='trust-constr')
        x_res = opt_res.x
        
        #Save results:
        fs_fit[i] = x_res[0]
        # amps_fit[:, i] = x_res[2:]
        
        if plots:
            pf = pfig.portable_fig()
            pf.set_figsize((12,9))
            for i in range(fft_abs_plot.shape[0]):
                pf.plot(freq_vec_plot, fft_abs_plot[i,:])
            pf.plot(freq_vec_plot, fs_peaks(freq_vec_plot, x_res[0], x_res[1], x_res[2:]), 'k:')
            pf.ylabel('Amplitude [counts]')
            pf.xlabel('Baseband frequency [Hz]')
            pf.title('Time = ' + str(t_centre) + ' ms')
            pf.gen_plot() 

    return fs_fit

def sample_fs_harmonics(params, fs, acq_start_time, f_samp, cmplx_data):
    N_times = params['t_centres'].shape[0]
    amps_samp = np.zeros([params['N_harmonics_samp'], N_times])
    ctime = acq_start_time + 1e3 * np.arange(cmplx_data.shape[0])/f_samp
    
    for i in range(N_times):
        #Obtain frequency spectrum centered around time sample of interest:
        t_centre = params['t_centres'][i]
        t_plot_indices = (ctime > t_centre - params['t_half_span']) & (ctime <= t_centre + params['t_half_span'])
        N_window = sum(t_plot_indices)
        freq_vec_unsort = np.fft.fftfreq(N_window, 1/params['f_samp'])
        fft_vec_unsort = np.fft.fft(cmplx_data[t_plot_indices] * np.hanning(N_window))
        
        #Sort result so that points are ascending in frequency:
        sort_indices = np.argsort(freq_vec_unsort)
        freq_vec = freq_vec_unsort[sort_indices]
    
        fft_vec = fft_vec_unsort[sort_indices]
        fft_abs = np.abs(fft_vec)
        
        f_plot_indices = (freq_vec > -params['fmax']) & (freq_vec <= params['fmax'])
        freq_vec_plot = freq_vec[f_plot_indices]
        fft_abs_plot = fft_abs[f_plot_indices]
        
        for j in range(params['N_harmonics_samp']):
            if j == 0:
                h = 0
            else:
                h = np.floor((j+1)/2)
                #even indices are negative multiples:
                if (j % 2) == 0:
                    h = -h  
            f_h = h * fs[i]
            index_h = np.argmin(np.abs(f_h - freq_vec_plot))
            amps_samp[j, i] = np.mean(fft_abs_plot[index_h-params['samp_half_span']:index_h+params['samp_half_span']+1])
        
    return amps_samp