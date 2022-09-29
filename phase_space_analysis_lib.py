# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:17:04 2022

@author: JohnG
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

import longitudinal_tomography.tomography.tomography as tomography
import longitudinal_tomography.tracking.particles as parts
import longitudinal_tomography.tracking.tracking as tracking
import longitudinal_tomography.data.data_treatment as dtreat
import longitudinal_tomography.utils.tomo_input as tomoin
import longitudinal_tomography.utils.tomo_output as tomoout

def gaussian_square_error(x, p_vs_r):
    A = x[0]
    sigma = x[1]
    r = np.arange(len(p_vs_r))
    square_error = np.power(p_vs_r - A * np.exp(-0.5*np.power(r/sigma, 2)) / (np.power(2*np.pi, 0.5) * sigma), 2)    
    return np.mean(square_error)

def read_input_parameters(filename):
    parameter_lines = 98
    input_parameters = []
    with open(filename, 'r') as line:
        for i in range(parameter_lines):
            input_parameters.append(line.readline().strip())
    return input_parameters
    
def read_and_analyse_phase_space(filename, fallback_filename, display_name, analysis, plots):
    #Read parameters:            
    input_parameters = read_input_parameters(filename)

    #Read raw data:
    raw_data = np.genfromtxt(filename, skip_header=98, dtype=np.float32)

    #Create dummy machine object to obtain parameters from tomo file:
    try:
        dummy_machine, frames = tomoin.txt_input_to_machine(input_parameters)
    except Exception as msg:
        print(msg)
        
        #Found invalid parameters in input file. Override.
        print('Falling back on parameters from ' + fallback_filename)
        
        #Reload input parameters from fallback file.
        #Same input parameters will be used later in bunch-by-bunch analysis.
        input_parameters = read_input_parameters(fallback_filename)
        dummy_machine, frames = tomoin.txt_input_to_machine(input_parameters)
        
    dummy_machine.nbins = frames.nbins_frame
    frames.skip_bins_start = 0
    frames.skip_bins_end = 0
        
    measured_waterfall = frames.to_waterfall(raw_data)

    #Grid coordinates are needed even if not plotting:
    plot_y, plot_x = np.mgrid[0:dummy_machine.nprofiles*dummy_machine.dturns:dummy_machine.dturns,\
                                  0:((dummy_machine.nbins-0.5)*dummy_machine.dtbin):dummy_machine.dtbin]

    if plots:
        plt.figure(figsize=(10,10))
        plt.pcolormesh(plot_x, plot_y, measured_waterfall, cmap='hot', shading='flat')
        plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
        plt.xlabel('Time [s]')
        plt.ylabel('Turn')
        plt.title(display_name + ', full data')
        plt.colorbar()
        plt.rc('font', size=16)
        plt.show()
        plt.close()

    waterfall_window = measured_waterfall[analysis['mode_window_start']:analysis['mode_window_end'],:]
    plot_x_window = plot_x[analysis['mode_window_start']:analysis['mode_window_end'],:]
    plot_y_window = plot_y[analysis['mode_window_start']:analysis['mode_window_end'],:]

    if plots:
        plt.figure(figsize=(10,10))
        plt.pcolormesh(plot_x_window, plot_y_window, waterfall_window, cmap='hot', shading='flat')
        plt.axis([plot_x_window.min(), plot_x_window.max(), plot_y_window.min(), plot_y_window.max()])
        plt.xlabel('Time [s]')
        plt.ylabel('Turn')
        plt.title(display_name + ', data in time window : profiles ' +\
                  str(analysis['mode_window_start']) + ' to ' + str(analysis['mode_window_end']))
        plt.colorbar()
        plt.rc('font', size=16)
        plt.show()
        plt.close()


    #Use mean profile to cut waterfall into individual bunches:
    mean_profile = np.mean(waterfall_window, axis=0)
    profile_max = np.max(mean_profile)
    profile_min = np.min(mean_profile)
    profile_level = 0.25
    profile_threshold = profile_level * profile_max + (1-profile_level) * profile_min
    
    profile_above_mid = np.array(mean_profile >= profile_threshold, dtype='int')
    edges = np.diff(profile_above_mid)
    rising_edges = np.hstack(np.argwhere(edges == 1))
    falling_edges = np.hstack(np.argwhere(edges == -1))

    #Indices to cut waterfall at:
    cut_indices = np.round(0.5*(rising_edges[1:] + falling_edges[0:-1]))
    N_buckets = len(cut_indices) + 1

    first_bucket_centre_index = np.round(0.5*(rising_edges[0] + falling_edges[0]))
    last_bucket_centre_index = np.round(0.5*(rising_edges[-1] + falling_edges[-1]))

    #Try to cut first and last buckets such that the bunch is roughly in the middle of the bins:
    first_bucket_lower_cut_index = np.max([0, first_bucket_centre_index - (cut_indices[0] - first_bucket_centre_index)])
    last_bucket_upper_cut_index = np.min([mean_profile.shape[0], \
                                        last_bucket_centre_index + (last_bucket_centre_index - cut_indices[-1])])
        
    cut_indices_edges = np.append(np.insert(cut_indices, 0, first_bucket_lower_cut_index), last_bucket_upper_cut_index)
    
    mean_profile_t = dummy_machine.dtbin * np.arange(dummy_machine.nbins)

    if plots:
        plt.figure(figsize=(10,10))   
        plt.plot(mean_profile_t, mean_profile, label='Profile')
        plt.plot(mean_profile_t[rising_edges], mean_profile[rising_edges], 'x', label='Rising edges')
        plt.plot(mean_profile_t[falling_edges], mean_profile[falling_edges], 'x', label='Falling edges')
        for i in cut_indices_edges:
            plt.axvline(mean_profile_t[int(i)], linestyle='--', color='k', alpha=0.4)
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.title(display_name + ', mean profile in window')
        plt.legend(loc=1, fontsize='medium')
        plt.rc('font', size=16)
        plt.show()

    waterfall_bucket = [None for _ in range(N_buckets)]
    bucket_phase_space = [None for _ in range(N_buckets)]
    
    for bucket in range(N_buckets):    
        waterfall_bucket[bucket] = waterfall_window[:,int(cut_indices_edges[bucket]):int(cut_indices_edges[bucket+1])]
        plot_x_bucket = plot_x_window[:,int(cut_indices_edges[bucket]):int(cut_indices_edges[bucket+1])]
        plot_y_bucket = plot_y_window[:,int(cut_indices_edges[bucket]):int(cut_indices_edges[bucket+1])]
        
        plt.figure(figsize=(10,10))
        plt.pcolormesh(plot_x_bucket, plot_y_bucket, waterfall_bucket[bucket], cmap='hot', shading='flat')
        plt.axis([plot_x_bucket.min(), plot_x_bucket.max(), plot_y_bucket.min(), plot_y_bucket.max()])
        plt.xlabel('Time [s]')
        plt.ylabel('Turn')
        plt.title(display_name + ', bucket ' + str(bucket) + ' : profiles ' + \
                  str(analysis['mode_window_start']) + ' to ' + str(analysis['mode_window_end']))
        plt.colorbar()
        plt.rc('font', size=16)
        plt.show()
        plt.close()

    #Maximum radius must be at most half of the width of the narrowest bucket:
    R_max = int(np.floor(np.min([b.shape[1] for b in waterfall_bucket]) / 2))

    mode_vs_r = np.zeros([N_buckets, len(analysis['fs_harms']), R_max], dtype='complex')
    mode_integ = np.zeros([N_buckets, len(analysis['fs_harms'])], dtype='complex') 

    #Calculate phase space for each bucket, perform polar sampling
    for bucket in range(N_buckets):
        #Re-generate machine object for each bucket:
        machine, frames = tomoin.txt_input_to_machine(input_parameters)
        
        machine.nprofiles = analysis['mode_window_end'] - analysis['mode_window_start']
        machine._nbins = waterfall_bucket[bucket].shape[1]
        machine.nbins = waterfall_bucket[bucket].shape[1]
        machine.full_pp_flag = True
        machine.values_at_turns()
    
        if 'tomogprahy_rebin_override' in analysis.keys():
            frames.rebin = analysis['tomogprahy_rebin_override']
        frames.nbins_frame = waterfall_bucket[bucket].shape[1]
        frames.skip_bins_start = 0
        frames.skip_bins_end = 0
        
        # Creating profiles object
        profiles = tomoin.raw_data_to_profiles(
            waterfall_bucket[bucket], machine,
            frames.rebin, frames.sampling_time)
    
        profiles.calc_profilecharge()
        
        if profiles.machine.synch_part_x < 0:
            fit_info = dtreat.fit_synch_part_x(profiles)
            machine.load_fitted_synch_part_x_ftn(fit_info)
        
        tracker = tracking.Tracking(machine)
        tracker.enable_fortran_output(profiles.profile_charge)
        
        # For including self fields during tracking
        # FOR SELF FIELDS
        if machine.self_field_flag:
            profiles.calc_self_fields()
            tracker.enable_self_fields(profiles)
        
        #Pick middle frame as reference:
        ref_frame = int(np.round(waterfall_bucket[bucket].shape[0] / 2))
        
        xp, yp = tracker.track(ref_frame)
        
        # Converting from physical coordinates ([rad], [eV])
        # to phase space coordinates.
        if not tracker.self_field_flag:
            xp, yp = parts.physical_to_coords(
                xp, yp, machine, tracker.particles.xorigin,
                tracker.particles.dEbin)
        
        # Filters out lost particles, transposes particle matrix, casts to np.int32
        xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)
        
        # Reconstructing phase space
        tomo = tomography.TomographyCpp(profiles.waterfall, xp)
        weight = tomo.run(niter=machine.niter, verbose=True)
        
        # Creating image for fortran style presentation of phase space.
        image = tomoout.create_phase_space_image(
            xp, yp, weight, machine.nbins, ref_frame)
        tomoout.show(image, tomo.diff, profiles.waterfall[ref_frame])
        
        # dEtbin = machine.dEbin * machine.dtbin
        
        #Save data:
        bucket_phase_space[bucket] = image
        
        #nterpolation of phase space plots:
        #Make sure synchronous particle is at origin.
        x_coords = np.arange(bucket_phase_space[bucket].shape[0]) - machine.synch_part_x
        y_coords = np.arange(bucket_phase_space[bucket].shape[1]) - machine.synch_part_y
        
        phase_space_spline = scipy.interpolate.RectBivariateSpline(x_coords, y_coords, bucket_phase_space[bucket], kx=3, ky=3)
          
        
        #Polar sampling:
        #Each data point should occupy the same area of phase space.
        phi_samp = [None for _ in range(R_max)]
        data_vs_phi = [None for _ in range(R_max)]
        fft_vs_r = [None for _ in range(R_max)]
        
        for r_index in range(R_max):
            if r_index == 0:
                N_phi_samp = 1
                x_samp = np.zeros(1)
                y_samp = np.zeros(1)
                phi_samp[r_index] = np.zeros(1)
                sample_area = np.pi*0.5**2
            else:
                N_phi_samp = int(np.ceil(2*np.pi*r_index))
                phi_samp[r_index] = np.linspace(0, 2*np.pi, N_phi_samp, endpoint=False)
                
                x_samp = r_index * np.cos(phi_samp[r_index])
                y_samp = r_index * np.sin(phi_samp[r_index])
                
                sample_area = np.pi*((r_index+0.5)**2 - (r_index-0.5)**2) / N_phi_samp
            
            #Sample values in polar coordinates from interpolating spline object:
            #Multiply values at each radius by scaling factor such that summing over
            #polar and cartesian axes gives the same result.
            data_vs_phi[r_index] = np.zeros(N_phi_samp)
            for phi_index in range(N_phi_samp):
                data_vs_phi[r_index][phi_index] = phase_space_spline(x_samp[phi_index], y_samp[phi_index]) * sample_area
            
            #Perform FFT along azimuth coordinate:
            fft_vs_r[r_index] = np.fft.rfft(data_vs_phi[r_index])# / N_phi_samp
            
            #Use FFT data to get mode spectra vs. radius:
            fft_result = fft_vs_r[r_index]
            for mode_index, mode in enumerate(analysis['fs_harms']):
                if mode < fft_result.shape[0]:
                    mode_vs_r[bucket,mode_index,r_index] = fft_result[mode]
                else:
                    mode_vs_r[bucket,mode_index,r_index] = 0
                
        #Integrate complex mode amplitudes over radius:
        #Note: Complex values used as oscillations at different radii that are out of phase are 
        #expected to cancel out in terms of impedance effects.
        #FFT data should not be normalised, a larger radius has more samples which corresponds
        #to a larger area in phase space.
        mode_integ[bucket,:] = np.sum(mode_vs_r[bucket,:,:], axis=1) * machine.dtbin
        
        #Plot sampled points vs. radius:
        if plots:
            r_axis = np.arange(R_max) * machine.dtbin
                
            plt.figure()
            for r_index in [1, 5, 10, 15, 20]:
                phi_360 = np.append(phi_samp[r_index], 2*np.pi)
                data_360 = np.append(data_vs_phi[r_index], data_vs_phi[r_index][0])
                
                plt.plot(phi_360, data_360, label='r = ' + str(r_index))    
            plt.xlabel('Phi [rad]')
            plt.ylabel('Phase space density')
            plt.title(display_name + ', bucket ' + str(bucket))
            plt.legend(loc=0, fontsize='medium')
            plt.show()
                
            #Mode amplitudes vs. radius:
            plt.figure()
            for mode_index, mode in enumerate(analysis['fs_harms']):
                plt.plot(r_axis, np.abs(mode_vs_r[bucket,mode_index,:]), label='Mode = ' + str(mode))    
            plt.xlabel('Radius [s]')
            plt.ylabel('Phase space density')
            plt.title(display_name + ', bucket ' + str(bucket))
            plt.legend(loc=0, fontsize='medium')
            plt.show()
            
        #Fit matched Gaussian bunch to phase space plot:
        p_vs_r = [np.mean(data_vs_phi[r_index]) for r_index in range(R_max)]
           
        x0 = [p_vs_r[0], R_max / 2]
        x_bounds = scipy.optimize.Bounds([0, 0], [np.inf, R_max])
        opt_res = scipy.optimize.minimize(gaussian_square_error, x0, args=(p_vs_r), bounds=x_bounds, method='trust-constr')
        A_res = opt_res.x[0]
        sigma_res = opt_res.x[1]
        

    #Statistics of mode amplitude vs radius:
    percentiles = [25, 50, 75]
    mode_amp_vs_r_percentiles = [np.percentile(np.abs(mode_vs_r), pc, axis=0) for pc in percentiles]
    
    if plots:
        plt.figure()
        for mode_index, mode in enumerate(analysis['fs_harms']):
            plt.plot(r_axis, mode_amp_vs_r_percentiles[1][mode_index,:], label='Mode = ' + str(mode)) 
            plt.fill_between(r_axis, mode_amp_vs_r_percentiles[0][mode_index,:], \
                             mode_amp_vs_r_percentiles[2][mode_index,:], alpha=0.2, antialiased=True)
        plt.xlabel('Radius [s]')
        plt.ylabel('Phase space density')
        plt.legend(loc=0, fontsize='medium')
        plt.show()
        
        plt.figure()
        for mode_index, mode in enumerate(analysis['fs_harms']):
            plt.semilogy(r_axis, mode_amp_vs_r_percentiles[1][mode_index,:], label='Mode = ' + str(mode)) 
            plt.fill_between(r_axis, mode_amp_vs_r_percentiles[0][mode_index,:], \
                             mode_amp_vs_r_percentiles[2][mode_index,:], alpha=0.2, antialiased=True)
        plt.xlabel('Radius [s]')
        plt.ylabel('Phase space density')
        plt.legend(loc=0, fontsize='medium')
        plt.show()
    
    #Mode spectra:
    mode_names = ['Monopole', 'Dipole', 'Quadrupole', 'Sextupole', 'Octupole', 'Decapole', '12-pole']
    mode_spectrum = np.zeros([analysis['N_buckets_fft'], len(analysis['fs_harms'])], dtype='complex')
    
    for mode_index, mode in enumerate(analysis['fs_harms']):
        mode_amp_cmplx_padded = np.zeros(analysis['N_buckets_fft'], dtype='complex')
        mode_amp_cmplx_padded[0:N_buckets] = mode_integ[:,mode_index]
        mode_spectrum[:,mode_index] = np.fft.fft(mode_amp_cmplx_padded)/analysis['N_buckets_fft']
        
        if plots:
            plt.figure()
            plt.bar(np.arange(analysis['N_buckets_fft']), np.abs(mode_spectrum[:,mode_index]))
            plt.xlabel('Mode')
            plt.ylabel('Mode amplitude [s]')
            plt.title(display_name + ', ' + str(mode_names[mode]) + ' mode spectrum')
            plt.show()
            
    return mode_spectrum