# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:43:16 2022

@author: JohnG
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

def scatter_plot_with_linreg(x_datasets, y_datasets, xlabel, ylabel, plot_labels=None, alpha=1.0, title=None, filename=None):
    plt.figure(figsize=(10,10))
    
    regs = []
    for idx, x_data in enumerate(x_datasets):
        y_data = y_datasets[idx]
        reg = scipy.stats.linregress(x_data, y_data)
        regs.append(reg)
        
        fit_label = 'slope = ' + f'{reg.slope:.2f}' + ', R^2 = ' + f'{np.power(reg.rvalue,2):.2f}'
        
        if plot_labels != None:
            plot_label = plot_labels[idx] + ', ' + fit_label
        else:
            plot_label = fit_label
            
        [scatterplt] = plt.plot(x_data, y_data, '.', label=plot_label, alpha=alpha)
        x_line = np.array([0, max(x_data)])
        plt.plot(x_line, x_line * reg.slope + reg.intercept, color=scatterplt.get_color())
        plt.fill_between(x_line, x_line * (reg.slope - reg.stderr) + (reg.intercept - reg.intercept_stderr),\
                         x_line * (reg.slope + reg.stderr) + (reg.intercept + reg.intercept_stderr),\
                         alpha=0.2, color=scatterplt.get_color(), antialiased=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    
    if title != None:
        plt.title(title)
        
    plt.legend(loc=0, fontsize='medium')
    plt.rc('font', size=16)
    
    if filename != None:
        plt.savefig(filename)
    plt.show()
    
    return regs
    

def gpr_plot(x_datasets, y_datasets, xlabel, ylabel, plot_labels=None, alpha=1.0, title=None, filename=None, prior_plot=False):
    #x_datasets and y_datasets should be lists of 1d numpy arrays
    assert len(x_datasets) == len(y_datasets), 'Mismatch in number of x and y datasets!'
    
    N_x_samp = 100
        
    plt.figure(figsize=(10,10))
    
    gprs = []
    slopes = []

    #Gaussian process regression of each dataset:
    for idx, x_data in enumerate(x_datasets):
        y_data = y_datasets[idx]
        
        x_scale = max(x_data)
        y_scale = max(y_data)
        
        X_data = x_data.reshape(-1, 1) / x_scale
        Y_data = y_data.reshape(-1, 1) / y_scale

        #Linear model with gain noise and offset noise       
        kernel = (WhiteKernel(noise_level=1) * DotProduct(sigma_0=1e-2)) + DotProduct(sigma_0=1)

        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        gprs.append(gpr)
        
        gpr.fit(X_data, Y_data)
            
        x_samp = np.linspace(0, x_scale, N_x_samp)
        X_samp = x_samp.reshape(-1, 1) / x_scale
    
        mean, std = gpr.predict(X_samp, return_std=True)
        
        if plot_labels != None:
            plot_label = plot_labels[idx]
        else:
            plot_label = None
        
        [scatterplt] = plt.plot(x_data, y_data, '.', alpha=alpha, label=plot_label)
        plt.plot(x_samp, mean[:,0] * y_scale, color=scatterplt.get_color())
        
        std_upper = (mean[:,0]+std) * y_scale
        std_lower = (mean[:,0]-std) * y_scale
        
        plt.fill_between(x_samp, std_lower, std_upper,\
                         alpha=0.2, color=scatterplt.get_color(), antialiased=True)
        
        #Find range of slopes allowed by 1-sigma bounds:
        mean_slope = (mean[-1,0] - mean[0,0]) * y_scale / (x_samp[-1] - x_samp[0])
        upper_slope = (std_upper[-1] - std_lower[0]) / (x_samp[-1] - x_samp[0])
        lower_slope = (std_lower[-1] - std_upper[0]) / (x_samp[-1] - x_samp[0])
        slopes.append([mean_slope, lower_slope, upper_slope])
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=0, fontsize='medium')
    if title != None:
        plt.title(title)
    if filename != None:
        plt.savefig(filename)
    plt.rc('font', size=16)
    plt.show()
    
    return [gprs, slopes]