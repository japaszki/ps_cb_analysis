# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 08:59:03 2022

@author: JohnG
"""

def scan_point_label(scan_param_names, scan_param):
    label = ''
    for param_index, param_name in enumerate(scan_param_names):
        if param_index != 0:
            label = label + ', '
        label += param_name + ' = ' + str(scan_param[param_index])
    return label

def gen_1d_scans(scan_settings, scan_param_names):
    N_params = len(scan_settings[0])
    
    if N_params == 1:
        y_indices = [[[scan_index for scan_index, scan_param in enumerate(scan_settings)]]]
        x_axis = [[[scan_param[0] for scan_index, scan_param in enumerate(scan_settings)]]]
        labels = [['']]
    else:
            
        y_indices = [None for _ in range(N_params)]
        x_axis = [None for _ in range(N_params)]
        labels = [None for _ in range(N_params)]

        #Use each parameter as the x-axis in turn:
        for x_param_index in range(N_params):
            #parameters not acting as x-axis:
            rem_param_indices = [param_index for param_index, param_name in enumerate(scan_param_names)\
                                 if param_index != x_param_index]
    
            #Iterate through all scan settings and find unique values of non-x-axis parameters:
            #Record scan_index of every instance of each value
            rem_unique_settings = []
            y_indices[x_param_index] = []
            for scan_index, scan_param in enumerate(scan_settings):
                rem_setting = [scan_param[i] for i in rem_param_indices]
                
                #If this setting hase already been found, find correct entry in scan_index_list and add to it:
                if rem_setting in rem_unique_settings:
                    y_indices[x_param_index][rem_unique_settings.index(rem_setting)].append(scan_index)
                #If not already found, add entry to both lists:
                else:
                    rem_unique_settings.append(rem_setting)
                    y_indices[x_param_index].append([scan_index])
            
            x_axis[x_param_index] = []
            labels[x_param_index] = []
            for unique_val_index, unique_val in enumerate(rem_unique_settings):
                x_axis[x_param_index].append([scan_settings[index][x_param_index]\
                                                           for index in y_indices[x_param_index][unique_val_index]])
                
                label = ''
                for label_index, rem_param_index in enumerate(rem_param_indices):
                    label += scan_param_names[rem_param_index] + \
                        ' = ' + str(unique_val[label_index])
                        
                labels[x_param_index].append(label)   
                
    return [x_axis, y_indices, labels]