# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:07:25 2022

@author: JohnG
"""
import yaml
import matplotlib.pyplot as plt
import numpy as np

class portable_fig:        
    def __init__(self, filename=None):
        if filename == None:
            self.plot_list = []
            self.xlabel_params = None
            self.ylabel_params = None
            self.title_params = None
            self.legend_params = None
            self.xlim_params = None
            self.ylim_params = None
            self.axis_params = None
            self.colorbar_params =  None
            self.fontsize = None
            self.filename = ''
            self.figsize = (8,6)
        else:
            with open(filename, 'r') as f:
                all_data = yaml.load(f)
            
            self.plot_list = all_data['plot_list'] if 'plot_list' in all_data else None
            self.xlabel_params = all_data['xlabel_params'] if 'xlabel_params' in all_data else None
            self.ylabel_params = all_data['ylabel_params'] if 'ylabel_params' in all_data else None
            self.title_params = all_data['title_params'] if 'title_params' in all_data else None
            self.legend_params = all_data['legend_params'] if 'legend_params' in all_data else None
            self.xlim_params = all_data['xlim_params'] if 'xlim_params' in all_data else None
            self.ylim_params = all_data['ylim_params'] if 'ylim_params' in all_data else None
            self.axis_params = all_data['axis_params'] if 'axis_params' in all_data else None
            self.colorbar_params = all_data['colorbar_params'] if 'colorbar_params' in all_data else None
            self.fontsize = all_data['fontsize'] if 'fontsize' in all_data else None
            self.filename = filename
            self.figsize = all_data['figsize'] if 'figsize' in all_data else (8,6)
        
    def plot(self, x, y, fmt=None, **kwargs):
        
        if isinstance(x, np.ndarray):
            x_cnv = x.tolist()
        else:
            x_cnv = x
            
        if isinstance(y, np.ndarray):
            y_cnv = y.tolist()
        else:
            y_cnv = y
        
        self.plot_list.append(('plot', {'x' : x_cnv, 'y' : y_cnv, 'fmt' : fmt, 'kwargs' : kwargs}))
        
    def stairs(self, x, y, fmt=None, **kwargs):
        
        if isinstance(x, np.ndarray):
            x_cnv = x.tolist()
        else:
            x_cnv = x
            
        if isinstance(y, np.ndarray):
            y_cnv = y.tolist()
        else:
            y_cnv = y
            
        self.plot_list.append(('stairs', {'x' : x_cnv, 'y' : y_cnv, 'fmt' : fmt, 'kwargs' : kwargs}))
    
    def semilogx(self, x, y, fmt=None, data=None, **kwargs):

        if isinstance(x, np.ndarray):
            x_cnv = x.tolist()
        else:
            x_cnv = x
            
        if isinstance(y, np.ndarray):
            y_cnv = y.tolist()
        else:
            y_cnv = y
            
        self.plot_list.append(('semilogx', {'x' : x_cnv, 'y' : y_cnv, 'fmt' : fmt, 'kwargs' : kwargs}))
    
    def semilogy(self, x, y, fmt=None, data=None, **kwargs):

        if isinstance(x, np.ndarray):
            x_cnv = x.tolist()
        else:
            x_cnv = x
            
        if isinstance(y, np.ndarray):
            y_cnv = y.tolist()
        else:
            y_cnv = y
            
        self.plot_list.append(('semilogy', {'x' : x_cnv, 'y' : y_cnv, 'fmt' : fmt, 'kwargs' : kwargs}))
    
    def loglog(self, x, y, fmt=None, data=None, **kwargs):

        if isinstance(x, np.ndarray):
            x_cnv = x.tolist()
        else:
            x_cnv = x
            
        if isinstance(y, np.ndarray):
            y_cnv = y.tolist()
        else:
            y_cnv = y
            
        self.plot_list.append(('loglog', {'x' : x_cnv, 'y' : y_cnv, 'fmt' : fmt, 'kwargs' : kwargs}))
    
    def bar(self, x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs):
        
        if isinstance(x, np.ndarray):
            x_cnv = x.tolist()
        else:
            x_cnv = x
            
        if isinstance(height, np.ndarray):
            height_cnv = height.tolist()
        else:
            height_cnv = height
            
        if isinstance(width, np.ndarray):
            width_cnv = width.tolist()
        else:
            width_cnv = width
            
        if isinstance(bottom, np.ndarray):
            bottom_cnv = bottom.tolist()
        else:
            bottom_cnv = bottom
            
        self.plot_list.append(('bar', {'x' : x_cnv,\
                                       'height' : height_cnv,\
                                       'width' : width_cnv,\
                                       'bottom' : bottom_cnv,\
                                       'align' : align,\
                                       'data' : data,\
                                       'kwargs' : kwargs}))
        
    def axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        self.plot_list.append(('axvline', {'x' : x, 'ymin' : ymin, 'ymax' : ymax, 'kwargs' : kwargs}))
        
    def fill_between(self, x, y1, y2=0, where=None, interpolate=False, \
                     step=None, *, data=None, **kwargs):
        
        if isinstance(x, np.ndarray):
            x_cnv = x.tolist()
        else:
            x_cnv = x
            
        if isinstance(y1, np.ndarray):
            y1_cnv = y1.tolist()
        else:
            y1_cnv = y1
            
        if isinstance(y2, np.ndarray):
            y2_cnv = y2.tolist()
        else:
            y2_cnv = y2
            
        self.plot_list.append(('fill_between', \
                               {'x' : x_cnv, \
                                'y1' : y1_cnv, \
                                'y2' : y2_cnv, \
                                'where' : where,\
                                'interpolate' : interpolate, \
                                'step' : step, \
                                'data' : data, \
                                'kwargs' : kwargs}))
    
    def pcolormesh(self, X, Y, C, **kwargs):
        
        if isinstance(X, np.ndarray):
            X_cnv = X.tolist()
        else:
            X_cnv = X
            
        if isinstance(Y, np.ndarray):
            Y_cnv = Y.tolist()
        else:
            Y_cnv = Y
            
        if isinstance(C, np.ndarray):
            C_cnv = C.tolist()
        else:
            C_cnv = C
            
        self.plot_list.append(('pcolormesh', \
                               {'X' : X_cnv,\
                                'Y' : Y_cnv,\
                                'C' : C_cnv,\
                                'kwargs' : kwargs}))
        
    
    def xlabel(self, xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        self.xlabel_params = {'xlabel' : xlabel,\
                                'fontdict' : fontdict,\
                                'labelpad' : labelpad,\
                                'loc' : loc,\
                                'kwargs' : kwargs}
            
    def ylabel(self, ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        self.ylabel_params = {'ylabel' : ylabel,\
                                'fontdict' : fontdict,\
                                'labelpad' : labelpad,\
                                'loc' : loc,\
                                'kwargs' : kwargs}
            
    def title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
        self.title_params = {'label' : label,\
                             'fontdict' : fontdict,\
                             'loc' : loc,\
                             'pad' : pad,\
                             'y' : y,\
                             'kwargs' : kwargs}
            
    def legend(self, *args, **kwargs):
        self.legend_params = {'args' : args, 'kwargs' : kwargs}
        
    def xlim(self, *args, **kwargs):
        self.xlim_params = {'args' : args, 'kwargs' : kwargs}
        
    def ylim(self, *args, **kwargs):
        self.ylim_params = {'args' : args, 'kwargs' : kwargs}
    
    def axis(self, *args, emit=True, **kwargs):
        self.axis_params = {'args' : args, 'emit' : emit, 'kwargs' : kwargs}
        
    def colorbar(self, mappable=None, cax=None, ax=None, **kwargs):
        self.colorbar_params = {'mappable' : mappable,\
                                'cax' : cax,\
                                'ax' : ax,\
                                'kwargs' : kwargs}
            
    def set_figsize(self, figsize):
        self.figsize = figsize
        
    def set_fontsize(self, fontsize):
        self.fontsize = fontsize
    
    def set_filename(self, filename):
        self.filename = filename
        
    def save_yaml(self):
        all_data = {'plot_list' : self.plot_list,\
                    'xlabel_params' : self.xlabel_params,\
                    'ylabel_params' : self.ylabel_params,\
                    'title_params' : self.title_params,\
                    'legend_params' : self.legend_params,\
                    'fontsize' : self.fontsize,\
                    'figsize' : self.figsize}
        
        with open(self.filename + '.yaml', 'w') as f:
            yaml.dump(all_data, f)
            
    def gen_plot(self, filetype='.png'):
        plt.figure(figsize=self.figsize)
        
        if self.xlabel_params != None:
            plt.xlabel(self.xlabel_params['xlabel'],\
                       fontdict=self.xlabel_params['fontdict'],\
                       labelpad=self.xlabel_params['labelpad'],\
                       loc=self.xlabel_params['loc'],\
                       **self.xlabel_params['kwargs'])
            
        if self.ylabel_params != None:
            plt.ylabel(self.ylabel_params['ylabel'],\
                       fontdict=self.ylabel_params['fontdict'],\
                       labelpad=self.ylabel_params['labelpad'],\
                       loc=self.ylabel_params['loc'],\
                       **self.ylabel_params['kwargs'])
        
        if self.title_params != None:
            plt.title(self.title_params['label'],\
                  fontdict=self.title_params['fontdict'],\
                  loc=self.title_params['loc'],\
                  pad=self.title_params['pad'],\
                  y=self.title_params['y'],\
                  **self.title_params['kwargs'])
                      
        if self.fontsize != None:
            plt.rc('font', size=self.fontsize)
        
        for plot_elem in self.plot_list:
            if plot_elem[0] == 'plot':
                if plot_elem[1]['fmt'] != None:
                    plt.plot(plot_elem[1]['x'],\
                         plot_elem[1]['y'],\
                         plot_elem[1]['fmt'],\
                         **plot_elem[1]['kwargs'])
                else:
                    plt.plot(plot_elem[1]['x'],\
                         plot_elem[1]['y'],\
                         **plot_elem[1]['kwargs'])
                    
            elif plot_elem[0] == 'stairs':
                if plot_elem[1]['fmt'] != None:
                    plt.stairs(plot_elem[1]['x'],\
                         plot_elem[1]['y'],\
                         plot_elem[1]['fmt'],\
                         **plot_elem[1]['kwargs'])
                else:
                    plt.stairs(plot_elem[1]['x'],\
                         plot_elem[1]['y'],\
                         **plot_elem[1]['kwargs'])
            
            elif plot_elem[0] == 'semilogx':
                if plot_elem[1]['fmt'] != None:
                    plt.semilogx(plot_elem[1]['x'],\
                                 plot_elem[1]['y'],\
                                 plot_elem[1]['fmt'],\
                                 **plot_elem[1]['kwargs'])
                else:
                    plt.semilogx(plot_elem[1]['x'],\
                                 plot_elem[1]['y'],\
                                 **plot_elem[1]['kwargs'])
            
            elif plot_elem[0] == 'semilogy':
                if plot_elem[1]['fmt'] != None:
                    plt.semilogy(plot_elem[1]['x'],\
                                 plot_elem[1]['y'],\
                                 plot_elem[1]['fmt'],\
                                 **plot_elem[1]['kwargs'])
                else:
                    plt.semilogy(plot_elem[1]['x'],\
                                 plot_elem[1]['y'],\
                                 **plot_elem[1]['kwargs'])
                        
            elif plot_elem[0] == 'loglog':
                if plot_elem[1]['fmt'] != None:
                    plt.loglog(plot_elem[1]['x'],\
                                 plot_elem[1]['y'],\
                                 plot_elem[1]['fmt'],\
                                 **plot_elem[1]['kwargs'])
                else:
                    plt.loglog(plot_elem[1]['x'],\
                                 plot_elem[1]['y'],\
                                 **plot_elem[1]['kwargs'])
            
            elif plot_elem[0] == 'bar':
                plt.bar(plot_elem[1]['x'],\
                        plot_elem[1]['height'],\
                        width=plot_elem[1]['width'],\
                        bottom=plot_elem[1]['bottom'],\
                        align=plot_elem[1]['align'],\
                        data=plot_elem[1]['data'],\
                        **plot_elem[1]['kwargs'])
                
            elif plot_elem[0] == 'axvline':
                plt.axvline(plot_elem[1]['x'],\
                            plot_elem[1]['ymin'],\
                            plot_elem[1]['ymax'],\
                            **plot_elem[1]['kwargs'])
                        
            elif plot_elem[0] =='fill_between':
                plt.fill_between(plot_elem[1]['x'],\
                                 plot_elem[1]['y1'],\
                                 y2=plot_elem[1]['y2'],\
                                 where=plot_elem[1]['where'],\
                                 interpolate=plot_elem[1]['interpolate'],\
                                 step=plot_elem[1]['step'],\
                                 data=plot_elem[1]['data'],\
                                 **plot_elem[1]['kwargs'])
                    
            elif plot_elem[0] == 'pcolormesh':
                plt.pcolormesh(plot_elem[1]['X'],\
                               plot_elem[1]['Y'],\
                               plot_elem[1]['C'],\
                               **plot_elem[1]['kwargs'])
        
            else:
                raise ValueError('Unknown plot type')

        if self.legend_params != None:
            plt.legend(*self.legend_params['args'], **self.legend_params['kwargs'])

        if self.xlim_params != None:
            plt.xlim(*self.xlim_params['args'], **self.xlim_params['kwargs'])
        
        if self.ylim_params != None:
            plt.ylim(*self.ylim_params['args'], **self.ylim_params['kwargs'])
        
        if self.axis_params != None:
            plt.axis(*self.axis_params['args'],\
                     emit=self.axis_params['emit'],\
                     **self.axis_params['kwargs'])
            
        if self.colorbar_params != None:
            plt.colorbar(mappable=self.colorbar_params['mappable'],\
                         cax=self.colorbar_params['cax'],\
                         ax=self.colorbar_params['ax'],\
                         **self.colorbar_params['kwargs'])

        if self.filename != None:
            plt.savefig(self.filename + filetype)
            
        plt.show()
        