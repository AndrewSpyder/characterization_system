# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 13:16:28 2021

@author: Andrew
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

from control import Control

class FeatureEngineering(Control):
    """ GET STOCK DATA. """
    
    
    def __init__(self, **kwargs):
        """
        
        
        Class init Routine.
        
        ----------
        **kwargs : TYPE
            DESCRIPTION:
                If the user inputs any variable that is already a defined class
                object, then the following loop will provide the modification to
                the class self.
            
        """
        super().__init__(**kwargs)
        self.__dict__.update(**self.__dict__)
    
    def temporal_features(
            self, 
            dataframe: pd.DataFrame = pd.DataFrame()
            ) -> pd.DataFrame:
        dataframe[self.weekday_denotation] = dataframe.index.weekday
        dataframe[self.day_denotation] = dataframe.index.day
        return dataframe
    
    def rolling_average_features(
            self, 
            dataframe: pd.DataFrame = pd.DataFrame(),
            rolling_periods: list = [7, 15, 30]
            ) -> pd.DataFrame:
        rolling_results = list()
        for rolling_period in rolling_periods:
            to_average = dataframe.copy()
            rolling = to_average.rolling(rolling_period).mean()
            rolling = rolling.add_suffix(f" Rolling {rolling_period}")
            rolling_results.append(rolling)
        for rolling_df in rolling_results:
            dataframe = dataframe.join(rolling_df)
        return dataframe
    
    def persistence_features(
            self, 
            dataframe: pd.DataFrame = pd.DataFrame(),
            shift: int = 7
            ) -> pd.DataFrame:
        shifts = range(1, shift+1)
        shifted_results = list()
        for shift_it in shifts:
            to_shift = dataframe.copy()
            shifted = to_shift.shift(shift_it)
            shifted = shifted.add_suffix(f" Shifted {shift_it}")
            shifted_results.append(shifted)
        for shifted_df in shifted_results:
            dataframe = dataframe.join(shifted_df)
        return dataframe
    
    def butterworth_filtering(
            self,
            dataframe: pd.DataFrame = pd.DataFrame(),
            columns_to_filter: list = list(),
            btype: str = str(),
            lowcut: float or None = None,
            highcut: float or None = None,
            fs: float = np.pi,
            order: int = 4,
            log_condition: bool = False,
            plot_it: bool = False
            ) -> pd.DataFrame:
        """
        
        Butterworth Filters
        
        :param dataframe: DESCRIPTION, defaults to pd.DataFrame()
        :type dataframe: pd.DataFrame, optional
        :param columns_to_filter: DESCRIPTION, defaults to list()
        :type columns_to_filter: list, optional
        :param btype: DESCRIPTION, defaults to str()
        :type btype: str, optional
        :param lowcut: DESCRIPTION, defaults to None
        :type lowcut: float or None, optional
        :param highcut: DESCRIPTION, defaults to None
        :type highcut: float or None, optional
        :param fs: DESCRIPTION, defaults to np.pi
        :type fs: float, optional
        :param order: DESCRIPTION, defaults to 4
        :type order: int, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        from scipy import signal
        if not isinstance(dataframe, pd.DataFrame):
            return
        if not isinstance(order, int):
            order = 2 # Safe Guess
        if isinstance(fs, float):
            nyq = 0.5 * fs
        if isinstance(lowcut, float):
            low = lowcut / nyq
        if isinstance(highcut, float):
            high = highcut / nyq
        if 'low' in btype.lower():
            sos = signal.butter(order, low, btype='lowpass', output='sos')
        if 'high' in btype.lower():
            sos = signal.butter(order, high, btype='highpass', output='sos')
        if 'band' in btype.lower():
            sos = signal.butter(order, [low, high], 
                                btype='bandpass', output='sos')
        
        dataframe_to_filter = dataframe.copy()
        if not isinstance(columns_to_filter, list):
            columns_to_filter = dataframe_to_filter.columns
        if len(columns_to_filter) == 0:
            columns_to_filter = dataframe_to_filter.columns
        dataframe_to_filter = dataframe_to_filter[columns_to_filter]
        
        dataframe_to_filter = dataframe_to_filter.resample(
            self.frequency_string).mean()
        dataframe_to_filter = dataframe_to_filter.interpolate(
            method='polynomial', order=3)
        dataframe_to_filter = dataframe_to_filter.dropna()
        
        if log_condition:
            dataframe_to_filter = dataframe_to_filter.abs().apply(np.log10)
            rename = dict()
            for col in dataframe_to_filter.columns:
                rename[col] = f"{col} {self.log10_denotation}"
            dataframe_to_filter = dataframe_to_filter.rename(columns=rename)
            columns_to_filter = dataframe_to_filter.columns # Redefine
        
        columns_to_not_filter = [col for col in dataframe.columns
                                 if not col in columns_to_filter]
        unfiltered_dataframe = dataframe[columns_to_not_filter]
        
        filter_dataframes = dataframe_to_filter.copy()
        length = int(len(columns_to_filter) - 1)
        for it, column in enumerate(columns_to_filter):
            data = np.ravel(dataframe_to_filter[column].copy())
            dictionary = dict()
            Y, X, period = list(), list(), 60 # Double the required pad length
            iterations = range(period, len(data)-1)
            for it in iterations:
                s = int(it-period)
                to_filter = data[s:it]
                X.append(to_filter)
            F = signal.sosfiltfilt(sos, X, padtype='constant')
            Y = [f[-1] for f in F]
            data = data[int(period+1):]
            index = dataframe_to_filter.index[int(period+1):]
            filter_signal = data - Y
            filtered_name = f"{btype} filtered {column}"
            filter_name = f"{btype} filter {column}"
            dictionary[filtered_name] = Y
            dictionary[filter_name] = filter_signal
            filter_dataframe = pd.DataFrame(data=dictionary,
                                            index=index)
            if plot_it:
                val_dataframe = filter_dataframe.join(
                    dataframe[[column]].apply(np.log10))
                val_dataframe.plot(figsize=[15,11])
                plt.show()
            filter_dataframe = filter_dataframe.dropna()
            if len(filter_dataframe.index) > 0:
                filter_dataframes = filter_dataframes.join(
                    filter_dataframe)
        
        threshold = int(len(filter_dataframes.index)*0.8)
        trainable_columns = list()
        for col in filter_dataframes.columns:
            length = len(filter_dataframes[col].dropna().index)
            if length >= threshold:
                trainable_columns.append(col)
        final = filter_dataframes[trainable_columns]
        final = unfiltered_dataframe.join(final)
        return final