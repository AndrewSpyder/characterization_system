# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 15:25:55 2021

@author: Andrew
"""
import numpy as np
import pandas as pd
from os import listdir
from datetime import timedelta

from functions import Functions

class PreProcess(Functions):
    """ Functions """
    
    
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
        self.__dict__.update(**kwargs)
    
    def prepare_raw_experiments(
            self,
            dataframe: pd.DataFrame = pd.DataFrame()
            ) -> dict:
        """
        
        Prepare Raw Experiments
        
        :param dataframe: DESCRIPTION, defaults to pd.DataFrame()
        :type dataframe: pd.DataFrame, optional
        :return: DESCRIPTION
        :rtype: dict

        """
        raw_experiments = dict()
        directory = self.raw_experiment_directory
        if not self.update_data:
            # Save Experiments TODO: Update Experiment Data
            current_years = listdir(directory)
            for filename in current_years:
                raw_experiments[filename.split('.')[0]] = self.load_data(
                    directory=directory, filename=filename)
            if len(raw_experiments.keys()) == len(current_years):
                return raw_experiments
        
        dataframe_columns = dataframe.columns
        
        # First - ID Starting Timestamps of Each Ticker.
        start_timestamps = list()
        for column in dataframe_columns:
            timestamp = dataframe[column].dropna().index[0]
            start_timestamps.append(timestamp)
        start_timestamps = np.unique(start_timestamps)
        start_timestamps = [
            pd.Timestamp(ts) for ts in np.sort(start_timestamps)]
        last_viable_timestamp = pd.Timestamp.now() - timedelta(days=90)
        last_viable_timestamp = pd.Timestamp(last_viable_timestamp)
        
        # Second - Establish Experiments
        for start_timestamp in start_timestamps:
            columns = list()
            for column in dataframe_columns:
                instance = dataframe[column].dropna()
                timestamp = instance.index[0]
                end_timestamp = instance.index[-1]
                conditions = list()
                conditions.append(timestamp <= start_timestamp)
                conditions.append(end_timestamp >= last_viable_timestamp)
                if all(conditions):
                    columns.append(column)
            if not start_timestamp in list(raw_experiments.keys()):
                experiment = dataframe[columns].loc[start_timestamp:]
                raw_experiments[start_timestamp] = experiment
        
        # Third - Reduce Number of Experiments - VIA Year
        years = dict()
        for start_timestamp in start_timestamps:
            year = start_timestamp.year
            if not year in list(years.keys()):
                years[year] = raw_experiments[start_timestamp]
        raw_experiments = years
        
        # Save Experiments TODO: Update Experiment Data
        for year in list(raw_experiments.keys()):
            filename = f"{year}.pkl"
            experiment = raw_experiments[year]
            self.dump_data(
                data=experiment,
                directory=directory,
                filename=filename)
        
        return raw_experiments
    
    def valiable_symbols(
            self,
            dataframe: pd.DataFrame = pd.DataFrame()
            ) -> list:
        datatypes = [self.volume_denotation,
                     self.close_denotation,
                     self.open_denotation,
                     self.adjclose_denotation]
        dictionary = dict()
        for datatype in datatypes:
            inspect = dataframe[[col for col in dataframe.columns
                                 if datatype in col]]
            inspect = inspect.replace(0, np.nan)
            keep_columns, threshold = list(), int(len(inspect.index)*0.6)
            for column in inspect.columns:
                view = inspect[column]
                if isinstance(view, pd.DataFrame):
                    if len(view.columns) > 0:
                        view = view.T
                        view.index = range(len(view.index))
                        view = view.T
                        prior, keep = 0, None
                        for t in view:
                            if isinstance(keep, type(None)):
                                keep = t
                            length = len(view[t].dropna())
                            if length > prior:
                                prior = length
                                keep = t
                        inspect = inspect.drop(columns=[column])
                        inspect[column] = view[keep] 
                
                length = len(inspect[column].dropna().index)
                if length >= threshold:
                    col = column.split(f" {datatype}")[0]
                    keep_columns.append(col)
            keep_columns = np.unique(keep_columns)
            columns_to_keep = list()
            for symbol in keep_columns:
                datatype_to_keep = [col for col in dataframe.columns
                                    if f"{symbol} " in col]
                columns_to_keep = np.append(columns_to_keep, datatype_to_keep)
            columns_to_keep = [col for col in np.unique(columns_to_keep)]
            dictionary[datatype] = columns_to_keep
        columns = list()
        for column in dataframe.columns:
            condition = list()
            for datatype in datatypes:
                condition.append(column in dictionary[datatype])
            if all(condition):
                columns.append(column)
        return columns
    
    def select_train_liquid_features(
            self, 
            train: pd.DataFrame = pd.DataFrame(),
            days: int = 30 # Defaults to a month look-back.
            ) -> list:
        """
        
        :param train: DESCRIPTION, defaults to pd.DataFrame()
        :type train: pd.DataFrame, optional
        :param days: DESCRIPTION, defaults to 30 # Defaults to a month look-back.
        :type days: int, optional
        :return: DESCRIPTION
        :rtype: list

        """
        signals = list()
        try:
            volume_df = train.copy()
            target_columns = [
                col for col in volume_df.columns
                if self.volume_denotation in col]
            volume_df = volume_df[target_columns]
            threshold = volume_df.mean(axis=1).describe()['25%']
            tail = volume_df.tail(days).mean(axis=0)
            tail = tail.dropna()
            tail = tail.loc[tail > threshold]
            split_by = f" {self.volume_denotation}"
            liquid_signals = [sym.split(split_by)[0] for sym in tail.index]
            columns = train.columns
            for column in columns:
                for liquid_signal in liquid_signals:
                    if not liquid_signal in column: continue
                    signals.append(column)
            signals = [s for s in np.unique(signals)]
        except Exception as e:
            print(e)
        return signals
    
    def apply_filtering(
            self,
            dataframe: pd.DataFrame = pd.DataFrame()
            ) -> pd.DataFrame:
        """
        
        :param dataframe: DESCRIPTION, defaults to pd.DataFrame()
        :type dataframe: pd.DataFrame, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        output = None
        columns_to_filter = list()
        # Highpass Filter Inputs
        order = 9
        fs = self.fs
        btype = 'band'
        log_condition = False
        lowcut = 1/(3600*24*24)
        highcut = 1/(3600*24*7) # Weekly Period
        highcut = 1/(3600*24*3) # Weekly Period
        output = self.butterworth_filtering(
            dataframe=dataframe.copy(),
            columns_to_filter=columns_to_filter,
            fs=fs,
            btype=btype,
            lowcut=lowcut,
            highcut=highcut,
            order=order,
            log_condition=log_condition,
            plot_it=False)
        return output
    
    def generate_filename(
            self,
            year: int or str = str(),
            target_column: str = str(),
            filter_condition: bool = True,
            persistence_condition: bool = True
            ) -> str:
        """
        
        :param year: DESCRIPTION, defaults to str()
        :type year: int or str, optional
        :param filter_condition: DESCRIPTION, defaults to True
        :type filter_condition: bool, optional
        :param persistence_condition: DESCRIPTION, defaults to True
        :type persistence_condition: bool, optional
        :return: DESCRIPTION
        :rtype: str

        """
        filename = f"{target_column} {str(year)}"
        if filter_condition:
            filename = f"{filename} filtered"
        if persistence_condition:
            filename = f"{filename} persistence"
        filename = f"{filename}.pkl"
        return filename
    
    def build_target(
            self,
            dataframe: pd.DataFrame = pd.DataFrame()
            ) -> pd.DataFrame:
        df = dataframe
        symbols = np.unique([col.split(' ')[0] for col in dataframe.columns])
        for symbol in symbols:
            name = f"{symbol} {self.direction_denotation}"
            direction = df[f"{symbol} {self.close_denotation}"] -\
                df[f"{symbol} {self.open_denotation}"]
            dataframe[name] = direction
        return dataframe
    
    def extend_forecast(
            self,
            dataframe: pd.DataFrame = pd.DataFrame(),
            days_out: int = 7
            ) -> pd.DataFrame:
        try:
            start = dataframe.index[-1] + timedelta(days=1)
            end = dataframe.index[-1] + timedelta(days=days_out)
            extend_by = pd.date_range(start, end, freq=self.frequency_string)
            fill_name = 'Fill Frame'
            fill_frame = pd.DataFrame(
                np.zeros(len(extend_by)),
                columns=[fill_name],
                index=extend_by)
            for c in dataframe:
                fill_frame[c] = np.nan
            fill_frame = fill_frame.drop(columns=[fill_name])
            idx = [t for t in fill_frame.index if not t in dataframe.index]
            if len(idx) > 0:
                dataframe = pd.concat([dataframe, fill_frame.loc[idx]])
        except Exception as e:
            print(e)
            print('extend_forecast Error')
        return dataframe
    
    def broad_feature_engineering(
            self,
            dataframe: pd.DataFrame = pd.DataFrame(),
            target_columns: list = list(),
            filter_condition: bool = False,
            persistence_condition: bool = False,
            rolling_periods: list = [8, 13, 35, 62],
            shift: int = 7
            ) -> pd.DataFrame:
        try:
            original_idx = dict()
            for target_column in target_columns:
                idx = dataframe.dropna(subset=[target_column])
                original_idx[target_column] = idx.index
            dataframe = self.extend_forecast(dataframe=dataframe)
            original_dataframe = dataframe[target_columns].copy()
            dataframe = dataframe.resample(self.frequency_string).mean()
            dataframe = dataframe.interpolate(method='ffill')
            t1 = str(dataframe.index[0]) # Set original boundaries
            t2 = str(dataframe.index[-1]) # Set original boundaries
            idx = [t for t in dataframe.index
                   if not t.weekday() in self.weekend]
            dataframe = dataframe.loc[idx].copy()
            dataframe = self.temporal_features(dataframe=dataframe)
            if filter_condition:
                dataframe = self.apply_filtering(dataframe=dataframe)
            if persistence_condition:
                dataframe = self.rolling_average_features(
                    dataframe=dataframe.copy(),
                    rolling_periods=rolling_periods)
                dataframe = self.persistence_features(
                    dataframe=dataframe.copy(),
                    shift=shift)
            columns_to_replace = [
                col for col in target_columns
                if col in dataframe.columns]
            # This removes potential false positives. 
            # It is done for each target.
            for column_to_replace in columns_to_replace:
                df = original_dataframe[[column_to_replace]].copy()
                idx = original_idx[column_to_replace]
                df = df.loc[idx]
                df = df.loc[t1:t2]
                dataframe = dataframe.drop(columns=df.columns)
                # Maintain dataframe.index but ensure raw target information.
                dataframe = dataframe.join(df) # Maintain dataframe.index
        except Exception as e:
            print(e)
            print('Error in Broad Feature Engineering')
        return dataframe
    
    def prepare_engineered_experiments(
            self,
            raw_experiments: dict = dict(),
            train_test_split: str or float = 0.55,
            test_validation_split: str or float = 0.75,
            buffer: str or int = 7, # Include last # of days.
            select_liquid_signals: bool = True,
            filter_condition: bool = True,
            persistence_condition: bool = True
            ) -> dict:
        """
        
        Prepare Engineered Experiments
        
        :param raw_experiments: DESCRIPTION, defaults to dict()
        :type raw_experiments: dict, optional
        :param train_test_split: DESCRIPTION, defaults to 0.55
        :type train_test_split: str or float, optional
        :param test_validation_split: DESCRIPTION, defaults to 0.75
        :type test_validation_split: str or float, optional
        :param buffer: DESCRIPTION, defaults to 1
        :type buffer: str or int, optional
        :return: DESCRIPTION
        :rtype: dict
        
        train_test_split = 0.55
        select_liquid_signals = True

        """
        engineered_experiments = dict()
        directory = self.engineered_experiment_directory
        experiment_years = list(raw_experiments.keys())
        years = list()
        if isinstance(self.selected_year, list):
            for select_year in self.selected_year:
                for year in experiment_years:
                    if str(year) == str(select_year):
                        years.append(year)
        target_columns = None
        if not self.update_data:
            # Save Experiments TODO: Update Experiment Data
            target_columns = listdir(directory)
            for year in years:
                for target_column in target_columns:
                    target_column = target_column.split(' ')[0]
                    filename = self.generate_filename(
                        year=year,
                        target_column=target_column,
                        filter_condition=filter_condition,
                        persistence_condition=persistence_condition)
                    if not filename in target_columns: continue
                    try:
                        if not year in engineered_experiments:
                            engineered_experiments[year] = dict()
                        engineered_experiments[
                            year][target_column] = self.load_data(
                                directory=directory,
                                filename=filename)
                    except Exception as e:
                        print(e)
                        print(f"Failed to load in - {filename}")
            return engineered_experiments
        
        for year in years:
            # year = str(year)
            dataframe = raw_experiments[year].copy()
            if select_liquid_signals:
                # Make Selections Based on Current Liquidity Assumptions 
                liquid_signals = self.select_train_liquid_features(
                    train=dataframe)
                dataframe = dataframe[liquid_signals]
                liquid_signals = self.valiable_symbols(dataframe=dataframe)
                dataframe = dataframe[liquid_signals]
            
            dataframe = self.build_target(dataframe=dataframe)
            target_columns = [col for col in dataframe.columns
                              if self.direction_denotation in col]
            
            rolling_periods = [8, 13, 35, 62]
            shift = buffer
            dataframe = self.broad_feature_engineering(
                dataframe=dataframe,
                target_columns=target_columns,
                filter_condition=filter_condition,
                persistence_condition=persistence_condition,
                shift=shift,
                rolling_periods=rolling_periods)
            
            if isinstance(train_test_split, float):
                train_end = len(dataframe.index)*train_test_split
                train_end = int(np.floor(train_end))
                true_test_begin = dataframe.index[train_end+1]
                test_begin = dataframe.index[train_end-buffer] # Reverse the buffer and cut later.
                train_end = dataframe.index[train_end]
            if isinstance(test_validation_split, float):
                test_end = len(dataframe.index)*test_validation_split
                test_end = int(np.floor(test_end))
                true_validation_begin = dataframe.index[test_end+1]
                validation_begin = dataframe.index[test_end-buffer] # Reverse the buffer and cut later.
                test_end = dataframe.index[test_end]
            
            for target_column in target_columns:
                try:
                    target_column = target_column.split(' ')[0]
                    filename = self.generate_filename(
                        year=year,
                        target_column=target_column,
                        filter_condition=filter_condition,
                        persistence_condition=persistence_condition)
                    
                    data = dataframe[[
                        c for c in dataframe
                        if target_column == c.split(' ')[0]]]
                    
                    train = data.loc[:train_end]
                    test = data.loc[test_begin:test_end]
                    validation = data.loc[validation_begin:]
                    
                    engineered = dict()
                    engineered[self.train_denotation] = train
                    engineered[self.test_denotation] = test
                    engineered[self.validation_denotation] = validation
                    
                    val_columns = engineered[self.validation_denotation].columns
                    columns_train = engineered[self.train_denotation].columns
                    columns_test = engineered[self.test_denotation].columns
                    keep_columns = list()
                    for col in val_columns:
                        conditions = list()
                        conditions.append(col in columns_train)
                        conditions.append(col in columns_test)
                        if not all(conditions): continue
                        keep_columns.append(col)
                    engineered[self.train_denotation] = engineered[
                        self.train_denotation][keep_columns]
                    engineered[self.test_denotation] = engineered[
                        self.test_denotation][keep_columns].loc[true_test_begin:]
                    engineered[self.validation_denotation] = engineered[
                        self.validation_denotation][keep_columns].loc[
                            true_validation_begin:]
                    self.dump_data(
                        data=engineered,
                        directory=directory,
                        filename=filename)
                except Exception as e:
                    print(e)
        
        engineered_experiments = dict()
        target_columns = listdir(directory)
        for year in years:
            for target_column in target_columns:
                target_column = target_column.split(' ')[0]
                filename = self.generate_filename(
                    year=year,
                    target_column=target_column,
                    filter_condition=filter_condition,
                    persistence_condition=persistence_condition)
                if not filename in target_columns: continue
                try:
                    if not year in engineered_experiments:
                        engineered_experiments[year] = dict()
                    engineered_experiments[
                        year][target_column] = self.load_data(
                            directory=directory,
                            filename=filename)
                except Exception as e:
                    print(e)
                    print(f"Failed to load in - {filename}")
        
        return engineered_experiments
    