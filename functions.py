# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 12:58:02 2021

@author: Andrew
"""
import os
import gzip
import numpy as np
import pandas as pd
import _pickle as cpickle
from sklearn.preprocessing import PowerTransformer

from feature_engineering import FeatureEngineering

class Functions(FeatureEngineering):
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
    
    def scale_training_data(
            self, 
            dataframe_to_scale: pd.DataFrame = pd.DataFrame(),
            scaler_save_name: str = str()
            ) -> dict:
        """
        

        Scale Training DataFrame.
        
        ----------
        dataframe_to_scale : pd.DataFrame, optional
            DESCRIPTION. The default is pd.DataFrame().
                Scales the training dataframe and returns the output as both
                a Pandas DataFrame & Numpy Array.
                
                The scaler is saved as well.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        scaler = PowerTransformer()
        scaler.fit(dataframe_to_scale)
        columns = dataframe_to_scale.columns
        index = dataframe_to_scale.index
        scaled_data_numpy = scaler.transform(dataframe_to_scale)
        scaled_data_dataframe = pd.DataFrame(scaled_data_numpy,
                                             columns=columns,
                                             index=index)
        if type(scaler_save_name) == str:
            if scaler_save_name != str():
                with gzip.GzipFile(scaler_save_name, 'wb') as stream:
                    cpickle.dump(scaler, stream)
        
        return {'Pandas': scaled_data_dataframe,
                'Numpy': scaled_data_numpy,
                'Scaler': scaler}
    
    def transform_testing_data(
            self, 
            dataframe_to_scale: pd.DataFrame = pd.DataFrame(),
            scaler_save_name: str = str()
            ) -> dict:
        """
        

        Transform Testing DataFrame.
        
        ----------
        dataframe_to_scale : pd.DataFrame, optional
            DESCRIPTION. The default is pd.DataFrame().
                Attempts to extract the saved scaler and then apply it to the
                testing dataframe.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        if type(scaler_save_name) == str:
            if scaler_save_name != str():
                with gzip.GzipFile(scaler_save_name, 'rb') as stream:
                    scaler = cpickle.load(stream)
        
        try:
            scaled_data_numpy = scaler.transform(dataframe_to_scale)
        except Exception as transform_error:
            print(transform_error)
            return {'Failed': None}
        
        columns = dataframe_to_scale.columns
        index = dataframe_to_scale.index
        scaled_data_dataframe = pd.DataFrame(scaled_data_numpy,
                                             columns=columns,
                                             index=index)
        
        return {'Pandas': scaled_data_dataframe,
                'Numpy': scaled_data_numpy}

    def build_fill_frame(
            self, 
            start: str or pd.Timestamp or None = None,
            end: str or pd.Timestamp or None = None,
            fill_name: str = 'Fill'
            ) -> pd.DataFrame:
        """
        
        Generic DataFrame Build
        
        :param start: DESCRIPTION, defaults to None
        :type start: str or pd.Timestamp or None, optional
        :param end: DESCRIPTION, defaults to None
        :type end: str or pd.Timestamp or None, optional
        :param fill_name: DESCRIPTION, defaults to 'Fill'
        :type fill_name: str, optional
        :return: DESCRIPTION
        :rtype: TYPE
    
        """
        check = all([not isinstance(start, type(None)),
                     not isinstance(end, type(None))])
        if not check:
            return
        date_range = pd.date_range(
            start=start,
            end=end,
            freq=self.frequency_string)
        fill_frame = pd.DataFrame(
            data={fill_name:np.zeros(len(date_range))},
            index=date_range)
        return fill_frame
    
    def prepare_dataframe(
            self,
            data: dict = dict()
            ) -> pd.DataFrame or None:
        """
        
        Prepare Asset DataFrame
        
        :param data: DESCRIPTION, defaults to dict()
        :type data: dict, optional
        :return: DESCRIPTION
        :rtype: TYPE
    
        """
        condition = False
        try:
            if isinstance(data, dict):
                data_keys = list(data.keys())
                condition = len(data_keys) > 0
        except Exception as e:
            print(e)
        if not condition: return
        
        asset_df_dict = dict()
        for asset in data_keys:
            asset_df = None
            try:
                asset_keys = list(data[asset].keys())
                if not self.price_denotation in asset_keys: continue
                asset_df = pd.DataFrame(
                    data[asset][self.price_denotation])
                asset_df = asset_df.drop(self.date_denotation, axis=1).\
                    set_index(self.index_denotation)
                rename_columns = dict()
                for col in asset_df.columns:
                    rename_columns[col] = f"{asset} {col}"
                asset_df = asset_df.rename(columns=rename_columns)
            except Exception as e:
                print(e)
            if not isinstance(asset_df, type(None)):
                asset_df_dict[asset] = asset_df
        list_of_dataframes = list()
        assets = list(asset_df_dict.keys())
        for asset in assets:
            df = asset_df_dict[asset]
            if not isinstance(df, pd.DataFrame): continue
            list_of_dataframes.append(df)
        dataframe = self.build_dataframe(
            list_of_dataframes=list_of_dataframes)
        return dataframe
            
    def build_dataframe(
            self,
            list_of_dataframes: list = list()
            ) -> pd.DataFrame or None:
        dataframe = None
        start_timestamps, end_timestamps = list(), list()
        for df in list_of_dataframes:
            if not isinstance(df, pd.DataFrame): continue
            start_timestamps.append(pd.Timestamp(df.index[0]))
            end_timestamps.append(pd.Timestamp(df.index[-1]))
        
        start, end = np.min(start_timestamps), np.max(end_timestamps)
        fill_name = 'Fill'
        fill_frame = self.build_fill_frame(start=start,
                                           end=end,
                                           fill_name=fill_name)
        for df in list_of_dataframes:
            try:
                df.index = [pd.Timestamp(ts) for ts in df.index]
                fill_frame = fill_frame.join(df)
            except Exception as e:
                print(e)
        if len(fill_frame.columns) == 1: return
        dataframe = fill_frame.drop(columns=[fill_name])
        return dataframe
    
    def load_data(
            self,
            directory: str = str(),
            filename: str = str()
            ) -> pd.DataFrame or dict or None:
        """
        
        Load Stock Related Data
        
        :param directory: DESCRIPTION, defaults to str()
        :type directory: str, optional
        :param filename: DESCRIPTION, defaults to str()
        :type filename: str, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        data, filename_directory = None, f"{directory}{filename}"
        try:
            if filename in os.listdir(directory):
                with open(filename_directory, 'rb') as stream:
                    data = cpickle.load(stream)
            conditions = list()
            conditions.append(isinstance(data, dict))
            conditions.append(isinstance(data, pd.DataFrame))
            if any(conditions):
                return data
        except Exception as e:
            print(e)
    
    def dump_data(
            self,
            data: dict or pd.DataFrame = pd.DataFrame(),
            directory: str = str(),
            filename: str = str()
            ):
        """
        
        Dump Stock Related Data
        
        :param data: DESCRIPTION, defaults to pd.DataFrame()
        :type data: dict or pd.DataFrame, optional
        :param directory: DESCRIPTION, defaults to str()
        :type directory: str, optional
        :param filename: DESCRIPTION, defaults to str()
        :type filename: str, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        filename_directory = f"{directory}{filename}"
        try:
            with open(filename_directory, 'wb') as stream:
                cpickle.dump(data, stream)
        except Exception as e:
            print(e)
            