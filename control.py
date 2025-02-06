# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 18:26:41 2021

@author: Andrew
"""
import os
import pandas as pd
from datetime import datetime as datetime_object


class Control(object):
    """ GET STOCK DATA. """
    
    drive = 'C'
    current_directory = f"{os.getcwd()}\\"
    project_directory = current_directory
    # project_directory = f"{drive}:\\Projects\\Hobby\\"
    
    log_directory = f"{project_directory}\\log\\"
    data_directory = f"{project_directory}data\\"
    csv_directory = f"{project_directory}csvs\\"
    pickle_directory = f"{project_directory}pickles\\"
    experiment_directory = f"{pickle_directory}experiments\\"
    raw_experiment_directory = f"{experiment_directory}raw\\"
    engineered_experiment_directory = f"{experiment_directory}engineered\\"
    results_pickle_directory = f"{pickle_directory}results\\"
    model_results_pickle_directory = f"{results_pickle_directory}models\\"
    realization_results_pickle_directory = f"{results_pickle_directory}realization\\"
    
    #TWS
    streaming_data_directory = f"{pickle_directory}streaming\\"
    account_streaming_data_directory = f"{streaming_data_directory}account\\"
    streaming_actions_directory = f"{account_streaming_data_directory}actions\\"
    order_status_directory =  f"{streaming_actions_directory}order_status\\"
    exec_details_directory = f"{streaming_actions_directory}exec_details\\"
    open_order_directory =  f"{streaming_actions_directory}open_order\\"
    
    # Actions
    action_directory = f"{pickle_directory}action\\"
    
        
    model_directory = f"{project_directory}models\\"
    keras_directory = f"{model_directory}Keras\\"
    
    sklearn_directory = f"{model_directory}Sklearn\\"
    sklearn_gboost_directory = f"{sklearn_directory}GBoost\\"
    sklearn_gboost_ml_directory = f"{sklearn_gboost_directory}ml\\"
    sklearn_gboost_scaler_directory = f"{sklearn_gboost_directory}scalers\\"
    sklearn_gboost_features_directory = f"{sklearn_gboost_directory}features\\"
    
    xgboost_directory = f"{model_directory}XGBoost\\"
    xgboost_ml_directory = f"{xgboost_directory}ml\\"
    xgboost_scaler_directory = f"{xgboost_directory}scalers\\"
    xgboost_features_directory = f"{xgboost_directory}features\\"
    
    xgboost_rf_directory = f"{model_directory}XGBoost RF\\"
    xgboost_rf_ml_directory = f"{xgboost_rf_directory}ml\\"
    xgboost_rf_scaler_directory = f"{xgboost_rf_directory}scalers\\"
    xgboost_rf_features_directory = f"{xgboost_rf_directory}features\\"
    
    scaler_save_name = 'scaler.pklz'
    features_save_name = 'features.pklz'
    
    train_denotation = 'train'
    test_denotation = 'test'
    validation_denotation = 'validation'
    
    # Feature Denotations
    hour_denotation = 'hour'
    day_denotation = 'day'
    weekday_denotation = 'weekday'
    month_denotation = 'month'
    year_denotation = 'year'
    week_denotation = 'week'
    create_timestamp_denotation = 'create_timestamp'
    
    errors_dictionary = dict()
    directory_contents = dict()
    
    start_date = '2016-01-01'
    end_date = '2021-08-01'
    time_interval = 'daily'
    assets = ['TSLA', 'MSFT']
    
    # Denotations when formatting json dictionary data.
    price_denotation = 'prices'
    date_denotation = 'date'
    index_denotation = 'formatted_date'
    
    period_denotation = 'period'
    direction_denotation = 'direction'
    baseline_denotation = 'baseline'
    backtest_denotation = 'backtest'
    
    train_denotation = 'train'
    test_denotation = 'test'
    validation_denotation = 'validation'
    volume_denotation = 'volume'
    close_denotation = 'close'
    open_denotation = 'open'
    adjclose_denotation = 'adjclose'
    log10_denotation = 'Log10'
    
    reset_data = False # If True Get new stock raw data - not content data.
    refresh = False # If True - Get new stock content data.
    update_data = False # If True - Updates Stock DataFrame.
    model_reset = False # If True - Retrains models.
    backtesting_condition = False # If True - Uses Iterative Periods for Validation (More Frequent Model Training)
    update_backtest = False
    visualization = False
    
    frequency_string = '1D'
    fs = 1/(3600*24)
    weekend = [5, 6]
    
    # IBAPI Denotations
    attributes_denotation = 'attributes'
    presets_denotation = 'presets'
    timestamp_denotation = 'timestamp'
    buy_denotation = 'BUY'
    sell_denotation = 'SELL'
    short_denotation = 'SHORT'
    move_denotation = 'move'
    quantity_denotation = 'quantity'
    enter_denotation = 'enter'
    exit_denotation = 'exit'
    # Buy or Sell
    enter_move_denotation = f"{enter_denotation} {move_denotation}"
    exit_move_denotation = f"{exit_denotation} {move_denotation}"
    # Volume
    enter_quantity_denotation = f"{enter_denotation} {quantity_denotation}"
    exit_quantity_denotation = f"{exit_denotation} {quantity_denotation}"
    # Contract Parameters
    symbol_denotation = 'SYMBOL'
    secType_denotation = 'secType' 
    exchange_denotation = 'exchange'
    currency_denotation = 'currency'
    
    orderType_denotation = 'orderType'
    tif_denotation = 'tif'
    orderId_denotation = 'orderId'
    action_denotation = 'action'
    totalQuantity_denotation = 'totalQuantity'
    lmtPrice_denotation = 'lmtPrice'
    
    
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
        self.__dict__.update(**kwargs)
        self.directory_onboarding()
        os.chdir(self.project_directory)
        self.derive_frequency_string()
        self.tickers_list = self.inspect_tickers()
    
    def check_directory(self, directory_to_check: str = str()):
        """
        

        Check Directory Information.
        
        ----------
        directories_to_check : list, optional
            DESCRIPTION:
                Checks a directory to determine if it exists or not. 
                    If it does exist then the content of the directory will
                        be saved as list into a dictionary.
                    If it does not exits then the directory will be created.
                
        """
        try:
            check_condition = False
            try:
                if type(directory_to_check) == str:
                    if len(directory_to_check) > 0:
                        check_condition = os.path.isdir(directory_to_check)
            except Exception as check_condition_error:
                logging_identifier = f"Check Directory Error  |  {datetime_object.now()}  |  Directory: {check_condition_error}    Check Condition: {check_condition}"
                self.errors_dictionary[logging_identifier] = \
                    check_condition_error
            
            if not check_condition:
                try:
                    os.mkdir(directory_to_check)
                except Exception as make_directory_error:
                    logging_identifier = f"Make Directory Error  |  {datetime_object.now()}  |  Directory: {directory_to_check}    Check Condition: {check_condition}"
                    self.errors_dictionary[logging_identifier] = \
                        make_directory_error
            else:
                try:
                    self.directory_contents[directory_to_check] = \
                        os.listdir(directory_to_check)
                except Exception as listdir_error:
                    logging_identifier = f"List Directory Content Error  |  {datetime_object.now()}  |  Directory: {directory_to_check}    Check Condition: {check_condition}"
                    self.errors_dictionary[logging_identifier] = listdir_error
                    
        except Exception as check_directory_error:
            logging_identifier = f"Make Directory Error  |  {datetime_object.now()}  |  Directory: {directory_to_check}"
            self.errors_dictionary[logging_identifier] = check_directory_error
    
    def directory_onboarding(self):
        """
        

        Onboard Directory Information to the Project.
        
        ----------
        Class Objects : self
            DESCRIPTION:
                
                Passes directories to check_directory function. The passed
                directory is checked and if it does not exist, the directory
                is create.
                
                Entire function lives in __init__ function.
                
        """
        self.check_directory(
            directory_to_check=self.project_directory)
        self.check_directory(
            directory_to_check=self.log_directory)
        self.check_directory(
            directory_to_check=self.data_directory)
        self.check_directory(
            directory_to_check=self.csv_directory)
        self.check_directory(
            directory_to_check=self.pickle_directory)
        self.check_directory(
            directory_to_check=self.experiment_directory)
        self.check_directory(
            directory_to_check=self.raw_experiment_directory)
        self.check_directory(
            directory_to_check=self.engineered_experiment_directory)
        self.check_directory(
            directory_to_check=self.results_pickle_directory)
        
        # TWS
        self.check_directory(
            directory_to_check=self.streaming_data_directory)
        self.check_directory(
            directory_to_check=self.account_streaming_data_directory)
        self.check_directory(
            directory_to_check=self.streaming_actions_directory)
        self.check_directory(
            directory_to_check=self.order_status_directory)
        self.check_directory(
            directory_to_check=self.exec_details_directory)
        self.check_directory(
            directory_to_check=self.open_order_directory)
        
        # Action
        self.check_directory(
            directory_to_check=self.action_directory)
        
        self.check_directory(
            directory_to_check=self.model_results_pickle_directory)
        self.check_directory(
            directory_to_check=self.realization_results_pickle_directory)
        self.check_directory(
            directory_to_check=self.model_directory)
        self.check_directory(
            directory_to_check=self.keras_directory)
        self.check_directory(
            directory_to_check=self.sklearn_directory)
        self.check_directory(
            directory_to_check=self.sklearn_gboost_directory)
        self.check_directory(
            directory_to_check= self.sklearn_gboost_ml_directory)
        self.check_directory(
            directory_to_check=self.sklearn_gboost_scaler_directory)
        self.check_directory(
            directory_to_check=self.sklearn_gboost_features_directory)
        self.check_directory(
            directory_to_check=self.xgboost_directory)
        self.check_directory(
            directory_to_check=self.xgboost_ml_directory)
        self.check_directory(
            directory_to_check=self.xgboost_scaler_directory)
        self.check_directory(
            directory_to_check=self.xgboost_features_directory)
        self.check_directory(
            directory_to_check=self.xgboost_rf_directory)
        self.check_directory(
            directory_to_check=self.xgboost_rf_ml_directory)
        self.check_directory(
            directory_to_check=self.xgboost_rf_scaler_directory)
        self.check_directory(
            directory_to_check=self.xgboost_rf_features_directory)
    
    def derive_frequency_string(self):
        if self.time_interval == 'hourly':
            self.frequency_string = 'h'
        if self.time_interval == 'daily':
            self.frequency_string = 'D'
        if self.time_interval == 'weekly':
            self.frequency_string = 'D'
        if self.time_interval == 'monthly':
            self.frequency_string = 'm'
        if self.time_interval == 'yearly':
            self.frequency_string = 'y'
    
    def inspect_tickers(self):
        tickers_file = f"{self.csv_directory}nasdaq_screener.csv"
        tickers = pd.read_csv(tickers_file)
        tickers['value'] = [
            float(t.split('$')[-1]) for t in tickers['Last Sale']]
        tickers = tickers.loc[(tickers['value'] > 5)
                              &
                              (tickers['value'] < 700)]
        tickers = tickers.loc[tickers['Country'] == 'United States']
        tickers = tickers.loc[tickers['Market Cap'] > 0]
        volume_min_threshold = 2500000 # Legacy
        volume_max_threshold = 50000000 # Legacy
        volume_min_threshold = 500000
        volume_max_threshold = 5000000000000
        tickers = tickers.loc[tickers['Volume'] > volume_min_threshold]
        tickers = tickers.loc[tickers['Volume'] < volume_max_threshold]
        symbols = tickers[['Symbol']]
        symbols_file = f"{self.csv_directory}symbols.csv"
        symbols.to_csv(symbols_file, index=False)
        
        if len(symbols.index) > 0:
            tickers.index = tickers[['Symbol']]
            self.tickers_df = symbols
            self.stock_information = tickers
        return symbols