# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:00:59 2025

@author: atara
"""


import pandas as pd

import xgboost as xgb
import lightgbm as lgbm

from get_data import GetData

import warnings
warnings.filterwarnings("ignore")

class CARTModels(GetData):
    
    visualize = False
    
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
    
    def lgboost_model(
            self,
            engineered_experiment: dict = dict(),
            target_column: str = str(),
            starting_point: float = 1000.0,
            threshold_str: str = '50%',
            parameters: dict = dict(),
            native_lgboost_params: dict = dict(),
            eval_metric: str = 'rmse',
            early_stopping_rounds: int = 4,
            confirmations: bool = False,
            verbose_eval: int  = 100,
            og_validation: pd.DataFrame = pd.DataFrame()
            ) -> dict:
        """
        

        Parameters
        ----------
        engineered_experiment : dict, optional
            DESCRIPTION. The default is dict().
        target_column : str, optional
            DESCRIPTION. The default is str().
        starting_point : float, optional
            DESCRIPTION. The default is 1000.0.
        threshold_str : str, optional
            DESCRIPTION. The default is '50%'.
        parameters : dict, optional
            DESCRIPTION. The default is dict().
        native_lgboost_params : dict, optional
            DESCRIPTION. The default is dict().
        eval_metric : str, optional
            DESCRIPTION. The default is 'rmse'.
        early_stopping_rounds : int, optional
            DESCRIPTION. The default is 4.
        confirmations : bool, optional
            DESCRIPTION. The default is False.
        verbose_eval : int, optional
            DESCRIPTION. The default is 100.
        og_validation : pd.DataFrame, optional
            DESCRIPTION. The default is pd.DataFrame().

        Returns
        -------
        dict
            DESCRIPTION.

        """
        output = dict()
        current_dataset_keys = list(engineered_experiment.keys())
        if not self.validation_denotation in current_dataset_keys: return output
        validation = engineered_experiment[self.validation_denotation].copy()
        
        train, test = None, None
        if self.model_reset:
            if self.train_denotation in current_dataset_keys:
                train = engineered_experiment[self.train_denotation].copy()
            if self.test_denotation in current_dataset_keys:
                test = engineered_experiment[self.test_denotation].copy()
        
        train[self.day_denotation] = train.index.day
        train[self.weekday_denotation] = train.index.weekday
        
        test[self.day_denotation] = test.index.day
        test[self.weekday_denotation] = test.index.weekday
        
        validation[self.day_denotation] = validation.index.day
        validation[self.weekday_denotation] = validation.index.weekday
        
        validation_df = validation.copy()
        train_columns = [c for c in validation_df if c != target_column]
        validation_df = validation_df[train_columns+[target_column]]
        # df = df.dropna()
        x_validation = validation_df[train_columns]
        
        if self.model_reset:
            y_validation = validation_df[[target_column]]
            y_validation = y_validation.fillna(0)
        
        if self.model_reset:
            df = train.copy().dropna()
            y_train = df[[target_column]]
            x_train = df[train_columns]
            lgbm_train = lgbm.Dataset(
                x_train, label=y_train, params={'verbose': -1})
            x_train, y_train = None, None
            
            df = test.copy().dropna()
            y_test = df[[target_column]]
            x_test = df[train_columns]
            lgbm_test = lgbm.Dataset(
                x_test, label=y_test, params={'verbose': -1})
            x_test, y_test = None, None
            
            if native_lgboost_params == dict():
                native_lgboost_params = {
                    'objective_type': 'regression',
                    'boosting_type': 'gbdt',
                    'num_iterations': 20000,
                    'eta': 0.1,
                    'tree': 'serial', 
                    'metric_type': 'rmse',
                    'early_stopping_round': early_stopping_rounds,
                    'verbose': -1}
        
        early_stopping = lgbm.early_stopping(
            stopping_rounds=early_stopping_rounds)
        
        selected_model = lgbm.train(
            native_lgboost_params,
            train_set=lgbm_train,
            valid_sets=[lgbm_test],
            callbacks=[early_stopping])
        
        forecast_column_name = f"LGBoost Forecast {target_column}"
        
        forecast_dataframe = pd.DataFrame(
            selected_model.predict(x_validation),
            index=x_validation.index,
            columns=[forecast_column_name])
        
        output['Raw Forecast'] = forecast_dataframe
        
        return output
    
    def xgboost_model(
            self,
            engineered_experiment: dict = dict(),
            target_column: str = str(),
            starting_point: float = 1000.0,
            threshold_str: str = '50%',
            parameters: dict = dict(),
            eval_metric: str = 'rmse',
            early_stopping_rounds: int = 4,
            confirmations: bool = False,
            verbose_eval: int  = 100,
            og_validation: pd.DataFrame = pd.DataFrame()
            ) -> dict:
        """
        

        Parameters
        ----------
        engineered_experiment : dict, optional
            DESCRIPTION. The default is dict().
        target_column : str, optional
            DESCRIPTION. The default is str().
        starting_point : float, optional
            DESCRIPTION. The default is 1000.0.
        threshold_str : str, optional
            DESCRIPTION. The default is '50%'.
        parameters : dict, optional
            DESCRIPTION. The default is dict().
        eval_metric : str, optional
            DESCRIPTION. The default is 'rmse'.
        early_stopping_rounds : int, optional
            DESCRIPTION. The default is 4.
        confirmations : bool, optional
            DESCRIPTION. The default is False.
        verbose_eval : int, optional
            DESCRIPTION. The default is 100.
        og_validation : pd.DataFrame, optional
            DESCRIPTION. The default is pd.DataFrame().

        Returns
        -------
        dict
            DESCRIPTION.

        """
        output = dict()
        booster = None
        current_dataset_keys = list(engineered_experiment.keys())
        if not self.validation_denotation in current_dataset_keys: return output
        validation = engineered_experiment[self.validation_denotation].copy()
        # original_validation = validation.copy()
        train, test = None, None
        if self.model_reset:
            if self.train_denotation in current_dataset_keys:
                train = engineered_experiment[self.train_denotation].copy()
            if self.test_denotation in current_dataset_keys:
                test = engineered_experiment[self.test_denotation].copy()
        
        train[self.day_denotation] = train.index.day
        train[self.weekday_denotation] = train.index.weekday
        
        test[self.day_denotation] = test.index.day
        test[self.weekday_denotation] = test.index.weekday
        
        validation[self.day_denotation] = validation.index.day
        validation[self.weekday_denotation] = validation.index.weekday
        
        validation_df = validation.copy()
        train_columns = [c for c in validation_df if c != target_column]
        validation_df = validation_df[train_columns+[target_column]]
        # df = df.dropna()
        x_validation = validation_df[train_columns]
        
        if self.model_reset:
            y_validation = validation_df[[target_column]]
            y_validation = y_validation.fillna(0)
            dval_predict = xgb.DMatrix(
                data=x_validation, label=y_validation)
        else:
            dval_predict = xgb.DMatrix(data=x_validation)
        
        if self.model_reset:
            df = train.copy().dropna()
            y_train = df[[target_column]]
            x_train = df[train_columns]
            dtrain = xgb.DMatrix(data=x_train, label=y_train)
            x_train, y_train = None, None
            
            df = test.copy().dropna()
            y_test = df[[target_column]]
            x_test = df[train_columns]
            dtest =  xgb.DMatrix(data=x_test, label=y_test)
            x_test, y_test = None, None
            
            if parameters == dict():
                parameters = {
                    'objective': 'reg:squarederror', # reg:pseudohubererror | reg:squarederror | reg:tweedie
                    'tree_method': 'exact',
                    'seed_per_iteration': True, # Default is False
                    'eta': 1,
                    'gamma': 1, 
                    'lambda': 1,
                    'alpha': 1,
                    'max_depth': 9,
                    'min_child_weight': 0,
                    'max_delta_step': 0,
                    'subsample': 0.91,
                    'colsample_bytree': 0.98,
                    'colsample_bylevel': 0.98,
                    'colsample_bynode': 0.98}
        
        num_boost_round = 100000
        if not self.visualize: verbose_eval = None
        
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name=eval_metric,
            data_name='test')
        
        booster = xgb.train(
            parameters, 
            dtrain,
            num_boost_round=num_boost_round,
            evals=[
                (dtrain, 'train'), 
                (dtest, 'test'), 
                (dval_predict, 'validation')],
            callbacks=[early_stop],
            maximize=True,
            verbose_eval=verbose_eval)
        
        forecast = booster.predict(dval_predict)
        forecast_column_name = f"XGBoost Forecast {target_column}"
        forecast_dataframe = pd.DataFrame(
            forecast,
            index=x_validation.index,
            columns=[forecast_column_name])
        
        output['Raw Forecast'] = forecast_dataframe
        
        return output
