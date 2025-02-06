# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:04:50 2025

@author: atara
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

from cart_models import CARTModels


class ModelOrchestration(CARTModels):
    
    starting_point = 1000
    filter_condition = False
    persistence_condition = True
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
    
    def backtesting_method(
            self,
            dataframe: pd.DataFrame = pd.DataFrame(),
            engineered_experiment: dict = dict(),
            target_columns: list = list(),
            period: int = 7,
            delay: int = 1,
            starting_point: float = 1000.0,
            year: int or str = 2016
            ):
        # Calculate Models
        horizon = 1000
        view_it = False
        normal_split = False
        if view_it: pnls = pd.DataFrame()
        percent_split = 0.5
        found_target_columns = list()
        for target_column in target_columns:
            if 'NEE' in target_column: continue
            if 'NEP' in target_column: continue
            found_target_columns.append(target_column)
        for target_column in found_target_columns:
            try:
                directory = self.engineered_experiment_directory
                if self.direction_denotation in target_column:
                    target_column = target_column.split(' ')[0]
                filename = self.generate_filename(
                    year=year,
                    target_column=target_column,
                    filter_condition=self.filter_condition,
                    persistence_condition=self.persistence_condition)
                try:
                    content = self.load_data(
                        directory=directory,
                        filename=filename)
                except Exception as e:
                    print(e)
                    print(f"Failed to load in - {filename}")
                target_column = f"{target_column} {self.direction_denotation}"
                train = content[self.train_denotation].copy()
                test = content[self.test_denotation].copy()
                validation = content[self.validation_denotation].copy()
                content = None
                train.index = [pd.Timestamp(ts) for ts in train.index]
                test.index = [pd.Timestamp(ts) for ts in test.index]
                validation.index = [
                    pd.Timestamp(ts) for ts in validation.index]
                information = pd.concat(
                    [train.copy(),
                     test.copy(),
                     validation.copy()])
                information.index = [
                    pd.Timestamp(ts) for ts in information.index
                    if not ts.weekday in self.weekend]
                information = information.reset_index(
                    ).sort_values('index').drop_duplicates(
                        'index').set_index('index')
                
                model_results = dict()
                try:
                    start_ts = pd.Timestamp(
                        validation.dropna(
                            subset=[target_column]).index[-1]
                        ) - timedelta(days=420)
                    end_ts = pd.Timestamp(
                        validation.dropna(
                            subset=[target_column]).index[-1]
                        ) + timedelta(days=period)
                    start_periods = [
                        pd.Timestamp(ts) for ts in 
                        pd.date_range(
                            start_ts, end_ts, freq=self.frequency_string)]
                    start_periods = [
                        t for t in start_periods
                        if not t.weekday() in self.weekend]
                    filename = f"{target_column}.pkl"
                    prior_results = self.load_data(
                        directory=self.model_results_pickle_directory,
                        filename=filename)
                    conditions = list()
                    conditions.append(isinstance(prior_results, dict))
                    conditions.append(self.update_backtest)
                    conditions.append(not view_it)
                    if all(conditions):
                        prior_periods = [t for t in prior_results][:-10]
                        start_periods = [
                            t for t in start_periods
                            if not t in prior_periods]
                        start_periods = [
                            pd.Timestamp(ts) for ts in
                            np.sort(start_periods)]
                        model_results = prior_results # Reassign and Continue
                except Exception as e:
                    print(e)
                    print(f"Establishing iterations Error on {target_column}")
                daily = pd.DataFrame()
                windows = dict()
                window_descriptions_train = dict()
                window_descriptions_test = dict()
                xgb_instance, lgb_instance = dict(), dict()
                for start_ts in start_periods:
                    try:
                        info = information.copy()
                        new_validation = info.copy().loc[
                            start_ts:]
                        
                        train_test_ts = info.copy().loc[
                            :start_ts].index[-2]
                        train_test = info.copy().loc[
                            :train_test_ts]
                        
                        to_filter = train_test[[target_column]].copy()
                        
                        order = 3
                        fs = self.fs
                        btype = 'band'
                        log_condition = False
                        
                        lowcut = 1/(3600*24*24)
                        highcut = 1/(3600*24*2.1) # Weekly Period
                        
                        output = self.butterworth_filtering(
                            dataframe=to_filter.copy(),
                            columns_to_filter=[target_column],
                            fs=fs,
                            btype=btype,
                            lowcut=lowcut,
                            highcut=highcut,
                            order=order,
                            log_condition=log_condition,
                            plot_it=False)
                        
                        train_test[target_column] = output[
                            f"band filter {target_column}"]
                        
                        unique = train_test[target_column].copy().dropna()
                        unique *= 10
                        unique = unique.round(0).sort_values()
                        unique_values = [v for v in unique.unique()]
                        simplify = pd.DataFrame() 
                        new_train, new_test = pd.DataFrame(), pd.DataFrame()
                        for unique_value in unique_values:
                            try:
                                vals = unique.loc[unique == unique_value]
                                vals = vals.to_frame()
                                if len(vals.index) < 10:
                                    simplify = pd.concat([simplify, vals])
                                else:
                                    if len(simplify.index) == 0:
                                        simplify = vals
                                    simplify['ts'] = simplify.index
                                    simplify = simplify.sort_values(by='ts')
                                    
                                    if normal_split:
                                        split_ts = simplify.index[
                                            int(len(
                                                simplify.index)*percent_split)]
                                        new_train_set = simplify[[
                                            target_column]].loc[:split_ts]
                                        split_ts = simplify.loc[
                                            split_ts:].index[1]
                                        new_test_set = simplify[[
                                            target_column]].loc[split_ts:]
                                    else:
                                        idx = simplify.index
                                        length = len(idx)
                                        groups = int(np.floor(
                                            length * percent_split))
                                        steps = length - groups
                                        batch = int(np.floor(length/steps))
                                        s_b, s_e, e = 0, batch, batch+1
                                        train_ids, test_ids = [], []
                                        for step in range(steps):
                                            try:
                                                while e >= len(idx):
                                                    s_e -= 1
                                                    e -= 1
                                                train_id = [
                                                    t for t in idx[s_b:s_e]]
                                                train_id = [
                                                    t for t in train_id
                                                    if not t in train_ids]
                                                test_id = [
                                                    t for t in [idx[e]]]
                                                test_id = [
                                                    t for t in test_id
                                                    if not t in test_ids]
                                                train_ids += train_id
                                                test_ids += test_id
                                                s_b = e+1
                                                s_e = s_b + batch
                                                e = s_e + 1
                                            except Exception as e:
                                                print(e)
                                            
                                        new_train_set = simplify[[
                                            target_column]].loc[train_ids]
                                        new_test_set = simplify[[
                                            target_column]].loc[test_ids]
                                    
                                    new_train_set = info.loc[
                                        new_train_set.index]
                                    new_test_set = info.loc[
                                        new_test_set.index]
                                    
                                    new_train = pd.concat(
                                        [new_train, new_train_set])
                                    new_test = pd.concat(
                                        [new_test, new_test_set])
                                    simplify = pd.DataFrame()
                            except Exception as e:
                                print(e)
                        
                        stock = target_column.split(' ')[0]
                        only_target_data = True
                        upper = 4
                        train_columns = [
                            col for col in new_validation.columns
                            if col[-1] in [str(x) for x in range(delay, upper)]
                            or col == self.weekday_denotation
                            or col == self.day_denotation
                            or col == self.week_denotation]
                        train_columns = [col for col in train_columns
                                         if 'Shifted' in col
                                         and 'Rolling' in col]
                                         # or col == self.weekday_denotation
                                         # or col == self.day_denotation]
                        if only_target_data:
                            train_columns = [
                                col for col in train_columns
                                if stock == col.split(' ')[0]
                                or self.weekday_denotation in col
                                or self.day_denotation in col
                                or col == self.week_denotation]
                        train_columns += [target_column]
                        
                        new_train = new_train.reset_index(
                            ).sort_values('index').set_index('index')
                        new_test = new_test.reset_index(
                            ).sort_values('index').set_index('index')
                        
                        
                        new_train_lgb = new_train.loc[new_train.index[-38]:]
                        new_test_lgb = new_test.loc[new_test.index[-38]:]
                        
                        descriptions = dict()
                        N = 90
                        for n in range(20, N):
                            ins_train = new_train.loc[new_train.index[-n]:]
                            descriptions[n] = ins_train[
                                target_column].abs().describe().loc['std']
                        descriptions = pd.DataFrame.from_dict(
                            descriptions, orient='index')
                        
                        threshold = descriptions[0].describe().loc['25%']
                        select = [i for i in 
                                  descriptions.loc[
                                      descriptions[0] >= threshold].index]
                        select = int(np.nanmean(select))
                        
                        new_train_lgb = new_train.loc[
                            new_train.index[-select]:]
                        new_test_lgb = new_test.loc[
                            new_test.index[-select]:]
                        
                        windows[start_ts] = select
                        window_descriptions_train[start_ts] = new_train_lgb[
                            target_column].describe()
                        window_descriptions_test[start_ts] = new_test_lgb[
                            target_column].describe()
                        
                        new_train_xgb = new_train.loc[
                            new_train.index[-select]:]
                        new_test_xgb = new_test.loc[
                            new_test.index[-select]:]
                        
                        lgb_instance[self.train_denotation] = new_train_lgb[
                            train_columns]
                        lgb_instance[self.test_denotation] = new_test_lgb[
                            train_columns]
                        lgb_instance[
                            self.validation_denotation] = new_validation[
                                train_columns]
                        
                        xgb_instance[self.train_denotation] = new_train_xgb[
                            train_columns]
                        xgb_instance[self.test_denotation] = new_test_xgb[
                            train_columns]
                        xgb_instance[
                            self.validation_denotation] = new_validation[
                                train_columns]
                        threshold_str = 'min' # 25% 50% 75%
                        
                        og_validation = information.copy().loc[
                            new_validation.index]
                        
                        eval_metric = 'rmse'
                        parameters = {
                            'objective': 'reg:squarederror', # reg:pseudohubererror | reg:squarederror
                            'tree_method': 'hist',
                            'seed_per_iteration': True, # Default is False
                            'eta': 0.1,
                            'gamma': 0, 
                            'lambda': 0,
                            'alpha': 0,
                            'max_depth': 9,
                            'min_child_weight': 0,
                            'max_delta_step': 0,
                            'subsample': 0.92,
                            'colsample_bytree': 0.98,
                            'colsample_bylevel': 0.98,
                            'colsample_bynode': 0.98,
                            'eval_metric': eval_metric}
                        
                        for esr in [9]:
                            early_stopping_rounds = esr
                            
                            eval_metric = 'mae'
                            parameters = {
                                'objective': 'reg:squarederror', # reg:pseudohubererror | reg:squarederror
                                'tree_method': 'exact',
                                'seed_per_iteration': True, # Default is False
                                'eta': 1,
                                'gamma': 0, 
                                'lambda': 0,
                                'alpha': 0,
                                'max_depth': 25,
                                'min_child_weight': 0,
                                'max_delta_step': 0,
                                'subsample': 0.5,
                                'colsample_bytree': 0.99,
                                'colsample_bylevel': 0.99,
                                'colsample_bynode': 0.99,
                                'eval_metric': eval_metric}
                            
                            native_lgboost_params = {
                                'objective_type': 'regression',
                                'boosting_type': 'gbdt',
                                'num_iterations': 20000,
                                'eta': 0.1,
                                'tree': 'serial', 
                                'metric_type': 'rmse',
                                'early_stopping_round': early_stopping_rounds,
                                'verbose': -1}
                            
                            engineered_experiment = xgb_instance.copy()
                            confirmations = self.model_reset
                            xgboost_output = self.xgboost_model(
                                engineered_experiment=engineered_experiment,
                                target_column=target_column,
                                starting_point=starting_point,
                                threshold_str=threshold_str,
                                parameters=parameters,
                                eval_metric=eval_metric,
                                early_stopping_rounds=early_stopping_rounds,
                                confirmations=confirmations,
                                og_validation=og_validation)
                            
                            early_stopping_rounds = esr
                            engineered_experiment = lgb_instance.copy()
                            confirmations = self.model_reset
                            lgboost_output = self.lgboost_model(
                                engineered_experiment=engineered_experiment,
                                target_column=target_column,
                                starting_point=starting_point,
                                threshold_str=threshold_str,
                                native_lgboost_params=native_lgboost_params,
                                eval_metric=eval_metric,
                                early_stopping_rounds=early_stopping_rounds,
                                confirmations=confirmations,
                                og_validation=og_validation)
                            
                            x_view = xgboost_output['complex']['pnl'].copy()
                            l_view = lgboost_output['complex']['pnl'].copy()
                            
                            day_x_view = x_view.loc[start_ts].to_frame().T
                            day_l_view = l_view.loc[start_ts].to_frame().T
                            day_x_view = day_x_view.add_suffix('_XGB')
                            day_l_view = day_l_view.add_suffix('_LGB')
                            day_view = day_x_view.join(day_l_view)
                            daily = pd.concat([daily, day_view])
                            
                            if view_it:
                                daily.fillna(0).cumsum().plot(
                                    figsize=[20,10],
                                    linewidth=8,
                                    fontsize=28)
                                plt.title(target_column, fontsize=28)
                                plt.grid(True, which='both')
                                plt.show()
                        
                        if not view_it:
                            model_results[start_ts] = dict()
                            model_results[start_ts]['xgboost'] = xgboost_output
                            model_results[start_ts]['lgboost'] = lgboost_output
                            
                            model_results[start_ts]['stats'] = dict()
                            model_results[start_ts]['stats']['daily'] = daily
                            model_results[start_ts]['stats'][
                                'windows'] = windows
                            model_results[start_ts]['stats'][
                                'window_descriptions_train'] = \
                                window_descriptions_train
                            model_results[start_ts]['stats'][
                                'window_descriptions_test'] = \
                                window_descriptions_test
                            
                            filename = f"{target_column}.pkl"
                            self.dump_data(
                                data=model_results,
                                directory=self.model_results_pickle_directory,
                                filename=filename)
                        xgboost_output = None
                    except Exception as e:
                        print(e)
                        print(f"backtesting iteration error on - {target_column} - {start_ts}")
            except Exception as e:
                print(e)
                print(f"backtesting iteration error on - {target_column}")
        try:
            if view_it:
                
                win_train = pd.DataFrame.from_dict(
                    window_descriptions_train, orient='index')
                
                win_train_std = win_train[['mean']]
                
                win = pd.DataFrame.from_dict(windows, orient='index')
                com = daily.copy().join(win).join(win_train_std)
                
                lgb_pos = com.loc[com['MSFT direction_LGB'] > 0]
                xgb_pos = com.loc[com['MSFT direction_XGB'] > 0]
                lgb_neg = com.loc[com['MSFT direction_LGB'] < 0]
                xgb_neg = com.loc[com['MSFT direction_XGB'] < 0]
                
                print(lgb_pos.describe())
                print(xgb_pos.describe())
                print(lgb_neg.describe())
                print(xgb_neg.describe())
                print(com.describe())
                
                pnls.head(7).sum().hist()
                plt.show()
                pnls.head(horizon).sum().hist()
                plt.show()
                pnls.sum().hist()
                plt.show()
                print(pnls.head(2).sum().describe())
                print(pnls.head(horizon).sum().describe())
                print(pnls.sum().describe())
        except Exception as e:
            print(e)
        """
        normal_split = False
        percent_split = 0.75
        count    196.000000
        mean       0.908206
        std       59.199609
        min     -249.923583
        25%      -29.569894
        50%        2.039158
        75%       28.816431
        max      347.261403
        
        normal_split = False
        percent_split = 0.5
        count    196.000000
        mean       6.024034
        std       56.253068
        min     -280.661420
        25%      -28.580068
        50%        0.135430
        75%       36.577358
        max      192.356157
        
        """
    
    
    def extract_target_columns(
            self,
            selected_year: str = str(),
            force_update_data: bool = False
            ) -> list:
        update_data = self.update_data
        self.update_data = force_update_data
        end_date = pd.Timestamp.now() + timedelta(days=1)
        time_interval = 'daily' # 'daily'
        filename = 'train.pkl'
        dataframe = self.run_get_data(
            end_date=end_date,
            time_interval=time_interval,
            filename=filename)
        raw_experiments = self.prepare_raw_experiments(
            dataframe=dataframe)
        self.selected_year = [selected_year]
        self.visualize = False
        filter_condition = False
        persistence_condition = True
        test_validation_split, buffer = 0.75, 7
        engineered_experiments = self.prepare_engineered_experiments(
            raw_experiments=raw_experiments,
            filter_condition=filter_condition,
            persistence_condition=persistence_condition,
            test_validation_split=test_validation_split,
            buffer=buffer)
        self.update_data = update_data
        
        experiments = list(engineered_experiments.keys())
        if isinstance(selected_year, str):
            if selected_year in experiments:
                experiments = [selected_year]
        for experiment in experiments:
            engineered_experiment = engineered_experiments[experiment]
            target_columns = list(engineered_experiment.keys())
            engineered_experiments[experiment] = None
            engineered_experiment = None
            return target_columns
    
    def update_etl_datasets(
            self,
            selected_year: str = str()
            ):
        self.update_data = True
        start_date = '2021-12-15'
        end_date = pd.Timestamp.now() + timedelta(days=1)
        time_interval = 'daily' # 'daily'
        filename = 'train.pkl'
        source_start = pd.Timestamp.now()
        dataframe = self.run_get_data(
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            filename=filename)
        source_end = pd.Timestamp.now()
        
        if len(dataframe) == 0: return
        
        raw_etl_start = pd.Timestamp.now()
        raw_experiments = self.prepare_raw_experiments(
            dataframe=dataframe)
        raw_etl_end = pd.Timestamp.now()
        
        self.selected_year = [selected_year]
        self.visualize = False
        filter_condition = self.filter_condition
        persistence_condition = self.persistence_condition
        test_validation_split, buffer = 0.95, 14
        etl_start = pd.Timestamp.now()
        engineered_experiments = self.prepare_engineered_experiments(
            raw_experiments=raw_experiments,
            filter_condition=filter_condition,
            persistence_condition=persistence_condition,
            test_validation_split=test_validation_split,
            buffer=buffer)
        etl_end = pd.Timestamp.now()
        print('##############################################################')
        print(f"Source Start: {source_start}")
        print(f"Source END:   {source_end}")
        print('##############################################################')
        print(f"RAW ETL Start: {raw_etl_start}")
        print(f"RAW ETL END:   {raw_etl_end}")
        print('##############################################################')
        print(f"ETL Start: {etl_start}")
        print(f"ETL END:   {etl_end}")
        print('##############################################################')
        
    def run_backtests(
            self,
            processes: list = list(),
            selected_year: str = str()
            ):
        """
        

        Parameters
        ----------
        processes : list, optional
            DESCRIPTION. The default is list().
        selected_year : str, optional
            DESCRIPTION. The default is str().

        Returns
        -------
        None.
        
        selected_year = '2016'
        processes = list()
        
        """
        update_data = self.update_data
        end_date = pd.Timestamp.now() + timedelta(days=1)
        time_interval = 'daily' # 'daily'
        filename = 'train.pkl'
        self.update_data = False # If False then run_get_data simply returns data.
        dataframe = self.run_get_data(
            end_date=end_date,
            time_interval=time_interval,
            filename=filename)
        raw_experiments = self.prepare_raw_experiments(
            dataframe=dataframe)
        self.selected_year = [selected_year]
        self.visualize = False
        filter_condition = self.filter_condition
        persistence_condition = self.persistence_condition
        test_validation_split, buffer = 0.95, 7
        self.update_data = False # If False then prepare_engineered_experiments simply returns data.
        engineered_experiments = self.prepare_engineered_experiments(
            raw_experiments=raw_experiments,
            filter_condition=filter_condition,
            persistence_condition=persistence_condition,
            test_validation_split=test_validation_split,
            buffer=buffer)
        self.update_data = update_data
        
        if len(processes) == 0:
            processes = self.extract_target_columns(
                selected_year=selected_year, force_update_data=False)
        
        years = list(engineered_experiments.keys())
        if isinstance(selected_year, str):
            if selected_year in years:
                years = [selected_year]
        for year in years:
            target_columns = list(engineered_experiments[year].keys())
            target_columns = [t for t in target_columns if t in processes]
            engineered_experiments[year] = None
            engineered_experiment = dict()
            period, delay, starting_point = 1, 1, 1
            self.backtesting_method(
                dataframe=dataframe,
                engineered_experiment=engineered_experiment,
                target_columns=target_columns,
                period=period,
                delay=delay,
                starting_point=starting_point, # Keep as 1 to scale it up later.
                year=year
                )
if __name__ == '__main__':
    update_data, model_reset, backtesting_condition = True, True, True
    update_backtest = True # True in Production
    visualization = True
    mo = ModelOrchestration(
        update_data=update_data,
        model_reset=model_reset,
        backtesting_condition=backtesting_condition,
        update_backtest=update_backtest,
        visualization=visualization)
    mo.update_etl_datasets(selected_year='2016')
    target_columns = mo.extract_target_columns(
        selected_year='2016', force_update_data=False)
    # models.run_backtests(
    #     selected_year='2016',
    #     processes=target_columns)