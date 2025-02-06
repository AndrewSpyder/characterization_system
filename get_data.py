# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 12:55:44 2021

@author: Andrew
"""
import pandas as pd
import _pickle as cpickle
from yahoofinancials import YahooFinancials
from datetime import timedelta
from multiprocessing.dummy import Pool as ThreadPool
import itertools

from preprocess import PreProcess

class GetData(PreProcess):
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
        self.__dict__.update(**kwargs)
    
    def set_assets(
            self,
            assets: list = list()
            ):
        """
        
        Set Assets to YahooFinancials
        
        :param assets: DESCRIPTION, defaults to list()
        :type assets: list, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.yahoofin = YahooFinancials(assets)
    
    def call_stock_content(
            self,
            assets: list = list()
            ) -> dict:
        """
        
        Gets Stock Information After Establishing the Class Object
        
        :param assets: DESCRIPTION, defaults to list()
        :type assets: list, optional
        :return: stock_content
        :rtype: dict

        """
        self.set_assets(assets=assets)
        stock_content = dict()
        try:
            stock_content['information'] = self.yahoofin.\
                get_stock_quote_type_data()
            stock_content['summary'] = self.yahoofin.get_summary_data()
        except Exception as e:
            print(e)
        return stock_content
    
    def mt_get_stock_content(
            self,
            tickers: list = list()
            ) -> dict:
        results = list()
        pool_N = int(len(tickers))
        mt_tickers = [[t] for t in tickers]
        if pool_N > 20: pool_N = 20
        pool = ThreadPool(pool_N)
        results = pool.starmap(
            self.get_stock_content,
            zip(mt_tickers)
            )
        pool.close()
        
        filename_directory = 'stock_results.pklz'
        with open(filename_directory, 'wb') as stream:
            cpickle.dump(results, stream)
        
        list_of_dataframes = list()
        for result in results:
            try:
                dataframe = self.prepare_dataframe(data=result)
                if not isinstance(dataframe, pd.DataFrame): continue
                list_of_dataframes.append(dataframe)
            except Exception as e:
                print(e)
    
    def get_stock_content(
            self,
            tickers: list = list()
            ) -> dict:
        """
        
        Iterate Stock Content Extract
        
        :return: DESCRIPTION
        :rtype: dict

        """
        stock_content = dict()
        for ticker in tickers:
            try:
                stock_content[ticker] = self.call_stock_content(
                    assets=[ticker])
            except Exception as e:
                print(e)
        return stock_content
    
    def stock_content_handle(
            self,
            filename_directory: str = str()
            ) -> dict:
        """
        
        Get Stock Content Information - Volume
        
        :param filename_directory: DESCRIPTION, defaults to str()
        :type filename_directory: str, optional
        :return: DESCRIPTION
        :rtype: dict

        """
        stock_content = self.load_volume_data(
            filename_directory=filename_directory)
        if self.refresh:
            stock_content = self.mt_get_stock_content(tickers=self.tickers_list)
            # stock_content = self.get_stock_content(tickers=self.tickers_list)
            self.dump_volume_data(stock_content=stock_content,
                                  filename_directory=filename_directory)
        return stock_content
    
    def stock_information_consolidation(
            self,
            stock_content: dict = dict(),
            extract_types: list = ['summary', 'information'],
            filename_directory: str = str()
            ) -> pd.DataFrame:
        """
        
        Builds Consolidated Stock Information DataFrame
        
        :param stock_content: DESCRIPTION, defaults to dict()
        :type stock_content: dict, optional
        :param extract_types: DESCRIPTION, defaults to ['summary', 'information']
        :type extract_types: list, optional
        :param filename_directory: DESCRIPTION, defaults to str()
        :type filename_directory: str, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        stock_content = self.stock_content_handle(
            filename_directory=filename_directory)
        tickers = list(stock_content.keys())
        stock_summaries, stock_information = dict(), dict()
        for extract_type in extract_types:
            if extract_type == 'summary':
                for ticker in tickers:
                    if extract_type in list(stock_content[ticker].keys()):
                        stock_summaries[ticker] = \
                            stock_content[ticker][extract_type][ticker]
            else:
                if extract_type == 'information':
                    for ticker in tickers:
                        if extract_type in list(stock_content[ticker].keys()):
                            stock_information[ticker] = \
                                stock_content[ticker][extract_type][ticker]
        stock_summaries = pd.DataFrame.from_dict(stock_summaries)
        stock_information = pd.DataFrame.from_dict(stock_information)
        stock_consolidated = stock_information.append(stock_summaries)
        stock_consolidated = stock_consolidated.transpose()
        return stock_consolidated
    
    def load_volume_data(
            self,
            filename_directory: str = str()
            ) -> dict:
        """
        
        Load Stock Data
            Loads in saved volume data for purpose of PoC.
        
        
        :param filename_directory: DESCRIPTION, defaults to str()
        :type filename_directory: str, optional
        :return: DESCRIPTION
        :rtype: dict

        """
        stock_content = dict()
        try:
            with open(filename_directory, 'rb') as stream:
                stock_content = cpickle.load(stream)
        except Exception as e:
            print(e)
        return stock_content
    
    def dump_volume_data(
            self,
            stock_content: dict = dict(),
            filename_directory: str = str()
            ):
        """
        
        Dump Stock Content
            Stock content contains volume data.
            This will most likely be updated at a later date.
                As in - update frequency of getting stock content for - Volume
                
        :param stock_content: DESCRIPTION, defaults to dict()
        :type stock_content: dict, optional
        :param filename_directory: DESCRIPTION, defaults to str()
        :type filename_directory: str, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        try:
            with open(filename_directory, 'wb') as stream:
                cpickle.dump(stock_content, stream)
        except Exception as e:
            print(e)
    
    def get_json_data(
            self, 
            start_date: str = str(),
            end_date: str = str(),
            time_interval: str = 'daily'
            ) -> dict:
        """
        
        Get JSON Stock Historical Data.
        
        :param start_date: DESCRIPTION, defaults to str()
        :type start_date: str, optional
        :param end_date: DESCRIPTION, defaults to str()
        :type end_date: str, optional
        :param time_interval: DESCRIPTION, defaults to 'daily'
        :type time_interval: str, optional
        :return: DESCRIPTION
        :rtype: dict

        """
        start_date = str(pd.Timestamp(start_date).date())
        end_date = str(pd.Timestamp(end_date).date())
        data = dict()
        try:
            data = self.yahoofin.get_historical_price_data(\
                                       start_date=start_date,
                                       end_date=end_date,
                                       time_interval=time_interval)
        except Exception as e:
            print(e)
            
        return data
    
    def get_data(
            self,
            assets: list = list(),
            start_date: str = str(),
            end_date: str = str(),
            time_interval: str = 'daily'
            ) -> dict or None:
        """
        
        Gets Historical Data
        
        :param stocks: DESCRIPTION, defaults to pd.DataFrame()
        :type stocks: pd.DataFrame, optional
        :param start_date: DESCRIPTION, defaults to str()
        :type start_date: str, optional
        :param end_date: DESCRIPTION, defaults to str()
        :type end_date: str, optional
        :param time_interval: DESCRIPTION, defaults to 'daily'
        :type time_interval: str, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        #TODO: Multi-Threading & Data Memory Size
        try:
            self.set_assets(assets=assets)
            data = self.get_json_data(start_date=start_date,
                                      end_date=end_date,
                                      time_interval=time_interval)
            return data
        except Exception as e:
            print(e)
    
    def multithread_get(
            self,
            tickers: list = list(),
            start_date: str = str(),
            end_date: str = str(),
            time_interval: str = 'daily'
            ) -> pd.DataFrame or None:
        dataframes, results = None, list()
        pool_N = int(len(tickers))
        if pool_N > 20: pool_N = 20
        pool = ThreadPool(pool_N)
        results = pool.starmap(
            self.get_data,
            zip(tickers,
                itertools.repeat(start_date),
                itertools.repeat(end_date),
                itertools.repeat(time_interval)
                )
            )
        pool.close()
        list_of_dataframes = list()
        for result in results:
            try:
                dataframe = self.prepare_dataframe(data=result)
                if not isinstance(dataframe, pd.DataFrame): continue
                list_of_dataframes.append(dataframe)
            except Exception as e:
                print(e)
        dataframes = self.build_dataframe(
            list_of_dataframes=list_of_dataframes)
        return dataframes
    
    def extract_data(
            self,
            stocks: pd.DataFrame = pd.DataFrame(),
            start_date: str = str(),
            end_date: str = str(),
            time_interval: str = 'daily',
            directory: str = str(),
            filename: str = 'train.pkl',
            update_data: bool = False
            ) -> pd.DataFrame:
        """
        
        Extracts & Updates Stock DataFrame
        
        :param stocks: DESCRIPTION, defaults to pd.DataFrame()
        :type stocks: pd.DataFrame, optional
        :param start_date: DESCRIPTION, defaults to str()
        :type start_date: str, optional
        :param end_date: DESCRIPTION, defaults to str()
        :type end_date: str, optional
        :param time_interval: DESCRIPTION, defaults to 'daily'.
        :directory: str = str()
        :type time_interval: str, optional
        :param filename: DESCRIPTION, defaults to 'train.pkl'
        :type filename: str, optional
        :param update_data: DESCRIPTION, defaults to False
        :type update_data: bool, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if directory == str(): return
        if end_date == str():
            end_date = pd.Timestamp.now() + timedelta(days=1)
        # If there is not any historical data, this will return None.
        old_dataframe = self.load_data(
            directory=directory,
            filename=filename)
        if not update_data: return old_dataframe
        conditions = list()
        conditions.append(not self.reset_data)
        conditions.append(isinstance(old_dataframe, pd.DataFrame))
        if all(conditions):
            conditions.append(len(old_dataframe.columns) > 0)
        if all(conditions):
            conditions.append(len(old_dataframe.index) > 0)
        if all(conditions):
            start_date = old_dataframe.index[-1] - timedelta(days=14)
        
        tickers = stocks.index
        new_dataframe = self.multithread_get(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval)
        if all(conditions):
            dataframe = pd.concat([old_dataframe, new_dataframe])
            dataframe.index.name = 'index'
            dataframe = dataframe.reset_index().\
                drop_duplicates('index', keep='last').set_index('index')
        else:
            dataframe = new_dataframe
        
        save_conditions = list()
        save_conditions.append(isinstance(dataframe, pd.DataFrame))
        if all(save_conditions):
            save_conditions.append(len(dataframe.columns) > 0)
        if all(save_conditions):
            save_conditions.append(len(dataframe.index) > 0)
        if all(save_conditions):
            self.dump_data(
                data=dataframe,
                directory=directory,
                filename=filename)
            return dataframe
    
    def run_get_data(
            self,
            start_date: str = str(),
            end_date: str = str(),
            time_interval: str = 'daily',
            filename: str = str()
            ) -> pd.DataFrame:
        """
        
        
        Run Routine.
        
        
        :param start_date: DESCRIPTION, defaults to str()
        :type start_date: str, optional
        :param end_date: DESCRIPTION, defaults to str()
        :type end_date: str, optional
        :param time_interval: DESCRIPTION, defaults to 'daily'
        :type time_interval: str, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if filename == str(): filename = 'train.pkl'
        stocks = self.stock_information
        directory = self.pickle_directory
        update_data = self.update_data
        dataframe = self.extract_data(
            stocks=stocks,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            directory=directory,
            filename=filename,
            update_data=update_data)
        return dataframe
        
if __name__ == '__main__':
    test_condition = True
    if test_condition:
        start_date = '2021-12-15'
        start_date = '2016-01-01'
        end_date = pd.Timestamp.now() + timedelta(days=1)
        time_interval = 'daily' # 'daily'
        get = GetData(reset_data=True,
                      update_data=True)
        get.run_get_data(start_date=start_date,
                         end_date=end_date,
                         time_interval=time_interval)
