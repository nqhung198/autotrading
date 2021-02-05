import csv
import numpy as np
from Environment.core import DataGenerator
import pandas as pd
from stockstats import StockDataFrame as Sdf
from sklearn import preprocessing
from indicator import *
import MetaTrader5 as mt5
class TAStreamer(DataGenerator):
    """Data generator from csv file.
    The csv file should no index columns.

    Args:
        filename (str): Filepath to a csv file.
        header (bool): True if the file has got a header, False otherwise
    """
    @staticmethod
    def _generator(filename,curr='GBPUSD',period=14, header=False, split=0.8, mode='train',spread=.005):
        df = pd.read_csv(filename)
        if "Name" in df:
            df.drop('Name',axis=1,inplace=True)
        _stock = Sdf.retype(df.copy())
        _stock.get('cci_14')
        _stock.get('rsi_14')
        _stock.get('dx_14')
        _stock = _stock.dropna(how='any')

        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        np_scaled = min_max_scaler.fit_transform(_stock[['rsi_14', 'cci_14','dx_14','volume']])
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized.columns = ['rsi_14', 'cci_14','dx_14','volume']
        df_normalized['bid'] = _stock['close'].values
        df_normalized['ask'] = df_normalized['bid'] + spread
        df_normalized['mid'] = (df_normalized['bid'] + df_normalized['ask'])/2

        split_len=int(split*len(df_normalized))

        if(mode=='train'):
            raw_data = df_normalized[['ask','bid','mid','rsi_14','cci_14','dx_14','volume']].iloc[:split_len,:]
        elif mode=='test':
            raw_data = df_normalized[['ask', 'bid', 'mid', 'rsi_14', 'cci_14','dx_14','volume']].iloc[split_len:,:]

        if mode=='trade':
            if not mt5.initialize():
                print("initialize() failed, error code =",mt5.last_error())
                quit()
            Done=True
            collect = pd.DataFrame(mt5.copy_rates_from_pos(curr, mt5.TIMEFRAME_M1, 0, 10))
            c=0
            open_list=collect['open'].values
            high_list=collect['high'].values
            low_list=collect['low'].values
            close_list=collect['close'].values
            rates = mt5.copy_rates_from_pos(curr, mt5.TIMEFRAME_M1, 0, 2)
            while Done:
                lasttick=mt5.symbol_info_tick(curr)
                bid, ask = round(lasttick.bid,5), round(lasttick.ask,5)
                mid = round((bid + ask)/2,5)
                
                check = rates[1][0]
                    #rates = mt5.copy_rates_from_pos("GBPUSD", mt5.TIMEFRAME_M1, 0, 1)
                while check==rates[1][0]:
                    rates = mt5.copy_rates_from_pos(curr, mt5.TIMEFRAME_M1, 0, 2)
                open, high, low, close, tickvol, spread = rates[0][1], rates[0][2], rates[0][3], rates[0][4], rates[0][5], rates[0][6]
                open_list = np.append(open_list, open)
                high_list = np.append(high_list,high)
                low_list = np.append(low_list, low)
                close_list = np.append(close_list,close)
                ADX = adx(high_list, low_list, close_list, period)
                cci = CCI(high_list, low_list, close_list, period)
                rsi = RSI(pd.Series(close_list), period)
                yield np.array([bid, ask, mid, round(rsi.values[-1],5), cci[-1]/100, ADX[-1]/100, tickvol])
                c+=1
        else:
            for index, row in raw_data.iterrows():
                yield row.to_numpy()


    def _iterator_end(self):
        """Rewinds if end of data reached.
        """
        # print("End of data reached, rewinding.")
        super(self.__class__, self).rewind()

    def rewind(self):
        """For this generator, we want to rewind only when the end of the data is reached.
        """
        self._iterator_end()