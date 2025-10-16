import requests
import pandas as pd
import datetime

import mplfinance as mpf
import matplotlib.pyplot as plt

url_template = 'http://iss.moex.com/iss/engines/stock/markets/shares/securities/{}/candles.json?from={}&till={}&interval={}'
def get_historical_data(ticker: str, start_date: str, end_date: str, interval: str):
  url = url_template.format(ticker, start_date, end_date, interval)

  response = requests.get(url).json()

  data = [{k: row[i] for i, k in enumerate(response['candles']['columns'])} for row in response['candles']['data']]

  return data

def candle_graphic_from_csv(ticker: str, plot_range):
  csv_file = f'{ticker}.csv'
  df = pd.read_csv(csv_file)
  
  start_slice = plot_range[0]
  end_slice = plot_range[1]
  df['date'] = pd.to_datetime(df['begin'])
  df.set_index('date', inplace=True)
  df_slice = df.loc[(df.index >= start_slice) & (df.index <= end_slice)]
  
  mpf.plot(df_slice, type='candle', style='yahoo', mav=(3, 6), title=f'{ticker} Candles', volume=True)
  # plt.show()
  

def download_historical_data_to_csv():
  # TODO: ticker, interval and so on as parameters
  ticker = 'SBER'
  interval = '1'

  start_date = datetime.date(2011, 1, 1) # to 2025-01-01 i string
  end_date = datetime.date(2025, 10, 1)
  time_delta = datetime.timedelta(days=1)

  df = pd.DataFrame()

  current_date = start_date
  while current_date < end_date:
    print(f"Downloading data for {current_date} to {current_date + time_delta}")
    data = get_historical_data(ticker, current_date.strftime('%Y-%m-%d'), (current_date + time_delta).strftime('%Y-%m-%d'), interval)
    current_df = pd.DataFrame(data)
    print(current_df.head())
    df = pd.concat([df, current_df], ignore_index=True)
    current_date += time_delta
  
  df.to_csv(f'{ticker}.csv', index=False)





  
def main():
  download_historical_data_to_csv()
  # candle_graphic_from_csv('SBER', (datetime.datetime(2025, 1, 6, 2, 0), datetime.datetime(2025, 1, 6, 13, 0)))
  



  
if __name__ == '__main__':
  main()