import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt


def draw_candle_graphic_from_csv(csv_file: str, plot_range=None):
  df = pd.read_csv(csv_file, index_col=0)

  df.index = pd.to_datetime(df.index)

  start_slice = df.index.min()
  end_slice = df.index.max()

  if plot_range:
    start_slice = plot_range[0]
    end_slice = plot_range[1]

  df_slice = df.loc[(df.index >= start_slice) & (df.index <= end_slice)]

  mpf.plot(df_slice, type='candle', style='yahoo', mav=(
      3, 6), title=f'{csv_file} Candles', volume=True)
  plt.show()


def draw_loss_curve(vals):

  training_loss = vals['training_loss']
  validation_loss = vals['validation_loss']

  epochs = range(1, len(training_loss) + 1)
  plt.plot(epochs, training_loss, 'b', label='Training loss')
  plt.plot(epochs, validation_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
