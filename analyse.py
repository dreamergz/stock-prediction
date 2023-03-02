import pandas as pd
import conf
import matplotlib.pyplot as plt

symbol = conf.SYMBOL_CMB
df = pd.read_csv(conf.CSV_SOURCE_DATA_PATH % symbol, index_col='date')
df = df[conf.DATA_COLUMNS]
df = df.sort_values('date')
print(df.shape)
print(df.info())
print(df.head())

plt.figure(figsize=(18, 9))
plt.plot(range(df.shape[0]), df['close'])
plt.xticks(range(0, df.shape[0], 500), df.index.values[::500], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.savefig(conf.FIGURE_PATH + 'source_data.png')
