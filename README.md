# Stock Prediction

Stock prediction with LSTM. Predict 21st day close price by input 1 - 20th days store data (data columns: 'open', 'high', 'low', 'close', 'adjusted_close', 'volume')

## Dataset

Select the past 15 years Daily Time Series for code 600036.SHH(China Merchants Bank) from Alpha Vantage. Split the dataset by ratio 8:1:1 to training set, validation set and test set.

## Outcome

### Training MSE

![training MSE](figures/train_losses.png 'Training MSE')

### Predictions

![predictions](figures/predictions.png 'Predictions')

## Usage

### Prepare Data

I got the data from alpha vantage and save copy at 'data' folder. Before train the model, need to run below script to split dataset and prepare data scaler.

```text
python prepare_data.py 'your alpha vantage api_key'
```

Alpha Vantage api_key is optional, needed only if you want to get data from Alpha Vantage yourself.

### Train and Test

There are train.py and test.py. Run the script accordingly. Or run main.py to train and test
