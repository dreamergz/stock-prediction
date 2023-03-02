SYMBOL = '002323.SHZ'
SYMBOL_CMB = '600036.SHH'
DATA_API = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=%s&outputsize=full&apikey=%s'
CSV_SOURCE_DATA_PATH = "data/%s.csv"
JSON_SOURCE_DATA_PATH = 'data/%s.json'
TRAIN_DATA_PATH = 'data/train_data.csv'
VALIDATION_DATA_PATH = 'data/validation_data.csv'
TEST_DATA_PATH = 'data/test_data.csv'
DATA_SCALER_PATH = 'data/scaler.save'
BATCH_SIZE = 50

DATA_COLUMNS = [
    'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
LABEL_COLUMNS = ['close']
NUM_UNROLL = 20

FEATURES_COUNT = 6

CHECKPOINT_PATH = 'checkpoint/lstm.cpt'
FIGURE_PATH = 'figures/'

HIDDEN_SIZE = 256
EPOCHS = 1000
LEARNING_RATE = 0.0001
MIN_LEARNING_RATE = 0.000001
TRAIN_LOSS_NON_DECREASE_THRESHOLD = 3
