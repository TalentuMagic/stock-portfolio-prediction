# stock-portfolio-prediction
## Author: David-Ioan STANESCU-FLOREA
LSTM Prediction Neural Network for my Investment pies

## The Data
 - It uses stock pies' that I personally have investments in whole historical data from Yahoo Finance API. It computes the price indicators: RSI, SMA (Simple Moving Average), EMA (Exponential Moving Average) and adds them to the dataset(s).
 - The *Tomorrow* column consists of shifting by one position the *Adj Close* column from the following day.
 - The *TargetClass* column consists of comparing whether the *Adj Close* of current iteration is bigger than the *Tomorrow* column. If so, there is a 1, else a 0.

## Network
 - The network is based on the LSTM (Long Short Term Memory) neural network  model.
 - For each stock in the pie, it creates an unique model.
 - The model must solve a binary classification problem.

### Input & Output Data
 - The input data for training consists of the whole history of data of a specific stock, except the *Volume* column, until one year before the current date.
 - The input data for testing and validation consists of the history of data of a specific stock, except the *Volume* column, starting from a year before the current date.
 - The output data for training consists of the whole history of data of a specific stock for the *TargetClass* column, until one year before the current date.
 - The output data for testing and validation consists of the history of data of a specific stock for the *TargetClass* column, starting from a year before the current date.