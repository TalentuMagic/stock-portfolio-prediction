from statistics import mode
import time
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, mixed_precision
import os
import analysePies

# using mixed precision for faster model training
mixed_precision.set_global_policy('mixed_float16')
start = datetime.now()


def plotModelPerformance(model, X_test, y_test, history, y_test_predictions, loss, accuracy):
    plt.plot(history.history['accuracy'],
             label='train_accuracy', linestyle='dashdot')
    plt.plot(history.history['val_accuracy'],
             label='val_accuracy', color='green')
    plt.plot(history.history['val_loss'],
             label='val_loss', linestyle="dashed", color='red')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test.index, y_test, alpha=0.5,
                color='green', label='Actual Values')
    plt.scatter(y_test.index, y_test_predictions, alpha=0.5,
                color='red', label='Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Actual vs. Predicted Values on Validation Set')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot histogram of predicted probabilities
    plt.figure(figsize=(10, 6))
    seaborn.histplot(y_test_predictions, bins=50, kde=True)
    plt.xlabel('Predicted Probabilities')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Probabilities on Validation Set')
    plt.grid()
    plt.show()

    # Plot the distribution of predicted probabilities
    plt.figure(figsize=(10, 6))
    seaborn.histplot(y_test_predictions, bins=30, kde=True,
                     color='blue', stat='density')
    plt.title('Distribution of Predicted Probabilities on Validation Set')
    plt.xlabel('Predicted Probabilities')
    plt.ylabel('Density')
    plt.grid()
    plt.show()


def read_datasets(user_choice: int = None, price_history: str = None, metrics: str = None, ticker: str = None):
    if user_choice is None and price_history is None and metrics is None:
        while True:
            print(
                "[1] Eurozone Investments\n[2] Own the World in 50\n[3] World ETFs\n[4] Crypto")
            user_choice = int(
                input("What pie do you want to analyze?(Enter a number from above)\n"))
            match user_choice:
                case 1:
                    ok = bool()
                    price_history = input(
                        "Do you want to plot the price history with indicators for this pie?[y/n]\n")
                    if price_history == 'n' or price_history == 'N' or price_history == '':
                        ok = False
                    elif price_history == 'y' or price_history == 'Y':
                        ok = True
                    analysePies.getPieData_EurozoneInvestments(ok=ok)
                    folder_path = r'./eurozone/'
                    files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                        os.path.join(folder_path, f))]

                    metrics = input(
                        "Do you want to plot the model performance metrics?[y/n]\n")
                    if metrics == "n" or metrics == 'N' or metrics == '':
                        ok = False
                    elif metrics == 'y' or metrics == 'Y':
                        ok = True
                    break
                case 2:
                    price_history = input(
                        "Do you want to plot the price history with indicators for this pie?[y/n]\n")
                    if price_history == 'n' or price_history == 'N' or price_history == '':
                        ok = False
                    elif price_history == 'y' or price_history == 'Y':
                        ok = True
                    analysePies.getPieData_OwnTheWorldIn50(ok=ok)
                    folder_path = r'./otw/'
                    files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                        os.path.join(folder_path, f))]

                    metrics = input(
                        "Do you want to plot the model performance metrics?[y/n]\n")
                    if metrics == "n" or metrics == 'N' or metrics == '':
                        ok = False
                    elif metrics == 'y' or metrics == 'Y':
                        ok = True
                    break
                case 3:
                    price_history = input(
                        "Do you want to plot the price history with indicators for this pie?[y/n]\n")
                    if price_history == 'n' or price_history == 'N' or price_history == '':
                        ok = False
                    elif price_history == 'y' or price_history == 'Y':
                        ok = True
                    analysePies.getPieData_ETFs(ok=ok)
                    folder_path = r'./etfs/'
                    files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                        os.path.join(folder_path, f))]

                    metrics = input(
                        "Do you want to plot the model performance metrics?[y/n]\n")
                    if metrics == "n" or metrics == 'N' or metrics == '':
                        ok = False
                    elif metrics == 'y' or metrics == 'Y':
                        ok = True
                    break
                case 4:
                    price_history = input(
                        "Do you want to plot the price history with indicators for this pie?[y/n]\n")
                    if price_history == 'n' or price_history == 'N' or price_history == '':
                        ok = False
                    elif price_history == 'y' or price_history == 'Y':
                        ok = True
                    analysePies.getPieData_Crypto(ok=ok)
                    folder_path = r'./crypto/'
                    files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                        os.path.join(folder_path, f))]

                    metrics = input(
                        "Do you want to plot the model performance metrics?[y/n]\n")
                    if metrics == "n" or metrics == 'N' or metrics == '':
                        ok = False
                    elif metrics == 'y' or metrics == 'Y':
                        ok = True
                    break
                case _:
                    print("Invalid choice. Try again...\n")
                    continue
    else:
        while True:
            match user_choice:
                case 1:
                    ok = bool()
                    if price_history == 'n' or price_history == 'N' or price_history == '':
                        ok = False
                    elif price_history == 'y' or price_history == 'Y':
                        ok = True
                    analysePies.getPieData_EurozoneInvestments(ok=ok)
                    folder_path = r'./eurozone/'
                    files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                        os.path.join(folder_path, f))]

                    if metrics == "n" or metrics == 'N' or metrics == '':
                        ok = False
                    elif metrics == 'y' or metrics == 'Y':
                        ok = True
                    break
                case 2:
                    if price_history == 'n' or price_history == 'N' or price_history == '':
                        ok = False
                    elif price_history == 'y' or price_history == 'Y':
                        ok = True
                    analysePies.getPieData_OwnTheWorldIn50(ok=ok)
                    folder_path = r'./otw/'
                    files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                        os.path.join(folder_path, f))]

                    if metrics == "n" or metrics == 'N' or metrics == '':
                        ok = False
                    elif metrics == 'y' or metrics == 'Y':
                        ok = True
                    break
                case 3:
                    if price_history == 'n' or price_history == 'N' or price_history == '':
                        ok = False
                    elif price_history == 'y' or price_history == 'Y':
                        ok = True
                    analysePies.getPieData_ETFs(ok=ok)
                    folder_path = r'./etfs/'
                    files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                        os.path.join(folder_path, f))]

                    if metrics == "n" or metrics == 'N' or metrics == '':
                        ok = False
                    elif metrics == 'y' or metrics == 'Y':
                        ok = True
                    break
                case 4:
                    if price_history == 'n' or price_history == 'N' or price_history == '':
                        ok = False
                    elif price_history == 'y' or price_history == 'Y':
                        ok = True
                    analysePies.getPieData_Crypto(ok=ok)
                    folder_path = r'./crypto/'
                    files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                        os.path.join(folder_path, f))]

                    if metrics == "n" or metrics == 'N' or metrics == '':
                        ok = False
                    elif metrics == 'y' or metrics == 'Y':
                        ok = True
                    break
                case _:
                    print("Invalid choice. Try again...\n")
                    continue
    return files, ok


def main(user_choice: int = None, price_history: str = None, metrics: str = None, ticker: str = None):
    files, ok = read_datasets(user_choice, price_history, metrics, ticker)

    index = 0
    if ticker is not None:
        try:
            for i, stock in enumerate(files):
                if ticker in stock.rstrip(".csv"):
                    index = i
        except:
            print("[INFO] The specified ticker NOT FOUND!")

    while index < len(files):
        try:
            print(files[index])

            scaler = StandardScaler(with_mean=False, with_std=False)

            # Load the dataset
            df = pd.read_csv(fr'./{files[index]}', index_col='Date')
            df = df.drop(['Volume', 'Close', 'Repaired?'], axis=1)
            df_columns, df_indexes = df.columns, df.index

            # Scale the data and pass it as the same DataFrame as before scaling
            # Handle empty dataset exception
            try:
                df = pd.DataFrame(scaler.fit_transform(
                    df), columns=df_columns, index=df_indexes)
            except:
                print("[INFO] The dataset is empty! Skipping...")
                index += 1
                continue

            X = df.iloc[:, :-1].astype('float16')
            y = df.iloc[:, -1].astype('float16')

            # select the past year for validation data
            X.index = pd.to_datetime(X.index)
            end_date = X.index.max()
            start_date = end_date - pd.DateOffset(years=1)
            X_past_1year = X[(X.index >= start_date)
                             & (X.index <= end_date)]
            # input data must be only until the validation set
            X = X[X.index <= start_date]

            y.index = pd.to_datetime(y.index)
            end_date = y.index.max()
            start_date = end_date - pd.DateOffset(years=1)
            y_past_1year = y[(y.index >= start_date) & (y.index <= end_date)]
            # output data must be only until validation set
            y_train = y[y.index <= start_date]

            # each row is a different sequence -> each column has 12 features
            X_train = np.array(X).reshape(X.shape[0], 1, X.shape[1])
            X_past_1year = np.array(X_past_1year).reshape(
                X_past_1year.shape[0], 1, X_past_1year.shape[1])

            X_val, X_test, y_val, y_test = train_test_split(
                X_past_1year, y_past_1year, test_size=0.3, random_state=42)

            # print("X_train shape:", X_train.shape)
            # print("X_val shape:", X_val.shape)
            # print("X_test shape:", X_test.shape)
            # print("y_train shape:", y_train.shape)
            # print("y_val shape:", y_val.shape)
            # print("y_test shape:", y_test.shape)

            # print(X_train.dtype)

            # Define the checkpoint callback
            filename = files[index][2:-4].split(".")
            if len(filename) > 1:
                checkpoint_path = rf'./models/classification/{filename[0]}_{filename[1]}.h5'
            else:
                checkpoint_path = rf'./models/classification/{filename[0]}.h5'

            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',  # You can use other metrics like 'val_loss'
                save_best_only=True,
                mode='max',  # 'max' for accuracy, 'min' for loss, 'auto' for automatic
                verbose=1
            )
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=25, restore_best_weights=True)

            # Define the model
            model = Sequential()
            print(model.dtype_policy)
            # Input layer
            model.add(LSTM(512, input_shape=(
                1, X_train.shape[2]), return_sequences=True, kernel_regularizer='l1_l2'))
            model.add(BatchNormalization())
            model.add(Dropout(0.15))

            # Output layer
            model.add(Dense(1, activation='sigmoid',
                            kernel_regularizer=regularizers.l2(0.01)))
            optimizer = Adam(learning_rate=0.00025, clipnorm=1.0)
            model.compile(optimizer=optimizer,
                          loss='binary_focal_crossentropy', metrics=['accuracy'])

            model.summary()
            history = model.fit(X_train, y_train, epochs=100, batch_size=64,
                                validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping], verbose=1)

            # load the best performing model
            if len(filename) > 1:
                model = load_model(
                    fr"./models/classification/{filename[0]}_{filename[1]}.h5")
            else:
                model = load_model(
                    fr"./models/classification/{filename[0]}.h5")

            loss, accuracy = model.evaluate(X_test, y_test)

            # Print the evaluation results
            print("Validation Loss:", loss)
            print("Validation Accuracy:", accuracy)

            time.sleep(1)
            # y_test_predictions = np.array(
            #     model.predict(X_test).astype('float32')).squeeze()
            y_test_predictions_scaled = np.array(model.predict(
                X_test).astype('float32')).reshape(-1, 1)

            # fit an output scaler with the test data
            scaler_y = StandardScaler(with_mean=False, with_std=False)
            scaler_y.fit(y_train.to_numpy().reshape(-1, 1))

            y_test_predictions = scaler_y.inverse_transform(
                y_test_predictions_scaled).squeeze()

            # Convert probabilities to binary predictions (0 or 1)
            y_test_predictions_binary = [
                1 if prediction > 0.5 else 0 for prediction in y_test_predictions]

            print("\nLast Year's Performance:",
                  mode(y_test_predictions_binary))
            print("Price Increase/Decrease Tomorrow:",
                  y_test_predictions_binary[-1], '\n')

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_test_predictions_binary)
            print("Confusion Matrix:")
            print(cm)

            # Classification Report
            print("Classification Report:")
            print(classification_report(y_test, y_test_predictions_binary))

            if ok:
                plotModelPerformance(model, X_test, y_test,
                                     history, y_test_predictions_binary, loss, accuracy)

            del model

            if ticker is not None:
                break
            index += 1

            print("Waiting 1 second(s) between trainings...\n")
            if index != len(files):
                time.sleep(1)

        except Exception as e:
            print("An error occured:", e, '\nRetrying...\n')
            continue
    print("\nTraining on the whole pie took:", datetime.now() - start)


if __name__ == "__main__":
    main()
