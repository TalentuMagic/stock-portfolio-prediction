import time
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers, mixed_precision
import os
import analysePies

# using mixed precision for faster model training
mixed_precision.set_global_policy('mixed_float16')


def plotModelPerformance(model, X_val, y_val, history, loss, accuracy):
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

    # Print the evaluation results
    print("Validation Loss:", loss)
    print("Validation Accuracy:", accuracy)

    y_val_predictions = model.predict(X_val).squeeze().astype('float32')

    # Convert probabilities to binary predictions (0 or 1)
    y_val_predictions_binary = (y_val_predictions > 0.5).astype(int).squeeze()

    # print(y_val_predictions_binary[:3])
    # print(y_val_predictions[:3])

    # Create confusion matrix
    cm = confusion_matrix(y_val, y_val_predictions_binary)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_val, y_val_predictions_binary))

    # Plot actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val_predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values on Validation Set')
    plt.show()

    # Plot histogram of predicted probabilities
    plt.figure(figsize=(10, 6))
    seaborn.histplot(y_val_predictions, bins=50, kde=True)
    plt.xlabel('Predicted Probabilities')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Probabilities on Validation Set')
    plt.show()

    # Flatten the probabilities
    flat_probabilities = y_val_predictions.flatten()

    # Plot the distribution of predicted probabilities
    plt.figure(figsize=(10, 6))
    seaborn.histplot(flat_probabilities, bins=30, kde=True,
                     color='blue', stat='density')
    plt.title('Distribution of Predicted Probabilities on Validation Set')
    plt.xlabel('Predicted Probabilities')
    plt.ylabel('Density')
    plt.show()


def main():
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
                folder_path = './eurozone/'
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
                folder_path = './otw/'
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
                folder_path = './etfs/'
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
                folder_path = './crypto/'
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

    index = 0
    while index < len(files):
        try:
            print(files[index])
            # Load the dataset
            df = pd.read_csv(f'./{files[index]}', index_col='Date')
            df = df.drop(['Volume', 'Close'], axis=1)

            X = df.iloc[:, :-1].astype('float16')
            y = df.iloc[:, -1]

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
                checkpoint_path = rf'./models/{filename[0]}_{filename[1]}.h5'
            else:
                checkpoint_path = rf'./models/{filename[0]}.h5'

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
                1, X_train.shape[2]), return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(0.15))

            # 2nd layer
            model.add(LSTM(256, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(0.15))

            # 3rd layer
            model.add(LSTM(128, return_sequences=True))
            model.add(BatchNormalization())

            # Output layer
            model.add(Dense(1, activation='sigmoid',
                            kernel_regularizer=regularizers.l2(0.01)))
            optimizer = Adam(learning_rate=0.00025)
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy', metrics=['accuracy'])

            model.summary()
            history = model.fit(X_train, y_train, epochs=100, batch_size=64,
                                validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping], verbose=1)

            loss, accuracy = model.evaluate(X_test, y_test)

            if ok:
                plotModelPerformance(model, X_test, y_test,
                                     history, loss, accuracy)

            del model

            print("Waiting 5 seconds between trainings...\n")
            index += 1

            if index != len(files):
                time.sleep(5)
        except Exception as e:
            print("An error occured:", e, '\nRetrying...\n')
            continue


if __name__ == "__main__":
    main()
