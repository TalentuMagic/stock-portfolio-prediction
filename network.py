from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import analysePies

while True:
    print(
        "[1] Eurozone Investments\n[2] Own the World in 50\n[3] World ETFs\n[4] Crypto")
    user_choice = int(
        input("What pie do you want to analyze?(Enter a number from above)\n"))
    match user_choice:
        case 1:
            price_history = input(
                "Do you want to plot the price history with indicators for this pie?[y/n]\n")
            if price_history == "n" or price_history == 'N' or price_history != 'y' or price_history != 'Y' or price_history == '' or price_history != '':
                ok = False
            else:
                ok = True
            analysePies.getPieData_EurozoneInvestments(ok=ok)
            folder_path = './eurozone/'
            files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                os.path.join(folder_path, f))]
            break
        case 2:
            price_history = input(
                "Do you want to plot the price history with indicators for this pie?[y/n]\n")
            if price_history == "n" or price_history == 'N' or price_history != 'y' or price_history != 'Y' or price_history == '' or price_history != '':
                ok = False
            else:
                ok = True
            analysePies.getPieData_OwnTheWorldIn50(ok=ok)
            folder_path = './otw/'
            files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                os.path.join(folder_path, f))]
            break
        case 3:
            price_history = input(
                "Do you want to plot the price history with indicators for this pie?[y/n]\n")
            if price_history == "n" or price_history == 'N' or price_history != 'y' or price_history != 'Y' or price_history == '' or price_history != '':
                ok = False
            else:
                ok = True
            analysePies.getPieData_ETFs(ok=ok)
            folder_path = './etfs/'
            files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                os.path.join(folder_path, f))]
            break
        case 4:
            price_history = input(
                "Do you want to plot the price history with indicators for this pie?[y/n]\n")
            if price_history == "n" or price_history == 'N' or price_history != 'y' or price_history != 'Y' or price_history == '' or price_history != '':
                ok = False
            else:
                ok = True
            analysePies.getPieData_Crypto(ok=ok)
            folder_path = './crypto/'
            files = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(
                os.path.join(folder_path, f))]
            break
        case _:
            print("Invalid choice. Try again...\n")
            continue

for file in files:
    # Load the dataset
    df = pd.read_csv(f'./{file}', index_col='Date')
    df = df.drop(['Volume', 'Close'], axis=1)

    X = df.iloc[:, :-1].astype('float32')
    # each row is a different sequence -> each column has 12 features
    X = np.array(X).reshape(X.shape[0], 1, X.shape[1])
    y = df.iloc[:, -1]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.25, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.3, random_state=42)

    # print("X_train shape:", X_train.shape)
    # print("X_val shape:", X_val.shape)
    # print("X_test shape:", X_test.shape)
    # print("y_train shape:", y_train.shape)
    # print("y_val shape:", y_val.shape)
    # print("y_test shape:", y_test.shape)

    # Define the checkpoint callback
    checkpoint_path = 'best_model.h5'
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
    model.add(LSTM(512, input_shape=(1, X.shape[2]), return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=100, batch_size=16,
                        validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping], verbose=1)

    loss, accuracy = model.evaluate(X_val, y_val)

    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

    # Print the evaluation results
    print("Validation Loss:", loss)
    print("Validation Accuracy:", accuracy)

    y_val_predictions = model.predict(X_val).squeeze()

    # Convert probabilities to binary predictions (0 or 1)
    y_val_predictions_binary = (y_val_predictions > 0.3).astype(int).squeeze()

    print(y_val_predictions_binary[:3])
    print(y_val_predictions[:3])

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

    break
