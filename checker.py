import os
from statistics import mode
import numpy as np

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import analysePies
import json
from keras import models, mixed_precision

mixed_precision.set_global_policy('mixed_float16')


def main():
    results = dict()
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
            X_past_1year = np.array(X_past_1year).reshape(
                X_past_1year.shape[0], 1, X_past_1year.shape[1])

            X_test = X_past_1year
            y_test = y_past_1year

            filename = files[index][2:-4].split(".")
            if len(filename) > 1:
                model_regression = models.load_model(
                    fr"./models/regression/{filename[0]}_{filename[1]}.h5")
                model_classification = models.load_model(
                    fr"./models/classification/{filename[0]}_{filename[1]}.h5")
            else:
                model_regression = models.load_model(
                    fr"./models/regression/{filename[0]}.h5")
                model_classification = models.load_model(
                    fr"./models/classification/{filename[0]}.h5")

            loss_c, accuracy = model_classification.evaluate(X_test, y_test)
            loss_r, mae, mse, rmse = model_regression.evaluate(X_test, y_test)

            # Print the classification evaluation results
            classification = dict()
            print("\n - Classification Results:")
            print("Validation Loss:", loss_c)
            print("Validation Accuracy:", accuracy)

            classification["Validation Loss"] = loss_c
            classification["Validation Accuracy"] = accuracy

            y_test_predictions = model_classification.predict(
                X_test).squeeze().astype('float32')

            # Convert probabilities to binary predictions (0 or 1)
            y_test_predictions_binary = (
                y_test_predictions >= 0.6).astype(int).squeeze()

            print("\nMost Occurring Prediction:",
                  mode(y_test_predictions_binary))

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_test_predictions_binary)
            print("Confusion Matrix:")
            print(cm)

            cm_values = dict()
            cm_values['True Negatives'] = cm[0][0].tolist()
            cm_values['False Negatives'] = cm[1][0].tolist()
            cm_values['True Positives'] = cm[1][1].tolist()
            cm_values['False Positives'] = cm[0][1].tolist()
            classification['Confusion Matrix'] = cm_values

            classification['Most Occuring Prediction'] = mode(
                y_test_predictions_binary).tolist()
            # Classification Report
            print("Classification Report:")
            print(classification_report(y_test, y_test_predictions_binary))

            print("\n - Regression Results:")
            regression = dict()
            # Make predictions on the test set

            # Print the evaluation results
            print("\nValidation Loss:", loss_r)
            print("Validation Mean Squared Error (MSE):", mse)
            print("Validation Mean Absolute Error (MAE):", mae)
            print("Validation Root Mean Sqared Error (RMSE):", rmse)

            regression['Validation Loss'] = loss_r
            regression['Validation Mean Squared Error (MSE)'] = mse
            regression['Validation Mean Absolute Error (MAE)'] = mae
            regression['Validation Root Mean Sqared Error (RMSE)'] = rmse

            y_test_predictions = np.array(model_regression.predict(
                X_test).astype('float32')).reshape(-1)
            # Plot actual vs. predicted values
            print("\nMost Occurring Prediction:", mode(
                y_test_predictions))

            regression['Most Occurring Prediction'] = mode(
                y_test_predictions).tolist()
            # Count occurrences within the threshold
            if np.abs(mse) + np.abs(mae) + np.abs(loss_r) - rmse + (rmse*0.25) <= rmse*0.25:
                threshold = np.abs(mse) + np.abs(mae) + np.abs(loss_r)
            else:
                threshold = np.abs(rmse) + np.abs(rmse*0.25)

            close_enough_count = np.sum(
                (np.abs(y_test) - np.abs(y_test_predictions)) <= threshold)
            close_enough_percentage = (np.sum(
                (np.abs(y_test) - np.abs(y_test_predictions)) <= threshold)/len(y_test))*100
            close_enough_indices = np.where(
                (np.abs(y_test) - np.abs(y_test_predictions)) <= threshold)
            print(
                f"Number of predictions close enough (to actual value within {threshold}): {close_enough_count}/{len(y_test)} ({close_enough_percentage}%)")
            mean_close_enough = np.mean(
                y_test_predictions[close_enough_indices])
            print(
                f"Mean of close enough (to actual value): {mean_close_enough}")

            print("\n------------------------\n")

            print("\nClassification <-> Regression")
            print(mode(y_test_predictions_binary), '<->', mean_close_enough)

            results['Classification'] = classification
            results['Regression'] = regression
            if classification['Most Occuring Prediction'] == 1 and regression['Most Occurring Prediction'] > 0:
                results['Both Models Agree'] = True
            elif classification['Most Occuring Prediction'] == 0 and regression['Most Occurring Prediction'] < 0:
                results['Both Models Agree'] = True
            else:
                results['Both Models Agree'] = False

            if len(filename) > 1:
                with open(f'./check_results/{filename[0]}_{filename[1]}.json', 'w') as file:
                    json.dump(results, file)
            else:
                with open(f'./check_results/{filename[0]}.json', 'w') as file:
                    json.dump(results, file)

            index += 1
        except Exception as e:
            print("An error occured:", e, '\nRetrying...\n')
            continue


if __name__ == "__main__":
    main()
