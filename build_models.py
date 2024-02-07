import network_regression
import network_classification
import checker
import argparse

parser = argparse.ArgumentParser(
    description="Build both Regression and Classification models for the parsed pie.")

parser.add_argument('Pie', metavar='pie', type=int,
                    help='Please select one of the available pies.')
parser.add_argument('--PriceHistory', metavar='price_history', type=str, default='n',
                    help="Choose 'y' if you want to see price history.")
parser.add_argument('--ModelPerformance', metavar='model_performance', type=str,
                    default='n', help="Choose 'y' if you want to see model(s) performance.")

args = parser.parse_args()


def build_models_regression(user_choice: int = None, price_history: str = None, metrics: str = None):
    network_regression.main(user_choice, price_history, metrics)


def build_models_classification(user_choice: int = None, price_history: str = None, metrics: str = None):
    network_classification.main(user_choice, price_history, metrics)


def compare_models(user_choice: int = None, price_history: str = None, metrics: str = None):
    checker.main(user_choice, price_history, metrics)


def main():
    build_models_classification(
        args.Pie, args.PriceHistory, args.ModelPerformance)
    build_models_regression(args.Pie, args.PriceHistory, args.ModelPerformance)
    compare_models(args.Pie, args.PriceHistory, args.ModelPerformance)


if __name__ == "__main__":
    main()
