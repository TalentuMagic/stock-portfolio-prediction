import network_regression
import network_classification
import checker


def build_models_regression(user_choice: int = None, price_history: str = None, metrics: str = None):
    network_regression.main(user_choice, price_history, metrics)


def build_models_classification(user_choice: int = None, price_history: str = None, metrics: str = None):
    network_classification.main(user_choice, price_history, metrics)


def compare_models(user_choice: int = None, price_history: str = None, metrics: str = None):
    checker.main(user_choice, price_history, metrics)


def main():
    build_models_classification(1, 'n', 'n')
    build_models_regression(1, 'n', 'n')
    compare_models(1, 'n', 'n')
