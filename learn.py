#!/usr/bin/proxychains python3

import ccxt
from time import sleep
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pickle
import json

def initialize_exchange():
    exchange_name = 'kucoin'
    exchange = getattr(ccxt, exchange_name)()
    exchange.set_sandbox_mode(enabled=False)
    exchange.apiKey = '657e612f747ae50001466868'
    exchange.secret = 'bcbf1f27-afe9-4343-b8c7-24f0d2e98d71'
    exchange.password = '@2Heroku'
    return exchange

def train_model(data):
    features = np.array([d[1:5] for d in data])
    target = np.array([1 if d[4] < d[1] else 0 for d in data])

    hyperparameters = {
        'hidden_layer_sizes': [(10,), (50,), (100,)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.001, 0.01, 0.1]
    }

    mlp = MLPClassifier()
    grid_search = GridSearchCV(mlp, hyperparameters, cv=5, n_jobs=-1)
    grid_search.fit(features, target)

    with open('hyperparameters.json', 'w') as f:
        json.dump(grid_search.best_params_, f)

    mlp.set_params(**grid_search.best_params_)
    mlp.fit(features, target)

    with open('15mincheck15minochlvmodel.pkl', 'wb') as f:
        pickle.dump(mlp, f)

    return mlp

def predict_market_direction(model, data):
    features_2d = np.array(data[-1][1:5]).reshape(-1, 4)
    prediction = model.predict(features_2d)
    return prediction

def main():
    exchange = initialize_exchange()

    while True:
        try:
            data = exchange.fetch_ohlcv('BTC/USDT', '15m')
            model = train_model(data)

            while True:
                current_time = datetime.now()
                print('# Market Data Print: Bullish [0] vs Bearish [1]')
                prediction = predict_market_direction(model, data)
                print("The market is ---> {}".format(prediction))
                print(current_time.strftime("%B %d, %Y %I:%M %p"))
                sleep(60)

        except:
            sleep(60)
            continue

if __name__ == "__main__":
    main()
