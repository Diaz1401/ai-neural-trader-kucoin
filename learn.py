#!/usr/bin/proxychains python3

## Things to improve on: 
# 1) use bayes for chosing hyperparamters or evolutions algorithm

import ccxt
from time import sleep
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pickle
import json


# Replace EXCHANGE_NAME with the name of the exchange you want to use
exchange_name = 'kucoin'

# Instantiate the Exchange class
exchange = getattr(ccxt, exchange_name)()

# Set sandbox mode to True or False
exchange.set_sandbox_mode(enabled=False)

# Set your API keys
exchange.apiKey = '657e612f747ae50001466868'
exchange.secret = 'bcbf1f27-afe9-4343-b8c7-24f0d2e98d71'
exchange.password = '@2Heroku'

# Set the symbol you want to trade on Kucoin
symbol = 'BTC/USDT'

# Set the amount of BTC you want to trade
amount = .005

# KuCoin fee per transcation
fee = .001

# Set the premium for the sell order
#print('# Set the premium for the sell order')
premium = 0.002 + fee

## Start the trading script
while True:
    try:
        # Batch streaming data from KuCoin it uses a pair
        # and a window time frame
        data = exchange.fetch_ohlcv('BTC/USDT', '15m')

        # predict if bullish or bearish
        def predict_market_direction(data):
            # extract the features and the target variable from the data (IMPROTRANT: chagned > to <)
            features = np.array([d[1:5] for d in data])
            target = np.array([1 if d[4] < d[1] else 0 for d in data])

            # specify the values for the hyperparameters that you want to tune
            hyperparameters = {
                'hidden_layer_sizes': [(10,), (50,), (100,)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.001, 0.01, 0.1]
            }
            # create a neural network model
            mlp = MLPClassifier()

            # use the GridSearchCV class to search through the combinations of hyperparameters
            # and evaluate the performance of the model on a validation set to find the best combination
            grid_search = GridSearchCV(mlp, hyperparameters, cv=5, n_jobs=-1)
            grid_search.fit(features, target)

            # save the best hyperparameters to a JSON file
            with open('hyperparameters.json', 'w') as f:
                json.dump(grid_search.best_params_, f)

            # print the best values for the hyperparameters
            #print(grid_search.best_params_)

            # update the model with the best values for the hyperparameters
            mlp.set_params(**grid_search.best_params_)

            # train the model using the updated hyperparameters and the features and target
            mlp.fit(features, target)

            # open the file in write mode
            f = open('15mincheck15minochlvmodel.pkl', 'wb')

            # save the trained model to the model file
            pickle.dump(mlp, f)

            # close the file
            f.close()

            # predict the market direction using the trained model
            # returns 0 for bullish
            # returns 1 for bearish
            # Use the reshape() method to convert the features[-1] array to a 2D shape with 4 features
            features_2d = np.array(features[-1]).reshape(-1, 4)

            # Use the predict() method with the reshaped array
            prediction = mlp.predict(features_2d)
            return prediction


        # Create an infinite loop to trade continuously
        while True:
            # Fetch the current ticker information for the symbol
            print('# Fetch the current ticker information for the symbol')
            ticker = exchange.fetch_ticker(symbol)

            # Check the current bid and ask prices
            print('# Check the current bid and ask prices')
            bid = ticker['bid']
            ask = ticker['ask']

            # Calculate the midpoint of the bid and ask prices
            print('# Calculate the midpoint of the bid and ask prices')
            midpoint = (bid + ask) / 2

            
            # Market Data Print
            current_time = datetime.now()
            print('# Market Data Print: Bullish [0] vs Bearish [1]')

            print("The market is ---> {}".format(predict_market_direction(data)))

            print(current_time.strftime("%B %d, %Y %I:%M %p"))
            # Market Data Print

            # Check if there are any open orders
            print('# Check if there are any open orders')
            try:
                open_orders = exchange.fetch_open_orders(symbol)
            except:
                sleep(60)
                open_orders = exchange.fetch_open_orders(symbol)

            #if not open_orders:
                #print('# Place a limit buy order at the midpoint price')
                #try:
                    # Check if it is bullish 1 or bearish 0 before buying
                    #if predict_market_direction(data).tolist()[0] == 0:
                        # Place a limit buy order at the midpoint price
                        #order_id = exchange.create_order(symbol, 'limit', 'buy', amount, midpoint)
                #except:
                    # We must own bitcoin and we want to sell it if the script
                    # tries to buy more bitcoin and has insufficent funds
                    #if not open_orders:
                        # Check if it is bullish 0 or bearish 1 before buying
                        #if predict_market_direction(data).tolist()[0] == 1:
                            # Place a limit sell order at the midpoint price plus the premium which includes the fee
                            #order_id = exchange.create_order(symbol, 'limit', 'sell', amount, midpoint * (1 + premium))
                            #order_id = exchange.create_order(symbol, 'limit', 'sell', amount, midpoint)
                #else:
                    ##Always run script even if error restart
                    #auto_start = True

            # Pause for a few seconds and check the status of the open orders before selling
            print('# Pause for a few seconds and check the status of the open orders')
            sleep(5)
            open_orders = exchange.fetch_open_orders(symbol)

            # Check if there are any open orders
            print('# Check if there are any open orders')
            #if not open_orders:
                # Place a limit sell order at the midpoint price plus the premium
                #try:
                    #if predict_market_direction(data).tolist()[0] == 1:
                        #order_id = exchange.create_order(symbol, 'limit', 'sell', amount, midpoint * (1 + premium))
                        #order_id = exchange.create_order(symbol, 'limit', 'sell', amount, midpoint)
                #except:
                    # Place a limit buy order at the midpoint price
                    # If for some reason the script doesnt have anything to sell
                    # It'll just buy it
                    #if predict_market_direction(data).tolist()[0] == 0:
                        #order_id = exchange.create_order(symbol, 'limit', 'buy', amount, midpoint)


            # Pause for a few seconds and check the status of the open orders XYZ
            print('# Pause for a few seconds and check the status of the open orders - checks every 60*15 min')
            sleep(60 * 15)
            try:
                open_orders = exchange.fetch_open_orders(symbol)
            except:
                sleep(5)
                open_orders = exchange.fetch_open_orders(symbol)
    except:
        sleep(60)
        continue