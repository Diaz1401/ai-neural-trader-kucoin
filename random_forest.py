#!/usr/bin/proxychains python3

from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, send_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from time import sleep
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import os
import telebot
import threading

BASE = 'USDT'
CRYPTO = 'ETH'
# BASE = 'USDC'
# CRYPTO = 'USDT'
SYMBOL = CRYPTO + '/' + BASE
TIMEFRAME = '15m'
SLEEP = 1
TEST  = False

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
API_KEY = os.getenv('API_KEY')
SECRET = os.getenv('SECRET')
PASSWORD = os.getenv('PASSWORD')
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')


def initialize_exchange():
    exchange_name = 'kucoin'
    exchange = getattr(ccxt, exchange_name)()
    exchange.set_sandbox_mode(enabled=False)
    exchange.apiKey = API_KEY
    exchange.secret = SECRET
    exchange.password = PASSWORD
    return exchange

def train_model(data):
    features = []
    target = []
    print(len(data))
    print(data[-1])
    for i in range(0, len(data)-1):
        features.append(data[i][1:])
        target.append(1 if data[i+1][-1] > data[i][-1] else 0)
    features = np.array(features)
    target = np.array(target)
    model = RandomForestClassifier(criterion='entropy', n_estimators=1000, max_depth=1000, max_features=150, n_jobs=-1)

    model = model.fit(features, target)
    return model

def calculate_sma_ema(prices):
    window_sizes = [i for i in range(2, 51)]
    sma_ema = []
    for window in window_sizes:
        alpha = 2 / (window + 1)
        initial_sma = np.mean(prices[-window:])
        ema_values = [initial_sma]
        for price in prices[-window + 1:]:
            ema_value = alpha * price + (1 - alpha) * ema_values[-1]
            ema_values.append(ema_value)
        sma_ema.append(round(ema_values[-1], 10))
        sma_ema.append(round(np.sum(prices[-window:]) / window, 10))
    return sma_ema

def calculate_rsi(prices):
    def calculate_single_rsi(prices, window_size):
        gains = []
        losses = []

        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            if diff >= 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))

        avg_gain = sum(gains[:window_size]) / window_size
        avg_loss = sum(losses[:window_size]) / window_size
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        rsi = (rsi - 50) * 2
        return rsi

    window_sizes = [i for i in range(2, 51)]
    rsi_values = [calculate_single_rsi(prices, frame_size) for frame_size in window_sizes]
    return rsi_values

def split_data(data, ratio=0.95):
    # data = data[:-150]
    # data = data[-1000:]
    format = []
    for i in range(49, len(data)):
        temp = [row[4] for row in data[i-49:i+1]]
        sma_ema = calculate_sma_ema(temp)
        rsi = calculate_rsi(temp)
        format.append([data[i][0], *sma_ema[:], *rsi[:], data[i][4]])
    split_index = int(len(format) * ratio)
    train_data = format[:split_index]
    test_data = format[split_index:]
    # train_data = format
    # test_data = format
    print(f'Train: {len(train_data)}\nTest: {len(test_data)}')
    return train_data, test_data, format

def predict_market_direction(model, data):
    print(len(data[-1]))
    features = np.array(data[-1][1:]).reshape(-1, 148)
    prediction = model.predict_proba(features)[0][1]
    return prediction

def plot_predictions(timestamps, predictions, prices):
    # Convert timestamps to dates
    if TEST:
        dates = [datetime.fromtimestamp(ts / 1000).strftime('%H\n%M\n%S') for ts in timestamps]
    else:
        dates = [ts.strftime('%H\n%M\n%S') for ts in timestamps]
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)  # Creating the first subplot for predictions
    plt.plot(dates, predictions, linestyle='-', color='b')
    plt.title('Market Predictions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Predictions')
    plt.grid(True)
    #plt.xticks(rotation='vertical')
    
    plt.subplot(2, 1, 2)  # Creating the second subplot for prices
    plt.plot(dates, prices, linestyle='-', color='r')
    plt.title('Prices Over Time')
    plt.xlabel('Time')
    plt.ylabel('Prices')
    plt.grid(True)
    #plt.xticks(rotation='vertical')
    
    plt.tight_layout()
    plt.savefig('static/output.png')  # Save the plot image
    if TEST:
        plt.show()
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot')
def plot():
    return send_file('static/output.png', mimetype='image/png')

def send_telegram_message(message):
    bot_token = BOT_TOKEN
    chat_id = CHAT_ID
    bot = telebot.TeleBot(bot_token)
    bot.send_message(chat_id, message)

def main():
    exchange = initialize_exchange()
    timestamps = []  # List to store timestamps
    predictions = []  # List to store predictions
    prices = []  # List to store prices

    while True:
        try:
            current_time = datetime.now()
            data = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1500)

            print('# Train data')
            # | Data for training | Data for testing | Full data without spliting
            train_data, test_data, full_data = split_data(data)

            if TEST:
                model = train_model(train_data)
                test_features = []
                test_target = []
                for i in range(0, len(test_data)-1):
                    test_features.append(test_data[i][1:])
                    test_target.append(1 if test_data[i+1][-1] > test_data[i][-1] else 0)
                test_features = np.array(test_features)
                test_target = np.array(test_target)

                # Checking test accuracy
                test_predictions = model.predict(test_features)
                test_accuracy = model.score(test_features, test_target)
                mse = np.mean((test_predictions - test_target) ** 2)
                mae = np.mean(np.abs(test_predictions - test_target))
                print('# Test data predictions')
                loop = 0
                win = 0
                loss = 0
                future_prices = []
                balance = 100
                max_up = 0
                max_down = 0
                total_percent = 0
                for row in test_data:
                    if loop < len(test_data)-5:
                        future_prices.append([test_data[loop+1][-1], test_data[loop+2][-1], test_data[loop+3][-1], test_data[loop+4][-1], test_data[loop+5][-1]])
                        max_up = ((max(future_prices[-1]) - row[-1]) / row[-1])
                        max_down = ((min(future_prices[-1]) - row[-1]) / row[-1])

                        # Check if the price increased or decreased
                        print('# Show increase/decrease potential in next 5 move')
                        print(f"==> Max increase by {round(max_up * 100, 3)}%")
                        print(f"==> Max decrease by {round(max_down * 100, 3)}%")
                    # Predict market direction for each row in test data
                    prediction = predict_market_direction(model, [row])

                    timestamps.append(row[0])  # Assuming timestamp is the first element
                    predictions.append(prediction)  # Append prediction
                    prices.append(row[-1])  # Assuming price data is in the fifth column (adjust if necessary)
                    if prediction > 0.5 and loop < len(test_data)-5:
                        if any(x > row[-1] for x in future_prices[-1]):
                            win += 1
                        else:
                            loss += 1
                        total_percent += round(max_up*100, 2)
                    print(f'BALANCE ${balance}')
                    print(f'Loop    : {loop}\nGuess   : {win + loss}\nWin     : {win}\nLoss    : {loss}\nMax     : {total_percent}\n')
                    loop += 1

                print(f'WIN {(win / (win + loss)) * 100}%')
                print(f"Mean Squared Error: {mse}")
                print(f"Mean Absolute Error: {mae}")
                print(f"Accuracy: {round(test_accuracy * 100, 1)}%")
            else:
                model = train_model(full_data)
                prediction = predict_market_direction(model, full_data)

                balance = exchange.fetch_balance()
                ticker = exchange.fetch_ticker(SYMBOL)
                print('# Check the current bid and ask prices')
                bid = ticker['bid']
                ask = ticker['ask']
                midpoint = (bid + ask) / 2
                print(f'ask ---> {ask}\nbid ---> {bid}\nmidpoint ---> {midpoint}')
                crypto_balance = balance[CRYPTO]['free']
                base_balance = balance[BASE]['free']
                capital = (crypto_balance * midpoint) + base_balance
                crypto_amount = (capital * 0.1) / midpoint
                base_amount = capital * 0.1
                print(f'# Balance:\n  {crypto_balance} {CRYPTO}\n  {base_balance} {BASE}\n# Total:\n  {capital} {BASE}')
                print(f"Market prediction : {prediction}")
                print(current_time.strftime("%B %d, %Y %I:%M %p"))

                # Your existing code to fetch predictions
                timestamps.append(current_time)  # Append current timestamp
                predictions.append(prediction)  # Append prediction
                prices.append(full_data[-1][-1])
                if prediction > 0.6: # buy
                    #exchange.create_order(SYMBOL, 'limit', 'buy', crypto_amount, midpoint)
                    print(f'BUY ZONE !!!\nCoin: {CRYPTO}\nPrice: {round(midpoint, 2)}\nScore: {round(prediction, 3)}')
                    #print(f'Bought {crypto_amount} {CRYPTO} at {midpoint} {BASE}\nScore: {round(prediction, 3)}')
                    #send_telegram_message(f'Bought {crypto_amount} {CRYPTO} at {midpoint} {BASE}\nScore: {round(prediction, 3)}')
                    send_telegram_message(f'BUY ZONE !!!\nCoin: {CRYPTO}\nPrice: {round(midpoint, 2)}\nScore: {round(prediction, 3)}')
                elif prediction < 0.4: # sell
                    print(f'SELL ZONE !!!\nCoin: {CRYPTO}\nPrice: {round(midpoint, 2)}\nScore: {round(prediction, 3)}')
                    send_telegram_message(f'SELL ZONE !!!\nCoin: {CRYPTO}\nPrice: {round(midpoint, 2)}\nScore: {round(prediction, 3)}')
                else:
                    print(f'Neutral\nCoin: {CRYPTO}\nPrice: {round(midpoint, 2)}\nScore: {round(prediction, 3)}')
                    send_telegram_message(f'Neutral\nCoin: {CRYPTO}\nPrice: {round(midpoint, 2)}\nScore: {round(prediction, 3)}')

            plot_thread = threading.Thread(target=plot_predictions, args=(timestamps[-50:], predictions[-50:], prices[-50:]))
            plot_thread.start()

            if TEST:
                sleep(999999999)
            else:
                sleep(SLEEP)
        except Exception as e:
            print(f'Error occurred: {e}')
            sleep(SLEEP)
            continue

if __name__ == "__main__":
    flask_thread = threading.Thread(target=app.run, kwargs={'port': 3000})
    flask_thread.start()
    main()
