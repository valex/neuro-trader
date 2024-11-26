## Overview
This repository contains pre-trained neural network models for trading on Binance Futures. Each model is accompanied by a Python script that enables automated trading. The neural network model independently decides whether to take a long or short position, while you only need to allocate the required amount of funds for trading.

Each model's recent performance is in [PERFORMANCE.md](PERFORMANCE.md) file.


## Repository Structure
All neural network models are stored in the `models` folder. Within this folder, models are organized into subfolders based on the data they were trained on.

For example, the model in the `RSRUSDT-4h` directory was trained using data for the trading pair RSR/USDT with a 4-hour timeframe.

Within each model's subfolder, there are additional directories named by the date when a specific model was trained (e.g., `2024-11-20`). This structure allows for multiple trained models to coexist for the same trading pair and timeframe, making it easier to manage and reference different versions.

Each such directory includes:

- the model file (`model.pkl`),
- a configuration file example (`config.example.cfg`),
- a list of required dependencies (`requirements.txt`),
- and the trading script (`trade.py`).


### Requirements
- **Python** 3.8+
- **pip** for dependency management

### Installation

#### 1. Clone the Repository
```sh
git clone https://github.com/valex/neuro-trader.git
cd neuro-trader
```

#### 2. Navigate to the desired model directory:
```sh
cd models
cd ZRXUSDT-4h
cd 2024-11-20
```

#### 3. Create and activate a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### 4. Install all dependencies:
```sh
pip install -r requirements.txt
```

#### 5. Rename `config.example.cfg` to `config.cfg`:
```sh
mv config.example.cfg config.cfg  # On Windows, use `rename config.example.cfg config.cfg`
```


#### 6. Open the `config.cfg` file in a text editor and replace the placeholders with your Binance Futures API credentials:
```sh
key=YOUR_BINANCE_FUTURE_API_KEY
secret=YOUR_BINANCE_FUTURE_API_SECRET
```

Make sure to keep your API key and secret secure and do not share them publicly.

To obtain your API keys, go to your Binance account and navigate to **[Account]** → **[API Management]**. Click **Create API** to generate new keys.

Make sure to check the box for **"Enable Futures"** to activate futures trading permissions.



### Usage

You can start trading by running the `trade.py` script.
```sh
python trade.py <usdt_to_trade>
```

Here, `<usdt_to_trade>` is the amount you allocate for trading, considering the selected leverage.

The script will make trading decisions once a new candle is formed based on the selected timeframe.




### Conclusions

1. By default, if the script is interrupted, it will automatically close any open positions. To change this behavior, open `trade.py` and set `CLOSE_ALL_ON_EXIT = False`. After making this change, you will need to manually close any open positions if the script is stopped.

2. Run only one script for one model per binance account.

If you encounter any issues or have suggestions for improvements, feel free to open an issue or contribute to the repository. Happy trading!

### DISCLAIMER FOR FINANCIAL MARKETS:
This script is distributed for educational purposes only and does not constitute financial advice. Trading cryptocurrencies is highly speculative and involves substantial risk of loss. The author of this script takes no responsibility for any financial losses resulting from its use. Users assume full responsibility for their trading decisions and are strongly encouraged to consult a licensed financial advisor before engaging in cryptocurrency trading.

### Donations
Bitcoin (**BTC**): bc1qt63aq5cpumgmv79ssy3ufy6qmawff5ulmx48cm  
Monero (**XMR**): 86AseotWnLaYoiMGgK79QiC9wRXfCvmSyLBHaa5Z66xWQL3sSruXjGfbLGA3VmJcVWKpmWgUApcNGBUjuCPDLJtNDeewEus  
Litecoin (**LTC**): MKd5bRmRQ8NxAbuP6rkWvF15BFYpG95KFs  




