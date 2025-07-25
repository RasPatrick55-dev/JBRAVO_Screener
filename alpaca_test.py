from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os

load_dotenv('.env')

API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
positions = trading_client.get_all_positions()
print('Positions:', positions)
