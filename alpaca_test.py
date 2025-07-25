from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os


def main() -> None:
    load_dotenv('.env')
    api_key = os.getenv('APCA_API_KEY_ID')
    api_secret = os.getenv('APCA_API_SECRET_KEY')
    if not api_key or not api_secret:
        raise ValueError('Missing Alpaca credentials')

    trading_client = TradingClient(api_key, api_secret, paper=True)
    positions = trading_client.get_all_positions()
    print('Positions:', positions)


if __name__ == '__main__':
    main()
