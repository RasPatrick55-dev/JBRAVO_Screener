import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

app = Flask(__name__)

api_key = os.getenv("APCA_API_KEY_ID")
api_secret = os.getenv("APCA_API_SECRET_KEY")
if not api_key or not api_secret:
    raise ValueError("Missing Alpaca credentials")

client = TradingClient(api_key, api_secret, paper=True)

@app.route('/webhook', methods=['POST'])
def webhook_handler():
    data = request.json
    action = data.get('action', '').lower()
    ticker = data.get('ticker')
    qty = 1  # customize quantity based on your strategy

    if action not in ('buy', 'sell'):
        return jsonify({'status': 'ignored', 'reason': 'Invalid action'}), 400

    try:
        order = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY if action == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        response = client.submit_order(order)
        return jsonify({'status': 'executed', 'order_id': response.id}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
