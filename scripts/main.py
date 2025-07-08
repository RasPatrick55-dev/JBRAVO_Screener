import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

dotenv_path = os.path.expanduser('/home/RasPatrick/jbravo_screener/.env')
load_dotenv(dotenv_path)

app = Flask(__name__)

api = tradeapi.REST(
    os.getenv("APCA_API_KEY_ID"),
    os.getenv("APCA_API_SECRET_KEY"),
    os.getenv("APCA_API_BASE_URL"),
    api_version="v2"
)

@app.route('/webhook', methods=['POST'])
def webhook_handler():
    data = request.json
    action = data.get('action', '').lower()
    ticker = data.get('ticker')
    qty = 1  # customize quantity based on your strategy

    if action not in ('buy', 'sell'):
        return jsonify({'status': 'ignored', 'reason': 'Invalid action'}), 400

    try:
        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side=action,
            type='market',
            time_in_force='day'
        )
        return jsonify({'status': 'executed', 'order_id': order.id}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)