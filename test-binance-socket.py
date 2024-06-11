import time

import websocket
import json
import requests
from multiprocessing import Process


def send_initial_payloads(ws):
    payload = {
        "id": "BTCUSDT",
        "method": "depth",
        "params": {
            "symbol": "BTCUSDT",
            "limit": 1000,
        }
    }
    payload1 = {
        "id": "ETHUSDT",
        "method": "depth",
        "params": {
            "symbol": "ETHUSDT",
            "limit": 100
        }
    }
    payload2 = {
        "id": "BNBUSDT",
        "method": "depth",
        "params": {
            "symbol": "BNBUSDT",
            "limit": 100
        }
    }
    ws.send(json.dumps(payload))
    ws.send(json.dumps(payload1))
    ws.send(json.dumps(payload2))


def send_payload(ws, symbol):
    payload = {
        "id": symbol,
        "method": "depth",
        "params": {
            "symbol": symbol,
            "limit": 100

        }
    }
    ws.send(json.dumps(payload))


def on_message(ws, message):
    data = json.loads(message)
    print(data)
    print(data["id"], len(data["result"]["bids"]))
    # send_payload(ws, data["id"])


def on_close(ws):
    print("Connection closed")
    print("Reconnecting in 5 seconds...")
    time.sleep(5)
    ws.run_forever()


def on_open(ws):
    send_initial_payloads(ws)
    print("Subscribed to ticker updates")


def on_error(ws, error):
    print("Error:", error)


if __name__ == "__main__":
    websocket_url = "wss://ws-api.binance.com:443/ws-api/v3"

    processes = []

    ws = websocket.WebSocketApp(websocket_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_open=on_open,
                                on_close=on_close
                                )
    ws.run_forever()

