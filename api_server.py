# api_server.py
# This creates an API that your React dashboard can connect to

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio
from datetime import datetime
from typing import Dict, List

# Import your trading components
from execution.broker_interface import BrokerInterface
from data.market_data import MarketDataManager
from monitoring.performance import PerformanceTracker

app = FastAPI()

# Allow React to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
broker = BrokerInterface()
market_data = MarketDataManager()


@app.get("/")
async def root():
    return {"message": "Trading System API", "status": "running"}


@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio data for dashboard"""
    try:
        account = broker.get_account_info()
        positions = broker.get_positions()

        # Format positions for React
        formatted_positions = []
        for pos in positions:
            formatted_positions.append({
                'symbol': pos['symbol'],
                'shares': pos['shares'],
                'avgCost': float(pos['avg_entry_price']),
                'currentPrice': float(pos['current_price']),
                'pnl': float(pos['unrealized_pl']),
                'pnlPercent': float(pos['unrealized_plpc']) * 100
            })

        return {
            'totalValue': float(account['portfolio_value']),
            'dayChange': float(account['equity']) - float(account['last_equity']),
            'dayChangePercent': ((float(account['equity']) - float(account['last_equity'])) / float(
                account['last_equity'])) * 100,
            'buyingPower': float(account['buying_power']),
            'positions': formatted_positions
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/signals")
async def get_signals():
    """Get current trading signals"""
    # This would come from your signal generator
    # For now, return mock data
    return {
        'signals': [
            {
                'symbol': 'TSLA',
                'type': 'BUY',
                'confidence': 0.82,
                'probability': 0.78,
                'expectedReturn': 0.065,
                'riskScore': 0.28
            },
            {
                'symbol': 'AMD',
                'type': 'BUY',
                'confidence': 0.75,
                'probability': 0.71,
                'expectedReturn': 0.055,
                'riskScore': 0.35
            }
        ]
    }


@app.get("/api/gpu-metrics")
async def get_gpu_metrics():
    """Get GPU performance metrics"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                'utilization': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            }
    except:
        return {
            'utilization': 0,
            'memory_used': 0,
            'memory_total': 8192,
            'temperature': 0
        }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Send updates every 5 seconds
            data = {
                'type': 'portfolio_update',
                'data': await get_portfolio(),
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send_json(data)
            await asyncio.sleep(5)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRADING SYSTEM API SERVER")
    print("=" * 60)
    print("API running at: http://localhost:8000")
    print("React can now connect to fetch data!")
    print("\nEndpoints:")
    print("  GET  /api/portfolio - Get portfolio data")
    print("  GET  /api/signals   - Get trading signals")
    print("  GET  /api/gpu-metrics - Get GPU metrics")
    print("  WS   /ws           - WebSocket for real-time updates")
    print("\nPress Ctrl+C to stop")

    uvicorn.run(app, host="0.0.0.0", port=8000)