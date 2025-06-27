// monitoring/dashboard_component.jsx

import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';
import {
  TrendingUp, TrendingDown, Activity, DollarSign,
  BarChart2, RefreshCw, Clock, Zap, Brain, Target
} from 'lucide-react';

const TradingDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [portfolio, setPortfolio] = useState({
    totalValue: 10000,
    dayChange: 124.50,
    dayChangePercent: 1.26,
    positions: [
      { symbol: 'AAPL', shares: 10, avgCost: 150, currentPrice: 155, pnl: 50 },
      { symbol: 'GOOGL', shares: 5, avgCost: 2800, currentPrice: 2850, pnl: 250 }
    ]
  });

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="w-8 h-8 text-purple-500" />
            Smart Swing Trader
          </h1>
          <div className="flex items-center gap-4">
            <Clock className="w-4 h-4" />
            <span>{new Date().toLocaleTimeString()}</span>
            <RefreshCw className="w-5 h-5 cursor-pointer" />
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="flex gap-1 px-4">
          {['overview', 'signals', 'positions'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-3 capitalize ${
                activeTab === tab ? 'text-purple-400 border-b-2 border-purple-400' : 'text-gray-400'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <main className="p-6">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Portfolio Summary */}
            <div className="lg:col-span-2 bg-gray-800 rounded-xl p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <DollarSign className="w-5 h-5 text-green-400" />
                Portfolio Summary
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-700 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">Total Value</p>
                  <p className="text-2xl font-bold">{formatCurrency(portfolio.totalValue)}</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">Day Change</p>
                  <p className={`text-2xl font-bold ${portfolio.dayChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {formatCurrency(portfolio.dayChange)}
                  </p>
                </div>
              </div>
            </div>

            {/* GPU Metrics */}
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                GPU Performance
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>GPU Utilization</span>
                    <span>78%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div className="bg-gradient-to-r from-green-400 to-yellow-400 h-2 rounded-full" style={{ width: '78%' }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Memory Usage</span>
                    <span>6.2/8 GB</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div className="bg-gradient-to-r from-blue-400 to-purple-400 h-2 rounded-full" style={{ width: '77.5%' }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'signals' && (
          <div className="bg-gray-800 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              AI Trading Signals
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-gray-400 border-b border-gray-700">
                    <th className="pb-3">Symbol</th>
                    <th className="pb-3">Signal</th>
                    <th className="pb-3">Confidence</th>
                    <th className="pb-3">Expected Return</th>
                    <th className="pb-3">Risk Score</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-700">
                    <td className="py-3">TSLA</td>
                    <td className="py-3">
                      <span className="px-2 py-1 rounded text-xs font-semibold bg-green-900 text-green-300">BUY</span>
                    </td>
                    <td className="py-3">82%</td>
                    <td className="py-3 text-green-400">+6.5%</td>
                    <td className="py-3">
                      <span className="text-green-400">Low</span>
                    </td>
                  </tr>
                  <tr className="border-b border-gray-700">
                    <td className="py-3">AMD</td>
                    <td className="py-3">
                      <span className="px-2 py-1 rounded text-xs font-semibold bg-green-900 text-green-300">BUY</span>
                    </td>
                    <td className="py-3">75%</td>
                    <td className="py-3 text-green-400">+5.5%</td>
                    <td className="py-3">
                      <span className="text-yellow-400">Medium</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'positions' && (
          <div className="bg-gray-800 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-blue-400" />
              Current Positions
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-gray-400 border-b border-gray-700">
                    <th className="pb-3">Symbol</th>
                    <th className="pb-3">Shares</th>
                    <th className="pb-3">Avg Cost</th>
                    <th className="pb-3">Current Price</th>
                    <th className="pb-3">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio.positions.map((pos, idx) => (
                    <tr key={idx} className="border-b border-gray-700">
                      <td className="py-3 font-medium">{pos.symbol}</td>
                      <td className="py-3">{pos.shares}</td>
                      <td className="py-3">${pos.avgCost}</td>
                      <td className="py-3">${pos.currentPrice}</td>
                      <td className={`py-3 font-medium ${pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {formatCurrency(pos.pnl)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default TradingDashboard;