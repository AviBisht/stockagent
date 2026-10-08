import React from 'react';
import { Briefcase, RotateCw, DollarSign, Clock, Layers } from 'lucide-react';

export default function PortfolioCard({ portfolioData, onRefresh, loading }) {
  const positions = portfolioData?.positions || {};
  const pendingOrders = portfolioData?.pending_orders || [];
  const entries = Object.entries(positions);
  const tradingActive = portfolioData?.trading_active ?? true;

  return (
    <div className="bento-card card-theme-green col-span-5">
      <div className="card-header-bar">
        <span className="card-badge-pill" style={{ color: 'var(--pastel-green-text)' }}>
          <Briefcase size={14} />
          Trading Book
        </span>
        <button
          className="refresh-icon-btn"
          onClick={onRefresh}
          title="Refresh portfolio"
        >
          <RotateCw size={13} className={loading ? 'spinner' : ''} />
          Refresh
        </button>
      </div>

      <h2 className="card-title" style={{ fontSize: '1.4rem' }}>Portfolio Holdings</h2>
      <p className="card-subtitle">
        Executed positions recorded by TradingExecutor in TradingStateManager.
      </p>

      {/* Positions List */}
      <div className="positions-list" style={{ flex: 1 }}>
        {entries.length > 0 ? (
          entries.map(([ticker, pos]) => (
            <div key={ticker} className="position-item-card">
              <div>
                <div className="position-ticker">{ticker}</div>
                <div className="position-meta">
                  <span>{pos.quantity} shares</span>
                  {pos.timestamp && (
                    <span style={{ marginLeft: '8px', opacity: 0.6 }}>
                      • {new Date(pos.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  )}
                </div>
              </div>

              <div className="position-price">
                ${Number(pos.entry_price).toFixed(2)}
                <div style={{ fontSize: '0.75rem', opacity: 0.65, fontWeight: 500 }}>
                  Entry Price
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="empty-state-box" style={{ background: '#FFFFFF', borderRadius: 'var(--radius-inner)' }}>
            <Layers size={28} style={{ opacity: 0.35, color: 'var(--pastel-green-text)' }} />
            <p>No active positions</p>
            <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>
              Orders executed by TradingExecutor will appear here automatically.
            </span>
          </div>
        )}
      </div>

      {/* Pending orders / Footer */}
      <div className="card-footer-refresh" style={{ color: 'var(--pastel-green-text)' }}>
        <span>Pending Orders: <strong>{pendingOrders.length}</strong></span>
        <span>Status: <strong>{tradingActive ? 'Trading Allowed' : 'Orders Blocked'}</strong></span>
      </div>
    </div>
  );
}
