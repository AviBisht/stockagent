import React from 'react';
import { Database, RotateCw, BookOpen, ShieldAlert } from 'lucide-react';

export default function MemoryCard({ memoryData, onRefresh, loading }) {
  const riskScores = memoryData?.risk_scores || {};
  const strategies = memoryData?.strategies || {};
  
  const riskEntries = Object.entries(riskScores);
  const strategyEntries = Object.entries(strategies);
  const totalEntries = riskEntries.length + strategyEntries.length;

  return (
    <div className="bento-card card-theme-lavender col-span-6">
      <div className="card-header-bar">
        <span className="card-badge-pill" style={{ color: 'var(--pastel-lavender-text)' }}>
          <Database size={14} />
          Knowledge Bank
        </span>
        <button
          className="refresh-icon-btn"
          onClick={onRefresh}
          title="Refresh memory"
        >
          <RotateCw size={13} className={loading ? 'spinner' : ''} />
          Refresh
        </button>
      </div>

      <h2 className="card-title" style={{ fontSize: '1.4rem' }}>Memory Curator</h2>
      <p className="card-subtitle">
        Archived risk scores and strategy definitions preserved across sessions in MEMORY_BANK.
      </p>

      {/* Metrics Row */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '14px' }}>
        <div style={{ flex: 1, background: 'rgba(255,255,255,0.7)', padding: '12px 16px', borderRadius: 'var(--radius-inner)' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 600, opacity: 0.7, textTransform: 'uppercase' }}>Archived Risks</div>
          <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--pastel-lavender-text)' }}>{riskEntries.length}</div>
        </div>
        <div style={{ flex: 1, background: 'rgba(255,255,255,0.7)', padding: '12px 16px', borderRadius: 'var(--radius-inner)' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 600, opacity: 0.7, textTransform: 'uppercase' }}>Stored Strategies</div>
          <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--pastel-lavender-text)' }}>{strategyEntries.length}</div>
        </div>
      </div>

      {/* Stack List */}
      <div className="items-stack">
        {totalEntries > 0 ? (
          <>
            {riskEntries.map(([ticker, data]) => (
              <div key={ticker} className="item-row">
                <div>
                  <strong style={{ color: 'var(--pastel-lavender-text)' }}>{ticker}</strong>
                  <span style={{ marginLeft: '8px', fontSize: '0.8rem', opacity: 0.65 }}>
                    Risk: {data.score} | Conf: {data.confidence}
                  </span>
                </div>
                <span style={{ fontSize: '0.75rem', opacity: 0.5 }}>
                  {data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : 'Archived'}
                </span>
              </div>
            ))}

            {strategyEntries.map(([regime, strat]) => (
              <div key={regime} className="item-row">
                <div>
                  <strong style={{ color: 'var(--pastel-lavender-text)' }}>{regime}</strong>
                  <span style={{ marginLeft: '8px', fontSize: '0.8rem', opacity: 0.65 }}>Strategy Profile</span>
                </div>
                <BookOpen size={14} style={{ opacity: 0.5 }} />
              </div>
            ))}
          </>
        ) : (
          <div className="empty-state-box" style={{ background: '#FFFFFF', borderRadius: 'var(--radius-inner)', padding: '24px 16px' }}>
            <Database size={26} style={{ opacity: 0.3, color: 'var(--pastel-lavender-text)' }} />
            <p>No archived memory entries</p>
            <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>
              MemoryCuratorAgent archives will appear here after evaluation loops run.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
