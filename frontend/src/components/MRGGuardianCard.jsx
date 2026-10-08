import React from 'react';
import { Shield, ShieldAlert, CheckCircle2, RotateCw, AlertTriangle, Layers } from 'lucide-react';

export default function MRGGuardianCard({ statusData, latestRisk, latestConfidence, onRefresh, loading }) {
  const isTradingActive = statusData?.trading_active ?? true;
  const pauseReason = statusData?.pause_reason;

  return (
    <div className="bento-card card-theme-lavender col-span-4">
      {/* Header */}
      <div className="card-header-bar">
        <span className="card-badge-pill" style={{ color: 'var(--pastel-lavender-text)' }}>
          <Shield size={14} />
          Safety Layer
        </span>
        <button
          className="refresh-icon-btn"
          onClick={onRefresh}
          title="Refresh system status"
        >
          <RotateCw size={13} className={loading ? 'spinner' : ''} />
          Refresh
        </button>
      </div>

      <h2 className="card-title" style={{ fontSize: '1.4rem' }}>MRG Guardian</h2>
      <p className="card-subtitle">
        Continuous risk-monitoring loop protecting autonomous trades from market regime shifts.
      </p>

      {/* Metric Grid */}
      <div className="mrg-metric-grid">
        <div className="mrg-metric-box">
          <div className="mrg-metric-label">System State</div>
          <div className="mrg-metric-value" style={{ fontSize: '1.2rem' }}>
            {isTradingActive ? (
              <span style={{ color: '#059669', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={18} /> Active
              </span>
            ) : (
              <span style={{ color: '#DC2626', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <ShieldAlert size={18} /> Paused
              </span>
            )}
          </div>
        </div>

        <div className="mrg-metric-box">
          <div className="mrg-metric-label">Max Cycles</div>
          <div className="mrg-metric-value" style={{ fontSize: '1.2rem', color: 'var(--pastel-lavender-text)' }}>
            <Layers size={18} style={{ opacity: 0.8 }} />
            <span>5 Iterations</span>
          </div>
        </div>

        <div className="mrg-metric-box">
          <div className="mrg-metric-label">Last Risk Score</div>
          <div className="mrg-metric-value" style={{ fontSize: '1.25rem' }}>
            {latestRisk !== null && latestRisk !== undefined ? (
              <span style={{ color: latestRisk >= 3 ? '#DC2626' : '#059669' }}>
                {latestRisk} <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>/ 5</span>
              </span>
            ) : (
              <span style={{ fontSize: '0.95rem', opacity: 0.5 }}>Awaiting Run</span>
            )}
          </div>
        </div>

        <div className="mrg-metric-box">
          <div className="mrg-metric-label">Confidence</div>
          <div className="mrg-metric-value" style={{ fontSize: '1.25rem' }}>
            {latestConfidence !== null && latestConfidence !== undefined ? (
              <span style={{ color: latestConfidence >= 0.7 ? '#059669' : '#D97706' }}>
                {Math.round(latestConfidence * 100)}%
              </span>
            ) : (
              <span style={{ fontSize: '0.95rem', opacity: 0.5 }}>Awaiting Run</span>
            )}
          </div>
        </div>
      </div>

      {/* Pause Alert Banner if paused */}
      {!isTradingActive && pauseReason && (
        <div className="mrg-status-banner" style={{ backgroundColor: '#FEE2E2', borderColor: '#FECACA' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#991B1B', fontSize: '0.85rem' }}>
            <AlertTriangle size={16} />
            <strong>Pause Reason:</strong> {pauseReason}
          </div>
        </div>
      )}
    </div>
  );
}
