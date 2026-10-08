import React from 'react';
import { Activity, RotateCw, History, CheckCircle2 } from 'lucide-react';

export default function ActivityCard({ auditData, onRefresh, loading }) {
  const auditLogs = auditData?.audit_logs || {};
  
  // Flatten entries into a chronological list
  const entries = [];
  Object.entries(auditLogs).forEach(([metricName, logsList]) => {
    if (Array.isArray(logsList)) {
      logsList.forEach((log) => {
        entries.push({
          metric: metricName,
          value: log.value,
          timestamp: log.timestamp,
        });
      });
    }
  });

  return (
    <div className="bento-card card-theme-pink col-span-6">
      <div className="card-header-bar">
        <span className="card-badge-pill" style={{ color: 'var(--pastel-pink-text)' }}>
          <Activity size={14} />
          Audit Trail
        </span>
        <button
          className="refresh-icon-btn"
          onClick={onRefresh}
          title="Refresh audit activity"
        >
          <RotateCw size={13} className={loading ? 'spinner' : ''} />
          Refresh
        </button>
      </div>

      <h2 className="card-title" style={{ fontSize: '1.4rem' }}>Agent Activity & Logs</h2>
      <p className="card-subtitle">
        Internal metric logs published by InternalMetricLogger tool in INTERNAL_AUDIT_LOG.
      </p>

      {/* Audit Log Stack */}
      <div className="items-stack">
        {entries.length > 0 ? (
          entries.map((entry, idx) => (
            <div key={idx} className="item-row">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CheckCircle2 size={16} style={{ color: '#E11D48', flexShrink: 0 }} />
                <div>
                  <div style={{ fontWeight: 700, color: 'var(--pastel-pink-text)' }}>
                    Metric: {entry.metric}
                  </div>
                  <div style={{ fontSize: '0.78rem', opacity: 0.65 }}>
                    Recorded value: <strong>{entry.value}</strong>
                  </div>
                </div>
              </div>

              <div style={{ fontSize: '0.78rem', opacity: 0.6 }}>
                {entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : 'Logged'}
              </div>
            </div>
          ))
        ) : (
          <div className="empty-state-box" style={{ background: '#FFFFFF', borderRadius: 'var(--radius-inner)', padding: '24px 16px' }}>
            <History size={26} style={{ opacity: 0.3, color: 'var(--pastel-pink-text)' }} />
            <p>No audit metrics recorded yet</p>
            <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>
              Internal audit log entries published during agent execution will stream here.
            </span>
          </div>
        )}
      </div>

      <div className="card-footer-refresh" style={{ color: 'var(--pastel-pink-text)' }}>
        <span>Total Logged Entries: <strong>{entries.length}</strong></span>
        <span>Source: <strong>INTERNAL_AUDIT_LOG</strong></span>
      </div>
    </div>
  );
}
