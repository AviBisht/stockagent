import React from 'react';
import { TrendingUp, ArrowDownRight, ArrowUpRight, Minus, AlertCircle, FileText, CheckCircle, Info } from 'lucide-react';

export default function AIAnalysisCard({ analysisResult, loading, error, analyzedQuery }) {
  // Parse signal if explicitly present in response
  const detectSignal = (text) => {
    if (!text) return null;
    const upper = text.toUpperCase();
    if (upper.includes('BUY') || upper.includes('RECOMMENDATION: BUY')) return 'BUY';
    if (upper.includes('SELL') || upper.includes('RECOMMENDATION: SELL')) return 'SELL';
    if (upper.includes('HOLD') || upper.includes('RECOMMENDATION: HOLD')) return 'HOLD';
    return null;
  };

  const signal = analysisResult?.response ? detectSignal(analysisResult.response) : null;

  return (
    <div className="bento-card card-theme-blue col-span-7">
      <div className="card-header-bar">
        <span className="card-badge-pill" style={{ color: 'var(--pastel-blue-text)' }}>
          <TrendingUp size={14} />
          Agent Reasoning & Verdict
        </span>

        {signal && (
          <span className={`signal-pill ${signal.toLowerCase()}`}>
            {signal === 'BUY' && <ArrowUpRight size={15} />}
            {signal === 'SELL' && <ArrowDownRight size={15} />}
            {signal === 'HOLD' && <Minus size={15} />}
            Signal: {signal}
          </span>
        )}
      </div>

      <h2 className="card-title" style={{ fontSize: '1.4rem' }}>AI Analysis Output</h2>
      <p className="card-subtitle">
        Direct multi-agent verdict from the SequentialAgent and TradingSystem pipeline.
      </p>

      {/* Loading State */}
      {loading && (
        <div className="analysis-output-container loading-indicator">
          <div className="spinner" style={{ color: 'var(--pastel-blue-text)' }} />
          <div>
            <div style={{ fontWeight: 700, fontSize: '0.98rem' }}>Multi-Agent Pipeline in Execution</div>
            <div style={{ fontSize: '0.85rem', opacity: 0.7, marginTop: '4px' }}>
              MRG Guardian checking risk across monitoring iterations...
            </div>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="error-banner">
          <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <div style={{ fontWeight: 700, marginBottom: '2px' }}>Analysis Request Error</div>
            <div style={{ fontSize: '0.85rem', opacity: 0.9 }}>{error}</div>
          </div>
        </div>
      )}

      {/* Success / Real Result */}
      {!loading && !error && analysisResult && (
        <div className="analysis-output-container">
          {analyzedQuery && (
            <div className="analysis-query-box">
              <span style={{ fontSize: '0.82rem', fontWeight: 700, opacity: 0.7, textTransform: 'uppercase' }}>
                Query Analyzed
              </span>
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--pastel-blue-text)' }}>
                "{analyzedQuery}"
              </span>
            </div>
          )}

          {/* Quick Metrics Bar if returned */}
          {(analysisResult.risk !== null || analysisResult.confidence !== null) && (
            <div style={{ display: 'flex', gap: '16px', marginBottom: '14px', flexWrap: 'wrap' }}>
              {analysisResult.risk !== null && (
                <div style={{ fontSize: '0.82rem', fontWeight: 600, background: '#EFF6FF', padding: '4px 12px', borderRadius: '12px' }}>
                  Risk Score: <strong>{analysisResult.risk}</strong>
                </div>
              )}
              {analysisResult.confidence !== null && (
                <div style={{ fontSize: '0.82rem', fontWeight: 600, background: '#EFF6FF', padding: '4px 12px', borderRadius: '12px' }}>
                  Confidence: <strong>{Math.round(analysisResult.confidence * 100)}%</strong>
                </div>
              )}
              {analysisResult.status && (
                <div style={{ fontSize: '0.82rem', fontWeight: 600, background: '#EFF6FF', padding: '4px 12px', borderRadius: '12px' }}>
                  System: <strong>{analysisResult.status}</strong>
                </div>
              )}
            </div>
          )}

          <div className="analysis-text-body">
            {analysisResult.response}
          </div>
        </div>
      )}

      {/* Empty State before any query */}
      {!loading && !error && !analysisResult && (
        <div className="analysis-output-container empty-state-box">
          <FileText size={32} style={{ opacity: 0.35, color: 'var(--pastel-blue-text)' }} />
          <p>No analysis performed yet.</p>
          <span style={{ fontSize: '0.82rem', opacity: 0.65 }}>
            Submit a query above to view the Gemini multi-agent reasoning trace and trade signal.
          </span>
        </div>
      )}
    </div>
  );
}
