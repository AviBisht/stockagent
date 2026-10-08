import React, { useState } from 'react';
import { Sparkles, ArrowRight, Loader2, Cpu, ShieldCheck } from 'lucide-react';

export default function StockAnalysisCard({ onAnalyze, loading, inputRef }) {
  const [query, setQuery] = useState('');

  const exampleQueries = [
    'The optimal trade is to BUY 100 shares of MSFT at the current market price.',
    'Evaluate momentum strategy on NVDA in Choppy market',
    'Assess Mean Reversion strategy for AAPL',
    'Check risk profile for TSLA with trending volatility',
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;
    onAnalyze(query);
  };

  const handleChipClick = (example) => {
    setQuery(example);
    onAnalyze(example);
  };

  return (
    <div className="bento-card card-theme-cream col-span-8">
      {/* Header Badge */}
      <div className="card-header-bar">
        <span className="card-badge-pill" style={{ color: 'var(--pastel-cream-text)' }}>
          <Sparkles size={14} />
          Multi-Agent Core
        </span>
      </div>

      {/* Main Title & Subtitle */}
      <h1 className="card-title">AI Stock Analyst</h1>
      <p className="card-subtitle">
        Analyze a stock using our multi-agent AI system. Autonomous Model Reliability Guardian (MRG) verifies risk before strategy execution.
      </p>

      {/* Reference Mockup Design Elements */}
      <div className="hero-mockup-pills" style={{ marginBottom: '20px' }}>
        <div className="thumb-tag">
          <ShieldCheck size={16} />
          <span>MRG Safety Loop</span>
        </div>
        <div className="pill-tag-text">
          <Cpu size={16} />
          <span>Gemini 2.5 Flash Lite + ADK</span>
        </div>
      </div>

      {/* Input & Form */}
      <div className="hero-input-container">
        <form onSubmit={handleSubmit} className="hero-input-pill">
          <input
            ref={inputRef}
            type="text"
            className="hero-input-field"
            placeholder="Enter ticker or ask StockAgent (e.g., BUY 100 shares of MSFT)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            className="hero-analyze-btn"
            disabled={loading || !query.trim()}
          >
            {loading ? (
              <>
                <Loader2 size={16} className="spinner" />
                <span>Evaluating...</span>
              </>
            ) : (
              <>
                <span>Analyze</span>
                <ArrowRight size={16} />
              </>
            )}
          </button>
        </form>

        {/* Suggestion Chips */}
        <div className="hero-chips-row">
          <span className="hero-chip-label">Try example:</span>
          {exampleQueries.map((ex, idx) => (
            <button
              key={idx}
              type="button"
              className="hero-example-chip"
              onClick={() => handleChipClick(ex)}
              disabled={loading}
            >
              {ex.length > 34 ? `${ex.slice(0, 34)}...` : ex}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
