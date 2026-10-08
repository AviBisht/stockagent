import React from 'react';
import { Bot, ShieldCheck, AlertCircle, ArrowUpRight } from 'lucide-react';

export default function Navbar({ systemStatus, activeTab, setActiveTab, onFocusInput }) {
  const isActive = systemStatus?.trading_active ?? true;
  const isKeyConfigured = systemStatus?.api_key_configured ?? false;

  return (
    <nav className="navbar-wrapper">
      {/* Left: Brand / Logo Pill */}
      <div className="nav-pill-logo">
        <Bot size={22} />
        <span>StockAgent</span>
      </div>

      {/* Center: Navigation Tabs */}
      <div className="nav-pill-links">
        {['Dashboard', 'Analysis', 'Portfolio', 'Activity'].map((tab) => (
          <button
            key={tab}
            className={`nav-tab-btn ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Right: System Status & Primary CTA */}
      <div className="nav-pill-actions">
        <div 
          className={`system-status-indicator ${isActive ? '' : 'paused'}`}
          title={systemStatus?.pause_reason || (isActive ? 'Guardian Active' : 'Guardian Paused')}
        >
          <span className={`status-dot ${isActive ? '' : 'paused'}`} />
          <span>{isActive ? 'System Active' : 'System Paused'}</span>
        </div>

        <button 
          className="nav-cta-btn"
          onClick={onFocusInput}
        >
          <span>Analyze Stock</span>
          <ArrowUpRight size={16} />
        </button>
      </div>
    </nav>
  );
}
