import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import StockAnalysisCard from './components/StockAnalysisCard';
import MRGGuardianCard from './components/MRGGuardianCard';
import AIAnalysisCard from './components/AIAnalysisCard';
import PortfolioCard from './components/PortfolioCard';
import MemoryCard from './components/MemoryCard';
import ActivityCard from './components/ActivityCard';

import { getStatus, getPortfolio, getMemory, getAuditLogs, sendChat } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [statusData, setStatusData] = useState(null);
  const [portfolioData, setPortfolioData] = useState(null);
  const [memoryData, setMemoryData] = useState(null);
  const [auditData, setAuditData] = useState(null);

  // Analysis / Chat state
  const [analyzedQuery, setAnalyzedQuery] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState(null);

  // General loading states
  const [statusLoading, setStatusLoading] = useState(false);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [auditLoading, setAuditLoading] = useState(false);

  const inputRef = useRef(null);

  // Initial load of dashboard data
  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    fetchStatusData();
    fetchPortfolioData();
    fetchMemoryData();
    fetchAuditData();
  };

  const fetchStatusData = async () => {
    setStatusLoading(true);
    try {
      const data = await getStatus();
      setStatusData(data);
    } catch (err) {
      console.error('Failed to load status:', err);
    } finally {
      setStatusLoading(false);
    }
  };

  const fetchPortfolioData = async () => {
    setPortfolioLoading(true);
    try {
      const data = await getPortfolio();
      setPortfolioData(data);
    } catch (err) {
      console.error('Failed to load portfolio:', err);
    } finally {
      setPortfolioLoading(false);
    }
  };

  const fetchMemoryData = async () => {
    setMemoryLoading(true);
    try {
      const data = await getMemory();
      setMemoryData(data);
    } catch (err) {
      console.error('Failed to load memory:', err);
    } finally {
      setMemoryLoading(false);
    }
  };

  const fetchAuditData = async () => {
    setAuditLoading(true);
    try {
      const data = await getAuditLogs();
      setAuditData(data);
    } catch (err) {
      console.error('Failed to load audit logs:', err);
    } finally {
      setAuditLoading(false);
    }
  };

  // Trigger analysis via real FastAPI POST /api/chat
  const handleAnalyze = async (queryText) => {
    setChatLoading(true);
    setChatError(null);
    setAnalyzedQuery(queryText);

    try {
      const result = await sendChat(queryText);
      setAnalysisResult(result);
      
      // Auto-refresh state after agent run finishes
      fetchAllData();
    } catch (err) {
      console.error('Chat API Error:', err);
      setChatError(err.message || 'Failed to complete analysis with StockAgent.');
      setAnalysisResult(null);
    } finally {
      setChatLoading(false);
    }
  };

  const focusSearchInput = () => {
    if (inputRef.current) {
      inputRef.current.focus();
      inputRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // Determine latest risk and confidence to pass to MRG Guardian
  const latestRisk = analysisResult?.risk ?? 
    (memoryData?.risk_scores && Object.keys(memoryData.risk_scores).length > 0 
      ? Object.values(memoryData.risk_scores)[Object.values(memoryData.risk_scores).length - 1]?.score 
      : null);

  const latestConfidence = analysisResult?.confidence ?? 
    (memoryData?.risk_scores && Object.keys(memoryData.risk_scores).length > 0 
      ? Object.values(memoryData.risk_scores)[Object.values(memoryData.risk_scores).length - 1]?.confidence 
      : (auditData?.audit_logs?.confidence && auditData.audit_logs.confidence.length > 0
          ? auditData.audit_logs.confidence[auditData.audit_logs.confidence.length - 1]?.value
          : null));

  return (
    <div className="bento-app-container">
      {/* Top Navbar */}
      <Navbar
        systemStatus={statusData}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onFocusInput={focusSearchInput}
      />

      {/* Main Bento Grid Layout */}
      <main className="bento-grid">
        {/* Row 1: Hero Card (col-span 8) + MRG Guardian Card (col-span 4) */}
        <StockAnalysisCard
          onAnalyze={handleAnalyze}
          loading={chatLoading}
          inputRef={inputRef}
        />

        <MRGGuardianCard
          statusData={statusData}
          latestRisk={latestRisk}
          latestConfidence={latestConfidence}
          onRefresh={fetchStatusData}
          loading={statusLoading}
        />

        {/* Row 2: AI Analysis Output (col-span 7) + Portfolio Holdings (col-span 5) */}
        <AIAnalysisCard
          analysisResult={analysisResult}
          loading={chatLoading}
          error={chatError}
          analyzedQuery={analyzedQuery}
        />

        <PortfolioCard
          portfolioData={portfolioData}
          onRefresh={fetchPortfolioData}
          loading={portfolioLoading}
        />

        {/* Row 3: Memory Curator (col-span 6) + Agent Activity & Audit (col-span 6) */}
        <MemoryCard
          memoryData={memoryData}
          onRefresh={fetchMemoryData}
          loading={memoryLoading}
        />

        <ActivityCard
          auditData={auditData}
          onRefresh={fetchAuditData}
          loading={auditLoading}
        />
      </main>
    </div>
  );
}
