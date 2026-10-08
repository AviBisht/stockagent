/**
 * StockAgent API Client Layer
 * Communicates with the FastAPI backend endpoints.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson && errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string' 
          ? errorJson.detail 
          : JSON.stringify(errorJson.detail);
      }
    } catch {
      // Body was not JSON
    }
    const err = new Error(errorDetail);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

/**
 * Fetch current system and trading status
 * GET /api/status
 */
export async function getStatus() {
  const response = await fetch(`${API_BASE_URL}/api/status`);
  return handleResponse(response);
}

/**
 * Fetch portfolio positions and pending orders
 * GET /api/portfolio
 */
export async function getPortfolio() {
  const response = await fetch(`${API_BASE_URL}/api/portfolio`);
  return handleResponse(response);
}

/**
 * Fetch archived memory bank (risk scores and strategies)
 * GET /api/memory
 */
export async function getMemory() {
  const response = await fetch(`${API_BASE_URL}/api/memory`);
  return handleResponse(response);
}

/**
 * Fetch internal audit log timeseries
 * GET /api/audit-logs
 */
export async function getAuditLogs() {
  const response = await fetch(`${API_BASE_URL}/api/audit-logs`);
  return handleResponse(response);
}

/**
 * Send user query / trade command to StockAgent multi-agent runner
 * POST /api/chat
 * @param {string} query
 */
export async function sendChat(query) {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  });
  return handleResponse(response);
}
