"""Engineering Dashboard Styles for UAV Mission Control — Light Theme.

Clean, high-contrast, crisp technical engineering console styling strictly without
branding, logos, or decorative fluff.
"""

MISSION_CONTROL_CSS = """
<style>
/* Mission Control Technical Light Architecture */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0f172a;
    background-color: #f8fafc;
}

.stApp {
    background-color: #f8fafc;
}

/* Header bar */
.telemetry-header {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 12px 20px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.header-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #0369a1;
    text-transform: uppercase;
}

.header-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #475569;
}

/* Card components */
.dashboard-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 16px 18px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #475569;
    margin-bottom: 12px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
}

/* Status Badges */
.status-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.status-active {
    background: #dcfce7;
    color: #15803d;
    border: 1px solid #86efac;
}

.status-standby {
    background: #e0f2fe;
    color: #0369a1;
    border: 1px solid #7dd3fc;
}

.status-training {
    background: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fca5a5;
}

.badge-device {
    background: #f1f5f9;
    color: #334155;
    border: 1px solid #cbd5e1;
}

/* Context Switch Event Banner */
.transition-banner {
    background: #f0f9ff;
    border-left: 4px solid #0284c7;
    border-top: 1px solid #e0f2fe;
    border-right: 1px solid #e0f2fe;
    border-bottom: 1px solid #e0f2fe;
    padding: 12px 16px;
    border-radius: 0 4px 4px 0;
    margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
}

.transition-header {
    color: #0284c7;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.transition-body {
    font-size: 0.8rem;
    color: #1e293b;
    line-height: 1.5;
}

/* Telemetry stat display */
.telemetry-stat-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 3px;
    padding: 10px 12px;
    text-align: center;
}

.telemetry-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
}

.telemetry-stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 4px;
    letter-spacing: 0.05em;
}

/* Disclaimer notice */
.disclaimer-banner {
    background: #fefce8;
    border-left: 3px solid #ca8a04;
    border-top: 1px solid #fef08a;
    border-right: 1px solid #fef08a;
    border-bottom: 1px solid #fef08a;
    padding: 8px 12px;
    font-size: 0.76rem;
    color: #854d0e;
    font-family: 'JetBrains Mono', monospace;
    margin: 8px 0;
}

/* Timeline */
.timeline-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    padding: 8px 12px;
    border-left: 3px solid #cbd5e1;
    margin-left: 6px;
    color: #475569;
    background: #ffffff;
    border-bottom: 1px solid #f1f5f9;
}

.timeline-item.active {
    border-left: 3px solid #0284c7;
    color: #0f172a;
    background: #f0f9ff;
}

.timeline-time {
    color: #0284c7;
    font-weight: 600;
}

/* Table styling */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    background: #ffffff;
}

.styled-table th {
    background: #f1f5f9;
    color: #334155;
    padding: 8px 10px;
    border: 1px solid #cbd5e1;
    text-align: left;
    font-weight: 600;
}

.styled-table td {
    background: #ffffff;
    color: #0f172a;
    padding: 8px 10px;
    border: 1px solid #e2e8f0;
}

.styled-table tr:nth-child(even) td {
    background: #f8fafc;
}

.styled-table tr:hover td {
    background: #f1f5f9;
}

/* Remove excessive padding */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
</style>
"""
