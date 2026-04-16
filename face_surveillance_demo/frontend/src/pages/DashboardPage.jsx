import { useEffect, useState } from 'react';
import axios from 'axios';
import {
  Users, UserCheck, Clock, UserX, AlertCircle,
  Camera, Activity, Eye, TrendingUp, BarChart3,
  Percent, Timer, Wifi, WifiOff,
} from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, ArcElement, Filler,
} from 'chart.js';
import { Line, Doughnut, Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, ArcElement, Filler
);

const API = 'http://localhost:5000';

const chartOpts = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#fff',
      titleColor: '#111827',
      bodyColor: '#6b7280',
      borderColor: '#e5e7eb',
      borderWidth: 1,
      padding: 10,
      cornerRadius: 8,
      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
      titleFont: { weight: 600, size: 13 },
    },
  },
  scales: {
    x: { grid: { color: '#f3f4f6', drawBorder: false }, ticks: { color: '#9ca3af', font: { size: 11 } } },
    y: { grid: { color: '#f3f4f6', drawBorder: false }, ticks: { color: '#9ca3af', font: { size: 11 } } },
  },
};

export default function DashboardPage() {
  const [data, setData] = useState(null);
  const [live, setLive] = useState(false);
  const [fps, setFps] = useState(0);
  const [uptime, setUptime] = useState(0);

  useEffect(() => {
    const poll = async () => {
      try {
        const [a, s] = await Promise.all([
          axios.get(`${API}/api/analytics`),
          axios.get(`${API}/api/status`),
        ]);
        setData(a.data);
        setLive(s.data.status === 'running');
        setFps(s.data.fps || 0);
        setUptime(s.data.uptime || 0);
      } catch {
        setData(prev => prev || { error: 'Cannot connect to the backend server.' });
        setLive(false);
      }
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  if (!data) {
    return (
      <div className="center-screen">
        <Activity size={36} className="spinner-icon" />
        <p style={{ color: '#9ca3af' }}>Connecting to recognition engine…</p>
      </div>
    );
  }

  if (data.error) {
    return (
      <div className="center-screen">
        <AlertCircle size={44} color="#dc2626" />
        <h2 className="error-title">Backend Offline</h2>
        <p className="error-msg">{data.error}</p>
        <p className="error-hint">Start the server with <code>python app.py</code></p>
      </div>
    );
  }

  const { stats = {}, charts = {}, entryExitLogs = [], insights = [] } = data;
  const fph = charts.facesPerHour || [];
  const kv = charts.knownVsUnknown || { known: 0, unknown: 0 };
  const mf = charts.mostFrequent || [];

  const attendRate = stats.totalStudents
    ? Math.round(((stats.presentToday + stats.lateToday) / stats.totalStudents) * 100)
    : 0;

  // Format uptime
  const mins = Math.floor(uptime / 60);
  const hrs = Math.floor(mins / 60);
  const uptimeStr = hrs > 0 ? `${hrs}h ${mins % 60}m` : `${mins}m`;

  const timelineData = {
    labels: fph.map(f => `${f.hour}:00`),
    datasets: [{
      label: 'Detections',
      data: fph.map(f => f.count),
      borderColor: '#4f46e5',
      backgroundColor: 'rgba(79,70,229,0.06)',
      pointBackgroundColor: '#4f46e5',
      pointRadius: 4, pointHoverRadius: 6,
      tension: 0.4, fill: true,
      borderWidth: 2,
    }],
  };

  const ratioData = {
    labels: ['Recognized', 'Unknown'],
    datasets: [{
      data: [kv.known, kv.unknown],
      backgroundColor: ['#4f46e5', '#ef4444'],
      hoverBackgroundColor: ['#6366f1', '#f87171'],
      borderWidth: 0, spacing: 4,
    }],
  };

  const freqData = {
    labels: mf.map(f => f.person_name),
    datasets: [{
      data: mf.map(f => f.count),
      backgroundColor: '#4f46e5',
      hoverBackgroundColor: '#6366f1',
      borderRadius: 6, barThickness: 20,
    }],
  };

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h1 className="page-title">Dashboard</h1>
            <p className="page-subtitle">Real-time attendance monitoring & analytics</p>
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <div className="badge badge-active" style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '6px 14px' }}>
              <Timer size={14} /> {uptimeStr}
            </div>
            <div className="badge badge-present" style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '6px 14px' }}>
              {live ? <Wifi size={14} /> : <WifiOff size={14} />}
              {fps.toFixed(1)} FPS
            </div>
          </div>
        </div>
      </div>

      {/* ── KPIs ─────────────────────────────────── */}
      <div className="stats-row">
        <div className="card stat-card purple">
          <div className="stat-top">
            <div className="stat-icon purple"><Eye size={20} /></div>
            <span className="stat-label">Visits</span>
          </div>
          <div className="stat-value">{stats.totalVisitsToday ?? 0}</div>
        </div>
        <div className="card stat-card green">
          <div className="stat-top">
            <div className="stat-icon green"><UserCheck size={20} /></div>
            <span className="stat-label">Present</span>
          </div>
          <div className="stat-value">{stats.presentToday ?? 0} <span className="stat-sub">/ {stats.totalStudents ?? 0}</span></div>
        </div>
        <div className="card stat-card amber">
          <div className="stat-top">
            <div className="stat-icon amber"><Clock size={20} /></div>
            <span className="stat-label">Late</span>
          </div>
          <div className="stat-value">{stats.lateToday ?? 0}</div>
        </div>
        <div className="card stat-card red">
          <div className="stat-top">
            <div className="stat-icon red"><UserX size={20} /></div>
            <span className="stat-label">Absent</span>
          </div>
          <div className="stat-value">{stats.absentToday ?? 0}</div>
        </div>
      </div>

      {/* ── Attendance Rate Bar ──────────────────── */}
      <div className="card card-body" style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Percent size={16} style={{ color: '#4f46e5' }} />
            <span style={{ fontWeight: 600, fontSize: 14 }}>Today's Attendance Rate</span>
          </div>
          <span style={{ fontWeight: 800, fontSize: 20, color: attendRate >= 75 ? '#059669' : '#dc2626' }}>
            {attendRate}%
          </span>
        </div>
        <div className="progress-bar-bg">
          <div
            className="progress-bar-fill"
            style={{
              width: `${attendRate}%`,
              background: attendRate >= 75
                ? 'linear-gradient(90deg, #059669, #34d399)'
                : 'linear-gradient(90deg, #dc2626, #f87171)',
            }}
          />
        </div>
      </div>

      {/* ── Live Feed + Entry/Exit ──────────────── */}
      <div className="grid-2-1">
        <div className="card card-body">
          <div className="card-title"><Camera size={15} /> Live Camera Feed</div>
          <div className="video-wrapper">
            <div className="video-badge">
              {live ? <><div className="live-dot" /> LIVE</> : <><div className="offline-dot" /> OFFLINE</>}
            </div>
            <img src={`${API}/video_feed`} alt="Camera" className="video-feed"
              onError={e => { e.target.style.opacity = '0.1'; }} />
          </div>
        </div>

        <div className="card card-body">
          <div className="card-title"><TrendingUp size={15} /> Entry / Exit Log</div>
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th>Person</th><th>First</th><th>Last</th><th>Visits</th></tr></thead>
              <tbody>
                {entryExitLogs.length === 0 && (
                  <tr><td colSpan="4" style={{ textAlign: 'center', padding: 28, color: '#9ca3af' }}>No visits yet</td></tr>
                )}
                {entryExitLogs.slice(0, 8).map((l, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 600, color: '#111827' }}>{l.person_name}</td>
                    <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{l.first_seen}</td>
                    <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{l.last_seen}</td>
                    <td><span className="badge badge-active">{l.total_visits}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ── Charts ────────────────────────────────── */}
      <div className="grid-3">
        <div className="card card-body">
          <div className="card-title"><Activity size={15} /> Detections / Hour</div>
          <div className="chart-area">
            <Line data={timelineData} options={chartOpts} />
          </div>
        </div>

        <div className="card card-body">
          <div className="card-title"><BarChart3 size={15} /> Most Frequent</div>
          <div className="chart-area">
            <Bar data={freqData} options={{ ...chartOpts, indexAxis: 'y', scales: { ...chartOpts.scales, x: { ...chartOpts.scales.x, beginAtZero: true } } }} />
          </div>
        </div>

        <div className="card card-body">
          <div className="card-title"><Eye size={15} /> Known vs Unknown</div>
          <div className="chart-area-sm" style={{ display: 'flex', justifyContent: 'center' }}>
            <Doughnut data={ratioData} options={{ responsive: true, maintainAspectRatio: false, cutout: '72%', plugins: { legend: { display: false }, tooltip: chartOpts.plugins.tooltip } }} />
          </div>
          <div className="chart-legend">
            <span><span className="chart-legend-dot" style={{ background: '#4f46e5' }} /> Known ({kv.known})</span>
            <span><span className="chart-legend-dot" style={{ background: '#ef4444' }} /> Unknown ({kv.unknown})</span>
          </div>
        </div>
      </div>

      {/* ── Insights ──────────────────────────────── */}
      {insights.length > 0 && (
        <div className="card card-body">
          <div className="card-title" style={{ color: '#dc2626' }}>
            <AlertCircle size={15} /> Low Attendance Alerts (&lt;75%)
          </div>
          {insights.map((ins, i) => (
            <div className="insight-item" key={i}>
              <div className="insight-icon-wrap"><AlertCircle size={18} /></div>
              <div className="insight-text">
                <h4>{ins.name}</h4>
                <p>{ins.attendance_percentage}% attendance — {ins.insight}</p>
              </div>
              <div style={{ marginLeft: 'auto' }}>
                <div className="progress-bar-bg" style={{ width: 80 }}>
                  <div className="progress-bar-fill" style={{ width: `${ins.attendance_percentage}%`, background: 'linear-gradient(90deg, #dc2626, #f87171)' }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
