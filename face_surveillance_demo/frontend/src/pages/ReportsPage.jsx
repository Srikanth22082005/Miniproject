import { useEffect, useState, useMemo } from 'react';
import axios from 'axios';
import {
  Calendar, Download, FileSpreadsheet, Filter,
  ChevronLeft, ChevronRight, TrendingUp, Percent, Users,
} from 'lucide-react';
import { motion } from 'framer-motion';
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, Title, Tooltip, Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const API = 'http://localhost:5000';

/* ── Helper: group by a key ──────────────────────── */
function groupBy(arr, fn) {
  const map = {};
  arr.forEach(item => {
    const key = fn(item);
    if (!map[key]) map[key] = [];
    map[key].push(item);
  });
  return map;
}

/* ── Helper: get week number ─────────────────────── */
function getWeek(dateStr) {
  const d = new Date(dateStr);
  const onejan = new Date(d.getFullYear(), 0, 1);
  return Math.ceil(((d - onejan) / 86400000 + onejan.getDay() + 1) / 7);
}

/* ── Helper: get month label ─────────────────────── */
function getMonthLabel(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleString('default', { month: 'long', year: 'numeric' });
}

const TABS = ['Daily', 'Weekly', 'Monthly', 'Semester'];

export default function ReportsPage() {
  const [logs, setLogs] = useState([]);
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState('Daily');
  const [statusFilter, setStatusFilter] = useState('All');

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/attendance`),
      axios.get(`${API}/api/students`),
    ])
      .then(([a, s]) => { setLogs(a.data); setStudents(s.data); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  /* ── Filter by status ──────────────────────────── */
  const filtered = statusFilter === 'All'
    ? logs
    : logs.filter(l => l.status === statusFilter);

  const counts = {
    All: logs.length,
    Present: logs.filter(l => l.status === 'Present').length,
    Late: logs.filter(l => l.status === 'Late').length,
  };

  /* ── Compute per-student attendance % ──────────── */
  const totalDays = useMemo(() => {
    const dates = new Set(logs.map(l => l.date));
    return dates.size || 1;
  }, [logs]);

  const studentStats = useMemo(() => {
    const byStudent = groupBy(logs, l => l.student_id);
    return students.map(s => {
      const records = byStudent[s.student_id] || [];
      const present = records.filter(r => r.status === 'Present').length;
      const late = records.filter(r => r.status === 'Late').length;
      const total = present + late;
      const pct = Math.round((total / totalDays) * 100);
      return { ...s, present, late, absent: totalDays - total, pct };
    }).sort((a, b) => b.pct - a.pct);
  }, [logs, students, totalDays]);

  /* ── Group records for each view ───────────────── */
  const grouped = useMemo(() => {
    if (tab === 'Daily') {
      return groupBy(filtered, l => l.date);
    } else if (tab === 'Weekly') {
      return groupBy(filtered, l => `Week ${getWeek(l.date)} - ${new Date(l.date).getFullYear()}`);
    } else if (tab === 'Monthly') {
      return groupBy(filtered, l => getMonthLabel(l.date));
    } else {
      return { 'Full Semester': filtered };
    }
  }, [filtered, tab]);

  const groupKeys = Object.keys(grouped).sort().reverse();

  /* ── Chart: per-student attendance % ──────────── */
  const chartData = {
    labels: studentStats.map(s => s.name.split(' ')[0]),
    datasets: [
      {
        label: 'Present',
        data: studentStats.map(s => s.present),
        backgroundColor: '#059669',
        borderRadius: 4, barThickness: 18,
      },
      {
        label: 'Late',
        data: studentStats.map(s => s.late),
        backgroundColor: '#d97706',
        borderRadius: 4, barThickness: 18,
      },
      {
        label: 'Absent',
        data: studentStats.map(s => s.absent),
        backgroundColor: '#e5e7eb',
        borderRadius: 4, barThickness: 18,
      },
    ],
  };

  const chartOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top', labels: { usePointStyle: true, pointStyle: 'circle', padding: 16, font: { size: 12 } } },
      tooltip: { backgroundColor: '#fff', titleColor: '#111827', bodyColor: '#6b7280', borderColor: '#e5e7eb', borderWidth: 1, padding: 10, cornerRadius: 8 },
    },
    scales: {
      x: { stacked: true, grid: { display: false }, ticks: { color: '#9ca3af', font: { size: 11 } } },
      y: { stacked: true, grid: { color: '#f3f4f6' }, ticks: { color: '#9ca3af', font: { size: 11 } } },
    },
  };

  /* ── CSV Download ──────────────────────────────── */
  const downloadCSV = () => {
    const header = 'Date,Time,Student ID,Name,Department,Status\n';
    const rows = filtered.map(l =>
      `${l.date},${l.time},${l.student_id},${l.name},${l.department || ''},${l.status}`
    ).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attendance_${tab.toLowerCase()}_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return <div className="center-screen"><FileSpreadsheet size={36} className="spinner-icon" /><p style={{ color: '#9ca3af' }}>Loading reports...</p></div>;
  }

  return (
    <>
      {/* ── Header ──────────────────────────────── */}
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h1 className="page-title">Attendance Reports</h1>
            <p className="page-subtitle">{totalDays} working days · {logs.length} records · {students.length} students</p>
          </div>
          <motion.button 
            whileHover={{ scale: 1.05 }} 
            whileTap={{ scale: 0.95 }}
            className="btn btn-primary" 
            onClick={downloadCSV}
          >
            <Download size={16} /> Export CSV
          </motion.button>
        </div>
      </div>

      {/* ── Period Tabs ─────────────────────────── */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 18, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <motion.button key={t}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={tab === t ? 'btn btn-primary' : 'btn btn-outline'}
            style={{ padding: '8px 20px', fontSize: 13 }}
            onClick={() => setTab(t)}>
            {t}
          </motion.button>
        ))}
        <div style={{ flex: 1 }} />
        {['All', 'Present', 'Late'].map(s => (
          <motion.button key={s}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={statusFilter === s ? 'btn btn-primary' : 'btn btn-outline'}
            style={{ padding: '7px 14px', fontSize: 12 }}
            onClick={() => setStatusFilter(s)}>
            {s} ({counts[s]})
          </motion.button>
        ))}
      </div>

      {/* ── Student Attendance Summary ──────────── */}
      <div className="grid-1-1" style={{ marginBottom: 18 }}>
        <div className="card card-body">
          <div className="card-title"><Percent size={15} /> Per-Student Attendance %</div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr><th>Student</th><th>Present</th><th>Late</th><th>Absent</th><th>%</th><th>Status</th></tr>
              </thead>
              <tbody>
                {studentStats.map(s => (
                  <tr key={s.student_id}>
                    <td style={{ fontWeight: 600, color: '#111827' }}>{s.name}</td>
                    <td><span className="badge badge-present">{s.present}</span></td>
                    <td><span className="badge badge-late">{s.late}</span></td>
                    <td><span className="badge badge-absent">{s.absent}</span></td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div className="progress-bar-bg" style={{ width: 60, height: 5 }}>
                          <div className="progress-bar-fill" style={{
                            width: `${s.pct}%`,
                            background: s.pct >= 75 ? '#059669' : '#dc2626',
                          }} />
                        </div>
                        <span style={{ fontWeight: 700, fontSize: 13, color: s.pct >= 75 ? '#059669' : '#dc2626' }}>
                          {s.pct}%
                        </span>
                      </div>
                    </td>
                    <td>
                      {s.pct >= 85 ? <span className="badge badge-present">Good</span>
                        : s.pct >= 75 ? <span className="badge badge-late">Warning</span>
                        : <span className="badge badge-absent">Critical</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card card-body">
          <div className="card-title"><TrendingUp size={15} /> Attendance Breakdown</div>
          <div style={{ height: 280 }}>
            <Bar data={chartData} options={chartOpts} />
          </div>
        </div>
      </div>

      {/* ── Grouped Records ────────────────────── */}
      {groupKeys.length === 0 ? (
        <div className="card card-body">
          <div className="empty-state">
            <FileSpreadsheet size={36} />
            <p>No records found for the selected filter.</p>
          </div>
        </div>
      ) : (
        groupKeys.map(key => (
          <div className="card" key={key} style={{ marginBottom: 14 }}>
            <div style={{
              padding: '14px 22px',
              background: '#f8f9fb',
              borderBottom: '1px solid #e5e7eb',
              borderRadius: '16px 16px 0 0',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              <span style={{ fontWeight: 700, fontSize: 14, color: '#111827', display: 'flex', alignItems: 'center', gap: 8 }}>
                <Calendar size={15} color="#4f46e5" /> {key}
              </span>
              <span className="badge badge-active">{grouped[key].length} records</span>
            </div>
            <div className="card-body" style={{ padding: '0' }}>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr><th>Date</th><th>Time</th><th>Student ID</th><th>Name</th><th>Status</th></tr>
                  </thead>
                  <tbody>
                    {grouped[key].map((l, i) => (
                      <tr key={i}>
                        <td style={{ fontSize: 13 }}>{l.date}</td>
                        <td style={{ fontFamily: 'monospace', fontSize: 12, color: '#9ca3af' }}>{l.time}</td>
                        <td style={{ fontFamily: 'monospace', color: '#9ca3af', fontSize: 12 }}>{l.student_id}</td>
                        <td style={{ fontWeight: 600, color: '#111827' }}>{l.name}</td>
                        <td><span className={`badge badge-${l.status.toLowerCase()}`}>{l.status}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ))
      )}
    </>
  );
}
