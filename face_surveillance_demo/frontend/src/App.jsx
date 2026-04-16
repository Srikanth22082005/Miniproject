import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Users, FileText, ShieldAlert,
  ScanFace, Settings, Bell,
} from 'lucide-react';
import DashboardPage from './pages/DashboardPage';
import StudentsPage from './pages/StudentsPage';
import ReportsPage from './pages/ReportsPage';
import AlertsPage from './pages/AlertsPage';

export default function App() {
  return (
    <Router>
      <div className="app-layout">
        <aside className="sidebar">
          <div className="sidebar-brand">
            <div className="sidebar-brand-icon">
              <ScanFace size={20} color="#fff" />
            </div>
            <span className="sidebar-brand-text">SmartAttend</span>
          </div>

          <div className="nav-section">Main</div>
          <nav className="sidebar-nav">
            <NavLink end to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <LayoutDashboard size={18} />
              <span>Dashboard</span>
            </NavLink>
            <NavLink to="/students" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Users size={18} />
              <span>Students</span>
            </NavLink>
          </nav>

          <div className="nav-section">Data</div>
          <nav className="sidebar-nav">
            <NavLink to="/reports" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <FileText size={18} />
              <span>Reports</span>
            </NavLink>
            <NavLink to="/alerts" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <ShieldAlert size={18} />
              <span>Alerts</span>
            </NavLink>
          </nav>

          <div className="sidebar-footer">
            <div className="sidebar-footer-badge">
              <div className="dot" />
              <span>System Online</span>
            </div>
          </div>
        </aside>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/students" element={<StudentsPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/alerts" element={<AlertsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
