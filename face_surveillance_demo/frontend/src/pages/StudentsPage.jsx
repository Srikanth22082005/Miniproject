import { useEffect, useState } from 'react';
import axios from 'axios';
import { Search, Info, GraduationCap, UserCircle } from 'lucide-react';

const API = 'http://localhost:5000';

const AVATAR_COLORS = [
  '#4f46e5', '#059669', '#d97706', '#dc2626', '#7c3aed',
  '#2563eb', '#db2777', '#0891b2', '#65a30d', '#ea580c',
];

function getInitials(name) {
  return name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
}

export default function StudentsPage() {
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');

  useEffect(() => {
    axios.get(`${API}/api/students`)
      .then(res => setStudents(res.data))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const filtered = students.filter(s =>
    s.name.toLowerCase().includes(query.toLowerCase()) ||
    s.department?.toLowerCase().includes(query.toLowerCase()) ||
    s.student_id?.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h1 className="page-title">Students</h1>
            <p className="page-subtitle">Manage enrolled students · {students.length} registered</p>
          </div>
          <div className="search-box">
            <Search size={15} />
            <input className="search-input" placeholder="Search by name, ID, dept…"
              value={query} onChange={e => setQuery(e.target.value)} />
          </div>
        </div>
      </div>

      <div className="card card-body">
        {loading ? (
          <div className="empty-state"><GraduationCap size={36} /><p>Loading…</p></div>
        ) : filtered.length === 0 ? (
          <div className="empty-state"><GraduationCap size={36} /><p>No students found</p></div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr><th></th><th>Name</th><th>ID</th><th>Department</th><th>Year</th><th>Status</th></tr>
              </thead>
              <tbody>
                {filtered.map((s, i) => (
                  <tr key={s.student_id}>
                    <td>
                      <div className="avatar" style={{ background: `${AVATAR_COLORS[i % AVATAR_COLORS.length]}15`, color: AVATAR_COLORS[i % AVATAR_COLORS.length] }}>
                        {getInitials(s.name)}
                      </div>
                    </td>
                    <td style={{ fontWeight: 600, color: '#111827' }}>{s.name}</td>
                    <td style={{ fontFamily: 'monospace', color: '#9ca3af', fontSize: 12 }}>{s.student_id}</td>
                    <td>{s.department}</td>
                    <td>{s.year}</td>
                    <td><span className="badge badge-active">Active</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="info-box">
        <Info size={18} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>Place face images in <code>dataset/&lt;student_name&gt;/</code> then run <code>python register.py</code> to build embeddings.</div>
      </div>
    </>
  );
}
