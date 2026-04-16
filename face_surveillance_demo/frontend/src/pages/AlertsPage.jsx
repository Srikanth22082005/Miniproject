import { useEffect, useState } from 'react';
import axios from 'axios';
import { ShieldAlert, ShieldCheck, ImageOff } from 'lucide-react';

const API = 'http://localhost:5000';

export default function AlertsPage() {
  const [faces, setFaces] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = () => {
      axios.get(`${API}/api/unknown_faces`)
        .then(res => setFaces(res.data))
        .catch(console.error)
        .finally(() => setLoading(false));
    };
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <ShieldAlert size={24} color="#dc2626" /> Alerts & Unknown Faces
            </h1>
            <p className="page-subtitle">Auto-captured snapshots of unrecognized individuals · {faces.length} captured</p>
          </div>
        </div>
      </div>

      <div className="card card-body">
        {loading ? (
          <div className="empty-state"><ShieldAlert size={36} /><p>Loading…</p></div>
        ) : faces.length === 0 ? (
          <div className="empty-state">
            <ShieldCheck size={48} />
            <p>No unknown faces detected</p>
            <p className="sub">All clear — security perimeter is secure.</p>
          </div>
        ) : (
          <div className="faces-gallery">
            {faces.map(f => (
              <div className="face-item" key={f.id}>
                <img
                  src={`${API}${f.image_path}`}
                  alt="Unknown"
                  onError={e => { e.target.style.display = 'none'; }}
                />
                <div className="face-item-overlay">
                  <span>{f.date} · {f.time}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
