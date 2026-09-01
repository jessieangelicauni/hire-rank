import { useState } from 'react';
import lgesLogo from './assets/President_University.jpg';
import { ROLES, APPLICANTS } from './data';
import { colorAccent, colorAccentSoft, colorBg, colorBorder } from './tokens';
import Dashboard from './views/Dashboard';
import Leaderboard from './views/Leaderboard';
import Comparison from './views/Comparison';
import Placeholder from './views/Placeholder';

export type View = 'dashboard' | 'leaderboard' | 'comparison' | 'placeholder';
export type NavTab = 'roles' | 'analytics' | 'settings' | 'notifications';

const NAV_TABS: { id: NavTab; label: string }[] = [
  { id: 'roles', label: 'Roles' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'settings', label: 'Settings' },
  { id: 'notifications', label: 'Notifications' },
];

function navTabStyle(active: boolean): React.CSSProperties {
  return {
    fontSize: 14,
    fontWeight: 500,
    padding: '5px 5px',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
    flexShrink: 0,
    background: active ? colorAccentSoft : undefined,
    color: active ? colorAccent : '#000000',
  };
}

export default function App() {
  const [view, setView] = useState<View>('dashboard');
  const [navTab, setNavTab] = useState<NavTab>('roles');
  const [activeRoleId, setActiveRoleId] = useState<string | null>(null);
  const [activeApplicantId, setActiveApplicantId] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  const goDashboard = () => {
    setView('dashboard');
    setNavTab('roles');
  };

  const handleNavTab = (id: NavTab) => {
    if (id === 'roles') goDashboard();
    else if (id === 'analytics') {
      setView('comparison');
      setNavTab('analytics');
    } else {
      setView('placeholder');
      setNavTab(id);
    }
  };

  const selectRole = (roleId: string) => {
    setView('leaderboard');
    setNavTab('roles');
    setActiveRoleId(roleId);
    setActiveApplicantId(null);
  };

  const backToDashboard = () => {
    setView('dashboard');
    setActiveApplicantId(null);
  };

  const selectApplicant = (applicantId: string) => {
    setActiveApplicantId(applicantId);
  };

  const closeDetail = () => setActiveApplicantId(null);

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100%', minWidth: 1180, background: colorBg, color: '#000000', overflow: 'hidden' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* TOP BAR */}
        <div style={{ height: 72, minHeight: 72, display: 'flex', alignItems: 'center', padding: '0 16px', borderBottom: `1px solid ${colorBorder}`, background: '#fff', gap: 8 }}>
          <div style={{ flex: '0 0 auto', display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }} onClick={goDashboard}>
            <img src={lgesLogo} alt="" style={{ height: 40, width: 'auto', flexShrink: 0 }} />
            <div style={{ lineHeight: 1.2 }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: '#000000' }}>Hire-Rank</div>
              <div style={{ fontSize: 14, color: '#000000' }}>Applicant Ranking</div>
            </div>
          </div>
          <div style={{ flex: '1 1 160px', minWidth: 160, maxWidth: 220, position: 'relative' }}>
            <span style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: '#000000', fontSize: 14 }}>⌕</span>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search applicants..."
              style={{
                width: '100%', height: 38, border: '1px solid #E7E1DB',
                padding: '0 16px 0 36px', fontSize: 14, fontFamily: 'inherit', outline: 'none',
                background: '#FAF8F6', boxSizing: 'border-box',
              }}
            />
          </div>
          <div style={{ flex: '0 0 auto', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 2, marginLeft: 'auto' }}>
            {NAV_TABS.map((tab) => (
              <div key={tab.id} onClick={() => handleNavTab(tab.id)} style={navTabStyle(navTab === tab.id)}>
                {tab.label}
              </div>
            ))}
          </div>
          <div style={{ flex: '0 0 auto', display: 'flex', alignItems: 'center', gap: 10, paddingLeft: 8, borderLeft: `1px solid ${colorBorder}`, whiteSpace: 'nowrap' }}>
            <div style={{ width: 32, height: 32, background: colorAccent, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 500, fontSize: 14, flexShrink: 0 }}>
              HR
            </div>
            <div style={{ lineHeight: 1.2 }}>
              <div style={{ fontSize: 14, fontWeight: 500 }}>HR LG Sinarmas</div>
              <div style={{ fontSize: 14, color: '#000000' }}>Talent Partner</div>
            </div>
          </div>
        </div>

        {/* CONTENT */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '32px 40px 60px' }}>
          {view === 'dashboard' && (
            <Dashboard roles={ROLES} applicants={APPLICANTS} search={search} onSelectRole={selectRole} />
          )}
          {view === 'leaderboard' && (
            <Leaderboard
              roles={ROLES}
              applicants={APPLICANTS}
              activeRoleId={activeRoleId ?? ROLES[0].id}
              activeApplicantId={activeApplicantId}
              onBackToDashboard={backToDashboard}
              onSelectApplicant={selectApplicant}
              onCloseDetail={closeDetail}
            />
          )}
          {view === 'comparison' && <Comparison roles={ROLES} />}
          {view === 'placeholder' && <Placeholder label={navTab === 'settings' ? 'Settings' : 'Notifications'} />}
        </div>
      </div>
    </div>
  );
}
