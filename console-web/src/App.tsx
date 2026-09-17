import { useState } from 'react';
import lgesLogo from './assets/lges-logo.svg';
import { ROLES, APPLICANTS } from './data';
import { colorAccent, colorBg, colorBorder, colorSurface, colorSurfaceMuted, colorText, colorTextMuted, colorTextSoft, radiusPill } from './tokens';
import Dashboard from './views/Dashboard';
import Leaderboard from './views/Leaderboard';
import Comparison from './views/Comparison';
import Placeholder from './views/Placeholder';

export type View = 'dashboard' | 'leaderboard' | 'comparison' | 'placeholder';
export type NavTab = 'roles' | 'analytics' | 'settings' | 'notifications';

const NAV_TABS: { id: NavTab; label: string; icon: string }[] = [
  { id: 'roles', label: 'Roles', icon: '⌂' },
  { id: 'analytics', label: 'Analytics', icon: '▤' },
  { id: 'settings', label: 'Settings', icon: '⚙' },
  { id: 'notifications', label: 'Notifications', icon: '🔔' },
];

function navTabStyle(active: boolean): React.CSSProperties {
  return {
    fontSize: 12,
    fontWeight: active ? 700 : 600,
    padding: '0 16px',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
    borderBottom: active ? `2px solid ${colorText}` : '2px solid transparent',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
    flexShrink: 0,
    color: active ? colorText : colorTextMuted,
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
    <div style={{ display: 'flex', height: '100vh', width: '100%', minWidth: 1180, background: colorBg, color: colorText, overflow: 'hidden' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* TOP BAR */}
        <div style={{ height: 60, minHeight: 60, display: 'flex', alignItems: 'stretch', padding: '0 16px', borderBottom: `1px solid ${colorBorder}`, background: colorSurface, gap: 8 }}>
          <div style={{ flex: '0 0 auto', display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', alignSelf: 'center' }} onClick={goDashboard}>
            <img src={lgesLogo} alt="" style={{ height: 32, width: 'auto', flexShrink: 0, borderRadius: radiusPill }} />
            <div style={{ lineHeight: 1.2 }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: colorText }}>Hire-Rank</div>
              <div style={{ fontSize: 12, color: colorTextMuted }}>Applicant Ranking</div>
            </div>
          </div>
          <div style={{ flex: '1 1 200px', minWidth: 180, maxWidth: 280, position: 'relative', alignSelf: 'center' }}>
            <span style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: colorTextSoft, fontSize: 14 }}>⌕</span>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search applicants..."
              style={{
                width: '100%', height: 34, border: 'none', borderRadius: radiusPill,
                padding: '0 16px 0 36px', fontSize: 14, fontFamily: 'inherit',
                background: colorSurfaceMuted, color: colorText, boxSizing: 'border-box',
              }}
            />
          </div>
          <div style={{ flex: '0 0 auto', display: 'flex', alignItems: 'stretch', justifyContent: 'flex-end', gap: 2, marginLeft: 'auto' }}>
            {NAV_TABS.map((tab) => (
              <div
                key={tab.id}
                className={navTab === tab.id ? undefined : 'row-hover'}
                onClick={() => handleNavTab(tab.id)}
                style={navTabStyle(navTab === tab.id)}
              >
                <span style={{ fontSize: 16, lineHeight: 1 }}>{tab.icon}</span>
                <span>{tab.label}</span>
              </div>
            ))}
          </div>
          <div style={{ flex: '0 0 auto', display: 'flex', alignItems: 'center', gap: 10, paddingLeft: 8, borderLeft: `1px solid ${colorBorder}`, whiteSpace: 'nowrap', alignSelf: 'center' }}>
            <div style={{ width: 32, height: 32, borderRadius: radiusPill, background: colorAccent, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 13, flexShrink: 0 }}>
              HR
            </div>
            <div style={{ lineHeight: 1.2 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: colorText }}>HR LG Sinarmas</div>
              <div style={{ fontSize: 12, color: colorTextMuted }}>Talent Partner</div>
            </div>
          </div>
        </div>

        {/* CONTENT */}
        <div key={view} style={{ flex: 1, overflowY: 'auto', padding: '24px 24px 60px', animation: 'fadeIn 0.15s ease' }}>
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
