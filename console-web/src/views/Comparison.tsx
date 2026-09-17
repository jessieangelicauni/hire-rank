import { COMPARISON, type Role } from '../data';
import { avatarColorOf, initialsOf } from '../lib/avatar';
import { colorBorder, colorSurface, colorText, colorTextMuted, radius, radiusPill, shadowMicro } from '../tokens';

interface Props {
  roles: Role[];
}

function metricFor(role: Role) {
  const m = COMPARISON[role.id] ?? { jdId: role.id, kendallTau: null, deltaU: null, faithfulness: null };
  return {
    tau: m.kendallTau,
    deltaU: m.deltaU,
    faithfulness: m.faithfulness,
  };
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      flex: 1, minWidth: 0, background: colorSurface, border: `1px solid ${colorBorder}`,
      borderRadius: radius, boxShadow: shadowMicro, padding: '18px 20px',
    }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: colorText, letterSpacing: '-0.01em' }}>{value}</div>
      <div style={{ fontSize: 13, color: colorTextMuted, marginTop: 4 }}>{label}</div>
    </div>
  );
}

export default function Comparison({ roles }: Props) {
  const metrics = roles.map(metricFor);
  const taus = metrics.map((m) => m.tau).filter((v): v is number => v !== null);
  const deltas = metrics.map((m) => m.deltaU).filter((v): v is number => v !== null);
  const avgTau = taus.length ? taus.reduce((a, b) => a + b, 0) / taus.length : null;
  const avgDelta = deltas.length ? deltas.reduce((a, b) => a + b, 0) / deltas.length : null;

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, letterSpacing: '-0.01em', color: colorText, margin: '0 0 4px' }}>Analytics</h1>
      <p style={{ fontSize: 14, color: colorTextMuted, margin: '0 0 20px', maxWidth: 640, lineHeight: 1.5 }}>
        How independent fact-checking and tournament re-ranking change each role's results.
      </p>

      <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
        <StatCard label="Avg. Kendall's τ across roles" value={avgTau !== null ? avgTau.toFixed(3) : '—'} />
        <StatCard label="Avg. Δu across roles" value={avgDelta !== null ? `${avgDelta >= 0 ? '+' : ''}${avgDelta.toFixed(3)}` : '—'} />
        <StatCard label="Roles evaluated" value={String(roles.length)} />
      </div>

      <div style={{ fontSize: 16, fontWeight: 700, color: colorText, margin: '0 0 12px' }}>By role</div>
      <div style={{ background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro, overflow: 'hidden' }}>
        {roles.map((role, i) => {
          const m = metricFor(role);
          const deltaU = m.deltaU !== null ? `${m.deltaU >= 0 ? '+' : ''}${m.deltaU.toFixed(4)}` : '—';
          const faithfulness = m.faithfulness !== null ? `${(m.faithfulness * 100).toFixed(1)}%` : '—';
          return (
            <div
              key={role.id}
              className="row-hover"
              style={{
                display: 'flex', alignItems: 'center', gap: 14, padding: '14px 20px',
                borderTop: i === 0 ? undefined : `1px solid ${colorBorder}`,
              }}
            >
              <div style={{
                flex: '0 0 auto', width: 40, height: 40, borderRadius: radiusPill,
                background: avatarColorOf(role.title), color: '#fff', fontWeight: 700, fontSize: 13,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                {initialsOf(role.title)}
              </div>
              <div style={{
                flex: 1, minWidth: 0, fontSize: 14, fontWeight: 600, color: colorText,
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>
                {role.title}
              </div>
              <div style={{ flex: '0 0 auto', display: 'flex', gap: 20, fontSize: 13, color: colorTextMuted, fontVariantNumeric: 'tabular-nums' }}>
                <span><span style={{ color: colorTextMuted }}>τ </span><strong style={{ color: colorText, fontWeight: 600 }}>{m.tau !== null ? m.tau.toFixed(4) : '—'}</strong></span>
                <span><span style={{ color: colorTextMuted }}>Δu </span><strong style={{ color: colorText, fontWeight: 600 }}>{deltaU}</strong></span>
                <span style={{ minWidth: 52, textAlign: 'right' }}>{faithfulness}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
