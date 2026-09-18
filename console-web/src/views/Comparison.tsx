import { COMPARISON, type Role } from '../data';
import { avatarColorOf, initialsOf } from '../lib/avatar';
import { colorAccent, colorBorder, colorSurface, colorText, colorTextMuted, radius, radiusPill, shadowMicro } from '../tokens';

interface Props {
  roles: Role[];
}

function metricFor(role: Role) {
  const m = COMPARISON[role.id] ?? { jdId: role.id, meanFitScore: null, meetsMinRate: null, hireRate: null };
  return {
    meanFitScore: m.meanFitScore,
    meetsMinRate: m.meetsMinRate,
    hireRate: m.hireRate,
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

function RateBar({ value, color }: { value: number | null; color: string }) {
  return (
    <div style={{ flexShrink: 0, width: 72, height: 6, borderRadius: radiusPill, background: colorBorder, overflow: 'hidden' }}>
      <div style={{ width: `${value !== null ? value * 100 : 0}%`, height: '100%', borderRadius: radiusPill, background: color }} />
    </div>
  );
}

export default function Comparison({ roles }: Props) {
  const metrics = roles.map(metricFor);
  const fitScores = metrics.map((m) => m.meanFitScore).filter((v): v is number => v !== null);
  const meetsMinRates = metrics.map((m) => m.meetsMinRate).filter((v): v is number => v !== null);
  const hireRates = metrics.map((m) => m.hireRate).filter((v): v is number => v !== null);
  const avgFitScore = fitScores.length ? fitScores.reduce((a, b) => a + b, 0) / fitScores.length : null;
  const avgMeetsMinRate = meetsMinRates.length ? meetsMinRates.reduce((a, b) => a + b, 0) / meetsMinRates.length : null;
  const avgHireRate = hireRates.length ? hireRates.reduce((a, b) => a + b, 0) / hireRates.length : null;

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, letterSpacing: '-0.01em', color: colorText, margin: '0 0 4px' }}>Analytics</h1>
      <p style={{ fontSize: 14, color: colorTextMuted, margin: '0 0 20px', maxWidth: 640, lineHeight: 1.5 }}>
        How Jev's structured assessment scores every role's shortlisted candidates.
      </p>

      <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
        <StatCard label="Avg. fit score across roles" value={avgFitScore !== null ? avgFitScore.toFixed(1) : '—'} />
        <StatCard label="Avg. meets-minimum rate" value={avgMeetsMinRate !== null ? `${(avgMeetsMinRate * 100).toFixed(0)}%` : '—'} />
        <StatCard label="Avg. hire-recommendation rate" value={avgHireRate !== null ? `${(avgHireRate * 100).toFixed(0)}%` : '—'} />
      </div>

      <div style={{ fontSize: 16, fontWeight: 700, color: colorText, margin: '0 0 12px' }}>By role</div>
      <div style={{ background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro, overflow: 'hidden' }}>
        {roles.map((role, i) => {
          const m = metricFor(role);
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
              <div style={{ flex: '0 0 auto', display: 'flex', alignItems: 'center', gap: 20, fontSize: 13, color: colorTextMuted, fontVariantNumeric: 'tabular-nums' }}>
                <span style={{ minWidth: 56, textAlign: 'right' }}>
                  <strong style={{ color: colorText, fontWeight: 600 }}>{m.meanFitScore !== null ? m.meanFitScore.toFixed(1) : '—'}</strong> fit
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <RateBar value={m.meetsMinRate} color={colorAccent} />
                  <span style={{ minWidth: 36, textAlign: 'right' }}>{m.meetsMinRate !== null ? `${(m.meetsMinRate * 100).toFixed(0)}%` : '—'}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <RateBar value={m.hireRate} color={colorAccent} />
                  <span style={{ minWidth: 36, textAlign: 'right' }}>{m.hireRate !== null ? `${(m.hireRate * 100).toFixed(0)}%` : '—'}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
