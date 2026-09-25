import { Fragment, useState } from 'react';
import { COMPARISON, EVALUATION_SUMMARY, type Role } from '../data';
import { avatarColorOf, initialsOf } from '../lib/avatar';
import { roleBreakdownFor } from '../lib/roleBreakdown';
import {
  colorAccent, colorBorder, colorDanger, colorSurface, colorSurfaceMuted,
  colorText, colorTextMuted, radius, radiusPill, shadowMicro,
} from '../tokens';

interface Props {
  roles: Role[];
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' ? value : null;
}

function metricFor(role: Role) {
  const m = COMPARISON[role.id];
  return {
    meanFitScore: numberOrNull(m?.meanFitScore),
    rankingStability: numberOrNull(m?.rankingStability),
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
  const [expandedRoleId, setExpandedRoleId] = useState<string | null>(null);
  const metrics = roles.map(metricFor);
  const fitScores = metrics.map((m) => m.meanFitScore).filter((v): v is number => v !== null);
  const avgFitScore = fitScores.length ? fitScores.reduce((a, b) => a + b, 0) / fitScores.length : null;

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, letterSpacing: '-0.01em', color: colorText, margin: '0 0 4px' }}>Analytics</h1>
      <p style={{ fontSize: 14, color: colorTextMuted, margin: '0 0 20px', maxWidth: 640, lineHeight: 1.5 }}>
        How Jev's structured assessment scores every role's shortlisted candidates.
      </p>

      <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
        <StatCard label="Avg. fit score across roles" value={avgFitScore !== null ? `${avgFitScore.toFixed(1)}/100` : '—'} />
      </div>

      <div style={{ fontSize: 16, fontWeight: 700, color: colorText, margin: '0 0 12px' }}>By role</div>
      <div style={{ background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro, overflow: 'hidden' }}>
        {roles.map((role, i) => {
          const m = metricFor(role);
          const isExpanded = expandedRoleId === role.id;
          return (
            <Fragment key={role.id}>
            <div
              className="row-hover"
              onClick={() => setExpandedRoleId(isExpanded ? null : role.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 14, padding: '14px 20px', cursor: 'pointer',
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
                <span style={{ minWidth: 76, textAlign: 'right' }}>
                  <strong style={{ color: colorText, fontWeight: 600 }}>{m.meanFitScore !== null ? `${m.meanFitScore.toFixed(1)}/100` : '—'}</strong> fit
                </span>
                <span style={{ minWidth: 44, textAlign: 'right' }}>
                  <span style={{ color: colorTextMuted }}>τ </span>
                  <strong style={{ color: colorText, fontWeight: 600 }}>{m.rankingStability !== null ? m.rankingStability.toFixed(3) : '—'}</strong>
                </span>
              </div>
            </div>
            {isExpanded && <RoleBreakdownPanel role={role} />}
            </Fragment>
          );
        })}
      </div>

      {EVALUATION_SUMMARY && (
        <>
          <div style={{ fontSize: 16, fontWeight: 700, color: colorText, margin: '28px 0 4px' }}>Reliability &amp; validation</div>
          <p style={{ fontSize: 13, color: colorTextMuted, margin: '0 0 12px', maxWidth: 640, lineHeight: 1.5 }}>
            From a dedicated evaluation study: {EVALUATION_SUMMARY.nPairs ?? '—'} pairs, each called {EVALUATION_SUMMARY.nRepeats ?? '—'}x independently.
          </p>
          <div style={{ display: 'flex', gap: 12 }}>
            <StatCard
              label="Mean ranking stability (τ)"
              value={EVALUATION_SUMMARY.meanRankingConvergence !== null ? EVALUATION_SUMMARY.meanRankingConvergence.toFixed(3) : '—'}
            />
            <StatCard
              label="Composite-score stdev across repeats (0-100 scale)"
              value={EVALUATION_SUMMARY.compositeScoreStdev !== null ? `${EVALUATION_SUMMARY.compositeScoreStdev.toFixed(2)} pts` : '—'}
            />
            <StatCard
              label="Requirement/composite coherence (ρ)"
              value={EVALUATION_SUMMARY.coherenceSpearmanRho !== null ? EVALUATION_SUMMARY.coherenceSpearmanRho.toFixed(3) : '—'}
            />
          </div>
        </>
      )}
    </div>
  );
}

function RoleBreakdownPanel({ role }: { role: Role }) {
  const { seniorityMean, educationMean, requirementMeans } = roleBreakdownFor(role);

  return (
    <div style={{ padding: '16px 20px 20px', background: colorSurfaceMuted, borderTop: `1px solid ${colorBorder}` }}>
      {(seniorityMean !== null || educationMean !== null) && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
          {seniorityMean !== null && (
            <StatCard label="Avg. years-of-experience match" value={`${seniorityMean.toFixed(1)}/100`} />
          )}
          {educationMean !== null && (
            <StatCard label="Avg. education match" value={`${educationMean.toFixed(1)}/100`} />
          )}
        </div>
      )}

      {requirementMeans.length > 0 && (
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: colorText, marginBottom: 6 }}>Avg. score per requirement</div>
          {requirementMeans.map(([requirement, score]) => (
            <div key={requirement} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '6px 0', fontSize: 13, color: colorText }}>
              <span style={{ flex: 1, minWidth: 0 }}>{requirement}</span>
              <div style={{ flexShrink: 0, width: 100, height: 6, borderRadius: radiusPill, background: colorBorder, overflow: 'hidden' }}>
                <div style={{
                  width: `${Math.max(0, Math.min(100, score))}%`, height: '100%', borderRadius: radiusPill,
                  background: score >= 50 ? colorAccent : colorDanger,
                }} />
              </div>
              <span style={{ flexShrink: 0, width: 52, textAlign: 'right', color: colorTextMuted, fontSize: 12 }}>{score.toFixed(0)}/100</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
