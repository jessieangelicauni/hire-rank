import { Fragment, useState } from 'react';
import { COMPARISON, reliabilityFor, type Role } from '../data';
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
    meanFitScore: numberOrNull(m?.meanCriterionScore) !== null ? numberOrNull(m?.meanCriterionScore)! * 100 : null,
    totalAssessed: numberOrNull(m?.totalAssessed),
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
                <span style={{ minWidth: 60, textAlign: 'right' }}>
                  <strong style={{ color: colorText, fontWeight: 600 }}>{m.totalAssessed ?? '—'}</strong> assessed
                </span>
              </div>
            </div>
            {isExpanded && <RoleBreakdownPanel role={role} />}
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}

const CONFIDENCE_BUCKET_ORDER = ['<0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'];

function MiniStat({ label, value, sub, warn }: { label: string; value: string; sub: string; warn?: boolean }) {
  return (
    <div style={{ flex: 1, minWidth: 0, background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, padding: '10px 14px' }}>
      <div style={{ fontSize: 18, fontWeight: 700, color: warn ? colorDanger : colorText, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
      <div style={{ fontSize: 11, color: colorTextMuted, marginTop: 2 }}>{label}</div>
      <div style={{ fontSize: 11, color: warn ? colorDanger : colorTextMuted, marginTop: 1 }}>{sub}</div>
    </div>
  );
}

function ReliabilityPanel({ roleId }: { roleId: string }) {
  const reliability = reliabilityFor(roleId);
  if (!reliability) return null;

  const { tieringValidity, confidenceCalibration, rankingConsistency, rankingRepeatability, coverage } = reliability;
  const hasStats = tieringValidity || rankingConsistency || rankingRepeatability;

  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 700, color: colorText, marginBottom: 10 }}>Reliability &amp; QA</div>
      {hasStats && (
        <div style={{ display: 'flex', gap: 10, marginBottom: confidenceCalibration ? 16 : 0 }}>
          {tieringValidity && (
            <MiniStat
              label="Tiering agreement"
              value={tieringValidity.agreementRate !== null ? `${(tieringValidity.agreementRate * 100).toFixed(0)}%` : '—'}
              sub={`${tieringValidity.agreed}/${tieringValidity.total} pairs`}
            />
          )}
          {rankingConsistency && (
            <MiniStat
              label="Ranking stability"
              value={`${((1 - rankingConsistency.inconsistentRate) * 100).toFixed(0)}%`}
              sub={`cosmetic ${((1 - rankingConsistency.cosmeticInconsistentRate) * 100).toFixed(0)}%`}
            />
          )}
          {rankingRepeatability && (
            <MiniStat
              label="Repeatability (τ)"
              value={rankingRepeatability.meanKendallTau !== null ? rankingRepeatability.meanKendallTau.toFixed(2) : '—'}
              warn={rankingRepeatability.nCandidatesInTieClusters === 0}
              sub={
                rankingRepeatability.nCandidatesInTieClusters != null && rankingRepeatability.nCandidates != null
                  ? rankingRepeatability.nCandidatesInTieClusters === 0
                    ? 'trivial — 0 live-tested'
                    : `${rankingRepeatability.nCandidatesInTieClusters}/${rankingRepeatability.nCandidates} live-tested`
                  : `${rankingRepeatability.nRepeatPairs} repeat pairs`
              }
            />
          )}
        </div>
      )}
      {confidenceCalibration && (
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: colorTextMuted, marginBottom: 6 }}>Confidence calibration</div>
          {CONFIDENCE_BUCKET_ORDER.filter((bucket) => confidenceCalibration.buckets[bucket]).map((bucket) => {
            const { mean, n } = confidenceCalibration.buckets[bucket];
            const pct = mean * 100;
            return (
              <div key={bucket} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '5px 0', fontSize: 13, color: colorText }}>
                <span style={{ flex: '0 0 64px' }}>{bucket}</span>
                <div style={{ flex: 1, height: 6, borderRadius: radiusPill, background: colorBorder, overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.max(0, Math.min(100, pct))}%`, height: '100%', borderRadius: radiusPill,
                    background: pct >= 50 ? colorAccent : colorDanger,
                  }} />
                </div>
                <span style={{ flexShrink: 0, width: 84, textAlign: 'right', color: colorTextMuted, fontSize: 12 }}>{pct.toFixed(0)}% agree · n={n}</span>
              </div>
            );
          })}
        </div>
      )}
      {coverage && <div style={{ fontSize: 12, color: colorTextMuted, marginTop: 10 }}>{coverage}</div>}
    </div>
  );
}

function RoleBreakdownPanel({ role }: { role: Role }) {
  const { requirementMeans } = roleBreakdownFor(role);
  const hasReliability = reliabilityFor(role.id) !== null;

  return (
    <div style={{ padding: '16px 20px 20px', background: colorSurfaceMuted, borderTop: `1px solid ${colorBorder}` }}>
      {requirementMeans.length > 0 && (
        <div style={{ marginBottom: hasReliability ? 20 : 0 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: colorText, marginBottom: 6 }}>Avg. expected strength per requirement</div>
          {requirementMeans.map(([requirement, score]) => {
            const pct = score * 100;
            return (
              <div key={requirement} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '6px 0', fontSize: 13, color: colorText }}>
                <span style={{ flex: 1, minWidth: 0 }}>{requirement}</span>
                <div style={{ flexShrink: 0, width: 100, height: 6, borderRadius: radiusPill, background: colorBorder, overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.max(0, Math.min(100, pct))}%`, height: '100%', borderRadius: radiusPill,
                    background: pct >= 50 ? colorAccent : colorDanger,
                  }} />
                </div>
                <span style={{ flexShrink: 0, width: 52, textAlign: 'right', color: colorTextMuted, fontSize: 12 }}>{pct.toFixed(0)}/100</span>
              </div>
            );
          })}
        </div>
      )}
      <ReliabilityPanel roleId={role.id} />
    </div>
  );
}
