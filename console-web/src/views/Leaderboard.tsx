import { useEffect, useState } from 'react';
import {
  assessmentFor, reliabilityFor,
  type Applicant, type AssessmentDetail, type Role, type SkillGroup,
} from '../data';
import { avatarColorOf, initialsOf } from '../lib/avatar';
import {
  colorAccent, colorAccentSoft, colorBorder, colorDanger,
  colorSurface, colorSurfaceMuted, colorText, colorTextMuted, colorTextSoft,
  radius, radiusPill, radiusSm, shadowMicro,
} from '../tokens';

const PAGE_SIZE = 20;

interface Props {
  roles: Role[];
  applicants: Applicant[];
  activeRoleId: string;
  activeApplicantId: string | null;
  onBackToDashboard: () => void;
  onSelectApplicant: (applicantId: string) => void;
  onCloseDetail: () => void;
}

type DescriptionBlock =
  | { type: 'heading' | 'p'; text: string }
  | { type: 'bullets'; items: string[] };

function parseDescription(role: Role): DescriptionBlock[] {
  const withoutTitle = role.description.startsWith(role.title)
    ? role.description.slice(role.title.length)
    : role.description;

  const blocks: DescriptionBlock[] = [];
  for (const rawLine of withoutTitle.split('\n')) {
    const line = rawLine.trim();
    if (!line) continue;

    if (line.startsWith('- ')) {
      const last = blocks[blocks.length - 1];
      const item = line.slice(2).trim();
      if (last?.type === 'bullets') last.items.push(item);
      else blocks.push({ type: 'bullets', items: [item] });
    } else if (line.endsWith(':') && line.length < 40) {
      blocks.push({ type: 'heading', text: line });
    } else {
      blocks.push({ type: 'p', text: line });
    }
  }
  return blocks;
}

function RoleDescription({ role }: { role: Role }) {
  const blocks = parseDescription(role);
  if (blocks.length === 0) return null;

  return (
    <div style={{ maxWidth: 720, margin: '0 0 14px' }}>
      {blocks.map((block, i) => {
        if (block.type === 'bullets') {
          return (
            <ul key={i} style={{ margin: '0 0 10px', paddingLeft: 18, fontSize: 14, color: colorTextMuted, lineHeight: 1.6, textAlign: 'justify' }}>
              {block.items.map((item, j) => (
                <li key={j} style={{ marginBottom: 4 }}>{item}</li>
              ))}
            </ul>
          );
        }
        if (block.type === 'heading') {
          return (
            <div key={i} style={{ fontSize: 14, fontWeight: 700, color: colorText, margin: '0 0 6px' }}>{block.text}</div>
          );
        }
        return (
          <p key={i} style={{ fontSize: 14, color: colorTextMuted, lineHeight: 1.6, margin: '0 0 10px', textAlign: 'justify' }}>{block.text}</p>
        );
      })}
    </div>
  );
}

export default function Leaderboard({
  roles, applicants, activeRoleId, activeApplicantId,
  onBackToDashboard, onSelectApplicant, onCloseDetail,
}: Props) {
  const role = roles.find((r) => r.id === activeRoleId)!;
  const leaderboard = applicants
    .filter((c) => c.roleId === activeRoleId)
    .sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity));
  const detailApplicant = activeApplicantId ? applicants.find((c) => c.id === activeApplicantId) ?? null : null;
  const splitFlex = detailApplicant ? '1.1' : '1';

  const [page, setPage] = useState(1);
  useEffect(() => setPage(1), [activeRoleId]);

  const totalPages = Math.max(1, Math.ceil(leaderboard.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageStart = (currentPage - 1) * PAGE_SIZE;
  const pageItems = leaderboard.slice(pageStart, pageStart + PAGE_SIZE);

  return (
    <div style={{ maxWidth: 1128, margin: '0 auto', display: 'flex', gap: 24, alignItems: 'flex-start' }}>
      <div style={{ flex: splitFlex, minWidth: 0 }}>
        <div style={{ fontSize: 13, color: colorAccent, fontWeight: 700, cursor: 'pointer', marginBottom: 10 }} onClick={onBackToDashboard}>
          ← All roles
        </div>
        <div style={{
          background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius,
          boxShadow: shadowMicro, padding: '20px 24px', marginBottom: 16,
        }}>
          <h1 style={{ fontSize: 24, fontWeight: 700, letterSpacing: '-0.01em', color: colorText, margin: '0 0 4px' }}>{role.title}</h1>
          <RoleDescription role={role} />
        </div>
        <div style={{ fontSize: 14, color: colorTextMuted, marginBottom: 16 }}>
          {leaderboard.length} applicants, ranked
        </div>

        <ReliabilityPanel roleId={activeRoleId} />

        <div style={{ background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro, overflow: 'hidden' }}>
          {pageItems.map((applicant) => {
            const isActive = applicant.id === activeApplicantId;
            return (
              <div
                key={applicant.id}
                className={isActive ? undefined : 'row-hover'}
                style={{
                  display: 'flex', alignItems: 'center', gap: 14,
                  borderBottom: `1px solid ${colorBorder}`, padding: '14px 20px', cursor: 'pointer',
                  background: isActive ? colorAccentSoft : undefined,
                }}
                onClick={() => onSelectApplicant(applicant.id)}
              >
                <span style={{
                  flex: '0 0 auto', display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  width: 24, height: 24, borderRadius: radiusPill, fontSize: 11, fontWeight: 700,
                  fontVariantNumeric: 'tabular-nums',
                  background: isActive ? colorSurface : colorSurfaceMuted,
                  color: isActive ? colorAccent : colorTextMuted,
                }}>
                  {applicant.rank ?? '—'}
                </span>
                <div style={{
                  flex: '0 0 auto', width: 40, height: 40, borderRadius: radiusPill,
                  background: avatarColorOf(applicant.name), color: '#fff', fontWeight: 700, fontSize: 14,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {initialsOf(applicant.name)}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: colorText, letterSpacing: '0.01em' }}>{applicant.name}</div>
                  <div style={{ fontSize: 12, color: colorTextMuted, marginTop: 1 }}>Rank {applicant.rank ?? '—'} of {leaderboard.length}</div>
                </div>
                <div
                  className="ghost-btn"
                  style={{
                    flex: '0 0 auto', fontSize: 13, fontWeight: 700, color: colorAccent, textAlign: 'center',
                    border: `1px solid ${colorAccent}`, borderRadius: radiusPill,
                    padding: '5px 14px', whiteSpace: 'nowrap',
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    window.open('#', '_blank');
                  }}
                >
                  View CV
                </div>
              </div>
            );
          })}
        </div>

        <Pager page={currentPage} totalPages={totalPages} onChange={setPage} />
      </div>

      {detailApplicant && (
        <DetailPanel
          applicant={detailApplicant}
          rank={detailApplicant.rank}
          total={leaderboard.length}
          onClose={onCloseDetail}
        />
      )}
    </div>
  );
}

function StatTile({
  label, value, sublabel, tone = 'neutral',
}: {
  label: string;
  value: string;
  sublabel: string;
  tone?: 'good' | 'warn' | 'neutral';
}) {
  const valueColor = tone === 'good' ? colorAccent : tone === 'warn' ? colorDanger : colorText;
  return (
    <div style={{ flex: '1 1 200px', minWidth: 160, padding: '12px 14px', background: colorSurfaceMuted, borderRadius: radiusSm }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: colorTextMuted, textTransform: 'uppercase', letterSpacing: '0.03em', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: valueColor, lineHeight: 1.1 }}>{value}</div>
      <div style={{ fontSize: 12, color: colorTextMuted, marginTop: 4 }}>{sublabel}</div>
    </div>
  );
}

function ReliabilityPanel({ roleId }: { roleId: string }) {
  const reliability = reliabilityFor(roleId);

  if (!reliability) {
    return (
      <div style={{
        background: colorSurfaceMuted, border: `1px dashed ${colorBorder}`, borderRadius: radius,
        padding: '14px 20px', marginBottom: 16, fontSize: 13, color: colorTextMuted,
      }}>
        Reliability checks (ground-truth-free QA) not yet run for this role.
      </div>
    );
  }

  const { tieringValidity, confidenceCalibration, rankingConsistency, rankingRepeatability } = reliability;
  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

  return (
    <div style={{
      background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius,
      boxShadow: shadowMicro, padding: '16px 20px', marginBottom: 16,
    }}>
      <div style={{ fontSize: 15, fontWeight: 700, color: colorText, marginBottom: 2 }}>Ranking reliability</div>
      <div style={{ fontSize: 12, color: colorTextMuted, marginBottom: 14 }}>
        Ground-truth-free QA: measures whether the ranking signal is internally consistent, not whether it's correct.
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: confidenceCalibration ? 14 : 0 }}>
        {tieringValidity && (
          <StatTile
            label="Tiering validity"
            value={tieringValidity.agreementRate !== null ? pct(tieringValidity.agreementRate) : '—'}
            sublabel={`Jev agrees with transparent score on ${tieringValidity.agreed}/${tieringValidity.total} clear-cut pairs`}
            tone={tieringValidity.agreementRate !== null && tieringValidity.agreementRate < 0.85 ? 'warn' : 'good'}
          />
        )}
        {rankingConsistency && (
          <StatTile
            label="Test-retest consistency"
            value={pct(1 - rankingConsistency.inconsistentRate)}
            sublabel={`${rankingConsistency.flippedPairs}/${rankingConsistency.totalPairs} pairs flip winner on repeat`}
            tone={rankingConsistency.inconsistentRate > 0.15 ? 'warn' : 'good'}
          />
        )}
        {rankingConsistency && (
          <StatTile
            label="Cosmetic invariance"
            value={pct(1 - rankingConsistency.cosmeticInconsistentRate)}
            sublabel={`${rankingConsistency.cosmeticFlipped}/${rankingConsistency.cosmeticTotal} pairs flip when evidence order is shuffled`}
            tone={rankingConsistency.cosmeticInconsistentRate > 0.15 ? 'warn' : 'good'}
          />
        )}
        {rankingRepeatability && (
          <StatTile
            label="Full-ranking repeatability"
            value={rankingRepeatability.meanKendallTau !== null ? rankingRepeatability.meanKendallTau.toFixed(3) : '—'}
            sublabel={`Kendall's tau across ${rankingRepeatability.nRepeatPairs} independent re-run pair(s)`}
            tone={rankingRepeatability.meanKendallTau !== null && rankingRepeatability.meanKendallTau < 0.7 ? 'warn' : 'good'}
          />
        )}
      </div>

      {confidenceCalibration && Object.keys(confidenceCalibration.buckets).length > 0 && (
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colorTextMuted, textTransform: 'uppercase', letterSpacing: '0.03em', marginBottom: 8 }}>
            Confidence calibration — agreement rate by original confidence bucket
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {(['<0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'] as const)
              .filter((bucket) => confidenceCalibration.buckets[bucket])
              .map((bucket) => {
                const b = confidenceCalibration.buckets[bucket];
                return (
                  <div key={bucket} style={{
                    flex: '1 1 110px', minWidth: 100, padding: '8px 10px',
                    background: colorSurfaceMuted, borderRadius: radiusSm,
                  }}>
                    <div style={{ fontSize: 11, color: colorTextMuted, marginBottom: 2 }}>conf {bucket} (n={b.n})</div>
                    <div style={{
                      fontSize: 15, fontWeight: 700,
                      color: b.mean < 0.7 ? colorDanger : colorAccent,
                    }}>
                      {pct(b.mean)}
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}

function Pager({
  page, totalPages, onChange,
}: {
  page: number;
  totalPages: number;
  onChange: (page: number) => void;
}) {
  const pages = Array.from({ length: totalPages }, (_, i) => i + 1);

  const itemStyle = (active: boolean, disabled: boolean): React.CSSProperties => ({
    fontSize: 14,
    fontWeight: active ? 700 : 500,
    padding: '4px 6px',
    borderRadius: radiusSm,
    cursor: disabled ? 'default' : 'pointer',
    color: disabled ? colorTextSoft : active ? colorAccent : colorTextMuted,
    textDecoration: active ? 'underline' : 'none',
    textUnderlineOffset: 3,
    userSelect: 'none',
  });

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, marginTop: 20, flexWrap: 'wrap',
    }}>
      <div className={page === 1 ? undefined : 'ghost-btn'} style={itemStyle(false, page === 1)} onClick={() => page > 1 && onChange(page - 1)}>
        ‹ Prev
      </div>
      {pages.map((p) => (
        <div key={p} className={p === page ? undefined : 'ghost-btn'} style={itemStyle(p === page, false)} onClick={() => onChange(p)}>
          {p}
        </div>
      ))}
      <div className={page === totalPages ? undefined : 'ghost-btn'} style={itemStyle(false, page === totalPages)} onClick={() => page < totalPages && onChange(page + 1)}>
        Next ›
      </div>
    </div>
  );
}

function DetailPanel({
  applicant, rank, total, onClose,
}: {
  applicant: Applicant;
  rank: number | null;
  total: number;
  onClose: () => void;
}) {
  const assessment = assessmentFor(applicant.id);

  return (
    <div style={{ flex: 1, minWidth: 0, position: 'sticky', top: 0 }}>
      <div style={{ fontSize: 13, color: colorAccent, fontWeight: 700, cursor: 'pointer', marginBottom: 10 }} onClick={onClose}>
        ✕ Close
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24 }}>
        <div style={{
          flex: '0 0 auto', width: 64, height: 64, borderRadius: radiusPill,
          background: avatarColorOf(applicant.name), color: '#fff', fontWeight: 700, fontSize: 20,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          {initialsOf(applicant.name)}
        </div>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 700, color: colorText, margin: 0 }}>{applicant.name}</h1>
          <div style={{ fontSize: 13, color: colorTextMuted, marginTop: 2 }}>{rank !== null ? `Rank ${rank} of ${total}` : 'Not ranked (no assessment available)'}</div>
        </div>
      </div>

      <ScoreSummary assessment={assessment} />
      <SkillsSection skillsGrouped={assessment.skillsGrouped} />
      <RequirementScores assessment={assessment} />
    </div>
  );
}

function ScoreSummary({ assessment }: { assessment: AssessmentDetail }) {
  return (
    <div style={{
      display: 'flex', gap: 16, alignItems: 'center', marginBottom: 20, padding: '16px 20px',
      background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro,
    }}>
      <div>
        <div style={{ fontSize: 32, fontWeight: 700, color: colorText, lineHeight: 1 }}>
          {(assessment.criterionScore * 100).toFixed(0)}
          <span style={{ fontSize: 16, fontWeight: 500, color: colorTextMuted }}>/100</span>
        </div>
        <div style={{ fontSize: 12, color: colorTextMuted, marginTop: 4 }}>
          Criterion score (transparent, probability-weighted) &middot; {assessment.criteriaMet}/{assessment.criteriaTotal} criteria with evidence
        </div>
      </div>
    </div>
  );
}

function SkillsSection({ skillsGrouped }: { skillsGrouped: SkillGroup[] }) {
  if (skillsGrouped.length === 0) return null;

  return (
    <div style={{ background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro, overflow: 'hidden', marginBottom: 16 }}>
      <div style={{ fontSize: 18, fontWeight: 700, color: colorText, padding: '16px 20px 4px' }}>
        Extracted skills
      </div>
      <div style={{ fontSize: 12, color: colorTextMuted, padding: '0 20px 12px', lineHeight: 1.5 }}>
        From the candidate's own CV, verbatim. Skills sharing the same sentence are grouped into one bullet.
      </div>
      <ul style={{ margin: 0, padding: '0 20px 16px 36px', listStyle: 'disc' }}>
        {skillsGrouped.map((group, i) => (
          <li key={i} style={{ fontSize: 13, color: colorText, marginBottom: 8, lineHeight: 1.5 }}>
            <strong>{group.names.join(', ')}</strong>
            <div style={{ color: colorTextMuted, fontSize: 12, marginTop: 1 }}>{group.context}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function RequirementScores({ assessment }: { assessment: AssessmentDetail }) {
  const entries = [...assessment.breakdown].sort((a, b) => b.expectedStrength - a.expectedStrength);
  const strengthColor = { strong: colorAccent, partial: colorTextMuted, none: colorDanger } as const;

  return (
    <div style={{ background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro, overflow: 'hidden' }}>
      <div style={{ fontSize: 18, fontWeight: 700, color: colorText, padding: '16px 20px 12px' }}>
        Requirement fit
      </div>
      <div>
        {entries.length === 0 ? (
          <div style={{ padding: '4px 20px 20px', fontSize: 14, color: colorTextMuted }}>No per-requirement breakdown available.</div>
        ) : (
          entries.map((item) => {
            const pct = item.expectedStrength * 100;
            return (
              <div
                key={item.requirement}
                style={{
                  display: 'flex', alignItems: 'center', gap: 12, padding: '10px 20px', fontSize: 14, color: colorText,
                  borderTop: `1px solid ${colorBorder}`,
                }}
              >
                <span style={{ flex: 1, minWidth: 0 }}>{item.requirement}</span>
                <span style={{
                  flexShrink: 0, fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.03em',
                  color: strengthColor[item.evidenceStrength],
                }}>
                  {item.evidenceStrength}
                </span>
                <div style={{ flexShrink: 0, width: 100, height: 6, borderRadius: radiusPill, background: colorSurfaceMuted, overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.max(0, Math.min(100, pct))}%`, height: '100%', borderRadius: radiusPill,
                    background: pct >= 50 ? colorAccent : colorDanger,
                  }} />
                </div>
                <span style={{ flexShrink: 0, width: 52, textAlign: 'right', color: colorTextMuted, fontSize: 13 }}>{pct.toFixed(0)}/100</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
