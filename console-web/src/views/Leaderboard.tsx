import { useEffect, useState } from 'react';
import { assessmentFor, type Applicant, type Assessment, type Role } from '../data';
import { avatarColorOf, initialsOf } from '../lib/avatar';
import {
  colorAccent, colorAccentSoft, colorBorder, colorDanger, colorDangerSoft,
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
      <RequirementScores requirementScores={assessment.requirement_scores} />
    </div>
  );
}

const RECOMMENDATION_LABEL: Record<Assessment['overall_recommendation'], string> = {
  hire: 'Hire',
  maybe: 'Maybe',
  no: 'No',
};

function ScoreSummary({ assessment }: { assessment: Assessment }) {
  const recommendationColor = assessment.overall_recommendation === 'hire'
    ? colorAccent
    : assessment.overall_recommendation === 'no' ? colorDanger : colorTextMuted;
  const recommendationSoft = assessment.overall_recommendation === 'hire'
    ? colorAccentSoft
    : assessment.overall_recommendation === 'no' ? colorDangerSoft : colorSurfaceMuted;

  return (
    <div style={{
      display: 'flex', gap: 16, alignItems: 'center', marginBottom: 20, padding: '16px 20px',
      background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro,
    }}>
      <div>
        <div style={{ fontSize: 32, fontWeight: 700, color: colorText, lineHeight: 1 }}>
          {assessment.overall_fit_score.toFixed(0)}
        </div>
        <div style={{ fontSize: 12, color: colorTextMuted, marginTop: 4 }}>Overall fit score</div>
      </div>
      <div style={{
        padding: '6px 14px', borderRadius: radiusPill, background: recommendationSoft,
        color: recommendationColor, fontWeight: 700, fontSize: 13,
      }}>
        {RECOMMENDATION_LABEL[assessment.overall_recommendation]}
      </div>
      <div style={{ fontSize: 13, color: assessment.meets_min_qualifications ? colorAccent : colorDanger, fontWeight: 600 }}>
        {assessment.meets_min_qualifications ? '✓ Meets minimum qualifications' : '✕ Does not meet minimum qualifications'}
      </div>
    </div>
  );
}

function RequirementScores({ requirementScores }: { requirementScores: Record<string, number> }) {
  const entries = Object.entries(requirementScores).sort(([, a], [, b]) => b - a);

  return (
    <div style={{ background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro, overflow: 'hidden' }}>
      <div style={{ fontSize: 18, fontWeight: 700, color: colorText, padding: '16px 20px 12px' }}>
        Requirement fit
      </div>
      <div>
        {entries.length === 0 ? (
          <div style={{ padding: '4px 20px 20px', fontSize: 14, color: colorTextMuted }}>No per-requirement breakdown available.</div>
        ) : (
          entries.map(([requirement, score]) => (
            <div
              key={requirement}
              style={{
                display: 'flex', alignItems: 'center', gap: 12, padding: '10px 20px', fontSize: 14, color: colorText,
                borderTop: `1px solid ${colorBorder}`,
              }}
            >
              <span style={{ flex: 1, minWidth: 0 }}>{requirement}</span>
              <div style={{ flexShrink: 0, width: 100, height: 6, borderRadius: radiusPill, background: colorSurfaceMuted, overflow: 'hidden' }}>
                <div style={{
                  width: `${Math.max(0, Math.min(100, score))}%`, height: '100%', borderRadius: radiusPill,
                  background: score >= 50 ? colorAccent : colorDanger,
                }} />
              </div>
              <span style={{ flexShrink: 0, width: 32, textAlign: 'right', color: colorTextMuted, fontSize: 13 }}>{score.toFixed(0)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
