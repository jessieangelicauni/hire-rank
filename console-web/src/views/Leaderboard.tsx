import { useEffect, useState } from 'react';
import { assessmentFor, type Applicant, type Role } from '../data';
import { colorAccent, colorAccentSoft, colorBorder, colorDanger, colorDangerSoft, colorSurfaceMuted } from '../tokens';

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
            <ul key={i} style={{ margin: '0 0 10px', paddingLeft: 18, fontSize: 14, color: '#000000', lineHeight: 1.55, textAlign: 'justify' }}>
              {block.items.map((item, j) => (
                <li key={j} style={{ marginBottom: 4 }}>{item}</li>
              ))}
            </ul>
          );
        }
        if (block.type === 'heading') {
          return (
            <div key={i} style={{ fontSize: 14, fontWeight: 500, margin: '0 0 6px' }}>{block.text}</div>
          );
        }
        return (
          <p key={i} style={{ fontSize: 14, color: '#000000', lineHeight: 1.55, margin: '0 0 10px', textAlign: 'justify' }}>{block.text}</p>
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
    <div style={{ maxWidth: 1440, margin: '0 auto', display: 'flex', gap: 28, alignItems: 'flex-start' }}>
      <div style={{ flex: splitFlex, minWidth: 0 }}>
        <div style={{ fontSize: 14, color: colorAccent, fontWeight: 500, cursor: 'pointer', marginBottom: 10 }} onClick={onBackToDashboard}>
          ← All roles
        </div>
        <h1 style={{ fontSize: 24, fontWeight: 500, margin: '0 0 4px' }}>{role.title}</h1>
        <RoleDescription role={role} />
        <div style={{ display: 'flex', gap: 10, marginBottom: 24 }}>
          <span style={{ fontSize: 14, color: '#000000', background: colorSurfaceMuted, padding: '5px 11px' }}>
            {leaderboard.length} applicants ranked
          </span>
        </div>

        <div style={{ background: '#fff', border: `1px solid ${colorBorder}`, overflow: 'hidden' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '44px 2.2fr 0.7fr', padding: '14px 24px', fontSize: 14, fontWeight: 500, letterSpacing: '0.08em', color: '#fff', background: colorAccent }}>
            <div>#</div>
            <div>APPLICANT</div>
            <div></div>
          </div>
          {pageItems.map((applicant) => {
            const isActive = applicant.id === activeApplicantId;
            return (
              <div
                key={applicant.id}
                style={{
                  display: 'grid', gridTemplateColumns: '44px 2.2fr 0.7fr', alignItems: 'center',
                  borderBottom: `1px solid ${colorBorder}`, padding: '16px 24px', cursor: 'pointer',
                  background: isActive ? colorAccentSoft : undefined,
                }}
                onClick={() => onSelectApplicant(applicant.id)}
              >
                <div style={{ fontSize: 14, fontWeight: 500, color: '#000000', fontVariantNumeric: 'tabular-nums' }}>{applicant.rank ?? '—'}</div>
                <div style={{ fontSize: 14, fontWeight: 500, letterSpacing: '0.01em' }}>{applicant.name}</div>
                <div
                  style={{ fontSize: 14, fontWeight: 500, color: colorAccent, textAlign: 'right' }}
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
    fontWeight: 500,
    padding: '6px 11px',
    cursor: disabled ? 'default' : 'pointer',
    color: disabled ? '#B8AFA5' : active ? '#fff' : '#000000',
    background: active ? colorAccent : undefined,
    userSelect: 'none',
  });

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 16, flexWrap: 'wrap' }}>
      <div style={itemStyle(false, page === 1)} onClick={() => page > 1 && onChange(page - 1)}>
        ‹ Prev
      </div>
      {pages.map((p) => (
        <div key={p} style={itemStyle(p === page, false)} onClick={() => onChange(p)}>
          {p}
        </div>
      ))}
      <div style={itemStyle(false, page === totalPages)} onClick={() => page < totalPages && onChange(page + 1)}>
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
      <div style={{ fontSize: 14, color: colorAccent, fontWeight: 500, cursor: 'pointer', marginBottom: 10 }} onClick={onClose}>
        ✕ Close
      </div>
      <h1 style={{ fontSize: 22, fontWeight: 500, margin: 0 }}>{applicant.name}</h1>
      <div style={{ fontSize: 14, color: '#000000', marginBottom: 24 }}>{rank !== null ? `Rank ${rank} of ${total}` : 'Not ranked (insufficient tournament comparisons)'}</div>

      <div style={{ display: 'flex', gap: 16, marginBottom: 20, alignItems: 'flex-start' }}>
        <AssessmentColumn title="Strengths" icon="✓" items={assessment.strengths} accent={colorAccent} accentSoft={colorAccentSoft} />
        <AssessmentColumn title="Weaknesses" icon="−" items={assessment.weaknesses} accent={colorDanger} accentSoft={colorDangerSoft} />
      </div>
    </div>
  );
}

function AssessmentColumn({
  title, icon, items, accent, accentSoft,
}: {
  title: string;
  icon: string;
  items: string[];
  accent: string;
  accentSoft: string;
}) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 14, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: '#000000', margin: '0 0 10px' }}>
        {title}
      </div>
      <div style={{ background: '#fff', border: `1px solid ${colorBorder}`, overflow: 'hidden' }}>
        {items.length === 0 ? (
          <div style={{ padding: '16px 20px', fontSize: 14, color: '#000000' }}>None identified.</div>
        ) : (
          items.map((item, i) => (
            <div
              key={i}
              style={{
                display: 'flex', gap: 10, padding: '12px 16px', fontSize: 14, color: '#000000', lineHeight: 1.5,
                borderBottom: i < items.length - 1 ? `1px solid ${colorBorder}` : undefined,
                borderLeft: `3px solid ${accent}`,
                background: accentSoft,
              }}
            >
              <span style={{ color: accent, fontWeight: 700, flexShrink: 0 }}>{icon}</span>
              <span>{item}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
