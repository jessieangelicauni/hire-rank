import type { Applicant, Role } from '../data';
import { avatarColorOf, initialsOf } from '../lib/avatar';
import { colorBorder, colorSurface, colorText, colorTextMuted, colorTextSoft, radius, shadowMicro } from '../tokens';

interface Props {
  roles: Role[];
  applicants: Applicant[];
  search: string;
  onSelectRole: (roleId: string) => void;
}

function roleSummary(role: Role): string {
  const withoutTitle = role.description.startsWith(role.title)
    ? role.description.slice(role.title.length)
    : role.description;
  const firstParagraph = withoutTitle.split('\n\n')[0] ?? '';
  return firstParagraph.replace(/\s+/g, ' ').trim();
}

export default function Dashboard({ roles, applicants, search, onSelectRole }: Props) {
  const q = search.toLowerCase();
  const filteredRoles = roles.filter((r) => !q || r.title.toLowerCase().includes(q));

  return (
    <div style={{ maxWidth: 1128, margin: '0 auto' }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, letterSpacing: '-0.01em', color: colorText, margin: '0 0 6px' }}>Open Roles</h1>
      <p style={{ fontSize: 14, color: colorTextMuted, margin: '0 0 28px' }}>
        {roles.length} roles · {applicants.length} applicants
      </p>
      <div style={{ background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro, overflow: 'hidden' }}>
        {filteredRoles.map((role) => {
          const count = applicants.filter((c) => c.roleId === role.id).length;
          return (
            <div
              key={role.id}
              className="row-hover"
              style={{ minWidth: 0, borderBottom: `1px solid ${colorBorder}`, padding: '14px 20px', display: 'flex', alignItems: 'center', gap: 14, cursor: 'pointer' }}
              onClick={() => onSelectRole(role.id)}
            >
              <div style={{
                flex: '0 0 auto', width: 40, height: 40, borderRadius: radius,
                background: avatarColorOf(role.title), color: '#fff', fontWeight: 700, fontSize: 15,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                {initialsOf(role.title)}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: colorText }}>{role.title}</div>
                <div
                  style={{
                    fontSize: 14, color: colorTextMuted, marginTop: 4, lineHeight: 1.5,
                    display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
                  }}
                >
                  {roleSummary(role)}
                </div>
                <div style={{ fontSize: 12, color: colorTextMuted, marginTop: 6 }}>
                  {count} applicants
                </div>
              </div>
              <span style={{ flex: '0 0 auto', fontSize: 18, color: colorTextSoft }}>›</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
