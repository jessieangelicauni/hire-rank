import type { Applicant, Role } from '../data';
import { colorAccent, colorBorder } from '../tokens';

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
    <div style={{ maxWidth: 1180, margin: '0 auto' }}>
      <h1 style={{ fontSize: 26, fontWeight: 500, margin: '0 0 6px' }}>Open Roles</h1>
      <p style={{ fontSize: 14, color: '#000000', margin: '0 0 28px' }}>
        {roles.length} roles · {applicants.length} applicants
      </p>
      <div style={{ background: '#fff', border: `1px solid ${colorBorder}`, overflow: 'hidden' }}>
        {filteredRoles.map((role) => {
          const count = applicants.filter((c) => c.roleId === role.id).length;
          return (
            <div
              key={role.id}
              style={{ minWidth: 0, borderBottom: `1px solid ${colorBorder}`, padding: '20px 24px', display: 'flex', alignItems: 'center', gap: 18, cursor: 'pointer' }}
              onClick={() => onSelectRole(role.id)}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 14, fontWeight: 700 }}>{role.title}</div>
                <div
                  style={{
                    fontSize: 14, color: '#000000', marginTop: 4, lineHeight: 1.5,
                    display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
                  }}
                >
                  {roleSummary(role)}
                </div>
              </div>
              <div style={{ flex: '0 0 120px', fontSize: 14, color: '#000000', textAlign: 'right' }}>{count} applicants</div>
              <div style={{ flex: '0 0 auto', fontSize: 14, fontWeight: 500, color: colorAccent }}>View →</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
