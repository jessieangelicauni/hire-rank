import { COMPARISON, type Role } from '../data';
import { colorAccent, colorBorder } from '../tokens';

interface Props {
  roles: Role[];
}

export default function Comparison({ roles }: Props) {
  return (
    <div style={{ maxWidth: 1180, margin: '0 auto' }}>
      <h1 style={{ fontSize: 26, fontWeight: 500, margin: '0 0 6px' }}>Evaluation</h1>
      <p style={{ fontSize: 14, color: '#000000', margin: '0 0 28px', maxWidth: 640, lineHeight: 1.5 }}>
        How independent fact-checking and tournament re-ranking change each role's results.
      </p>
      <div style={{ background: '#fff', border: `1px solid ${colorBorder}`, overflow: 'hidden' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '2.1fr 0.7fr 0.8fr 0.9fr', padding: '14px 24px', fontSize: 14, fontWeight: 500, letterSpacing: '0.08em', color: '#fff', background: colorAccent }}>
          <div>ROLE</div>
          <div>τ</div>
          <div>Δu</div>
          <div>FAITHFULNESS</div>
        </div>
        {roles.map((role) => {
          const m = COMPARISON[role.id] ?? { jdId: role.id, kendallTau: null, deltaU: null, faithfulness: null };
          const deltaU = m.deltaU !== null ? `${m.deltaU >= 0 ? '+' : ''}${m.deltaU.toFixed(4)}` : '—';
          const faithfulness = m.faithfulness !== null ? `${(m.faithfulness * 100).toFixed(1)}%` : '—';
          return (
            <div
              key={role.id}
              style={{ display: 'grid', gridTemplateColumns: '2.1fr 0.7fr 0.8fr 0.9fr', alignItems: 'center', borderBottom: `1px solid ${colorBorder}`, padding: '16px 24px' }}
            >
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 14, fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{role.title}</div>
              </div>
              <div style={{ fontSize: 14, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>{m.kendallTau !== null ? m.kendallTau.toFixed(4) : '—'}</div>
              <div style={{ fontSize: 14, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>{deltaU}</div>
              <div style={{ fontSize: 14, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>{faithfulness}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
