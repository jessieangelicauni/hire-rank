import { colorSurfaceMuted, colorText, colorTextMuted, colorTextSoft, radiusPill } from '../tokens';

interface Props {
  label: string;
}

export default function Placeholder({ label }: Props) {
  return (
    <div style={{ maxWidth: 1180, margin: '80px auto', textAlign: 'center' }}>
      <div style={{
        width: 56, height: 56, borderRadius: radiusPill, background: colorSurfaceMuted,
        display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px',
        fontSize: 22, color: colorTextSoft,
      }}>
        ⚙
      </div>
      <div style={{ fontSize: 18, fontWeight: 700, color: colorText, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 14, color: colorTextMuted }}>This area isn't part of the prototype yet.</div>
    </div>
  );
}
