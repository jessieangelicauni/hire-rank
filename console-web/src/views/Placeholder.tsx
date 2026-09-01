interface Props {
  label: string;
}

export default function Placeholder({ label }: Props) {
  return (
    <div style={{ maxWidth: 1180, margin: '60px auto', textAlign: 'center', color: '#000000' }}>
      <div style={{ fontSize: 18, fontWeight: 500, color: '#000000', marginBottom: 8 }}>{label}</div>
      <div style={{ fontSize: 14 }}>This area isn't part of the prototype yet.</div>
    </div>
  );
}
