import React, { useEffect, useState } from 'react';
import { Box, Button, Card, CardContent, Link, Stack, Typography } from '@mui/material';

interface SessionRow {
  id: string;
  created_at: string;
  ended_at?: string;
  source_lang: string;
  target_lang: string;
}

const SeamlessSessions: React.FC = () => {
  const [rows, setRows] = useState<SessionRow[]>([]);
  const [loading, setLoading] = useState(false);

  const apiBase = (import.meta as any).env.VITE_API_BASE || 'http://localhost:3000';

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${apiBase}/api/seamless/sessions`);
        const data = await res.json();
        setRows(Array.isArray(data) ? data : []);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    run();
  }, [apiBase]);

  const downloadTranscript = async (id: string) => {
    try {
      const res = await fetch(`${apiBase}/api/seamless/sessions/${id}/transcripts`);
      const data = await res.json();
      const finals = (Array.isArray(data) ? data : []).filter((t: any) => t.is_final === 1);
      const text = finals.map((t: any) => t.text).join('\n');
      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${id}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <Box p={3}>
      <Typography variant="h5" mb={2}>Seamless Sessions</Typography>
      {loading && <Typography>Loading...</Typography>}
      <Stack spacing={2}>
        {rows.map((r) => (
          <Card key={r.id}>
            <CardContent>
              <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography variant="subtitle1">{r.id}</Typography>
                  <Typography variant="body2">{r.source_lang} → {r.target_lang}</Typography>
                  <Typography variant="body2">{r.created_at}{r.ended_at ? ` → ${r.ended_at}` : ''}</Typography>
                </Box>
                <Stack direction="row" spacing={1}>
                  <Button size="small" variant="outlined" onClick={() => downloadTranscript(r.id)}>Download Transcript</Button>
                  <Link href={`${apiBase}/api/seamless/sessions/${r.id}/events`} target="_blank" rel="noreferrer">Events</Link>
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        ))}
        {rows.length === 0 && !loading && <Typography>No sessions yet.</Typography>}
      </Stack>
    </Box>
  );
};

export default SeamlessSessions;


