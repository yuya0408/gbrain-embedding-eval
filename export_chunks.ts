// Export gbrain's content_chunks table to JSONL for the standalone
// embedding-model comparison harness. Read-only against the PGLite brain.
import { PGlite } from '@electric-sql/pglite';
import { vector } from '@electric-sql/pglite/vector';
import { writeFileSync } from 'node:fs';

const BRAIN_DIR = process.env.GBRAIN_DIR;
if (!BRAIN_DIR) {
  throw new Error('GBRAIN_DIR env var must point at the gbrain PGLite data directory');
}
const OUT_PATH = process.argv[2] ?? 'chunks.jsonl';

const db = new PGlite(BRAIN_DIR, { extensions: { vector } });

const { rows } = await db.query<{
  chunk_id: number;
  page_id: number;
  slug: string;
  page_type: string;
  chunk_index: number;
  chunk_text: string;
  token_count: number | null;
}>(`
  SELECT
    c.id            AS chunk_id,
    c.page_id       AS page_id,
    p.slug          AS slug,
    p.type          AS page_type,
    c.chunk_index   AS chunk_index,
    c.chunk_text    AS chunk_text,
    c.token_count   AS token_count
  FROM content_chunks c
  JOIN pages p ON p.id = c.page_id
  ORDER BY c.page_id, c.chunk_index
`);

const lines = rows.map(r => JSON.stringify(r));
writeFileSync(OUT_PATH, lines.join('\n') + '\n', 'utf-8');

console.log(`Exported ${rows.length} chunks from ${new Set(rows.map(r => r.page_id)).size} pages -> ${OUT_PATH}`);
await db.close();
