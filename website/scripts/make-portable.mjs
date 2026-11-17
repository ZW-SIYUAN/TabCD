import { readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../dist');
const base = (process.env.BASE_PATH || '/').replace(/\/$/, '') + '/';
const escaped = base.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const assetPrefix = new RegExp(`(["'\\s,(])${escaped}_astro/`, 'g');
const publicPrefix = new RegExp(`(["'])${escaped}(ICDE2027\\.pdf|citation\\.bib|favicon\\.svg|tabcd-logo\\.svg)`, 'g');
const files = await readdir(root, { recursive: true });
let adjusted = 0;
for (const name of files) {
  if (!['.html', '.css'].includes(extname(name))) continue;
  const path = resolve(root, name);
  let text = await readFile(path, 'utf8');
  if (name.endsWith('.html')) {
    text = text.replace(assetPrefix, '$1./_astro/').replace(publicPrefix, '$1./$2');
  } else {
    text = text.replace(assetPrefix, '$1./');
  }
  await writeFile(path, text);
  adjusted++;
}
await writeFile(resolve(root, '.nojekyll'), '');
console.log(`Prepared ${adjusted} HTML/CSS files with relative assets for project-repository and local-file use.`);
