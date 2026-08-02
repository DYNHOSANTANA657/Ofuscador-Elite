import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const project = new URL("../", import.meta.url);

test("a versão web usa a interface e os metadados do Ofuscador Elite", async () => {
  const [page, layout, hosting] = await Promise.all([
    readFile(new URL("app/page.tsx", project), "utf8"),
    readFile(new URL("app/layout.tsx", project), "utf8"),
    readFile(new URL(".openai/hosting.json", project), "utf8"),
  ]);

  assert.match(page, /<EliteApp\s*\/>/);
  assert.match(layout, /title:\s*"Ofuscador Elite"/);
  assert.match(layout, /lang="pt-BR"/);
  assert.doesNotMatch(page + layout, /SkeletonPreview|Starter Project|codex-preview/);
  assert.deepEqual(JSON.parse(hosting), { d1: null, r2: null });
});

test("a interface compilada possui o fluxo principal", async () => {
  const index = await readFile(new URL("backend/static/index.html", project), "utf8");
  assert.match(index, /<html lang="pt-BR">/);
  assert.match(index, /<div id="root"><\/div>/);
  assert.match(index, /assets\/index-[^"']+\.js/);
  assert.match(index, /assets\/index-[^"']+\.css/);
});
