import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const project = new URL("../", import.meta.url);

test("a interface portátil monta o EliteApp com os estilos do projeto", async () => {
  const [entry, page] = await Promise.all([
    readFile(new URL("portable-ui/main.tsx", project), "utf8"),
    readFile(new URL("portable-ui/index.html", project), "utf8"),
  ]);

  assert.match(entry, /<EliteApp\s*\/>/);
  assert.match(entry, /import "@\/ui\/globals\.css"/);
  assert.match(page, /lang="pt-BR"/);
  assert.doesNotMatch(entry + page, /SkeletonPreview|Starter Project|codex-preview/);
});

test("a interface compilada possui o fluxo principal", async () => {
  const index = await readFile(new URL("backend/static/index.html", project), "utf8");
  assert.match(index, /<html lang="pt-BR">/);
  assert.match(index, /<div id="root"><\/div>/);
  assert.match(index, /assets\/index-[^"']+\.js/);
  assert.match(index, /assets\/index-[^"']+\.css/);
});
