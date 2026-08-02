import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

const project = new URL("../", import.meta.url);

test("oferece Piper local por padrão e Azure opcional", async () => {
  const source = await readFile(new URL("ui/EliteApp.tsx", project), "utf8");
  assert.match(source, /useState<Provider>\("piper"\)/);
  assert.match(source, /Piper local/);
  assert.match(source, /Azure Speech/);
  assert.match(source, /provider, volumePercent/);
  assert.match(source, /piperAvailable/);
  assert.doesNotMatch(source, /Modo de teste ativo/);
});

test("a compilação portátil contém interface e estilos", async () => {
  const staticRoot = new URL("backend/static/", project);
  const index = await readFile(new URL("index.html", staticRoot), "utf8");
  const assets = await readdir(new URL("assets/", staticRoot));
  assert.match(index, /<div id="root"><\/div>/);
  assert.ok(assets.some((name) => name.endsWith(".js")));
  assert.ok(assets.some((name) => name.endsWith(".css")));
});

test("a prévia possui reprodução manual e valida o WAV recebido", async () => {
  const source = await readFile(new URL("ui/EliteApp.tsx", project), "utf8");
  assert.match(source, /audioData\.byteLength < 48/);
  assert.match(source, /new Blob\(\[audioData\], \{ type: "audio\/wav" \}\)/);
  assert.match(source, /fetchWithNetworkRetry\("\/api\/preview"/);
  assert.match(source, /Mantenha a janela do Ofuscador Elite aberta/);
  assert.match(source, /A prévia anterior continua disponível; a nova tentativa falhou/);
  assert.doesNotMatch(source, /setNotice\(error instanceof Error \? error\.message : "A prévia não pôde ser criada\."\)/);
  assert.match(source, /Reproduzir/);
  assert.match(source, /Baixar prévia WAV/);
  assert.match(source, /player\.load\(\)/);
});

test("trocar o vídeo limpa o trabalho anterior e protege um processamento ativo", async () => {
  const source = await readFile(new URL("ui/EliteApp.tsx", project), "utf8");
  const sendVideo = source.slice(source.indexOf("function sendVideo"), source.indexOf("function onTextFile"));
  assert.match(sendVideo, /if \(jobActive \|\| subtitleScanActive\)/);
  assert.match(sendVideo, /setJob\(null\)/);
  assert.match(source, /disabled=\{jobActive \|\| subtitleScanActive\}/);
  assert.match(source, /Processar outro vídeo/);
});

test("oferece remoção local de legendas com revisão manual", async () => {
  const source = await readFile(new URL("ui/EliteApp.tsx", project), "utf8");
  assert.match(source, /"audio" \| "subtitles" \| "audio_and_subtitles"/);
  assert.match(source, /\/api\/subtitle-model\/install/);
  assert.match(source, /\/subtitle-scan/);
  assert.match(source, /drawPointer/);
  assert.match(source, /Gerar prévia antes e depois/);
  assert.match(source, /removeEmbedded/);
  assert.match(source, /removeBurnedIn/);
  assert.match(source, /HDR bloqueado nesta versão/);
});
