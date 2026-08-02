import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

const project = new URL("../", import.meta.url);

test("oferece Piper local por padrão e Azure opcional", async () => {
  const source = await readFile(new URL("ui/EliteApp.tsx", project), "utf8");
  assert.match(source, /useState<Provider>\("piper"\)/);
  assert.match(source, /Piper local/);
  assert.match(source, /Azure Speech/);
  assert.match(source, /provider, audioAssetId:/);
  assert.match(source, /volumePercent: volume/);
  assert.match(source, /piperAvailable/);
  assert.doesNotMatch(source, /Modo de teste ativo/);
});

test("aceita um arquivo de áudio próprio no lugar da voz sintetizada", async () => {
  const source = await readFile(new URL("ui/EliteApp.tsx", project), "utf8");
  assert.match(source, /"piper" \| "azure" \| "file"/);
  assert.match(source, /fetch\("\/api\/audio", \{ method: "POST"/);
  assert.match(source, /Meu áudio/);
  assert.match(source, /Volume do áudio enviado/);
  // Sem voz sintetizada não há texto nem prévia para pedir.
  assert.match(source, /usesAudio && !usesOwnAudio && <div className="step-card text-card">/);
});

test("processa vários vídeos em fila com o mesmo ajuste", async () => {
  const source = await readFile(new URL("ui/EliteApp.tsx", project), "utf8");
  assert.match(source, /function startBatch/);
  assert.match(source, /multiple=\{batchAllowed\}/);
  // Lote só no modo Áudio: legenda gravada precisa de revisão por vídeo.
  assert.match(source, /const batchAllowed = mode === "audio"/);
  assert.match(source, /Fila de processamento/);
  assert.match(source, /batchPendingIds/);
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
