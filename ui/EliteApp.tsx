"use client";

import { ChangeEvent, FormEvent, type PointerEvent as ReactPointerEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Provider = "piper" | "azure" | "file";
type AudioAsset = { audioId: string; filename: string; duration: number; size: number; codec: string };
type BatchItem = { key: string; name: string; size: number; status: string; progress: number; message: string; jobId?: string; outputName?: string; error?: string; warning?: string };
type ProcessMode = "audio" | "subtitles" | "audio_and_subtitles";
type Voice = { shortName: string; displayName: string; gender: "Male" | "Female"; locale: string; provider: Provider; local?: boolean };
type AudioTrack = { index: number; order: number; codec: string; channels: number; language: string; title: string };
type SubtitleTrack = { index: number; order: number; codec: string; language: string; title: string; default: boolean; forced: boolean };
type UploadInfo = { uploadId: string; filename: string; duration: number; size: number; audioTracks: AudioTrack[]; subtitleTracks: SubtitleTrack[]; resolution: { width: number; height: number }; fps: number; videoCodec: string; variableFrameRate: boolean; hdr: boolean };
type Job = { id: string; status: string; progress: number; message: string; outputName?: string; error?: string; warning?: string };
type ModelStatus = { installed: boolean; version?: string; status: string; progress: number; message: string; error?: string };
type SubtitleRegion = { id: string; x: number; y: number; width: number; height: number; startMs: number; endMs: number; source: "automatic" | "manual"; enabled: boolean; text: string; confidence: number };
type SubtitleScan = { id: string; uploadId: string; status: string; progress: number; message: string; fullScreen: boolean; regions: SubtitleRegion[]; sampleTimesMs: number[]; error?: string };
type AppStatus = { credentialConfigured: boolean; azureConfigured: boolean; azureRegion: string; piperAvailable: boolean; defaultProvider: Provider; outputDirectory: string; networkMode: boolean; pinRequired: boolean; subtitleModels?: ModelStatus };

const fallbackVoices: Voice[] = [
  { shortName: "pt_BR-faber-medium", displayName: "Faber — masculina (local)", gender: "Male", locale: "pt-BR", provider: "piper", local: true },
];

const localServiceError = "O serviço local não respondeu. Mantenha a janela do Ofuscador Elite aberta e tente novamente.";

function isNetworkFetchError(error: unknown) {
  return error instanceof TypeError || (error instanceof Error && /failed to fetch|networkerror|load failed/i.test(error.message));
}

async function fetchWithNetworkRetry(url: string, init: RequestInit, retries = 1): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await fetch(url, init);
    } catch (error) {
      lastError = error;
      if (!isNetworkFetchError(error) || attempt === retries) break;
      await new Promise((resolve) => window.setTimeout(resolve, 650));
    }
  }
  throw lastError;
}

async function apiJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, { credentials: "same-origin", ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  } catch (error) {
    if (isNetworkFetchError(error)) throw new Error(localServiceError);
    throw error;
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.message || "Não foi possível concluir esta ação.");
  return body as T;
}

function formatTime(seconds: number) {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 MB";
  return `${(bytes / 1024 / 1024).toFixed(bytes > 100 * 1024 * 1024 ? 0 : 1)} MB`;
}

function Logo() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 48 48" fill="none"><path d="M14 13.5v21M34 13.5v21M8.5 19h5.5m20 0h5.5M8.5 29H14m20 0h5.5" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/><path d="M20 9v30M28 9v30" stroke="currentColor" strokeWidth="5" strokeLinecap="round"/><path d="m19 24 4-4v8l6-6" stroke="#6EE7B7" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
    </div>
  );
}

export function EliteApp() {
  const [status, setStatus] = useState<AppStatus>({ credentialConfigured: false, azureConfigured: false, azureRegion: "", piperAvailable: true, defaultProvider: "piper", outputDirectory: "Saidas", networkMode: false, pinRequired: false });
  const [backendOnline, setBackendOnline] = useState(true);
  const [video, setVideo] = useState<File | null>(null);
  const [upload, setUpload] = useState<UploadInfo | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [audioTrack, setAudioTrack] = useState<number | null>(null);
  const [mode, setMode] = useState<ProcessMode>("audio");
  const [removeEmbedded, setRemoveEmbedded] = useState(true);
  const [removeBurnedIn, setRemoveBurnedIn] = useState(false);
  const [burnedInMode, setBurnedInMode] = useState<"auto" | "regions">("auto");
  const [scanFullScreen, setScanFullScreen] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ installed: false, status: "not_installed", progress: 0, message: "Pacote de IA ainda não instalado" });
  const [scan, setScan] = useState<SubtitleScan | null>(null);
  const [regions, setRegions] = useState<SubtitleRegion[]>([]);
  const [reviewTimeMs, setReviewTimeMs] = useState(0);
  const [previewStamp, setPreviewStamp] = useState(0);
  const [subtitlePreviewUrl, setSubtitlePreviewUrl] = useState("");
  const [subtitleBusy, setSubtitleBusy] = useState(false);
  const [textValue, setTextValue] = useState("");
  const [provider, setProvider] = useState<Provider>("piper");
  const [gender, setGender] = useState<"Female" | "Male">("Male");
  const [voices, setVoices] = useState<Voice[]>(fallbackVoices);
  const [voice, setVoice] = useState(fallbackVoices[0].shortName);
  const [volume, setVolume] = useState(5);
  const [job, setJob] = useState<Job | null>(null);
  const [audioAsset, setAudioAsset] = useState<AudioAsset | null>(null);
  const [audioUploading, setAudioUploading] = useState(false);
  const [batch, setBatch] = useState<BatchItem[]>([]);
  const [batchStarting, setBatchStarting] = useState(false);
  const [notice, setNotice] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [azureKey, setAzureKey] = useState("");
  const [azureRegion, setAzureRegion] = useState("");
  const [outputDirectory, setOutputDirectory] = useState("");
  const [savingSettings, setSavingSettings] = useState(false);
  const [removingAzure, setRemovingAzure] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewing, setPreviewing] = useState(false);
  const [previewMessage, setPreviewMessage] = useState("");
  const [voiceError, setVoiceError] = useState("");
  const [pinOpen, setPinOpen] = useState(false);
  const [networkPin, setNetworkPin] = useState("");
  const [authenticating, setAuthenticating] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const audioFileInput = useRef<HTMLInputElement>(null);
  const modelFileInput = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const previewObjectUrl = useRef("");
  const subtitlePreviewObjectUrl = useRef("");
  const drawStart = useRef<{ x: number; y: number } | null>(null);

  const jobActive = Boolean(job && ["queued", "processing"].includes(job.status));
  const subtitleScanActive = Boolean(scan && ["queued", "scanning"].includes(scan.status));
  const batchActive = batchStarting || batch.some((item) => ["queued", "processing", "enviando"].includes(item.status));

  const loadStatus = useCallback(async () => {
    try {
      const response = await fetch("/api/config/status", { credentials: "same-origin" });
      if (response.status === 401) {
        setPinOpen(true);
        setBackendOnline(true);
        return;
      }
      if (!response.ok) throw new Error();
      const next = await response.json() as AppStatus;
      setStatus(next);
      if (next.subtitleModels) setModelStatus(next.subtitleModels);
      setOutputDirectory(next.outputDirectory);
      setAzureRegion(next.azureRegion || "");
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadStatus(), 0);
    return () => window.clearTimeout(timer);
  }, [loadStatus]);

  useEffect(() => {
    if (!["downloading", "installing"].includes(modelStatus.status)) return;
    const timer = window.setInterval(() => {
      void apiJson<ModelStatus>("/api/subtitle-model/status").then(setModelStatus).catch(() => undefined);
    }, 900);
    return () => window.clearInterval(timer);
  }, [modelStatus.status]);

  useEffect(() => {
    if (!scan || !["queued", "scanning"].includes(scan.status)) return;
    const timer = window.setInterval(() => {
      void apiJson<SubtitleScan>(`/api/subtitle-scans/${scan.id}`).then((next) => {
        setScan(next);
        if (next.status === "completed") {
          setRegions(next.regions);
          setReviewTimeMs(next.sampleTimesMs[0] ?? 0);
        }
        if (next.status === "failed") setNotice(next.error || "O exame de legendas falhou.");
      }).catch((error) => setNotice(error instanceof Error ? error.message : "Não foi possível acompanhar o exame."));
    }, 850);
    return () => window.clearInterval(timer);
  }, [scan?.id, scan?.status]);

  useEffect(() => {
    // Com áudio próprio não existe voz para carregar; o endpoint só aceita piper/azure.
    if (provider === "file") return;
    let cancelled = false;
    apiJson<{ voices: Voice[] }>(`/api/voices?provider=${provider}&gender=${gender}`)
      .then(({ voices: available }) => {
        if (cancelled) return;
        setVoiceError("");
        setVoices(available);
        setVoice(available[0]?.shortName || "");
      })
      .catch((error) => {
        const available = provider === "piper" ? fallbackVoices.filter((item) => item.gender === gender) : [];
        setVoices(available);
        setVoice(available[0]?.shortName || "");
        setVoiceError(error instanceof Error ? error.message : "Não foi possível carregar as vozes.");
      });
    return () => { cancelled = true; };
  }, [gender, provider, status.credentialConfigured]);

  useEffect(() => {
    const jobId = job?.id;
    if (!jobId || !jobActive) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await apiJson<Job>(`/api/jobs/${jobId}`);
        setJob(next);
        if (["completed", "failed", "cancelled"].includes(next.status)) {
          window.clearInterval(timer);
          // Numa falha o vídeo e a análise de legendas continuam válidos no servidor.
          // Limpar tudo aqui obrigava a reenviar o arquivo e refazer o OCR do zero.
        }
      } catch (error) {
        setNotice(error instanceof Error ? error.message : "Não foi possível acompanhar o processamento.");
      }
    }, 900);
    return () => window.clearInterval(timer);
  }, [job?.id, jobActive]);

  // Depende só da lista de identificadores pendentes. Depender de `batch` inteiro
  // recriaria o intervalo a cada resposta do servidor.
  const batchPendingIds = useMemo(
    () => batch.filter((item) => item.jobId && ["queued", "processing"].includes(item.status)).map((item) => item.jobId).join(","),
    [batch],
  );

  useEffect(() => {
    if (!batchPendingIds) return;
    const ids = batchPendingIds.split(",");
    const timer = window.setInterval(async () => {
      try {
        const updates = await Promise.all(ids.map((id) => apiJson<Job>(`/api/jobs/${id}`)));
        setBatch((current) => current.map((item) => {
          const next = updates.find((update) => update.id === item.jobId);
          if (!next) return item;
          return { ...item, status: next.status, progress: next.progress, message: next.message, outputName: next.outputName, error: next.error, warning: next.warning };
        }));
      } catch (error) {
        setNotice(error instanceof Error ? error.message : "Não foi possível acompanhar a fila.");
      }
    }, 900);
    return () => window.clearInterval(timer);
  }, [batchPendingIds]);

  useEffect(() => {
    if (!previewUrl || !audioRef.current) return;
    const player = audioRef.current;
    player.load();
    setPreviewMessage("Prévia pronta. Se ela não iniciar sozinha, clique em Reproduzir.");
    void player.play()
      .then(() => setPreviewMessage("Reproduzindo a prévia gerada."))
      .catch(() => setPreviewMessage("Prévia pronta. Clique em Reproduzir para ouvir."));
  }, [previewUrl]);

  useEffect(() => () => {
    if (previewObjectUrl.current) URL.revokeObjectURL(previewObjectUrl.current);
    if (subtitlePreviewObjectUrl.current) URL.revokeObjectURL(subtitlePreviewObjectUrl.current);
  }, []);

  const selectedVoice = useMemo(() => voices.find((item) => item.shortName === voice), [voices, voice]);
  const usesAudio = mode !== "subtitles";
  const usesSubtitles = mode !== "audio";
  const subtitleReady = !usesSubtitles || ((removeEmbedded && Boolean(upload?.subtitleTracks.length)) || (removeBurnedIn && (burnedInMode === "auto" ? modelStatus.installed && !upload?.hdr : scan?.status === "completed" && regions.some((item) => item.enabled))));
  const usesOwnAudio = provider === "file";
  const voiceReady = usesOwnAudio ? Boolean(audioAsset) : Boolean(textValue.trim() && textValue.length <= 5000 && voice);
  const audioReady = !usesAudio || Boolean(voiceReady && audioTrack !== null);
  const canProcess = Boolean(upload && audioReady && subtitleReady && !uploading && !jobActive && !subtitleScanActive && !batchActive && job?.status !== "completed");
  // O lote repete o mesmo ajuste em vários vídeos, como o script que percorria a
  // pasta inteira. Só vale para o modo Áudio: legenda gravada exige revisão por vídeo.
  const batchAllowed = mode === "audio";

  function clearPreview() {
    if (previewObjectUrl.current) URL.revokeObjectURL(previewObjectUrl.current);
    previewObjectUrl.current = "";
    setPreviewUrl("");
    setPreviewMessage("");
  }

  async function sendAudio(file: File) {
    setNotice("");
    if (file.size > 300 * 1024 * 1024) {
      setNotice("O áudio ultrapassa o limite de 300 MB.");
      return;
    }
    if (!backendOnline) {
      setNotice("Abra esta interface pelo inicializador do aplicativo para enviar o áudio.");
      return;
    }
    setAudioUploading(true);
    try {
      const form = new FormData();
      form.append("audio", file);
      const response = await fetch("/api/audio", { method: "POST", body: form, credentials: "same-origin" });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || "O áudio não pôde ser analisado.");
      if (audioAsset?.audioId) {
        void fetch(`/api/audio/${audioAsset.audioId}`, { method: "DELETE", credentials: "same-origin" });
      }
      setAudioAsset(body as AudioAsset);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "O áudio não pôde ser analisado.");
    } finally {
      setAudioUploading(false);
    }
  }

  async function startBatch(files: File[]) {
    setNotice("");
    if (!backendOnline) {
      setNotice("Abra esta interface pelo inicializador do aplicativo para processar vídeos.");
      return;
    }
    if (usesOwnAudio ? !audioAsset : !textValue.trim() || (!usesOwnAudio && !voice)) {
      setNotice(usesOwnAudio ? "Envie o arquivo de áudio antes de processar o lote." : "Escreva o texto e escolha a voz antes de processar o lote.");
      return;
    }
    const invalid = files.filter((file) => !file.name.toLowerCase().endsWith(".mp4"));
    if (invalid.length) {
      setNotice(`Estes arquivos não são MP4 e foram ignorados: ${invalid.map((file) => file.name).join(", ")}`);
    }
    const accepted = files.filter((file) => file.name.toLowerCase().endsWith(".mp4") && file.size <= 2 * 1024 * 1024 * 1024);
    if (!accepted.length) {
      setNotice("Nenhum MP4 válido foi selecionado.");
      return;
    }
    setJob(null);
    setVideo(null);
    setUpload(null);
    setBatchStarting(true);
    setBatch(accepted.map((file, index) => ({ key: `${index}-${file.name}`, name: file.name, size: file.size, status: "enviando", progress: 0, message: "Enviando o vídeo" })));

    for (const [index, file] of accepted.entries()) {
      const key = `${index}-${file.name}`;
      const patch = (changes: Partial<BatchItem>) => setBatch((current) => current.map((item) => (item.key === key ? { ...item, ...changes } : item)));
      try {
        const form = new FormData();
        form.append("video", file);
        const uploadResponse = await fetch("/api/uploads", { method: "POST", body: form, credentials: "same-origin" });
        const uploaded = await uploadResponse.json().catch(() => ({}));
        if (!uploadResponse.ok) throw new Error(uploaded.detail || "O vídeo não pôde ser analisado.");
        const track = uploaded.audioTracks?.[0]?.index;
        if (track === undefined) throw new Error("Este vídeo não possui faixa de áudio para misturar.");
        const created = await apiJson<Job>("/api/jobs", {
          method: "POST",
          body: JSON.stringify({
            uploadId: uploaded.uploadId, mode: "audio",
            text: usesOwnAudio ? "" : textValue.trim(), voice: usesOwnAudio ? "" : voice, gender,
            provider, audioAssetId: usesOwnAudio ? audioAsset?.audioId : null,
            volumePercent: volume, audioStreamIndex: track,
            subtitleRemoval: { removeEmbedded: false, removeBurnedIn: false, scanId: null },
          }),
        });
        patch({ jobId: created.id, status: created.status, progress: created.progress, message: created.message });
      } catch (error) {
        patch({ status: "failed", message: "Falha", error: error instanceof Error ? error.message : "Não foi possível enfileirar este vídeo." });
      }
    }
    setBatchStarting(false);
  }

  function clearBatch() {
    if (batchActive) return;
    setBatch([]);
    setNotice("");
  }

  function chooseVideos(files: File[]) {
    if (!files.length) return;
    if (files.length === 1) {
      sendVideo(files[0]);
      return;
    }
    if (!batchAllowed) {
      setNotice("Vários vídeos de uma vez só funcionam no modo Áudio. A remoção de legenda precisa da revisão de cada vídeo.");
      return;
    }
    void startBatch(files);
  }

  function sendVideo(file: File) {
    if (jobActive || subtitleScanActive) {
      setNotice("Aguarde a análise ou o processamento atual terminar antes de trocar o vídeo.");
      return;
    }
    if (upload?.uploadId) {
      void fetch(`/api/uploads/${upload.uploadId}`, { method: "DELETE", credentials: "same-origin" });
    }
    setJob(null);
    setVideo(file);
    setUpload(null);
    setAudioTrack(null);
    setScan(null);
    setRegions([]);
    setReviewTimeMs(0);
    setPreviewStamp(0);
    if (subtitlePreviewObjectUrl.current) URL.revokeObjectURL(subtitlePreviewObjectUrl.current);
    subtitlePreviewObjectUrl.current = "";
    setSubtitlePreviewUrl("");
    setNotice("");
    if (!file.name.toLowerCase().endsWith(".mp4")) {
      setNotice("Escolha um vídeo no formato MP4.");
      return;
    }
    if (file.size > 2 * 1024 * 1024 * 1024) {
      setNotice("O vídeo ultrapassa o limite de 2 GB.");
      return;
    }
    if (!backendOnline) {
      setNotice("Abra esta interface pelo inicializador do aplicativo para analisar o vídeo.");
      return;
    }
    const form = new FormData();
    form.append("video", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/uploads");
    xhr.withCredentials = true;
    setUploading(true);
    setUploadProgress(0);
    xhr.upload.onprogress = (event) => event.lengthComputable && setUploadProgress(Math.round((event.loaded / event.total) * 100));
    xhr.onload = () => {
      setUploading(false);
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status < 200 || xhr.status >= 300) throw new Error(body.detail || "O vídeo não pôde ser analisado.");
        setUpload(body);
        setAudioTrack(body.audioTracks[0]?.index ?? null);
        setRemoveEmbedded(body.subtitleTracks.length > 0);
      } catch (error) {
        setNotice(error instanceof Error ? error.message : "O vídeo não pôde ser analisado.");
      }
    };
    xhr.onerror = () => { setUploading(false); setNotice("A conexão foi interrompida durante o envio do vídeo."); };
    xhr.send(form);
  }

  function onTextFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".txt") || file.size > 64 * 1024) {
      setNotice("Escolha um arquivo .txt pequeno, com até 5.000 caracteres.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setTextValue(String(reader.result || "").slice(0, 5000));
    reader.readAsText(file, "utf-8");
    event.target.value = "";
  }

  async function previewVoice() {
    if (!textValue.trim()) { setNotice("Digite um texto antes de ouvir a prévia."); return; }
    const previousPreviewAvailable = Boolean(previewObjectUrl.current);
    setPreviewing(true);
    setNotice("");
    setPreviewMessage(previousPreviewAvailable ? "Gerando uma nova prévia…" : "Gerando a prévia…");
    try {
      const response = await fetchWithNetworkRetry("/api/preview", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: textValue.slice(0, 500), voice, gender, provider }) });
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || "A prévia não pôde ser criada."); }
      const audioData = await response.arrayBuffer();
      if (audioData.byteLength < 48) throw new Error("A prévia recebida está vazia ou inválida.");
      const url = URL.createObjectURL(new Blob([audioData], { type: "audio/wav" }));
      clearPreview();
      previewObjectUrl.current = url;
      setPreviewUrl(url);
      setBackendOnline(true);
    } catch (error) {
      const message = isNetworkFetchError(error) ? localServiceError : error instanceof Error ? error.message : "A prévia não pôde ser criada.";
      if (isNetworkFetchError(error)) setBackendOnline(false);
      setNotice(message);
      setPreviewMessage(previousPreviewAvailable ? "A prévia anterior continua disponível; a nova tentativa falhou." : "");
    } finally { setPreviewing(false); }
  }

  async function playPreview() {
    if (!audioRef.current) return;
    try {
      await audioRef.current.play();
      setPreviewMessage("Reproduzindo a prévia gerada.");
    } catch {
      setNotice("O navegador bloqueou a reprodução. Use o botão de play no controle de áudio ou baixe a prévia WAV.");
    }
  }

  function prepareNextVideo() {
    if (jobActive) return;
    setJob(null);
    setVideo(null);
    setUpload(null);
    setAudioTrack(null);
    setScan(null);
    setRegions([]);
    setUploadProgress(0);
    setNotice("");
    window.setTimeout(() => fileInput.current?.click(), 0);
  }

  async function processVideo() {
    if (!canProcess || !upload) return;
    setNotice("");
    try {
      if (removeBurnedIn && burnedInMode === "regions" && scan?.status === "completed") {
        await saveRegions();
      }
      const next = await apiJson<Job>("/api/jobs", { method: "POST", body: JSON.stringify({
        uploadId: upload.uploadId, mode,
        text: usesAudio && !usesOwnAudio ? textValue : "", voice: usesAudio && !usesOwnAudio ? voice : "", gender,
        provider, audioAssetId: usesAudio && usesOwnAudio ? audioAsset?.audioId : null,
        volumePercent: volume, audioStreamIndex: usesAudio ? audioTrack : null,
        subtitleRemoval: { removeEmbedded: usesSubtitles && removeEmbedded, removeBurnedIn: usesSubtitles && removeBurnedIn, burnedInMode, scanId: removeBurnedIn && burnedInMode === "regions" ? scan?.id : null },
      }) });
      setJob(next);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Não foi possível iniciar o processamento.");
    }
  }

  async function installSubtitleModels() {
    setNotice("");
    try {
      const next = await apiJson<ModelStatus>("/api/subtitle-model/install", { method: "POST", body: "{}" });
      setModelStatus(next);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Não foi possível iniciar a instalação da IA.");
    }
  }

  async function importSubtitleModels(file: File) {
    setSubtitleBusy(true);
    setNotice("");
    const form = new FormData();
    form.append("package", file);
    try {
      const response = await fetch("/api/subtitle-model/import", { method: "POST", body: form, credentials: "same-origin" });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || "O pacote de IA não pôde ser importado.");
      setModelStatus(body);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "O pacote de IA não pôde ser importado.");
    } finally {
      setSubtitleBusy(false);
    }
  }

  async function startSubtitleScan() {
    if (!upload) return;
    setNotice("");
    try {
      const next = await apiJson<SubtitleScan>(`/api/uploads/${upload.uploadId}/subtitle-scan`, { method: "POST", body: JSON.stringify({ fullScreen: scanFullScreen }) });
      setScan(next);
      setRegions([]);
      setPreviewStamp(0);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Não foi possível iniciar o exame de legendas.");
    }
  }

  async function saveRegions() {
    if (!scan) return;
    const next = await apiJson<SubtitleScan>(`/api/subtitle-scans/${scan.id}`, { method: "PUT", body: JSON.stringify({ regions }) });
    setScan(next);
    setRegions(next.regions);
  }

  async function makeSubtitlePreview() {
    if (!scan) return;
    setSubtitleBusy(true);
    setNotice("");
    try {
      await saveRegions();
      const response = await fetch(`/api/subtitle-scans/${scan.id}/preview?timeMs=${reviewTimeMs}&side=after&check=${Date.now()}`, { credentials: "same-origin" });
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || "A prévia não pôde ser criada."); }
      const url = URL.createObjectURL(await response.blob());
      if (subtitlePreviewObjectUrl.current) URL.revokeObjectURL(subtitlePreviewObjectUrl.current);
      subtitlePreviewObjectUrl.current = url;
      setSubtitlePreviewUrl(url);
      setPreviewStamp(Date.now());
    } catch (error) {
      const previewError = error instanceof Error ? error.message : "A prévia de remoção não pôde ser criada.";
      setNotice(previewError);
    } finally { setSubtitleBusy(false); }
  }

  function updateRegion(id: string, changes: Partial<SubtitleRegion>) {
    setRegions((current) => current.map((item) => item.id === id ? { ...item, ...changes } : item));
    setPreviewStamp(0);
  }

  function drawPointer(event: ReactPointerEvent<HTMLDivElement>, ending: boolean) {
    if (!upload || !scan || scan.status !== "completed") return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const point = { x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)), y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)) };
    if (!ending) { drawStart.current = point; event.currentTarget.setPointerCapture(event.pointerId); return; }
    const start = drawStart.current;
    drawStart.current = null;
    if (!start) return;
    const x = Math.min(start.x, point.x), y = Math.min(start.y, point.y);
    const width = Math.abs(start.x - point.x), height = Math.abs(start.y - point.y);
    if (width < 0.01 || height < 0.01) return;
    setRegions((current) => [...current, { id: crypto.randomUUID().replace(/-/g, ""), x, y, width, height, startMs: 0, endMs: Math.round(upload.duration * 1000), source: "manual", enabled: true, text: "Área manual", confidence: 1 }]);
  }

  async function saveSettings(event: FormEvent) {
    event.preventDefault();
    setSavingSettings(true);
    setNotice("");
    try {
      const body: Record<string, string> = { outputDirectory };
      if (azureKey.trim()) { body.key = azureKey.trim(); body.region = azureRegion.trim(); }
      await apiJson("/api/config/azure", { method: "POST", body: JSON.stringify(body) });
      setAzureKey("");
      setAzureRegion("");
      setSettingsOpen(false);
      await loadStatus();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Não foi possível salvar as configurações.");
    } finally { setSavingSettings(false); }
  }

  async function removeAzureCredentials() {
    if (!window.confirm("Remover a chave Azure salva neste computador? O Piper local continuará funcionando.")) return;
    setRemovingAzure(true);
    setNotice("");
    try {
      await apiJson("/api/config/azure", { method: "DELETE" });
      setProvider("piper");
      setGender("Male");
      setAzureKey("");
      setAzureRegion("");
      clearPreview();
      await loadStatus();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Não foi possível remover a chave Azure.");
    } finally {
      setRemovingAzure(false);
    }
  }

  async function unlockNetwork(event: FormEvent) {
    event.preventDefault();
    setAuthenticating(true);
    setNotice("");
    try {
      await apiJson("/api/session", { method: "POST", body: JSON.stringify({ pin: networkPin }) });
      setPinOpen(false);
      setNetworkPin("");
      await loadStatus();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "PIN incorreto.");
    } finally { setAuthenticating(false); }
  }

  function chooseProvider(next: Provider) {
    setProvider(next);
    clearPreview();
    setNotice("");
    if (next === "piper") setGender("Male");
  }

  function removeOwnAudio() {
    if (!audioAsset || jobActive || batchActive) return;
    void fetch(`/api/audio/${audioAsset.audioId}`, { method: "DELETE", credentials: "same-origin" });
    setAudioAsset(null);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Ofuscador Elite"><Logo/><span><b>OFUSCADOR</b><em>ELITE</em></span></a>
        <div className="top-actions">
          <a className="colab-cta" href="https://colab.research.google.com/github/DYNHOSANTANA657/Ofuscador-Elite/blob/main/colab/Remover_Legenda_ProPainter_SEGURO.ipynb" target="_blank" rel="noopener noreferrer" title="Abrir o notebook do Colab (GPU grátis) para remover legenda — recomendado para PC sem GPU">☁ Rodar no Colab</a>
          <span className={`connection-pill ${backendOnline ? "online" : "offline"}`}><i />{backendOnline ? "Processamento local" : "Modo de apresentação"}</span>
          <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Abrir configurações">⚙</button>
        </div>
      </header>

      <section className="hero" id="top">
        <div>
          <span className="eyebrow">ESTÚDIO DE ÁUDIO LOCAL</span>
          <h1>Seu vídeo. Sua voz.<br/><span>Controle total.</span></h1>
          <p>Prepare uma faixa estéreo personalizada sem alterar o arquivo original. Todo o processamento acontece neste computador.</p>
        </div>
        <div className="phase-visual" aria-label="Representação do áudio em antifase">
          <div className="wave left"><span/><span/><span/><span/><span/><span/><span/><span/><span/></div>
          <div className="phase-center"><strong>L</strong><i>+</i><small>VOZ IA</small><i>+</i><strong>R</strong></div>
          <div className="wave right"><span/><span/><span/><span/><span/><span/><span/><span/><span/></div>
          <div className="phase-caption"><span>ORIGINAL</span><span>ANTIFASE</span></div>
        </div>
      </section>

      <div className={`diagnostic-banner ${status.piperAvailable ? "local-ready" : "local-error"}`}><span>◈</span><div><b>{status.piperAvailable ? "Piper local gratuito ativo" : "Piper local não encontrado"}</b><small>{status.piperAvailable ? "A fala é gerada neste computador, sem chave, limite mensal ou cobrança. A Azure continua disponível como opção." : "Este pacote não contém todos os componentes da voz local. Reinstale o aplicativo completo."}</small></div><button onClick={() => setSettingsOpen(true)}>Azure opcional</button></div>

      <section className="workspace">
        <div className="step-card video-card">
          <div className="step-heading"><span>01</span><div><h2>Vídeo original</h2><p>MP4 com até 2 GB</p></div></div>
          <input ref={fileInput} type="file" accept="video/mp4,.mp4" hidden multiple={batchAllowed} disabled={jobActive || subtitleScanActive || batchActive} onChange={(event) => { const files = Array.from(event.target.files || []); event.target.value = ""; chooseVideos(files); }}/>
          {video ? (
            <div className="file-panel">
              <div className="file-icon">▶</div><div className="file-data"><b>{video.name}</b><small>{formatBytes(video.size)}{upload ? ` · ${formatTime(upload.duration)}` : ""}</small></div>
              <button disabled={jobActive || subtitleScanActive} onClick={() => fileInput.current?.click()}>{jobActive ? "Em processamento" : subtitleScanActive ? "Analisando" : "Trocar"}</button>
            </div>
          ) : batch.length > 0 ? (
            <div className="file-panel"><div className="file-icon">≡</div><div className="file-data"><b>{batch.length} vídeos na fila</b><small>{batch.filter((item) => item.status === "completed").length} prontos · {batch.filter((item) => item.status === "failed").length} com falha</small></div><button disabled={batchActive} onClick={() => { clearBatch(); window.setTimeout(() => fileInput.current?.click(), 0); }}>{batchActive ? "Processando" : "Nova fila"}</button></div>
          ) : (
            <button className="drop-zone" disabled={batchActive} onClick={() => fileInput.current?.click()}><span className="upload-icon">↑</span><b>Escolher vídeo MP4</b><small>{batchAllowed ? "Pode escolher vários de uma vez · o original é preservado" : "O arquivo original será preservado"}</small></button>
          )}
          {uploading && <div className="progress-wrap"><div><span>Analisando o vídeo…</span><b>{uploadProgress}%</b></div><progress value={uploadProgress} max="100"/></div>}
          {upload && usesAudio && upload.audioTracks.length > 0 && <label className="field-label">Faixa de áudio original<select value={audioTrack ?? ""} onChange={(e) => setAudioTrack(Number(e.target.value))}>{upload.audioTracks.map((track) => <option key={track.index} value={track.index}>Faixa {track.order} · {track.codec.toUpperCase()} · {track.channels === 1 ? "Mono" : `${track.channels} canais`}{track.language !== "und" ? ` · ${track.language}` : ""}{track.title ? ` · ${track.title}` : ""}</option>)}</select></label>}
          {upload && <div className="video-metadata"><span>{upload.resolution.width}×{upload.resolution.height}</span><span>{upload.fps.toFixed(3)} FPS{upload.variableFrameRate ? " · VFR" : ""}</span><span>{upload.videoCodec.toUpperCase()}</span>{upload.hdr && <span className="danger-chip">HDR</span>}</div>}
        </div>

        <div className="step-card mode-card">
          <div className="step-heading"><span>02</span><div><h2>O que deseja processar?</h2><p>As opções podem funcionar separadas ou juntas</p></div></div>
          <div className="mode-toggle">
            <button className={mode === "audio" ? "active" : ""} onClick={() => setMode("audio")}><b>Áudio</b><small>Voz e antifase atuais</small></button>
            <button className={mode === "subtitles" ? "active" : ""} onClick={() => setMode("subtitles")}><b>Legendas</b><small>Preserva o áudio original</small></button>
            <button className={mode === "audio_and_subtitles" ? "active" : ""} onClick={() => setMode("audio_and_subtitles")}><b>Áudio + legendas</b><small>Executa as duas etapas</small></button>
          </div>
        </div>

        {usesSubtitles && upload && <div className="step-card subtitle-card">
          <div className="step-heading"><span>03</span><div><h2>Remoção de legendas</h2><p>Faixas separadas sem perda ou texto gravado com IA local</p></div></div>
          <div className="subtitle-options">
            <label className={upload.subtitleTracks.length ? "option-box" : "option-box disabled"}><input type="checkbox" checked={removeEmbedded} disabled={!upload.subtitleTracks.length} onChange={(event) => setRemoveEmbedded(event.target.checked)}/><span><b>Remover faixas separadas</b><small>{upload.subtitleTracks.length ? `${upload.subtitleTracks.length} faixa(s) detectada(s): ${upload.subtitleTracks.map((track) => `${track.codec.toUpperCase()}${track.language !== "und" ? ` · ${track.language}` : ""}`).join(", ")}. Vídeo e áudio serão copiados sem recodificação quando esta for a única remoção.` : "Nenhuma faixa separada foi encontrada."}</small></span></label>
            <label className="option-box"><input type="checkbox" checked={removeBurnedIn} onChange={(event) => { setRemoveBurnedIn(event.target.checked); setPreviewStamp(0); }}/><span><b>Apagar legenda gravada na imagem</b><small>Reconstrói o fundo com IA local (OCR + LaMa). O vídeo é recodificado em H.264.</small></span></label>
          </div>

          {removeBurnedIn && <div className="ai-removal-panel">
            {upload.hdr ? <div className="inline-warning"><b>HDR bloqueado nesta versão.</b> A remoção gravada não será iniciada para evitar alteração incorreta de cor.</div> : !modelStatus.installed ? <div className="model-installer">
              <div><b>Pacote gratuito de IA necessário</b><small>{modelStatus.error || modelStatus.message}. Ele roda somente neste computador e é instalado uma vez.</small></div>
              {!["downloading", "installing"].includes(modelStatus.status) ? <div className="model-actions"><button onClick={() => void installSubtitleModels()}>Baixar e instalar</button><button className="secondary" onClick={() => modelFileInput.current?.click()}>Importar ZIP</button><input ref={modelFileInput} type="file" hidden accept="application/zip,.zip" onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void importSubtitleModels(file); }}/></div> : <div className="model-progress"><span>{modelStatus.message}</span><b>{Math.round(modelStatus.progress)}%</b><progress value={modelStatus.progress} max="100"/></div>}
            </div> : <>
              <div className="provider-toggle burned-mode-toggle">
                <button type="button" className={burnedInMode === "auto" ? "active" : ""} onClick={() => { setBurnedInMode("auto"); setPreviewStamp(0); }}><b>Automático (recomendado)</b><small>Apaga o texto na tela inteira e protege o rosto — você não desenha nada.</small></button>
                <button type="button" className={burnedInMode === "regions" ? "active" : ""} onClick={() => setBurnedInMode("regions")}><b>Desenhar caixa (avançado)</b><small>Você marca as áreas e revê antes de processar.</small></button>
              </div>
              {burnedInMode === "auto" ? <div className="auto-removal-note">
                <p className="provider-note"><b>Modo automático ligado.</b> É a mesma receita testada no Colab: encontra a legenda de qualquer cor pela forma das letras, apaga no rodapé e no topo e <b>protege o rosto</b>. É só clicar em <b>Processar</b> mais abaixo — sem examinar nem desenhar.</p>
                <p className="provider-note azure-note">Sem uma placa de vídeo forte, o modo automático é <b>lento</b> (analisa quadro a quadro): num vídeo longo pode demorar bastante. O resultado é o mesmo do Colab, só mais devagar.</p>
              </div> : <>
              <div className="scan-toolbar"><label><input type="checkbox" checked={scanFullScreen} onChange={(event) => setScanFullScreen(event.target.checked)}/> Examinar a tela inteira <small>(o padrão examina a parte inferior)</small></label><button disabled={scan?.status === "queued" || scan?.status === "scanning"} onClick={() => void startSubtitleScan()}>{scan?.status === "completed" ? "Examinar novamente" : "Examinar legendas"}</button></div>
              {scan && ["queued", "scanning"].includes(scan.status) && <div className="model-progress"><span>{scan.message}</span><b>{Math.round(scan.progress)}%</b><progress value={scan.progress} max="100"/></div>}
              {scan?.status === "completed" && <div className="review-layout">
                <div className="review-stage">
                  <div className="timeline-row"><label>Quadro em {formatTime(reviewTimeMs / 1000)}</label><input type="range" min="0" max={Math.max(1, Math.round(upload.duration * 1000) - 1)} value={reviewTimeMs} onChange={(event) => { setReviewTimeMs(Number(event.target.value)); setPreviewStamp(0); setSubtitlePreviewUrl(""); }}/></div>
                  {scan.sampleTimesMs.length > 0 && <div className="sample-times">{scan.sampleTimesMs.slice(0, 18).map((time) => <button key={time} onClick={() => { setReviewTimeMs(time); setPreviewStamp(0); setSubtitlePreviewUrl(""); }}>{formatTime(time / 1000)}</button>)}</div>}
                  <p className="draw-help">Arraste sobre a imagem para desenhar uma nova área. As caixas verdes estão ativas neste instante.</p>
                  <div className="frame-editor" onPointerDown={(event) => drawPointer(event, false)} onPointerUp={(event) => drawPointer(event, true)}>
                    <img src={`/api/uploads/${upload.uploadId}/frame?timeMs=${reviewTimeMs}`} alt="Quadro para revisão das legendas" draggable={false}/>
                    <div className="region-overlay">{regions.filter((item) => item.enabled && item.startMs <= reviewTimeMs && item.endMs >= reviewTimeMs).map((item) => <span key={item.id} className={item.source === "manual" ? "manual" : ""} style={{ left: `${item.x * 100}%`, top: `${item.y * 100}%`, width: `${item.width * 100}%`, height: `${item.height * 100}%` }}/>)}</div>
                  </div>
                </div>
                <div className="region-list"><div className="region-list-heading"><b>{regions.length} região(ões)</b><button onClick={() => setRegions((current) => [...current, { id: crypto.randomUUID().replace(/-/g, ""), x: .15, y: .75, width: .7, height: .15, startMs: 0, endMs: Math.round(upload.duration * 1000), source: "manual", enabled: true, text: "Área manual", confidence: 1 }])}>+ Adicionar área</button></div>{regions.length === 0 ? <p>Nenhum texto foi detectado. Você pode desenhar uma área manualmente.</p> : regions.map((item, index) => <div className={`region-item ${item.enabled ? "" : "off"}`} key={item.id}><label><input type="checkbox" checked={item.enabled} onChange={(event) => updateRegion(item.id, { enabled: event.target.checked })}/><b>Área {index + 1}</b><small>{item.source === "manual" ? "Manual" : `${Math.round(item.confidence * 100)}% OCR`}</small></label><p>{item.text || "Texto detectado"}</p><div className="region-times"><label>Início (s)<input type="number" min="0" max={upload.duration} step="0.1" value={(item.startMs / 1000).toFixed(1)} onChange={(event) => updateRegion(item.id, { startMs: Math.max(0, Math.round(Number(event.target.value) * 1000)) })}/></label><label>Fim (s)<input type="number" min="0" max={upload.duration} step="0.1" value={(item.endMs / 1000).toFixed(1)} onChange={(event) => updateRegion(item.id, { endMs: Math.min(Math.round(upload.duration * 1000), Math.round(Number(event.target.value) * 1000)) })}/></label></div><button className="remove-region" onClick={() => setRegions((current) => current.filter((region) => region.id !== item.id))}>Excluir área</button></div>)}</div>
                <div className="preview-removal"><button disabled={subtitleBusy || !regions.some((item) => item.enabled && item.startMs <= reviewTimeMs && item.endMs >= reviewTimeMs)} onClick={() => void makeSubtitlePreview()}>{subtitleBusy ? "Reconstruindo…" : "Gerar prévia antes e depois"}</button>{previewStamp > 0 && subtitlePreviewUrl && <div><figure><img src={`/api/uploads/${upload.uploadId}/frame?timeMs=${reviewTimeMs}`} alt="Antes da remoção"/><figcaption>Antes</figcaption></figure><figure><img src={subtitlePreviewUrl} alt="Depois da remoção"/><figcaption>Depois — estimativa da IA</figcaption></figure></div>}</div>
              </div>}
              </>}
            </>}
            <div className="inline-warning"><b>Limite da reconstrução:</b> a IA estima o fundo que estava escondido. Ela não recupera exatamente pixels que não existem nos quadros originais e nunca troca uma falha por borrão ou corte.</div>
          </div>}
        </div>}

        {usesAudio && !usesOwnAudio && <div className="step-card text-card">
          <div className="step-heading"><span>02</span><div><h2>Texto para a voz</h2><p>Até 5.000 caracteres</p></div><label className="import-button">Importar .txt<input type="file" accept="text/plain,.txt" hidden onChange={onTextFile}/></label></div>
          <textarea value={textValue} maxLength={5000} onChange={(event) => setTextValue(event.target.value)} placeholder="Digite ou cole aqui o texto que será transformado em fala…"/>
          <div className="text-footer"><span>O texto será repetido até o vídeo terminar</span><b className={textValue.length >= 4900 ? "limit" : ""}>{textValue.length.toLocaleString("pt-BR")} / 5.000</b></div>
        </div>}

        {usesAudio && <div className="step-card voice-card">
          <div className="step-heading"><span>{usesOwnAudio ? "02" : "03"}</span><div><h2>{usesOwnAudio ? "Áudio e intensidade" : "Voz e intensidade"}</h2><p>{usesOwnAudio ? "Escolha o arquivo que será misturado" : "Escolha como a voz será aplicada"}</p></div></div>
          <div className="voice-controls">
            <label className="control-label">Origem do áudio</label>
            <div className="provider-toggle"><button className={provider === "piper" ? "active" : ""} onClick={() => chooseProvider("piper")}><b>Piper local</b><small>Grátis e sem internet</small></button><button className={provider === "azure" ? "active" : ""} onClick={() => chooseProvider("azure")}><b>Azure Speech</b><small>Opcional e mais natural</small></button><button className={provider === "file" ? "active" : ""} onClick={() => chooseProvider("file")}><b>Meu áudio</b><small>MP3 ou WAV do seu computador</small></button></div>

            {usesOwnAudio ? <div className="own-audio-panel">
              <input ref={audioFileInput} type="file" hidden accept="audio/*,.mp3,.wav,.m4a,.aac,.ogg,.flac" onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void sendAudio(file); }}/>
              {!audioAsset ? (
                <button className="drop-zone audio-drop" disabled={audioUploading} onClick={() => audioFileInput.current?.click()}><span className="upload-icon">♪</span><b>{audioUploading ? "Analisando o áudio…" : "Escolher arquivo de áudio"}</b><small>MP3, WAV, M4A, AAC, OGG ou FLAC · até 300 MB</small></button>
              ) : (
                <div className="file-panel"><div className="file-icon">♪</div><div className="file-data"><b>{audioAsset.filename}</b><small>{formatBytes(audioAsset.size)} · {formatTime(audioAsset.duration)} · {audioAsset.codec.toUpperCase()}</small></div><button disabled={audioUploading || jobActive || batchActive} onClick={() => audioFileInput.current?.click()}>Trocar</button><button className="secondary" disabled={audioUploading || jobActive || batchActive} onClick={removeOwnAudio}>Remover</button></div>
              )}
              <p className="provider-note">O áudio é repetido até o vídeo terminar e entra no mesmo volume ajustado abaixo. O áudio original do vídeo continua com a antifase.</p>
            </div> : <>
            <label className="control-label">Tipo de voz</label>
            <div className="gender-toggle"><button disabled={provider === "piper"} className={gender === "Female" ? "active" : ""} onClick={() => setGender("Female")} title={provider === "piper" ? "A voz pt-BR local licenciada deste pacote é masculina." : "Voz feminina"}><i>♀</i> Feminina</button><button className={gender === "Male" ? "active" : ""} onClick={() => setGender("Male")}><i>♂</i> Masculina</button></div>
            {provider === "piper" && <p className="provider-note">A voz pt-BR local incluída neste pacote é masculina. Para uma voz feminina, selecione Azure Speech.</p>}
            {provider === "azure" && !status.azureConfigured && <p className="provider-note azure-note">Cadastre uma chave Azure na engrenagem para carregar as vozes online.</p>}
            {voiceError && <p className="provider-note voice-error" role="alert">{voiceError}</p>}
            <label className="field-label">Voz em português do Brasil<select value={voice} disabled={!voices.length} onChange={(event) => setVoice(event.target.value)}>{voices.length ? voices.map((item) => <option value={item.shortName} key={item.shortName}>{item.displayName}</option>) : <option value="">Nenhuma voz disponível</option>}</select></label>
            <div className="preview-row"><button className="preview-button" onClick={previewVoice} disabled={previewing || !textValue.trim() || !voice}>{previewing ? "Criando prévia…" : "▶  Ouvir prévia"}</button><span>{selectedVoice?.provider === "piper" ? "Piper local · sem custo" : "Azure Speech · online"}</span></div>
            {previewUrl && <div className="preview-player"><audio ref={audioRef} src={previewUrl} controls preload="auto" className="audio-player" onPlay={() => setPreviewMessage("Reproduzindo a prévia gerada.")} onEnded={() => setPreviewMessage("Prévia concluída. Você pode reproduzi-la novamente.")} onError={() => setNotice("O navegador não conseguiu abrir a prévia WAV. Use a opção Baixar prévia.")}/><div className="preview-actions"><button type="button" onClick={() => void playPreview()}>▶ Reproduzir</button><a href={previewUrl} download="previa-ofuscador-elite.wav">↓ Baixar prévia WAV</a></div><small className="preview-status" aria-live="polite">{previewMessage}</small></div>}
            </>}
          </div>
          <div className="volume-block"><div><label htmlFor="volume">{usesOwnAudio ? "Volume do áudio enviado" : "Volume da voz gerada"}</label><output>{volume}%</output></div><input id="volume" type="range" min="0" max="25" value={volume} onChange={(event) => setVolume(Number(event.target.value))}/><div className="range-labels"><span>Discreto</span><span>Máximo seguro</span></div></div>
        </div>}
      </section>

      {notice && <div className="notice" role="alert"><span>!</span><p>{notice}</p><button onClick={() => setNotice("")}>×</button></div>}

      {batch.length > 0 && <section className="batch-panel">
        <div className="batch-heading">
          <div><b>Fila de processamento</b><small>Um vídeo por vez, com o mesmo ajuste de áudio. Os originais não são alterados.</small></div>
          <button disabled={batchActive} onClick={clearBatch}>{batchActive ? `${batch.filter((item) => ["completed", "failed"].includes(item.status)).length} de ${batch.length}` : "Limpar fila"}</button>
        </div>
        <ul className="batch-list">
          {batch.map((item) => (
            <li key={item.key} className={`batch-item ${item.status}`}>
              <div className="batch-name"><b>{item.name}</b><small>{formatBytes(item.size)}</small></div>
              <div className="batch-state">
                <span>{item.status === "completed" ? "Pronto" : item.status === "failed" ? "Falhou" : item.status === "enviando" ? "Enviando o vídeo" : item.message}</span>
                <progress value={item.status === "completed" ? 100 : item.progress} max="100"/>
              </div>
              {item.status === "completed" && item.jobId
                ? <a className="batch-action" href={`/api/jobs/${item.jobId}/download`}>↓ Baixar</a>
                : <span className="batch-action muted">{item.status === "failed" ? "—" : `${Math.round(item.progress)}%`}</span>}
              {item.error && <small className="batch-error">{item.error}</small>}
              {item.warning && <small className="batch-warning">{item.warning}</small>}
            </li>
          ))}
        </ul>
        <p className="destination">Os resultados são salvos em <b>{status.outputDirectory}</b></p>
      </section>}

      {batch.length === 0 && <section className="action-panel">
        {usesAudio ? <div className="signal-summary"><div><span className="channel l">L</span><p><b>Original + voz</b><small>Canal esquerdo</small></p></div><i>→</i><div><span className="channel r">R</span><p><b>Original invertido + voz</b><small>Canal direito</small></p></div></div> : <div className="signal-summary subtitle-summary"><div><span className="channel l">CC</span><p><b>Legendas removidas</b><small>Todas as faixas de áudio originais serão preservadas</small></p></div></div>}
        {job && <div className={`job-progress ${job.status}`}><div><span>{job.status === "completed" ? "Vídeo pronto" : job.status === "failed" ? "O processamento parou" : job.message}</span><b>{Math.round(job.progress)}%</b></div><progress value={job.progress} max="100"/>{job.error && <small>{job.error}</small>}{job.warning && <small className="job-warning">{job.warning}</small>}</div>}
        {job?.status === "completed" ? <div className="completion-actions"><a className="primary-action ready" href={`/api/jobs/${job.id}/download`}>↓ Baixar {job.outputName || "vídeo processado"}</a><button type="button" onClick={prepareNextVideo}>Processar outro vídeo</button></div> : <button className="primary-action" disabled={!canProcess} onClick={processVideo}>{jobActive ? "Processando…" : "Processar vídeo"}<span>→</span></button>}
        <p className="destination">O resultado será salvo em <b>{status.outputDirectory}</b></p>
      </section>}

      <section className="warning-card"><span>i</span><div><b>{usesAudio ? "Importante sobre a antifase" : "Importante sobre a remoção"}</b><p>{usesAudio ? "O efeito pode se cancelar quando o áudio é convertido para mono e pode mudar após uma nova compressão. Ele não garante como serviços de transcrição ou plataformas interpretarão o conteúdo." : "Faixas separadas são removidas sem perda. Para legendas gravadas, a IA reconstrói uma estimativa do fundo e o vídeo precisa ser recodificado; o arquivo original permanece intacto."}</p></div></section>

      <footer><div><Logo/><span><b>OFUSCADOR ELITE</b><small>Processamento privado, no seu computador</small></span></div><p>O vídeo original nunca é modificado.</p></footer>

      {settingsOpen && <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setSettingsOpen(false)}><form className="settings-modal" onSubmit={saveSettings}><div className="modal-heading"><div><span>CONFIGURAÇÕES</span><h2>Azure opcional e destino</h2></div><button type="button" onClick={() => setSettingsOpen(false)}>×</button></div><p>O Piper local não precisa de configuração. Preencha a Azure somente se quiser usar as vozes online mais naturais; a chave fica protegida no cofre deste computador.</p><label>Chave Azure Speech<input type="password" value={azureKey} onChange={(event) => setAzureKey(event.target.value)} placeholder={status.credentialConfigured ? "Chave já cadastrada — preencha para trocar" : "Opcional — cole sua chave"}/></label><label>Região Azure<input value={azureRegion} onChange={(event) => setAzureRegion(event.target.value)} placeholder="Ex.: brazilsouth"/></label><label>Pasta de saída<input value={outputDirectory} onChange={(event) => setOutputDirectory(event.target.value)} placeholder="Caminho da pasta Saidas"/></label><div className="security-note">🔒 O Piper funciona offline. A chave Azure só pode ser alterada diretamente neste computador.{status.credentialConfigured && <button className="remove-credential" type="button" disabled={removingAzure} onClick={() => void removeAzureCredentials()}>{removingAzure ? "Removendo…" : `Remover chave Azure${status.azureRegion ? ` (${status.azureRegion})` : ""}`}</button>}</div><div className="modal-actions"><button type="button" onClick={() => setSettingsOpen(false)}>Cancelar</button><button type="submit" disabled={savingSettings}>{savingSettings ? "Verificando…" : "Salvar configurações"}</button></div></form></div>}
      {pinOpen && <div className="modal-backdrop"><form className="settings-modal pin-modal" onSubmit={unlockNetwork}><div className="modal-heading"><div><span>ACESSO PELA REDE</span><h2>Digite o PIN temporário</h2></div></div><p>Veja o código de seis dígitos na janela do Ofuscador Elite no computador principal.</p><label>PIN<input autoFocus inputMode="numeric" pattern="[0-9]{6}" maxLength={6} value={networkPin} onChange={(event) => setNetworkPin(event.target.value.replace(/\D/g, ""))} placeholder="000000"/></label><div className="modal-actions"><button type="submit" disabled={authenticating || networkPin.length !== 6}>{authenticating ? "Verificando…" : "Entrar"}</button></div></form></div>}
    </main>
  );
}
