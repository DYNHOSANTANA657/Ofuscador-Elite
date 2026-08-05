# remove_v2.py - Remove legenda/tarja QUEIMADA com LaMa (inpaint por quadro). v2.3
# Roda no runtime do Colab (auto-baixa LaMa, auto-instala rapidocr, auto-pede o video).
#
# v2.3 - DETECTA LEGENDA EM QUALQUER POSICAO (cima / meio / baixo / bem embaixo).
#   Nao usa mais faixa fixa. Como nao da p/ "apagar todo texto" (apagaria a BIBLIA),
#   duas travas separam legenda de Biblia/fundo:
#     (1) TAMANHO  - so conta caixa de texto ALTA (legenda tem letra grande;
#                    a letrinha da Biblia e pequena -> fica protegida).      [MIN_TEXT_H]
#     (2) PERSISTENCIA - so apaga onde texto GRANDE aparece com frequencia no
#                    MESMO lugar (legenda fica parada; coisa passageira nao).  [PERSIST]
#   + UNIAO TEMPORAL causal: quando detecta o texto, a mascara "gruda" nos
#     quadros seguintes -> mata o "pisca" do karaoke (branco sobre camisa branca).

import os, glob, cv2, numpy as np, urllib.request, torch
from collections import deque
from google.colab.patches import cv2_imshow

# ---------------- parametros ----------------
MAX_SEGUNDOS = 0        # 0 = video INTEIRO
OCR_CADA     = 2        # OCR a cada N quadros (a uniao temporal cobre os pulados)
PERSIST      = 0.03     # fracao dos quadros amostrados p/ um ponto virar "zona de legenda" (baixo = pega legenda curta, ex. rodape em tela dividida)
MIN_TEXT_H   = 22       # ALTURA minima da caixa de texto, em px. PROTEGE a Biblia (letra pequena)
DIL          = 6        # dilatacao da mascara (engorda o texto p/ nao sobrar borda)
TEMPORAL_W   = 5        # quantas deteccoes recentes de OCR unir (mata o pisca do karaoke)
MIN_AREA     = 60
N_SAMP       = 160      # amostras na PASSA 1 (denso p/ achar legenda em QUALQUER posicao)
MARGEM       = 10       # engorda a zona detectada
MODEL_URL    = 'https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt'

# ---------------- 1) video ----------------
cands = [x for x in glob.glob('*.mp4') if 'sem_legenda' not in x]
if not cands:                                            # VM nova/reciclada: pede o upload
    print('>>> Nenhum .mp4 no /content. Clique em "Escolher arquivos" e envie o video <<<', flush=True)
    try:
        from google.colab import files; files.upload()
    except Exception as e:
        print('upload falhou:', e, flush=True)
    cands = [x for x in glob.glob('*.mp4') if 'sem_legenda' not in x]
assert cands, 'ERRO: nao achei o .mp4 original (reenvie o video).'
VID = cands[0]; print('video original:', VID)

# ---------------- 2) LaMa torchscript ----------------
if not os.path.exists('big-lama.pt'):
    print('baixando LaMa (~200MB)...', flush=True)
    urllib.request.urlretrieve(MODEL_URL, 'big-lama.pt')
dev  = 'cuda' if torch.cuda.is_available() else 'cpu'
lama = torch.jit.load('big-lama.pt', map_location=dev).eval()
print('LaMa pronto em', dev, flush=True)

def inpaint(bgr, mask):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if ph or pw:
        rgb  = np.pad(rgb,  ((0, ph), (0, pw), (0, 0)), 'symmetric')
        mask = np.pad(mask, ((0, ph), (0, pw)), 'constant')
    it = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).div(255).to(dev)
    mt = torch.from_numpy((mask > 0).astype('float32'))[None, None].to(dev)
    with torch.inference_mode():
        out = lama(it, mt)
    res = out[0].permute(1, 2, 0).detach().cpu().numpy()
    res = res * 255.0 if res.max() <= 1.5 else res
    return cv2.cvtColor(np.clip(res, 0, 255).astype('uint8')[:h, :w], cv2.COLOR_RGB2BGR)

# ---------------- 3) detector de texto (com trava de TAMANHO) ----------------
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    print('instalando rapidocr_onnxruntime (VM nova)...', flush=True)
    os.system('pip install -q rapidocr_onnxruntime')
    from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()

def boxes(img):
    out = ocr(img, use_det=True, use_rec=False, use_cls=False)
    res = out[0] if isinstance(out, tuple) else out
    bs = []
    if res:
        for it in res:
            a = np.asarray(it, dtype=np.float32)
            b = a if (a.ndim == 2 and a.shape == (4, 2)) else np.asarray(it[0], dtype=np.float32)
            if b.shape != (4, 2):                 continue
            if cv2.contourArea(b) < MIN_AREA:      continue
            if (b[:, 1].max() - b[:, 1].min()) < MIN_TEXT_H: continue   # trava: texto PEQUENO (Biblia) fica de fora
            bs.append(b)
    return bs

# ---------------- 4) info do video ----------------
cap   = cv2.VideoCapture(VID)
fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(3)); H = int(cap.get(4))
N = total if MAX_SEGUNDOS == 0 else min(total, int(MAX_SEGUNDOS * fps))
print(f'{total} quadros no total; processando {N}; {W}x{H} @ {fps:.1f}', flush=True)

# ---------------- 5) PASSA 1: ZONA de legenda (TELA TODA) + PROTECAO do ROSTO ----------
# texto GRANDE frequente em QUALQUER posicao = zona de legenda. E acha o ROSTO (Haar)
# p/ NUNCA apagar olhos/oculos (senao o detector confunde a armacao dos oculos c/ "texto").
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
HAS_FACE = not face_cascade.empty()
heat = np.zeros((H, W), np.float32)
facehits = np.zeros((H, W), np.float32)
samp = np.unique(np.linspace(0, N - 1, min(N_SAMP, N)).astype(int))
last = None
for i in samp:
    cap.set(1, int(i)); ok, fr = cap.read()
    if not ok: continue
    last = fr
    m = np.zeros((H, W), np.uint8)
    for b in boxes(fr): cv2.fillPoly(m, [b.astype(np.int32)], 255)
    heat += (m > 0)
    if HAS_FACE:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(70, 70)):
            y0 = max(0, y - int(0.15 * h)); y1 = min(H, y + int(1.05 * h))   # so ate o queixo
            x0 = max(0, x - int(0.20 * w)); x1 = min(W, x + int(1.20 * w))
            facehits[y0:y1, x0:x1] += 1
heat /= len(samp); facefreq = facehits / len(samp)
if not HAS_FACE:
    print('AVISO: sem detector de rosto -> protegendo o MEIO por seguranca', flush=True)
    facefreq[int(0.18 * H):int(0.58 * H), :] = 1.0
# LINHAS onde texto GRANDE aparece com frequencia -> FAIXA de LARGURA TOTAL (de um lado ao
# outro), garantindo que as PONTAS da legenda nao escapem. Melhor que uma zona so no formato.
rowscore = heat.max(axis=1)
onrows = np.convolve((rowscore >= PERSIST).astype(np.uint8),
                     np.ones(MARGEM * 2 + 1, np.uint8), mode='same') > 0
zb = np.zeros((H, W), bool)
zb[onrows, :] = True                                    # LARGURA TOTAL nas linhas do texto
face_zone = cv2.dilate((facefreq >= 0.10).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
zb[face_zone] = False                                   # NUNCA apaga o rosto (olhos/oculos)
if not zb.any():
    raise SystemExit('Nenhuma zona de legenda encontrada (baixe PERSIST ou MIN_TEXT_H).')
print(f'ZONA: {int(zb.sum())} px ({100.0 * zb.sum() / (H * W):.1f}%) | rosto protegido: {int(face_zone.sum())} px', flush=True)
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DIL * 2 + 1, DIL * 2 + 1))

if last is not None:
    ov = last.copy()
    ov[zb] = (0.4 * ov[zb].astype(np.float32) + 0.6 * np.array([0, 0, 255], np.float32)).astype('uint8')
    print('PREVIA da ZONA (vermelho = vai apagar). ROSTO e BIBLIA devem ficar FORA do vermelho:')
    cv2_imshow(ov)

# ---------------- 6) PASSA 2: OCR (tela toda) + uniao temporal + LaMa ----------------
os.makedirs('v2out', exist_ok=True)
cap.set(1, 0); feito = 0
hist = deque(maxlen=TEMPORAL_W)                          # ultimas deteccoes (listas de caixas)
for i in range(N):
    ok, fr = cap.read()
    if not ok: break
    if i % OCR_CADA == 0:
        hist.append(boxes(fr))                          # detecta na TELA TODA
    mm = np.zeros((H, W), np.uint8)
    for bs in hist:                                     # une as deteccoes recentes (mata o pisca)
        for b in bs: cv2.fillPoly(mm, [b.astype(np.int32)], 255)
    if mm.any():
        mm = cv2.dilate(mm, ker); mm[~zb] = 0           # so apaga DENTRO da zona de legenda
        if mm.any():
            fr = inpaint(fr, mm)
    cv2.imwrite('v2out/%05d.png' % i, fr); feito += 1
    if (i + 1) % 50 == 0 or i == N - 1:
        print('  quadro', i + 1, '/', N, flush=True)
cap.release()
print('inpaint pronto:', feito, 'quadros', flush=True)

# ---------------- 7) remonta video + audio ----------------
SAI = 'video_sem_legenda_v2.mp4'
os.system(f'ffmpeg -y -loglevel error -framerate {fps} -i v2out/%05d.png -i "{VID}" '
          f'-map 0:v -map 1:a? -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest "{SAI}"')
print('PRONTO ->', SAI, os.path.getsize(SAI), 'bytes', flush=True)

# ---------------- 8) VERIFICA o video TODO ----------------
capf = cv2.VideoCapture(SAI); nn = int(capf.get(cv2.CAP_PROP_FRAME_COUNT))
for t in [0.02, 0.35, 0.70, 0.99]:
    capf.set(1, int(nn * t)); ok, vf = capf.read()
    if ok:
        print(f'==== VERIFICACAO t={t}  (quadro {int(nn*t)}/{nn}) ===='); cv2_imshow(vf)
capf.release()

# ---------------- 9) baixa + faxina ----------------
from google.colab import files; files.download(SAI)
import shutil; shutil.rmtree('v2out', ignore_errors=True)
print('faxina: apaguei os quadros intermediarios; mantive so o video v2', flush=True)
