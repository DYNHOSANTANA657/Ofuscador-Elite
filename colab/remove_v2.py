# remove_v2.py - Remove legenda/tarja QUEIMADA com LaMa (inpaint por quadro). v2.2
# Roda no runtime QUENTE do Colab (video ja enviado, rapidocr ja instalado).
#
# ESTRUTURA que funciona (v2.0): OCR no quadro INTEIRO + LaMa no quadro INTEIRO.
# CORRECAO (v2.2): a ZONA e por LINHA e de LARGURA TOTAL -> pega as PONTAS das
#                  legendas compridas. Mira as DUAS faixas (topo=tarja, baixo=legenda),
#                  exclui o MEIO (rosto). LaMa apaga pelo redor -> sem borrao preto.

import os, glob, cv2, numpy as np, urllib.request, torch
from google.colab.patches import cv2_imshow

# ---------------- parametros ----------------
MAX_SEGUNDOS = 15      # 0 = video INTEIRO
OCR_CADA     = 1       # 1 = OCR todo quadro (mais preciso). 2/3 = mais rapido (reusa a mascara)
PERSIST      = 0.12
DIL          = 5
MIN_AREA     = 40
N_SAMP       = 70
MARGEM       = 14
MODEL_URL    = 'https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt'

# ---------------- 1) video ----------------
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

# ---------------- 3) detector de texto ----------------
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
            if b.shape == (4, 2) and cv2.contourArea(b) >= MIN_AREA:
                bs.append(b)
    return bs

# ---------------- 4) info do video ----------------
cap   = cv2.VideoCapture(VID)
fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(3)); H = int(cap.get(4))
N = total if MAX_SEGUNDOS == 0 else min(total, int(MAX_SEGUNDOS * fps))
print(f'{total} quadros no total; processando {N}; {W}x{H} @ {fps:.1f}', flush=True)

# ---------------- 5) PASSA 1: FAIXAS de texto fixo (por LINHA, largura total) ----------------
rowhits = np.zeros(H, np.float32)
samp = np.unique(np.linspace(0, N - 1, min(N_SAMP, N)).astype(int))
last = None
for i in samp:
    cap.set(1, int(i)); ok, fr = cap.read()
    if not ok: continue
    last = fr
    m = np.zeros((H, W), np.uint8)
    for b in boxes(fr): cv2.fillPoly(m, [b.astype(np.int32)], 255)
    rowhits += (m.max(1) > 0)
rowfreq = rowhits / len(samp)
rowfreq[int(0.28 * H):int(0.54 * H)] = 0                 # exclui o MEIO (rosto)
on = rowfreq >= PERSIST
faixas = []; y = 0
while y < H:
    if on[y]:
        y0 = y
        while y < H and on[y]: y += 1
        faixas.append([max(0, y0 - MARGEM), min(H, y + MARGEM)])
    else:
        y += 1
merged = []
for f in faixas:
    if merged and f[0] - merged[-1][1] <= 12: merged[-1][1] = f[1]
    else: merged.append(f)
faixas = [tuple(f) for f in merged]
print('FAIXAS de texto fixo (linhas):', faixas, flush=True)
if not faixas:
    raise SystemExit('Nenhuma faixa de texto fixo encontrada (baixe o PERSIST).')
zb = np.zeros((H, W), bool)
for (a, b) in faixas: zb[a:b, :] = True                   # LARGURA TOTAL nas linhas do texto
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DIL * 2 + 1, DIL * 2 + 1))

if last is not None:
    ov = last.copy()
    for (a, b) in faixas: ov[a:b] = (0.4 * ov[a:b] + 0.6 * np.array([0, 0, 255])).astype('uint8')
    print('PREVIA das FAIXAS (vermelho):'); cv2_imshow(ov)

# ---------------- 6) PASSA 2: OCR (quadro inteiro) + LaMa (quadro inteiro) ----------------
os.makedirs('v2out', exist_ok=True)
cap.set(1, 0); feito = 0; prev = None
for i in range(N):
    ok, fr = cap.read()
    if not ok: break
    if (i % OCR_CADA == 0) or (prev is None):
        mm = np.zeros((H, W), np.uint8)
        for b in boxes(fr): cv2.fillPoly(mm, [b.astype(np.int32)], 255)
        mm = cv2.dilate(mm, ker); mm[~zb] = 0
        prev = mm
    else:
        mm = prev
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
