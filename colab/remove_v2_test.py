# remove_v2_test.py - VALIDACAO RAPIDA (nao processa tudo):
# acha as FAIXAS (largura total) e testa o inpaint em 3 quadros do FIM,
# mostrando ANTES/DEPOIS da faixa da legenda (pra ver se as PONTAS somem).
# Estrutura = v2.0 que funcionou: OCR no quadro inteiro + LaMa no quadro inteiro.

import os, glob, cv2, numpy as np, urllib.request, torch
from google.colab.patches import cv2_imshow

PERSIST  = 0.12
DIL      = 5
MIN_AREA = 40
N_SAMP   = 70
MARGEM   = 14
SEG_TESTE = 15     # olha as faixas dentro dos primeiros 15s (onde vimos as pontas)
MODEL_URL = 'https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt'

cands = [x for x in glob.glob('*.mp4') if 'sem_legenda' not in x]
VID = cands[0]; print('video:', VID)

if not os.path.exists('big-lama.pt'):
    urllib.request.urlretrieve(MODEL_URL, 'big-lama.pt')
dev  = 'cuda' if torch.cuda.is_available() else 'cpu'
lama = torch.jit.load('big-lama.pt', map_location=dev).eval()
print('LaMa', dev, flush=True)

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

cap = cv2.VideoCapture(VID)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(3)); H = int(cap.get(4))
NN = min(total, int(SEG_TESTE * fps))
print(f'{total} quadros; janela de teste ate {NN}; {W}x{H} @ {fps:.1f}', flush=True)

# PASSA 1: frequencia por LINHA -> faixas de LARGURA TOTAL
rowhits = np.zeros(H, np.float32)
samp = np.unique(np.linspace(0, NN - 1, min(N_SAMP, NN)).astype(int))
for i in samp:
    cap.set(1, int(i)); ok, fr = cap.read()
    if not ok: continue
    m = np.zeros((H, W), np.uint8)
    for b in boxes(fr): cv2.fillPoly(m, [b.astype(np.int32)], 255)
    rowhits += (m.max(1) > 0)
rowfreq = rowhits / len(samp)
rowfreq[int(0.28 * H):int(0.54 * H)] = 0
on = rowfreq >= PERSIST
# faixas contiguas
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
print('FAIXAS (linhas):', faixas, flush=True)
# zona de LARGURA TOTAL nas linhas das faixas
zb = np.zeros((H, W), bool)
for (a, b) in faixas: zb[a:b, :] = True
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DIL * 2 + 1, DIL * 2 + 1))

# TESTE em 3 quadros do FIM da janela (onde vimos "er...ju")
for fidx in [NN - 45, NN - 22, NN - 4]:
    cap.set(1, int(fidx)); ok, fr = cap.read()
    if not ok: continue
    orig = fr.copy()
    mm = np.zeros((H, W), np.uint8)
    for b in boxes(fr): cv2.fillPoly(mm, [b.astype(np.int32)], 255)
    mm = cv2.dilate(mm, ker); mm[~zb] = 0
    out = inpaint(fr, mm) if mm.any() else fr
    a, b = faixas[-1]
    yA, yB = max(0, a - 12), min(H, b + 12)
    print(f'===== quadro {fidx}  ANTES ====='); cv2_imshow(orig[yA:yB])
    print(f'===== quadro {fidx}  DEPOIS (v2.2) ====='); cv2_imshow(out[yA:yB])
cap.release()
print('TESTE pronto. Se as pontas sumiram no DEPOIS, a correcao funciona.', flush=True)
