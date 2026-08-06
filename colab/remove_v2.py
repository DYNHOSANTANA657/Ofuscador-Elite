# remove_v2.py - Remove legenda/tarja QUEIMADA com LaMa (inpaint por quadro). v2.5
# Roda no runtime do Colab (auto-baixa LaMa, auto-instala rapidocr, auto-pede o video).
#
# v2.5 - CONSERTO do bug de VIDEO MULTI-CENA (varias pessoas / TELA DIVIDIDA).
#   A v2.4.x criava um "buraco FIXO do rosto" = uniao das posicoes de rosto do video INTEIRO
#   (amostrando ~200 quadros). Em video de UMA cena isso funciona; mas em video de VARIAS cenas
#   (uma pessoa numa hora, outra noutra, e TELA DIVIDIDA com 2 pessoas) essas posicoes se SOMAM e
#   viram uma TARJA no MEIO da tela (chegava a ~48% da area). Legenda que cai nessa faixa (ex.:
#   a legenda no meio da tela dividida, ou a legenda no peito) ficava DENTRO da zona protegida ->
#   NAO era apagada. Esse era o motivo de "a legenda nao sai quando tem 2 pessoas / tela dividida".
#   CONSERTO: acabou o buraco FIXO (e a PASSA 1 de amostragem). A protecao de rosto agora e SO por
#   quadro (na PASSA 2), onde o rosto realmente esta -> legenda em QUALQUER lugar/altura pode ser
#   apagada e o rosto/oculos seguem protegidos (validado em cena de mulher, homem e tela dividida).
#
# v2.4.1 / v2.4 - (historico) zona = tela toda menos um buraco FIXO do rosto. Substituido pela v2.5.
#   Travas que permanecem: (1) TAMANHO (MIN_TEXT_H) protege a letra pequena (Biblia); (2) ROSTO
#   detectado em CADA quadro (Haar) -> nunca apaga olhos/oculos. + OCR em TODOS os quadros
#   (OCR_CADA=1) e UNIAO TEMPORAL curta (tira o pisca do karaoke).

import os, glob, cv2, numpy as np, urllib.request, torch
from collections import deque
from google.colab.patches import cv2_imshow

# ---------------- parametros ----------------
MAX_SEGUNDOS = 0        # 0 = video INTEIRO
OCR_CADA     = 1        # OCR a cada N quadros. 1 = TODOS (pega o flash de 1 quadro). Mais lento.
MIN_TEXT_H   = 22       # ALTURA minima da caixa de texto, em px. PROTEGE a Biblia (letra pequena)
DIL          = 6        # dilatacao da mascara (engorda o texto p/ nao sobrar borda)
TEMPORAL_W   = 3        # quantas deteccoes recentes de OCR unir (tira o pisca; curto p/ nao arrastar)
MIN_AREA     = 60
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

# ---------------- 5) rosto: protecao SO por quadro (SEM buraco FIXO) ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
HAS_FACE = not face_cascade.empty()

def face_rects(bgr):
    if not HAS_FACE: return []
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(g, 1.2, 5, minSize=(90, 90))   # rosto grande (menos falso-positivo)

def stamp_face(dst, rects):                             # marca rosto: do cabelo ate o queixo
    for (x, y, w, h) in rects:
        y0 = max(0, y - int(0.20 * h)); y1 = min(H, y + int(1.00 * h))
        x0 = max(0, x - int(0.20 * w)); x1 = min(W, x + int(1.20 * w))
        dst[y0:y1, x0:x1] += 1

ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DIL * 2 + 1, DIL * 2 + 1))
if not HAS_FACE:
    print('AVISO: sem detector de rosto (Haar) -> o rosto NAO sera protegido.', flush=True)

# PREVIA (checkpoint): mostra o rosto detectado num quadro do meio. Agora NAO ha mais tarja fixa
# no meio -> legenda em qualquer altura sai; a protecao (verde) ACOMPANHA o rosto quadro a quadro.
cap.set(1, N // 2); okp, midf = cap.read()
if okp:
    ov = midf.copy(); fm0 = np.zeros((H, W), np.uint8); stamp_face(fm0, face_rects(midf))
    ov[fm0 > 0] = (0.4 * ov[fm0 > 0].astype(np.float32) + 0.6 * np.array([0, 255, 0], np.float32)).astype('uint8')
    print('PREVIA: VERDE = rosto protegido NESTE quadro (a protecao acompanha o rosto quadro a')
    print('quadro). Toda legenda FORA do verde sera apagada, em QUALQUER parte da tela:')
    cv2_imshow(ov)

# ---------------- 6) PASSA 2: OCR (tela toda) + uniao temporal + protege rosto por quadro + LaMa ----
os.makedirs('v2out', exist_ok=True)
cap.set(1, 0); feito = 0
hist = deque(maxlen=TEMPORAL_W)                          # ultimas deteccoes (listas de caixas)
for i in range(N):
    ok, fr = cap.read()
    if not ok: break
    if i % OCR_CADA == 0:
        hist.append(boxes(fr))                          # detecta na TELA TODA (qualquer cor)
    mm = np.zeros((H, W), np.uint8)
    for bs in hist:                                     # une as deteccoes recentes (tira o pisca)
        for b in bs: cv2.fillPoly(mm, [b.astype(np.int32)], 255)
    if mm.any():
        mm = cv2.dilate(mm, ker)
        if HAS_FACE:                                    # trava: rosto DESTE quadro (onde ele esta)
            fm = np.zeros((H, W), np.uint8)
            stamp_face(fm, face_rects(fr))
            mm[fm > 0] = 0
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
