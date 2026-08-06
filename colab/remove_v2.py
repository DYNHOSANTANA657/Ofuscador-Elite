# remove_v2.py - Remove legenda/tarja QUEIMADA com LaMa (inpaint por quadro). v2.7
# Roda no runtime do Colab (auto-baixa LaMa, auto-instala rapidocr, auto-pede o video).
#
# v2.7 - SEM TRAVA DE TAMANHO (a pedido do usuario). Removi o MIN_TEXT_H: agora a legenda e
#   apagada por MENOR que seja. Motivo: em video de baixa resolucao (ex. 360x516) a legenda tem
#   so ~15-20px de altura e escapava da trava de 22px, sobrevivendo. ATENCAO: nao ha mais protecao
#   de letra pequena, entao numero de versiculo da Biblia tambem pode ser apagado. Sobra so o
#   MIN_AREA (ignora ruido de poucos pixels do detector; nunca bloqueia legenda, que tem area grande).
#
# v2.6 - SEM PROTECAO DE ROSTO/AREA (a pedido do usuario). NAO existe mais nenhuma regra que
#   proteja o rosto ou qualquer regiao do video: a legenda e apagada em QUALQUER parte da tela,
#   inclusive POR CIMA do rosto (videos cinematograficos em que o rosto passa em cima da legenda
#   e depois desce). O resto continua IGUAL: deteccao de texto POR QUADRO (OCR DBNet, qualquer
#   cor), uniao temporal curta (tira o pisca do karaoke) e LaMa por quadro. Removi o detector de
#   rosto Haar (face_cascade / face_rects / stamp_face) e a marcacao de rosto por quadro na PASSA 2.
#
# v2.5 - (historico) tinha protecao de rosto SO por quadro (Haar na PASSA 2). Removida na v2.6.
# v2.4.x - (historico) zona = tela toda menos um buraco FIXO do rosto. Substituido pela v2.5/v2.6.

import os, glob, cv2, numpy as np, urllib.request, torch
from collections import deque
from google.colab.patches import cv2_imshow

# ---------------- parametros ----------------
MAX_SEGUNDOS = 0        # 0 = video INTEIRO
OCR_CADA     = 1        # OCR a cada N quadros. 1 = TODOS (pega o flash de 1 quadro). Mais lento.
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

# ---------------- 3) detector de texto (sem trava de tamanho; so ignora ruido por MIN_AREA) ----------------
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
            if cv2.contourArea(b) < MIN_AREA:      continue   # so ignora ruido minusculo; NAO ha mais trava de altura
            bs.append(b)
    return bs

# ---------------- 4) info do video ----------------
cap   = cv2.VideoCapture(VID)
fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(3)); H = int(cap.get(4))
N = total if MAX_SEGUNDOS == 0 else min(total, int(MAX_SEGUNDOS * fps))
print(f'{total} quadros no total; processando {N}; {W}x{H} @ {fps:.1f}', flush=True)

# ---------------- 5) kernel + PREVIA (checkpoint do que SERA APAGADO) ----------------
# SEM protecao de rosto/area: nao ha buraco nem zona verde. A previa mostra em VERMELHO o texto
# detectado num quadro do meio = exatamente o que a PASSA 2 vai apagar (em qualquer parte da tela).
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DIL * 2 + 1, DIL * 2 + 1))

cap.set(1, N // 2); okp, midf = cap.read()
if okp:
    mm0 = np.zeros((H, W), np.uint8)
    for b in boxes(midf):
        cv2.fillPoly(mm0, [b.astype(np.int32)], 255)
    if mm0.any():
        mm0 = cv2.dilate(mm0, ker)
    ov = midf.copy()
    ov[mm0 > 0] = (0.4 * ov[mm0 > 0].astype(np.float32) + 0.6 * np.array([0, 0, 255], np.float32)).astype('uint8')
    print('PREVIA: VERMELHO = texto que SERA apagado neste quadro. NAO ha protecao de rosto/area:')
    print('a legenda sai em QUALQUER parte da tela, inclusive por cima do rosto.')
    cv2_imshow(ov)

# ---------------- 6) PASSA 2: OCR (tela toda) + uniao temporal + LaMa (SEM protecao) ----------------
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
        fr = inpaint(fr, mm)                            # apaga em QUALQUER parte (rosto incluido)
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
