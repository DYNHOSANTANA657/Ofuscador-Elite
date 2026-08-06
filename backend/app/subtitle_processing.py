from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path
from typing import IO, Callable

from .config import find_binary
from .subtitle_models import SubtitleModelManager


class SubtitleRemovalError(RuntimeError):
    pass


def _read_exact(stream: IO[bytes], size: int) -> bytearray | None:
    """Lê exatamente um quadro do pipe. Devolve None no fim do vídeo.

    Um `read()` em pipe pode voltar parcial; sem este laço um quadro sairia
    deslocado e todos os seguintes ficariam corrompidos.
    """
    buffer = bytearray(size)
    view = memoryview(buffer)
    filled = 0
    while filled < size:
        received = stream.readinto(view[filled:])
        if not received:
            return None
        filled += received
    return buffer


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()
    except OSError:
        return ""


def regions_at(regions: list[dict[str, object]], time_ms: int) -> list[dict[str, object]]:
    return [
        region for region in regions
        if bool(region.get("enabled", True)) and int(region["startMs"]) <= time_ms <= int(region["endMs"])
    ]


class SubtitleFrameCleaner:
    def __init__(self, models: SubtitleModelManager, fps: float = 30.0) -> None:
        self.models = models
        self._session = None
        self._gpu = False
        self._detector = None   # None = ainda não tentou; False = indisponível
        self._last_mask = None  # máscara do último quadro-chave, para propagar
        # A recuperação temporal (fluxo óptico) deforma o quadro JÁ limpo do passo
        # anterior e ACUMULA erro: com a mão passando sobre a legenda o remendo arrastava
        # e virava uma faixa suja. Por isso o padrão é rodar o LaMa em TODO quadro
        # (`_max_propagation = 0`), o que só é viável com GPU. Sem GPU (ou no modo
        # "rapida") cai-se numa janela de propagação para não ficar lento demais.
        self._cpu_window = max(4, int(round(fps / 2.0)))
        self._max_propagation = self._cpu_window
        self._propagated = 0

    def _session_for_lama(self):
        if self._session is not None:
            return self._session
        model_dir = self.models.models_dir()
        if not model_dir:
            raise SubtitleRemovalError("O pacote de IA para remover legendas não está instalado.")
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, os.cpu_count() or 1)
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_cpu_mem_arena = False
            available = set(ort.get_available_providers())
            wanted: list[str] = []
            if os.environ.get("OFUSCADOR_FORCE_CPU") != "1":
                for name in ("DmlExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider"):
                    if name in available:
                        wanted.append(name)
            model_path = str(model_dir / "lama_fp32.onnx")
            self._session = None
            if wanted:
                try:
                    trial = ort.InferenceSession(model_path, sess_options=options, providers=wanted + ["CPUExecutionProvider"])
                    self._probe_lama(trial)  # roda uma inferência de verdade na GPU
                    self._session = trial
                    self._gpu = trial.get_providers()[0] != "CPUExecutionProvider"
                except Exception:
                    # A GPU não executa este modelo (ex.: DirectML não suporta as camadas
                    # FFC do LaMa). Cai para CPU em vez de quebrar no meio do vídeo.
                    self._session = None
            if self._session is None:
                self._session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
                self._gpu = False
        except SubtitleRemovalError:
            raise
        except Exception as exc:
            raise SubtitleRemovalError("O modelo LaMa não pôde ser iniciado. Reinstale o pacote de IA.") from exc
        # Modo de reconstrução: "alta" = LaMa todo quadro (mais nítido/discreto); "rapida"
        # = propaga (mais veloz); "auto" = alta no GPU, rápida no CPU.
        mode = os.environ.get("OFUSCADOR_RECON_MODE", "auto").strip().lower()
        if mode == "alta":
            self._max_propagation = 0
        elif mode == "rapida":
            self._max_propagation = self._cpu_window
        else:
            self._max_propagation = 0 if self._gpu else self._cpu_window
        return self._session

    @staticmethod
    def _probe_lama(session) -> None:
        """Roda uma inferência 512x512 mínima. Se o provider (GPU) não executar de fato
        este modelo, lança — e quem chama volta para o CPU antes de começar o vídeo."""
        import numpy as np
        inputs = session.get_inputs()
        names = {item.name.lower(): item.name for item in inputs}
        image_name = names.get("image", inputs[0].name)
        mask_name = names.get("mask", inputs[1].name if len(inputs) > 1 else inputs[0].name)
        image = np.zeros((1, 3, 512, 512), dtype=np.float32)
        mask = np.zeros((1, 1, 512, 512), dtype=np.float32)
        mask[..., 240:272, 200:312] = 1.0
        session.run(None, {image_name: image, mask_name: mask})

    @staticmethod
    def _text_strokes(roi):
        """Só as linhas de legenda dentro do retângulo, não o retângulo inteiro.

        Legendas gravadas são quase sempre texto claro (branco ou amarelo). O teste
        de cor (alta luminância + baixa saturação, ou amarelo karaokê) já é seletivo:
        pele e parede claras têm matiz e quase nunca passam. As letras do tamanho
        certo são agrupadas em LINHAS (conexão só na horizontal) e cada linha vira um
        retângulo cheio, com uma folga que engole o contorno escuro e a sombra.

        Por que retângulo cheio por linha, e não o traço da letra: o LaMa reconstrói
        um bloco contíguo a partir do fundo limpo em volta e não deixa fantasma; um
        traço fino de letra ele preenchia com pixels que ainda eram de outra letra e
        sobrava a borda. E por que agrupar só na horizontal: duas linhas próximas não
        podem virar um bloco único — um bloco grande dentro de um banner colorido faz
        o LaMa apagar o banner inteiro e deixar borrão, enquanto faixas finas por
        linha ele preenche com a cor em volta, mantendo a caixa. O tamanho de tudo é
        relativo à ALTURA DA LETRA detectada, não à faixa que o usuário desenhou, então
        uma caixa grande não engole o cenário e uma pequena ainda une as letras.
        """
        import cv2
        import numpy as np
        if roi.size == 0:
            return np.zeros(roi.shape[:2], dtype=np.uint8)
        b, g, r = cv2.split(roi.astype(np.int16))
        mx = np.maximum(np.maximum(b, g), r)
        mn = np.minimum(np.minimum(b, g), r)
        height, width = roi.shape[:2]
        white = (mn > 170) & ((mx - mn) < 45)
        yellow = (r > 150) & (g > 135) & (b < 120)
        bright = ((white | yellow).astype(np.uint8)) * 255

        # Componentes claros do tamanho de letra: não a faixa toda em altura, nem um
        # borrão gigante (parede/pele). O que separa letra de fundo é a ALTURA e a
        # área, não a largura (uma palavra pode ser larga).
        keep = np.zeros_like(bright)
        heights = []
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            box_h = stats[index, cv2.CC_STAT_HEIGHT]
            if area < 6 or box_h > height * 0.75 or area > 0.20 * height * width:
                continue
            keep[labels == index] = 255
            heights.append(int(box_h))
        if not heights:
            return np.zeros((height, width), dtype=np.uint8)
        glyph = int(np.median(heights))

        # Agrupa em linhas conectando só na horizontal (kernel de 1px de altura) e
        # preenche o retângulo de cada linha, com folga proporcional à letra.
        link = max(3, int(glyph * 1.2) | 1)
        linked = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (link, 1)))
        out = np.zeros((height, width), dtype=np.uint8)
        pad_x = max(2, int(glyph * 0.35))
        pad_y = max(2, int(glyph * 0.25))
        count, _, stats, _ = cv2.connectedComponentsWithStats(linked, 8)
        for index in range(1, count):
            x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
            w, h = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]
            if h > height * 0.85:
                continue
            if w < glyph * 0.5 and h < glyph * 0.5:
                continue
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
            out[y0:y1, x0:x1] = 255
        # Rede de segurança: todo pixel claro de texto detectado (inclusive pingos soltos
        # que não formaram linha — o "i", um acento, um resto de palavra anterior) entra na
        # máscara com uma folga que engole o contorno escuro. É só um halo por glifo, não um
        # bloco grande, então não dissolve tarja colorida (o fundo da tarja não é claro).
        halo = cv2.dilate(keep, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, ((pad_x * 2 + 1), (pad_y * 2 + 1))))
        return cv2.max(out, halo)

    @staticmethod
    def mask_for(frame, regions: list[dict[str, object]]):
        import cv2
        import numpy as np
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            x1 = max(0, min(width - 1, int(float(region["x"]) * width)))
            y1 = max(0, min(height - 1, int(float(region["y"]) * height)))
            x2 = max(x1 + 1, min(width, int(round((float(region["x"]) + float(region["width"])) * width))))
            y2 = max(y1 + 1, min(height, int(round((float(region["y"]) + float(region["height"])) * height))))
            strokes = SubtitleFrameCleaner._text_strokes(frame[y1:y2, x1:x2])
            mask[y1:y2, x1:x2] = cv2.max(mask[y1:y2, x1:x2], strokes)
        kernel = max(3, int(round(min(height, width) * 0.006)) | 1)
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))

    def _text_detector(self):
        """Detector de texto DBNet do RapidOCR (mesmo pacote do exame). Acha texto de
        QUALQUER cor pela forma, não pela cor. Só detecção, sem reconhecer as letras."""
        if self._detector is not None:
            return self._detector or None
        model_dir = self.models.models_dir()
        if not model_dir:
            self._detector = False
            return None
        try:
            from rapidocr import RapidOCR
            self._detector = RapidOCR(params={
                "Global.model_root_dir": str(model_dir),
                "Global.text_score": 0.4,
                "Global.log_level": "warning",
            })
        except Exception:
            self._detector = False
            return None
        return self._detector

    def _detect_boxes(self, crop):
        """Caixas de texto (qualquer cor) no recorte, ou None se o detector faltar."""
        import numpy as np
        detector = self._text_detector()
        if detector is None or crop.size == 0:
            return None
        try:
            result = detector(crop, use_det=True, use_rec=False, use_cls=False)
        except Exception:
            return None
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        return [np.asarray(box, dtype=np.float32) for box in boxes]

    def _subtitle_mask(self, frame, regions: list[dict[str, object]]):
        """Máscara do texto por REGIÃO: detector (qualquer cor) reforçado pela detecção
        de cor branco/amarelo. Cada caixa detectada vira um bloco cheio por linha — o
        que mata o fantasma da borda e mantém a discrição. Se o detector não estiver
        disponível, usa só a cor (comportamento da 1.3.3)."""
        import cv2
        import numpy as np
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            x1 = max(0, min(width - 1, int(float(region["x"]) * width)))
            y1 = max(0, min(height - 1, int(float(region["y"]) * height)))
            x2 = max(x1 + 1, min(width, int(round((float(region["x"]) + float(region["width"])) * width))))
            y2 = max(y1 + 1, min(height, int(round((float(region["y"]) + float(region["height"])) * height))))
            crop = frame[y1:y2, x1:x2]
            # Máscara por COR (branco/amarelo): justa e já ótima na 1.3.3 — preserva
            # tarjas coloridas. É a base.
            local = self._text_strokes(crop)
            boxes = self._detect_boxes(crop)
            if boxes:
                ch, cw = crop.shape[:2]
                for quad in boxes:
                    bxs, bys = quad[:, 0], quad[:, 1]
                    box_h = float(bys.max() - bys.min())
                    pad_x = max(2, int(box_h * 0.30))
                    pad_y = max(1, int(box_h * 0.12))
                    xa = max(0, int(bxs.min()) - pad_x)
                    xb = min(cw, int(bxs.max()) + pad_x)
                    ya = max(0, int(bys.min()) - pad_y)
                    yb = min(ch, int(bys.max()) + pad_y)
                    if xb <= xa or yb <= ya:
                        continue
                    # Se a COR já cobre bem esta caixa, é texto branco/amarelo: confia na
                    # máscara de cor (mais justa, não engrossa a tarja). Só usa a caixa do
                    # detector quando a cor NÃO pegou — aí é legenda de outra cor.
                    box_area = (yb - ya) * (xb - xa)
                    if int(cv2.countNonZero(local[ya:yb, xa:xb])) > 0.12 * box_area:
                        continue
                    local[ya:yb, xa:xb] = 255
            mask[y1:y2, x1:x2] = cv2.max(mask[y1:y2, x1:x2], local)
        kernel = max(3, int(round(min(height, width) * 0.006)) | 1)
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))

    def _region_bbox(self, shape, regions):
        height, width = shape[:2]
        xs1, ys1, xs2, ys2 = [], [], [], []
        for region in regions:
            xs1.append(max(0, min(width - 1, int(float(region["x"]) * width))))
            ys1.append(max(0, min(height - 1, int(float(region["y"]) * height))))
            xs2.append(max(1, min(width, int(round((float(region["x"]) + float(region["width"])) * width)))))
            ys2.append(max(1, min(height, int(round((float(region["y"]) + float(region["height"])) * height)))))
        return min(xs1), min(ys1), max(xs2), max(ys2)

    def clean(self, frame, regions: list[dict[str, object]], previous_input, previous_clean):
        import cv2
        import numpy as np
        if not regions:
            return frame.copy()
        candidate = None
        mask = None
        can_propagate = (
            previous_input is not None
            and previous_clean is not None
            and self._last_mask is not None
            and previous_input.shape == frame.shape
            and self._propagated < self._max_propagation
        )
        if can_propagate:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(previous_input, cv2.COLOR_BGR2GRAY)
            scene_change = float(np.mean(cv2.absdiff(current_gray, previous_gray)))
            # Gate LOCAL na CAIXA do usuário: se a área da legenda mudou muito (a mão
            # passou por cima, a cena mexeu ali), NÃO propague — o warp arrastaria pixels
            # da borda e deixaria uma faixa suja. Refaz o LaMa, que reconstrói limpo.
            rx1, ry1, rx2, ry2 = self._region_bbox(frame.shape, regions)
            region_change = float(np.mean(cv2.absdiff(
                current_gray[ry1:ry2, rx1:rx2], previous_gray[ry1:ry2, rx1:rx2])))
            if scene_change < 30 and region_change < 12:
                # Fluxo óptico em meia resolução: ~4x mais rápido, sem perda perceptível.
                fh, fw = current_gray.shape
                cur_s = cv2.resize(current_gray, (fw // 2, fh // 2))
                prev_s = cv2.resize(previous_gray, (fw // 2, fh // 2))
                flow_s = cv2.calcOpticalFlowFarneback(cur_s, prev_s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow = cv2.resize(flow_s, (fw, fh)) * 2.0
                grid_x, grid_y = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))
                map_x, map_y = grid_x + flow[..., 0], grid_y + flow[..., 1]
                candidate = cv2.remap(previous_clean, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                # A máscara acompanha o texto que se move (karaokê) pelo mesmo fluxo.
                mask = cv2.remap(self._last_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                self._propagated += 1
        if candidate is None:
            # Quadro-chave: detecta o texto (qualquer cor) e reconstrói com o LaMa.
            mask = self._subtitle_mask(frame, regions)
            self._last_mask = mask
            if cv2.findNonZero(mask) is None:
                return frame.copy()
            candidate = self._lama(frame, mask)
            self._propagated = 0
        if mask is None or cv2.countNonZero(mask) == 0:
            return frame.copy()
        # Casa o TOM do remendo com o fundo em volta antes de colar. Sem isto, a média de
        # brilho da reconstrução (ou do remendo propagado, que escurece um pouco a cada
        # warp) fica um tico diferente da roupa e a borda da máscara vira uma LINHA
        # horizontal — a "emenda". Desloca o remendo pela diferença de média entre um anel
        # fino de fundo REAL (fora) e o anel do remendo (dentro), colado à borda. Some a
        # emenda sem borrar a textura; o deslocamento é limitado para nunca inventar cor.
        ring = max(3, int(round(min(frame.shape[:2]) * 0.010)) | 1)
        elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring, ring))
        outer = cv2.subtract(cv2.dilate(mask, elem), mask)
        inner = cv2.subtract(mask, cv2.erode(mask, elem))
        if cv2.countNonZero(outer) > 20 and cv2.countNonZero(inner) > 20:
            background = frame.astype(np.float32)[outer > 0].mean(axis=0)
            patch = candidate.astype(np.float32)[inner > 0].mean(axis=0)
            shift = np.clip(background - patch, -40.0, 40.0)
            candidate = np.clip(
                candidate.astype(np.float32) + shift * (mask > 0)[..., None], 0, 255
            ).astype(np.uint8)
        feather_size = max(3, int(round(min(frame.shape[:2]) * 0.009)) | 1)
        alpha = cv2.GaussianBlur(mask, (feather_size, feather_size), 0).astype(np.float32)[..., None] / 255.0
        return np.clip(frame.astype(np.float32) * (1 - alpha) + candidate.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    def _run_lama_512(self, tile_bgr, tile_mask):
        """Reconstrói um bloco 512x512 exato (o modelo LaMa é fixo nesse tamanho).

        Recebe BGR uint8 e a máscara uint8, devolve BGR uint8 512x512.
        """
        import cv2
        import numpy as np
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        image_input = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        mask_input = (tile_mask.astype(np.float32) / 255.0)[None, None]
        session = self._session_for_lama()
        inputs = session.get_inputs()
        names = {item.name.lower(): item.name for item in inputs}
        image_name = names.get("image", inputs[0].name)
        mask_name = names.get("mask", inputs[1].name if len(inputs) > 1 else inputs[0].name)
        try:
            output = session.run(None, {image_name: image_input, mask_name: mask_input})[0]
        except Exception as exc:
            raise SubtitleRemovalError("O modelo LaMa falhou ao reconstruir o fundo. Nenhum borrão ou corte foi aplicado.") from exc
        result = np.asarray(output)[0]
        if result.shape[0] == 3:
            result = np.transpose(result, (1, 2, 0))
        if float(np.nanmax(result)) <= 1.5:
            result = result * 255.0
        return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _lama(self, frame, mask):
        """Reconstrói o fundo sob a máscara com o LaMa (512x512).

        Recorta a região do texto com uma folga de contexto e redimensiona para 512.
        A folga é MODERADA de propósito: contexto grande demais faz o LaMa preencher a
        legenda com o que domina em volta (ex.: a parede atrás de uma tarja colorida,
        dissolvendo a tarja); contexto local o bastante deixa a cor de fundo mandar
        (o vermelho da tarja continua vermelho, a camisa continua camisa). O rastro que
        aparecia em roupa lisa não vinha daqui, e sim da propagação temporal (resolvida
        em `clean`), então este caminho reconstrói limpo quadro a quadro.
        """
        import cv2
        points = cv2.findNonZero(mask)
        if points is None:
            return frame.copy()
        x, y, width, height = cv2.boundingRect(points)
        # Contexto pela ALTURA do texto, não pela largura: numa tarja larga, contexto
        # proporcional à largura subiria fundo na parede acima e o LaMa dissolveria a
        # tarja. Preso à altura, o recorte fica junto da legenda e a cor de fundo manda.
        context = max(40, int(height * 0.6))
        x1, y1 = max(0, x - context), max(0, y - context)
        x2, y2 = min(frame.shape[1], x + width + context), min(frame.shape[0], y + height + context)
        crop = frame[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        if crop.size == 0:
            raise SubtitleRemovalError("A região de legenda não pôde ser preparada para a IA.")
        image_512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
        mask_512 = cv2.resize(crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        result = self._run_lama_512(image_512, mask_512)
        result = cv2.resize(result, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
        candidate = frame.copy()
        candidate[y1:y2, x1:x2] = result
        return candidate

    # ------------------------------------------------------------------
    # Modo AUTOMÁTICO "tela toda menos rosto" (receita da Colab, v2.4.1).
    # Reaproveita o MESMO motor ONNX e o MESMO detector DBNet do modo de caixa; a
    # diferença é a estratégia: em vez de apagar só dentro de uma caixa desenhada,
    # varre a TELA TODA por texto de qualquer cor e protege o rosto.
    # ------------------------------------------------------------------
    def _auto_boxes(self, frame, min_text_h: int, min_area: int):
        """Caixas de texto de QUALQUER cor na TELA TODA (detector DBNet), já filtradas
        como na v2.4.1: descarta caixa baixa demais (< min_text_h → o texto pequeno da
        Bíblia na tela, que NÃO é legenda) e área minúscula (< min_area → ruído do
        detector). Devolve None quando o detector não está instalado — quem chama trata
        isso como erro, pois sem OCR não há como achar a legenda."""
        import cv2
        import numpy as np
        boxes = self._detect_boxes(frame)
        if boxes is None:
            return None
        kept = []
        for quad in boxes:
            box = np.asarray(quad, dtype=np.float32)
            if box.shape != (4, 2):
                continue
            if cv2.contourArea(box) < min_area:
                continue
            if (box[:, 1].max() - box[:, 1].min()) < min_text_h:
                continue
            kept.append(box)
        return kept

    def inpaint_components(self, frame, mask):
        """Inpaint POR COMPONENTE de texto: cada bloco no seu recorte LOCAL (contexto pela
        ALTURA), 512, colando só os pixels da máscara. Assim texto no topo E no rodapé não
        vira "a tela inteira esticada" e borrada — cada bloco fica nítido. É o mesmo motor
        (`_run_lama_512`) do modo de caixa, varrendo cada componente da máscara. Lê o
        contexto do quadro ORIGINAL e escreve numa cópia, igual à receita validada."""
        import cv2
        import numpy as np
        height, width = frame.shape[:2]
        binary = (mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        out = frame.copy()
        for index in range(1, count):
            x, y = int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP])
            w, h = int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT])
            context = max(40, int(h * 0.6))
            x1, y1 = max(0, x - context), max(0, y - context)
            x2, y2 = min(width, x + w + context), min(height, y + h + context)
            crop = frame[y1:y2, x1:x2]
            component = np.zeros((height, width), np.uint8)
            component[labels == index] = 255
            crop_mask = component[y1:y2, x1:x2]
            if crop.size == 0 or not crop_mask.any():
                continue
            image_512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
            mask_512 = cv2.resize(crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            result = self._run_lama_512(image_512, mask_512)
            result = cv2.resize(result, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
            selection = crop_mask > 0
            region = out[y1:y2, x1:x2]
            region[selection] = result[selection]
            out[y1:y2, x1:x2] = region
        return out


def _stream_frames(
    source: Path,
    destination: Path,
    probe: dict[str, object],
    process_frame: Callable[[object, int, int], object],
    progress: Callable[[float, str], None] | None = None,
    message: str = "Reconstruindo quadro",
) -> dict[str, float]:
    """Decodifica o vídeo com FFmpeg, entrega cada quadro BGR a ``process_frame(frame,
    índice, tempo_ms)`` e recodifica em H.264 o quadro devolvido. Concentra a leitura/escrita
    robusta (contagem exata de quadros, stderr em arquivo, limpeza em caso de falha) que os
    dois modos — caixa e automático — compartilham."""
    import numpy as np
    fps = float(probe["fps"])
    rate = str(probe.get("frameRateRational") or f"{fps:.8f}")
    width, height = int(probe["width"]), int(probe["height"])
    frame_bytes = width * height * 3
    video_duration = float(probe.get("videoDuration") or probe["duration"])
    total_frames = int(probe.get("frameCount") or round(video_duration * fps))

    # Decodificar com o mesmo FFmpeg que codifica: o OpenCV usa outro build de
    # libavcodec e pode devolver menos quadros em vídeos com B-frames ou GOP aberto,
    # encurtando o resultado. `-fps_mode cfr -r` também normaliza VFR na entrada,
    # então o número de quadros que entra é exatamente o que sai.
    decode_command = [
        find_binary("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error", "-i", str(source),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "cfr", "-r", rate,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    encode_command = [
        find_binary("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", rate, "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-movflags", "+faststart", str(destination),
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    logs = Path(tempfile.mkdtemp(prefix="ofuscador-ffmpeg-"))
    decode_log, encode_log = logs / "decode.txt", logs / "encode.txt"
    processed = 0
    decoder = encoder = None
    try:
        # stderr vai para arquivo, não para PIPE: um PIPE de erro que ninguém lê
        # enche e trava o FFmpeg no meio de um vídeo longo.
        with decode_log.open("wb") as decode_errors, encode_log.open("wb") as encode_errors:
            decoder = subprocess.Popen(decode_command, stdout=subprocess.PIPE, stderr=decode_errors, creationflags=flags)
            encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encode_errors, creationflags=flags)
            assert decoder.stdout is not None and encoder.stdin is not None
            while True:
                raw = _read_exact(decoder.stdout, frame_bytes)
                if raw is None:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                time_ms = int(round(processed / fps * 1000))
                cleaned = process_frame(frame, processed, time_ms)
                try:
                    encoder.stdin.write(cleaned.tobytes())
                except (BrokenPipeError, OSError) as exc:
                    raise SubtitleRemovalError(f"O codificador H.264 parou no quadro {processed}. {_tail(encode_log)}".strip()) from exc
                processed += 1
                if progress and (processed % max(1, int(fps // 2)) == 0 or processed == total_frames):
                    progress(min(1.0, processed / max(1, total_frames)), f"{message} {processed} de {total_frames}")
            decoder.stdout.close()
            try:
                decode_status = decoder.wait(timeout=120)
            except subprocess.TimeoutExpired as exc:
                raise SubtitleRemovalError("A leitura dos quadros não terminou no tempo esperado.") from exc
            if decode_status != 0:
                raise SubtitleRemovalError(f"O FFmpeg não conseguiu ler todos os quadros do vídeo. {_tail(decode_log)}".strip())
            encoder.stdin.close()
            try:
                # Fechar o stdin não encerra o codificador na hora: o libx264 ainda esvazia
                # o lookahead e o `+faststart` reescreve o índice do MP4 inteiro no fim.
                encode_status = encoder.wait(timeout=1800)
            except subprocess.TimeoutExpired as exc:
                raise SubtitleRemovalError("O codificador H.264 não terminou no tempo esperado.") from exc
            if encode_status != 0 or not destination.is_file() or destination.stat().st_size == 0:
                raise SubtitleRemovalError(f"O vídeo sem legendas não pôde ser codificado em H.264. {_tail(encode_log)}".strip())
        if processed == 0:
            raise SubtitleRemovalError("Nenhum quadro foi lido do vídeo. O arquivo pode estar corrompido.")
        # Falha aqui, com números, em vez de deixar o erro aparecer só no fim do mux.
        missing = total_frames - processed
        if missing > 0 and missing / max(1, total_frames) > 0.01 and missing / fps > 0.5:
            raise SubtitleRemovalError(
                f"A leitura terminou cedo: {processed} de {total_frames} quadros "
                f"({missing / fps:.2f}s a menos). {_tail(decode_log)}".strip()
            )
    except Exception:
        for child in (decoder, encoder):
            if child and child.poll() is None:
                child.kill()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        destination.unlink(missing_ok=True)
        raise
    finally:
        # No Windows um arquivo ainda aberto por um processo filho não pode ser apagado;
        # deixar esse OSError escapar aqui substituiria a exceção original por outra.
        try:
            shutil.rmtree(logs, ignore_errors=True)
        except OSError:
            pass
    return {
        "frames": processed,
        "expectedFrames": total_frames,
        "outputDuration": processed / fps,
        "sourceVideoDuration": video_duration,
    }


def remove_burned_subtitles(
    source: Path,
    destination: Path,
    probe: dict[str, object],
    regions: list[dict[str, object]],
    models: SubtitleModelManager,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, float]:
    """Modo CAIXA: reconstrói o fundo apenas DENTRO das regiões que o usuário revisou."""
    if bool(probe.get("hdr")):
        raise SubtitleRemovalError("Vídeos HDR ainda não são suportados na remoção de legenda gravada, pois a conversão poderia alterar as cores.")
    if not regions:
        raise SubtitleRemovalError("Nenhuma região ativa foi definida para remover da imagem.")
    if int(probe["width"]) % 2 or int(probe["height"]) % 2:
        raise SubtitleRemovalError("A resolução precisa ter largura e altura pares para manter H.264 em yuv420p.")
    fps = float(probe["fps"])
    cleaner = SubtitleFrameCleaner(models, fps)
    previous_input = None
    previous_clean = None

    def process(frame, index: int, time_ms: int):
        nonlocal previous_input, previous_clean
        active = regions_at(regions, time_ms)
        cleaned = cleaner.clean(frame, active, previous_input, previous_clean) if active else frame
        previous_input = frame
        previous_clean = cleaned
        return cleaned

    return _stream_frames(source, destination, probe, process, progress)


def _load_face_cascade(models: SubtitleModelManager):
    """CascadeClassifier de rosto do OpenCV, procurando o XML na pasta de modelos do app e
    nos dados empacotados do OpenCV. Devolve None se não achar — aí o chamador protege a
    FAIXA DO MEIO da tela por segurança, como faz a receita da Colab sem detector."""
    import cv2
    name = "haarcascade_frontalface_default.xml"
    candidates: list[Path] = []
    try:
        model_dir = models.models_dir()
    except Exception:
        model_dir = None
    if model_dir:
        candidates.append(Path(model_dir) / name)
    try:
        candidates.append(Path(cv2.data.haarcascades) / name)
    except Exception:
        pass
    for path in candidates:
        try:
            if path.is_file():
                cascade = cv2.CascadeClassifier(str(path))
                if not cascade.empty():
                    return cascade
        except Exception:
            continue
    return None


def _face_rects(cascade, frame):
    if cascade is None:
        return []
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))


def _stamp_face(target, rects, height: int, width: int, only_upper: bool = False) -> None:
    """Soma 1 na caixa AMPLIADA de cada rosto. ``only_upper`` ignora rostos na metade de
    baixo (na PASSA 1 evita confundir legenda embaixo com rosto)."""
    for (x, y, w, h) in rects:
        if only_upper and (y + 0.5 * h) > 0.50 * height:
            continue
        y0 = max(0, int(y) - int(0.20 * h))
        y1 = min(height, int(y) + int(1.00 * h))
        x0 = max(0, int(x) - int(0.20 * w))
        x1 = min(width, int(x) + int(1.20 * w))
        target[y0:y1, x0:x1] += 1


def _auto_params(options: dict[str, object] | None = None) -> dict[str, object]:
    """Parâmetros da receita automática, com os valores da v2.4.1 validada e overrides por
    variável de ambiente — dá para trocar velocidade por fidelidade sem recompilar."""
    options = options or {}

    def pick(key: str, env: str, default, cast):
        if options.get(key) is not None:
            try:
                return cast(options[key])
            except (TypeError, ValueError):
                pass
        raw = os.environ.get(env)
        if raw not in (None, ""):
            try:
                return cast(raw)
            except (TypeError, ValueError):
                pass
        return default

    return {
        "ocr_cada": max(1, pick("ocrEvery", "OFUSCADOR_OCR_CADA", 1, int)),
        "min_text_h": max(1, pick("minTextH", "OFUSCADOR_MIN_TEXT_H", 22, int)),
        "dil": max(0, pick("dil", "OFUSCADOR_DIL", 6, int)),
        "temporal_w": max(1, pick("temporalW", "OFUSCADOR_TEMPORAL_W", 3, int)),
        "min_area": max(1, pick("minArea", "OFUSCADOR_MIN_AREA", 60, int)),
        "n_samp": max(8, pick("faceSamples", "OFUSCADOR_FACE_SAMPLES", 200, int)),
        "face_freq": min(1.0, max(0.0, pick("faceFreq", "OFUSCADOR_FACE_FREQ", 0.30, float))),
    }


def _scan_faces(source: Path, probe: dict[str, object], cascade, n_samp: int, facehits, height: int, width: int) -> int:
    """PASSA 1: decodifica o vídeo em fps BAIXO (~n_samp amostras no total) e marca onde o
    rosto aparece. Usa o MESMO FFmpeg do app (não o VideoCapture do OpenCV), então funciona
    igual no pacote congelado. Devolve quantas amostras conseguiu ler."""
    import numpy as np
    fps = float(probe["fps"])
    frame_bytes = width * height * 3
    video_duration = float(probe.get("videoDuration") or probe["duration"]) or 1.0
    sample_fps = max(0.2, min(fps if fps > 0 else 2.0, n_samp / max(1.0, video_duration)))
    command = [
        find_binary("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error", "-i", str(source),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", f"fps={sample_fps:.6f}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    samples = 0
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags)
        assert process.stdout is not None
        while True:
            raw = _read_exact(process.stdout, frame_bytes)
            if raw is None:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            _stamp_face(facehits, _face_rects(cascade, frame), height, width, only_upper=True)
            samples += 1
        process.stdout.close()
        process.wait(timeout=120)
    except Exception:
        # A PASSA 1 é só uma otimização de segurança; se falhar, devolve o que leu e o
        # chamador protege a faixa do meio.
        if process and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    return samples


def remove_burned_subtitles_auto(
    source: Path,
    destination: Path,
    probe: dict[str, object],
    models: SubtitleModelManager,
    progress: Callable[[float, str], None] | None = None,
    options: dict[str, object] | None = None,
) -> dict[str, float]:
    """Modo AUTOMÁTICO "tela toda menos rosto" — a receita validada na Colab (v2.5), rodando
    no PC com o motor ONNX do app. Sem desenhar caixa: apaga texto de QUALQUER cor na tela
    inteira e protege o rosto POR QUADRO (onde ele realmente está). Por quadro: OCR + união
    temporal + trava de tamanho (protege a Bíblia) + trava de rosto por quadro + inpaint por
    componente. (v2.5 removeu o "buraco fixo" — a união de rostos do vídeo INTEIRO virava uma
    tarja no meio e protegia a legenda em vídeo multi-cena / tela dividida.)"""
    import cv2
    import numpy as np
    if bool(probe.get("hdr")):
        raise SubtitleRemovalError("Vídeos HDR ainda não são suportados na remoção de legenda gravada, pois a conversão poderia alterar as cores.")
    if int(probe["width"]) % 2 or int(probe["height"]) % 2:
        raise SubtitleRemovalError("A resolução precisa ter largura e altura pares para manter H.264 em yuv420p.")
    params = _auto_params(options)
    fps = float(probe["fps"])
    width, height = int(probe["width"]), int(probe["height"])

    cleaner = SubtitleFrameCleaner(models, fps)
    # Aquece o motor e o detector agora: falha cedo e com mensagem clara, não depois de
    # horas no meio do vídeo.
    cleaner._session_for_lama()
    if cleaner._text_detector() is None:
        raise SubtitleRemovalError("O detector de texto (RapidOCR) não está disponível. Reinstale o pacote de IA para usar o modo automático.")

    if progress:
        progress(0.0, "Preparando remoção automática")
    cascade = _load_face_cascade(models)
    has_face = cascade is not None
    # v2.5: SEM buraco FIXO de rosto. A PASSA 1 antiga (união das posições de rosto do vídeo
    # INTEIRO) virava uma tarja no MEIO em vídeo MULTI-CENA (várias pessoas / tela dividida) e
    # PROTEGIA a legenda do meio -> ela sobrevivia. Agora o rosto é protegido SÓ por quadro
    # (trava 2, onde ele realmente está). Só quando NÃO há detector de rosto é que caímos no
    # antigo escudo da FAIXA DO MEIO, pra degradar com segurança sem apagar o rosto.
    allowed = np.ones((height, width), bool)
    if not has_face:
        allowed[int(0.12 * height):int(0.60 * height), :] = False

    dil = int(params["dil"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil * 2 + 1, dil * 2 + 1)) if dil > 0 else None
    ocr_cada = int(params["ocr_cada"])
    min_text_h = int(params["min_text_h"])
    min_area = int(params["min_area"])
    history: deque = deque(maxlen=int(params["temporal_w"]))

    def process(frame, index: int, time_ms: int):
        if index % ocr_cada == 0:
            found = cleaner._auto_boxes(frame, min_text_h, min_area)
            if found is None:
                raise SubtitleRemovalError("O detector de texto (RapidOCR) parou de responder durante o processamento.")
            history.append(found)
        mask = np.zeros((height, width), np.uint8)
        for found in history:
            for box in found:
                cv2.fillPoly(mask, [box.astype(np.int32)], 255)
        if not mask.any():
            return frame
        if kernel is not None:
            mask = cv2.dilate(mask, kernel)
        mask[~allowed] = 0                       # só age SEM detector de rosto (escudo da faixa do meio)
        if has_face and mask.any():              # trava 2: rosto DESTE quadro
            dynamic = np.zeros((height, width), np.uint8)
            _stamp_face(dynamic, _face_rects(cascade, frame), height, width)
            mask[dynamic > 0] = 0
        if not mask.any():
            return frame
        return cleaner.inpaint_components(frame, mask)

    forward = (lambda fraction, message: progress(0.06 + fraction * 0.94, message)) if progress else None
    return _stream_frames(source, destination, probe, process, forward, message="Reconstruindo (tela toda) quadro")
