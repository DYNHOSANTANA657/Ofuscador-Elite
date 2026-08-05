from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.config import find_binary
from app.media import probe_video
from app.subtitle_models import SubtitleModelManager, _safe_member
from app.subtitle_processing import (
    SubtitleFrameCleaner,
    SubtitleRemovalError,
    _auto_params,
    _load_face_cascade,
    _stamp_face,
    regions_at,
    remove_burned_subtitles,
    remove_burned_subtitles_auto,
)
from app.subtitles import validate_region


def test_region_validation_and_time_lookup() -> None:
    region = validate_region({"x": 0.1, "y": 0.7, "width": 0.8, "height": 0.15, "startMs": 100, "endMs": 900, "source": "manual"}, 1000)
    assert region["source"] == "manual"
    assert len(regions_at([region], 500)) == 1
    assert regions_at([region], 950) == []
    with pytest.raises(ValueError, match="limites"):
        validate_region({"x": 0.8, "y": 0.8, "width": 0.4, "height": 0.1, "startMs": 0, "endMs": 100}, 1000)


def test_zip_path_validation_rejects_traversal() -> None:
    assert _safe_member("lama_fp32.onnx") == "lama_fp32.onnx"
    with pytest.raises(ValueError, match="inseguro"):
        _safe_member("../lama_fp32.onnx")
    with pytest.raises(ValueError, match="inesperado"):
        _safe_member("models/evil.onnx")


def test_model_import_rejects_malicious_zip(tmp_path: Path) -> None:
    package = tmp_path / "bad.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"version": "1", "files": []}))
        archive.writestr("../escape.onnx", b"bad")
    manager = SubtitleModelManager()
    with pytest.raises(ValueError, match="inseguro"):
        manager.install_zip(package)
    assert not (tmp_path.parent / "escape.onnx").exists()


class NoModelNeeded:
    def models_dir(self):
        return None


def test_mask_covers_text_but_spares_the_rest_of_the_band() -> None:
    """A máscara marca só as letras, não a faixa inteira.

    Era isso que causava as "falhas exageradas": o retângulo cheio apagava mão,
    rosto e cenário junto com a legenda. Aqui um bloco claro grande (pele/parede)
    convive com o texto; a máscara precisa cobrir o texto e poupar o bloco.
    """
    import cv2

    height, width = 160, 480
    frame = np.full((height, width, 3), (60, 90, 40), dtype=np.uint8)   # fundo escuro
    cv2.rectangle(frame, (20, 20), (170, 150), (150, 175, 190), -1)     # "pele": claro mas não branco
    cv2.putText(frame, "LEGENDA", (210, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 9)   # contorno
    cv2.putText(frame, "LEGENDA", (210, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)

    region = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    mask = SubtitleFrameCleaner.mask_for(frame, [region])

    # o texto (à direita) é marcado
    assert cv2.countNonZero(mask[70:130, 205:430]) > 500
    # o bloco claro tipo pele (à esquerda) é quase todo poupado
    skin = mask[20:150, 20:170]
    assert cv2.countNonZero(skin) < 0.15 * skin.size
    # e a máscara total fica longe de cobrir a faixa inteira
    assert cv2.countNonZero(mask) < 0.30 * mask.size


def test_mask_is_solid_per_line_and_keeps_lines_apart() -> None:
    """Cada linha vira um bloco cheio; duas linhas não se fundem.

    O bloco cheio é o que mata o fantasma da borda — o LaMa reconstrói uma área
    contígua a partir do fundo limpo. Já manter as linhas separadas é o que evita
    apagar um banner colorido inteiro: se as duas linhas virassem um bloco único, o
    LaMa trataria o banner como objeto e o removeria deixando borrão.
    """
    import cv2

    height, width = 240, 480
    frame = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)
    cv2.putText(frame, "LINHA UM", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, "LINHA DOIS", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    region = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    mask = SubtitleFrameCleaner.mask_for(frame, [region])

    # numa linha do meio do texto a máscara é uma BARRA CONTÍGUA (bloco, não traço):
    # sem buracos entre as letras é o que impede o fantasma da borda.
    row = mask[60]
    xs = np.nonzero(row)[0]
    assert xs.size > 150
    holes = (xs[-1] - xs[0] + 1) - xs.size
    assert holes < 6, "a máscara ficou vazada entre as letras (deixaria fantasma)"
    # existe uma faixa horizontal totalmente vazia ENTRE as duas linhas
    gap_rows = [row for row in range(95, 150) if cv2.countNonZero(mask[row:row + 1, :]) == 0]
    assert gap_rows, "as duas linhas de legenda se fundiram num bloco só"


def test_mask_covers_isolated_bright_speck() -> None:
    """Um pingo claro solto (acento, tittle do 'i', resto de palavra anterior) também
    entra na máscara. Sem isso ele sobrava como um pontinho amarelo/branco no resultado.
    O halo por glifo garante cobertura mesmo quando o pingo não forma linha."""
    import cv2

    height, width = 240, 480
    frame = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)
    # um único pingo amarelo isolado, longe de qualquer linha de texto
    cv2.circle(frame, (240, 120), 4, (40, 230, 240), -1)
    region = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    mask = SubtitleFrameCleaner.mask_for(frame, [region])
    assert mask[120, 240] == 255, "o pingo claro solto não foi coberto pela máscara"


def test_reconstruction_preserves_every_frame(tmp_path: Path) -> None:
    """Nenhum quadro pode se perder entre a leitura e a regravação.

    Antes a leitura era feita pelo OpenCV e a gravação pelo FFmpeg; cada quadro que o
    OpenCV não devolvia encurtava o resultado e derrubava a verificação final.
    A região fica fora da janela de tempo do vídeo de propósito: assim o caminho de
    quadros é exercitado sem exigir o pacote de IA instalado.
    """
    import subprocess

    source = tmp_path / "origem.mp4"
    destination = tmp_path / "limpo.mp4"
    subprocess.run([
        find_binary("ffmpeg"), "-hide_banner", "-y",
        "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30000/1001:d=4",
        "-c:v", "libx264", "-preset", "ultrafast", "-bf", "3", "-g", "250", "-pix_fmt", "yuv420p", str(source),
    ], capture_output=True, timeout=120, check=True)
    info = probe_video(source)
    far_future = {"x": 0.1, "y": 0.8, "width": 0.5, "height": 0.1, "startMs": 10_000_000, "endMs": 10_001_000, "enabled": True}
    stats = remove_burned_subtitles(source, destination, info, [far_future], NoModelNeeded())

    assert int(stats["frames"]) == int(info["frameCount"])
    result = probe_video(destination)
    assert int(result["frameCount"]) == int(info["frameCount"])
    assert abs(float(result["videoDuration"]) - float(info["videoDuration"])) < 0.05


def test_temporal_recovery_stops_before_the_patch_smears() -> None:
    """A recuperação temporal (modo rápido/CPU) não pode ser propagada sem limite.

    Cada propagação deforma o remendo do quadro anterior. Sem um teto, uma legenda
    de vinte segundos arrastava o mesmo remendo por centenas de quadros e o
    resultado virava uma mancha. Passado o teto, o LaMa precisa rodar de novo. Com a
    região estável (quadro anterior igual ao atual) a propagação de fato ocorre.
    """
    import cv2

    height, width = 90, 160
    background = np.full((height, width, 3), (80, 120, 160), dtype=np.uint8)
    frame = background.copy()
    cv2.putText(frame, "TESTE", (45, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    region = {"x": 0.2, "y": 0.55, "width": 0.65, "height": 0.35, "startMs": 0, "endMs": 60_000, "enabled": True}

    cleaner = SubtitleFrameCleaner(NoModelNeeded(), fps=10.0)
    # Sem modelo, a sessão nunca é criada: o teto fica na janela padrão de CPU (fps/2).
    assert cleaner._max_propagation == 5
    # Simula um quadro-chave anterior já resolvido, para exercitar só a propagação.
    cleaner._last_mask = SubtitleFrameCleaner.mask_for(frame, [region])
    for step in range(5):
        # quadro anterior IGUAL ao atual: a área da legenda está estável e propaga.
        cleaner.clean(frame, [region], frame, background)
        assert cleaner._propagated == step + 1
    # Esgotado o teto, o caminho volta para o LaMa — que aqui não está instalado.
    with pytest.raises(SubtitleRemovalError, match="não está instalado"):
        cleaner.clean(frame, [region], frame, background)


def test_temporal_recovery_changes_only_mask_area() -> None:
    import cv2
    height, width = 90, 160
    background = np.full((height, width, 3), (80, 120, 160), dtype=np.uint8)
    current = background.copy()
    cv2.putText(current, "TESTE", (45, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    region = {"x": 0.2, "y": 0.55, "width": 0.65, "height": 0.35, "startMs": 0, "endMs": 1000, "enabled": True}
    # quadro anterior IGUAL ao atual (região estável) => propaga o limpo anterior.
    cleaner = SubtitleFrameCleaner(NoModelNeeded())
    cleaner._last_mask = SubtitleFrameCleaner.mask_for(current, [region])
    cleaned = cleaner.clean(current, [region], current, background)
    mask = SubtitleFrameCleaner.mask_for(current, [region])
    assert np.array_equal(cleaned[mask == 0], current[mask == 0])
    assert float(np.mean(np.abs(cleaned[mask > 0].astype(float) - current[mask > 0].astype(float)))) > 1


# --------------------------------------------------------------------------
# Modo AUTOMÁTICO "tela toda menos rosto" (receita da Colab, v2.4.1).
# --------------------------------------------------------------------------
def test_auto_params_match_validated_recipe_and_override(monkeypatch) -> None:
    """Os valores-padrão têm de ser exatamente os da v2.4.1 validada; as opções e as
    variáveis de ambiente podem sobrepor para trocar velocidade por fidelidade."""
    params = _auto_params()
    assert params["ocr_cada"] == 1        # OCR em TODO quadro (pega o flash de 1 quadro)
    assert params["min_text_h"] == 22     # protege o texto pequeno da Bíblia
    assert params["dil"] == 6
    assert params["temporal_w"] == 3
    assert params["min_area"] == 60
    assert params["n_samp"] == 200
    assert abs(float(params["face_freq"]) - 0.30) < 1e-9

    assert _auto_params({"minTextH": 40})["min_text_h"] == 40
    monkeypatch.setenv("OFUSCADOR_OCR_CADA", "4")
    assert _auto_params()["ocr_cada"] == 4
    # a opção explícita ganha da variável de ambiente
    assert _auto_params({"ocrEvery": 2})["ocr_cada"] == 2


def test_stamp_face_expands_box_and_can_ignore_lower_half() -> None:
    height, width = 200, 400
    target = np.zeros((height, width), np.float32)
    _stamp_face(target, [(100, 60, 40, 40)], height, width)
    # caixa ampliada: y 52..99, x 92..147 (folga de 20% em volta) recebe +1
    assert target[70, 120] == 1
    assert target[10, 10] == 0
    # only_upper descarta rosto cuja metade cai na parte de baixo da tela
    lower = np.zeros((height, width), np.float32)
    _stamp_face(lower, [(100, 150, 40, 40)], height, width, only_upper=True)
    assert lower.sum() == 0


def test_load_face_cascade_uses_bundled_opencv_data() -> None:
    """O XML do Haar acompanha o opencv-python; o carregador tem de encontrá-lo mesmo sem
    a pasta de modelos (models_dir() == None), senão o rosto ficaria sem proteção."""
    cascade = _load_face_cascade(NoModelNeeded())
    assert cascade is not None


def test_inpaint_components_only_repaints_inside_the_mask() -> None:
    """O inpaint por componente só pode alterar os pixels da máscara; o resto do quadro
    (rosto, cenário) fica intacto. Dois blocos separados são reconstruídos cada um no seu
    recorte."""
    cleaner = SubtitleFrameCleaner(NoModelNeeded())
    # troca o motor LaMa por um substituto: devolve um bloco 512 de cor sólida, sem exigir
    # o modelo ONNX instalado — o que se testa aqui é a COLAGEM por componente.
    cleaner._run_lama_512 = lambda image_512, mask_512: np.full(image_512.shape, 200, np.uint8)
    frame = np.full((120, 200, 3), 50, np.uint8)
    mask = np.zeros((120, 200), np.uint8)
    mask[15:35, 20:70] = 255     # bloco A (rodapé esquerdo)
    mask[80:100, 140:190] = 255  # bloco B (canto oposto)
    out = cleaner.inpaint_components(frame, mask)
    assert np.array_equal(out[mask == 0], frame[mask == 0]), "mexeu fora da máscara"
    assert not np.array_equal(out[mask > 0], frame[mask > 0]), "não reconstruiu dentro da máscara"
    assert int(out[25, 45, 0]) == 200 and int(out[90, 165, 0]) == 200


def test_auto_mode_requires_the_ai_pack() -> None:
    """Sem o pacote de IA, o modo automático falha CEDO e com mensagem clara — não no meio
    do vídeo depois de horas."""
    probe = {
        "width": 320, "height": 180, "fps": 30.0, "duration": 4.0, "videoDuration": 4.0,
        "hdr": False, "frameCount": 120, "frameRateRational": "30/1",
    }
    with pytest.raises(SubtitleRemovalError):
        remove_burned_subtitles_auto(Path("nao-existe.mp4"), Path("saida.mp4"), probe, NoModelNeeded())
