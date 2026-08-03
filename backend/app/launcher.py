from __future__ import annotations

import argparse
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
import json
from pathlib import Path


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "endereço deste computador"


def serve(network: bool, port: int, pin: str | None) -> None:
    if network:
        os.environ["OFUSCADOR_NETWORK_MODE"] = "1"
        if pin:
            os.environ["OFUSCADOR_PIN"] = pin
    else:
        os.environ.pop("OFUSCADOR_NETWORK_MODE", None)
        os.environ.pop("OFUSCADOR_PIN", None)
    try:
        import uvicorn
        from .main import app
        uvicorn.run(
            app,
            host="0.0.0.0" if network else "127.0.0.1",
            port=port,
            log_config=None,
            access_log=False,
            server_header=False,
        )
    except BaseException:
        log_directory = Path(sys.executable).resolve().parent / "Dados" if getattr(sys, "frozen", False) else Path.cwd() / ".local-data"
        try:
            log_directory.mkdir(parents=True, exist_ok=True)
            (log_directory / "erro-inicializacao.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        raise


def child_command(port: int, network: bool, pin: str | None) -> list[str]:
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--serve", "--port", str(port)]
    else:
        command = [sys.executable, "-m", "app.launcher", "--serve", "--port", str(port)]
    if network:
        command.append("--lan")
        command.extend(["--pin", pin or ""])
    return command


def run_launcher() -> None:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("Ofuscador Elite")
    root.geometry("590x440")
    root.minsize(540, 410)
    root.configure(bg="#f2f5f4")
    process: subprocess.Popen[bytes] | None = None

    title = tk.Label(root, text="OFUSCADOR ELITE", fg="#0f6f5d", bg="#f2f5f4", font=("Segoe UI", 16, "bold"))
    title.pack(pady=(34, 3))
    subtitle = tk.Label(root, text="Processamento privado de vídeo e áudio", fg="#667281", bg="#f2f5f4", font=("Segoe UI", 10))
    subtitle.pack()
    panel = tk.Frame(root, bg="#ffffff", highlightbackground="#dbe2e2", highlightthickness=1)
    panel.pack(fill="both", expand=True, padx=34, pady=25)
    status = tk.Label(panel, text="Escolha como deseja abrir o aplicativo", fg="#1c2836", bg="#ffffff", font=("Segoe UI", 12, "bold"))
    status.pack(pady=(28, 8))
    detail = tk.Label(panel, text="O modo local é recomendado. Nada sai deste computador.", fg="#74808c", bg="#ffffff", font=("Segoe UI", 9), wraplength=450, justify="center")
    detail.pack(pady=(0, 20))
    pin_label = tk.Label(panel, text="", fg="#0f6f5d", bg="#ffffff", font=("Consolas", 18, "bold"))
    pin_label.pack()

    def set_buttons(enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        local_button.configure(state=state)
        lan_button.configure(state=state)

    def open_when_ready(url: str) -> None:
        for _ in range(80):
            try:
                urllib.request.urlopen(f"{url}/api/health", timeout=0.5).read()
                root.after(0, lambda: status.configure(text="Aplicativo em execução"))
                root.after(0, lambda: detail.configure(text="Seu navegador foi aberto. Mantenha esta janela aberta enquanto estiver usando o aplicativo."))
                webbrowser.open(url)
                return
            except Exception:
                time.sleep(0.2)
        root.after(0, lambda: set_buttons(True))
        root.after(0, lambda: messagebox.showerror("Não foi possível iniciar", "O serviço local não respondeu. Verifique se outro programa de segurança bloqueou o aplicativo."))

    def start(network: bool) -> None:
        nonlocal process
        if process and process.poll() is None:
            webbrowser.open(current_url.get())
            return
        port = available_port()
        pin = f"{secrets.randbelow(1_000_000):06d}" if network else None
        url = f"http://127.0.0.1:{port}"
        current_url.set(url)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(child_command(port, network, pin), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        except OSError as exc:
            messagebox.showerror("Não foi possível iniciar", str(exc))
            return
        set_buttons(False)
        status.configure(text="Iniciando com acesso pela rede…" if network else "Iniciando no modo local…")
        if network:
            pin_label.configure(text=f"PIN: {pin}")
            detail.configure(text=f"Outros aparelhos na mesma rede podem abrir http://{local_ip()}:{port} e informar este PIN temporário.")
        else:
            pin_label.configure(text="")
            detail.configure(text="Preparando a interface no navegador…")
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    current_url = tk.StringVar(value="")
    local_button = tk.Button(panel, text="Abrir somente neste computador", command=lambda: start(False), bg="#0f6f5d", fg="#ffffff", activebackground="#0a5146", activeforeground="#ffffff", relief="flat", bd=0, padx=22, pady=12, font=("Segoe UI", 10, "bold"), cursor="hand2")
    local_button.pack(pady=(22, 8))
    lan_button = tk.Button(panel, text="Permitir acesso pela rede com PIN", command=lambda: start(True), bg="#eef6f3", fg="#0f6f5d", activebackground="#dcefe8", activeforeground="#0f6f5d", relief="flat", bd=0, padx=22, pady=10, font=("Segoe UI", 9, "bold"), cursor="hand2")
    lan_button.pack()

    def close() -> None:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def self_test() -> int:
    from .config import find_binary, resource_root
    from .main import app
    from .media import has_h264_encoder
    from .piper_local import piper_available
    try:
        import cv2  # noqa: F401
        import onnxruntime  # noqa: F401
        import rapidocr  # noqa: F401
        subtitle_runtime = True
    except Exception:
        subtitle_runtime = False
    checks = {
        "api": app.title == "Ofuscador Elite",
        "interface": (resource_root() / "static" / "index.html").is_file(),
        "ffmpeg": bool(find_binary("ffmpeg")),
        "ffprobe": bool(find_binary("ffprobe")),
        "piper": piper_available(),
        "h264": has_h264_encoder(),
        "subtitle_runtime": subtitle_runtime,
    }
    print(json.dumps(checks))
    # Linha informativa: qual acelerador o LaMa usaria (não conta para passar/falhar,
    # já que o CPU é um fallback válido). Serve para o build/registro confirmar a GPU.
    try:
        import onnxruntime as ort
        available = list(ort.get_available_providers())
        gpu = next((name for name in ("DmlExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider") if name in available), "cpu")
        print(json.dumps({"lama_provider": gpu, "providers": available}))
    except Exception:
        pass
    return 0 if all(checks.values()) else 1


def main() -> None:
    parser = argparse.ArgumentParser(add_help=not getattr(sys, "frozen", False))
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--lan", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pin")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    if args.serve:
        serve(args.lan, args.port, args.pin)
    else:
        run_launcher()


if __name__ == "__main__":
    main()
