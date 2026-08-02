from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
import threading
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode

from fastapi import HTTPException, Request

COOKIE_NAME = "ofuscador_session"


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return host.lower() in {"localhost", "testclient"}


class SessionSecurity:
    def __init__(self, network_mode: bool = False, pin: str | None = None) -> None:
        self.network_mode = network_mode
        self.pin = pin or f"{secrets.randbelow(1_000_000):06d}"
        self.secret = secrets.token_bytes(32)
        self._attempts: dict[str, list[float]] = {}
        self._attempt_lock = threading.Lock()
        self.max_attempts = 5
        self.attempt_window_seconds = 60

    def issue(self) -> str:
        payload = f"{int(time.time())}:{secrets.token_hex(12)}".encode()
        signature = hmac.new(self.secret, payload, hashlib.sha256).digest()
        return urlsafe_b64encode(payload + b"." + signature).decode()

    def verify(self, token: str | None, max_age: int = 12 * 60 * 60) -> bool:
        if not token:
            return False
        try:
            raw = urlsafe_b64decode(token.encode())
            payload, signature = raw.split(b".", 1)
            if not hmac.compare_digest(signature, hmac.new(self.secret, payload, hashlib.sha256).digest()):
                return False
            issued = int(payload.split(b":", 1)[0])
            return 0 <= time.time() - issued <= max_age
        except (ValueError, TypeError):
            return False

    def request_allowed(self, request: Request) -> bool:
        if is_loopback(request.client.host if request.client else None):
            return True
        return self.network_mode and self.verify(request.cookies.get(COOKIE_NAME))

    def authenticate(self, host: str | None, pin: str, *, now: float | None = None) -> tuple[bool, int]:
        """Valida o PIN e limita tentativas por endereço na rede local."""
        current = time.time() if now is None else now
        client = host or "desconhecido"
        with self._attempt_lock:
            recent = [attempt for attempt in self._attempts.get(client, []) if current - attempt < self.attempt_window_seconds]
            if len(recent) >= self.max_attempts:
                self._attempts[client] = recent
                retry_after = max(1, int(self.attempt_window_seconds - (current - recent[0])) + 1)
                return False, retry_after
            if not self.network_mode or not hmac.compare_digest(pin, self.pin):
                recent.append(current)
                self._attempts[client] = recent
                return False, 0
            self._attempts.pop(client, None)
            return True, 0


def require_local(request: Request) -> None:
    if not is_loopback(request.client.host if request.client else None):
        raise HTTPException(status_code=403, detail="Esta configuração só pode ser alterada diretamente neste computador.")
