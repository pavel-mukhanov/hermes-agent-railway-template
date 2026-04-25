import asyncio
import base64
import json
import logging
import os
import re
import secrets
import signal
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    SimpleUser,
)
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.templating import Jinja2Templates

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_BUFFER_LINES = int(os.environ.get("LOG_BUFFER_LINES", "1000"))
SERVICE_LOGS: deque[str] = deque(maxlen=LOG_BUFFER_LINES)


class RingBufferLogHandler(logging.Handler):
    def __init__(self, buffer: deque[str]):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord):
        try:
            self.buffer.append(self.format(record))
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
)
logger = logging.getLogger("hermes-railway")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if not any(isinstance(h, RingBufferLogHandler) for h in logger.handlers):
    ring_buffer_handler = RingBufferLogHandler(SERVICE_LOGS)
    ring_buffer_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    ring_buffer_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ring_buffer_handler)

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(16)
    logger.warning("Generated admin password (ADMIN_PASSWORD was empty): %s", ADMIN_PASSWORD)

HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
ENV_FILE_PATH = Path(HERMES_HOME) / ".env"
CONFIG_FILE_PATH = Path(HERMES_HOME) / "config.yaml"
PAIRING_DIR = Path(HERMES_HOME) / "pairing"
CODE_TTL_SECONDS = 3600

# Registry of known Hermes env vars exposed in the UI.
# Each entry: (key, label, category, is_password)
ENV_VAR_DEFS = [
    # Model
    ("HERMES_MODEL", "Model", "model", False),
    # Providers
    ("OPENROUTER_API_KEY", "OpenRouter API Key", "provider", True),
    ("DEEPSEEK_API_KEY", "DeepSeek API Key", "provider", True),
    ("DASHSCOPE_API_KEY", "DashScope API Key", "provider", True),
    ("GLM_API_KEY", "GLM / Z.AI API Key", "provider", True),
    ("KIMI_API_KEY", "Kimi API Key", "provider", True),
    ("MINIMAX_API_KEY", "MiniMax API Key", "provider", True),
    ("HF_TOKEN", "Hugging Face Token", "provider", True),
    # Tools
    ("PARALLEL_API_KEY", "Parallel API Key", "tool", True),
    ("FIRECRAWL_API_KEY", "Firecrawl API Key", "tool", True),
    ("TAVILY_API_KEY", "Tavily API Key", "tool", True),
    ("FAL_KEY", "FAL API Key", "tool", True),
    ("BROWSERBASE_API_KEY", "Browserbase API Key", "tool", True),
    ("BROWSERBASE_PROJECT_ID", "Browserbase Project ID", "tool", False),
    ("GITHUB_TOKEN", "GitHub Token", "tool", True),
    ("VOICE_TOOLS_OPENAI_KEY", "OpenAI Voice Key", "tool", True),
    ("HONCHO_API_KEY", "Honcho API Key", "tool", True),
    # Messaging — Telegram
    ("TELEGRAM_BOT_TOKEN", "Telegram Bot Token", "messaging", True),
    ("TELEGRAM_ALLOWED_USERS", "Telegram Allowed Users", "messaging", False),
    # Messaging — Discord
    ("DISCORD_BOT_TOKEN", "Discord Bot Token", "messaging", True),
    ("DISCORD_ALLOWED_USERS", "Discord Allowed Users", "messaging", False),
    # Messaging — Slack
    ("SLACK_BOT_TOKEN", "Slack Bot Token", "messaging", True),
    ("SLACK_APP_TOKEN", "Slack App Token", "messaging", True),
    # Messaging — WhatsApp
    ("WHATSAPP_ENABLED", "WhatsApp Enabled", "messaging", False),
    # Messaging — Email
    ("EMAIL_ADDRESS", "Email Address", "messaging", False),
    ("EMAIL_PASSWORD", "Email Password", "messaging", True),
    ("EMAIL_IMAP_HOST", "Email IMAP Host", "messaging", False),
    ("EMAIL_SMTP_HOST", "Email SMTP Host", "messaging", False),
    # Messaging — Mattermost
    ("MATTERMOST_URL", "Mattermost URL", "messaging", False),
    ("MATTERMOST_TOKEN", "Mattermost Token", "messaging", True),
    # Messaging — Matrix
    ("MATRIX_HOMESERVER", "Matrix Homeserver", "messaging", False),
    ("MATRIX_ACCESS_TOKEN", "Matrix Access Token", "messaging", True),
    ("MATRIX_USER_ID", "Matrix User ID", "messaging", False),
    # Messaging — General
    ("GATEWAY_ALLOW_ALL_USERS", "Allow All Users", "messaging", False),
]

PASSWORD_KEYS = {key for key, _, _, is_pw in ENV_VAR_DEFS if is_pw}

PROVIDER_KEYS = [key for key, _, cat, _ in ENV_VAR_DEFS if cat == "provider"]
CHANNEL_KEYS = {
    "Telegram": "TELEGRAM_BOT_TOKEN",
    "Discord": "DISCORD_BOT_TOKEN",
    "Slack": "SLACK_BOT_TOKEN",
    "WhatsApp": "WHATSAPP_ENABLED",
    "Email": "EMAIL_ADDRESS",
    "Mattermost": "MATTERMOST_TOKEN",
    "Matrix": "MATRIX_ACCESS_TOKEN",
}
MODEL_PROVIDER_KEYS = [
    ("OPENROUTER_API_KEY", "openrouter"),
    ("DEEPSEEK_API_KEY", "deepseek"),
    ("DASHSCOPE_API_KEY", "alibaba"),
    ("GLM_API_KEY", "zai"),
    ("KIMI_API_KEY", "kimi-coding"),
    ("MINIMAX_API_KEY", "minimax"),
    ("HF_TOKEN", "huggingface"),
]


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def write_env_file(path: Path, env_vars: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)

    categories = {"model": "Model", "provider": "Providers", "tool": "Tools", "messaging": "Messaging"}
    grouped: dict[str, list[str]] = {cat: [] for cat in categories}
    known_keys = {key for key, _, _, _ in ENV_VAR_DEFS}
    key_to_cat = {key: cat for key, _, cat, _ in ENV_VAR_DEFS}

    for key, value in env_vars.items():
        if not value:
            continue
        cat = key_to_cat.get(key, "other")
        line = f"{key}={value}"
        if cat in grouped:
            grouped[cat].append(line)
        else:
            grouped.setdefault("other", []).append(line)

    lines = []
    for cat, heading in categories.items():
        entries = grouped.get(cat, [])
        if entries:
            lines.append(f"# {heading}")
            lines.extend(sorted(entries))
            lines.append("")

    other = grouped.get("other", [])
    if other:
        lines.append("# Other")
        lines.extend(sorted(other))
        lines.append("")

    path.write_text("\n".join(lines) + "\n" if lines else "")


def mask_secrets(env_vars: dict[str, str]) -> dict[str, str]:
    result = {}
    for key, value in env_vars.items():
        if key in PASSWORD_KEYS and value:
            result[key] = value[:8] + "***" if len(value) > 8 else "***"
        else:
            result[key] = value
    return result


def merge_secrets(new_vars: dict[str, str], existing_vars: dict[str, str]) -> dict[str, str]:
    result = {}
    for key, value in new_vars.items():
        if key in PASSWORD_KEYS and value.endswith("***"):
            result[key] = existing_vars.get(key, "")
        else:
            result[key] = value
    return result


def normalize_model_env_vars(env_vars: dict[str, str]) -> dict[str, str]:
    """
    Keep model env vars backward-compatible:
    - New key: HERMES_MODEL
    - Legacy key: LLM_MODEL
    """
    normalized = dict(env_vars)
    hermes_model = (normalized.get("HERMES_MODEL") or "").strip()
    legacy_model = (normalized.get("LLM_MODEL") or "").strip()
    model = hermes_model or legacy_model
    normalized["HERMES_MODEL"] = model
    # Kept for compatibility with older clients reading /api/config.
    normalized["LLM_MODEL"] = model
    return normalized


def log_model_settings(env_vars: dict[str, str], context: str):
    normalized = normalize_model_env_vars(env_vars)
    logger.info(
        "%s model configuration: HERMES_MODEL=%r, LLM_MODEL=%r, config_yaml=%s",
        context,
        normalized.get("HERMES_MODEL", ""),
        normalized.get("LLM_MODEL", ""),
        CONFIG_FILE_PATH,
    )


def count_configured_channels(env_vars: dict[str, str]) -> int:
    configured = 0
    for key in CHANNEL_KEYS.values():
        val = env_vars.get(key, "")
        if bool(val) and val.lower() not in ("false", "0", "no"):
            configured += 1
    return configured


def yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def infer_model_provider(env_vars: dict[str, str], model: str) -> str:
    explicit_provider = (env_vars.get("HERMES_INFERENCE_PROVIDER") or "").strip()
    if explicit_provider and explicit_provider != "auto":
        return explicit_provider

    configured = [provider for key, provider in MODEL_PROVIDER_KEYS if env_vars.get(key)]
    if len(configured) == 1:
        return configured[0]
    if env_vars.get("OPENROUTER_API_KEY") and "/" in model:
        return "openrouter"
    return configured[0] if configured else "openrouter"


def read_configured_model_from_config_file() -> str:
    if not CONFIG_FILE_PATH.exists():
        return ""

    lines = CONFIG_FILE_PATH.read_text().splitlines()
    in_model = False
    for line in lines:
        if line.strip() == "model:" and line == line.lstrip():
            in_model = True
            continue
        if in_model and line == line.lstrip() and line.strip():
            break
        if in_model:
            stripped = line.strip()
            if stripped.startswith("default:"):
                value = stripped.partition(":")[2].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1].replace("''", "'")
                return value
    return ""


def render_model_config_block(env_vars: dict[str, str]) -> list[str]:
    normalized = normalize_model_env_vars(env_vars)
    model = normalized.get("HERMES_MODEL", "").strip()
    if not model:
        return []

    provider = infer_model_provider(normalized, model)
    return [
        "model:",
        f"  default: {yaml_quote(model)}",
        f"  provider: {yaml_quote(provider)}",
    ]


def merge_model_config_block(existing_block: list[str], model_block: list[str]) -> list[str]:
    replacements = {}
    for line in model_block[1:]:
        key = line.strip().partition(":")[0]
        replacements[key] = line

    output = ["model:"]
    seen = set()
    for line in existing_block:
        stripped = line.strip()
        key = stripped.partition(":")[0]
        if key in replacements:
            output.append(replacements[key])
            seen.add(key)
        else:
            output.append(line)

    insert_at = 1
    for key in ("default", "provider"):
        if key not in seen:
            output.insert(insert_at, replacements[key])
            insert_at += 1
    return output


def sync_model_config_file(env_vars: dict[str, str], context: str):
    normalized = normalize_model_env_vars(env_vars)
    if not normalized.get("HERMES_MODEL", "").strip():
        configured_model = read_configured_model_from_config_file()
        if configured_model:
            normalized["HERMES_MODEL"] = configured_model
            normalized["LLM_MODEL"] = configured_model

    model_block = render_model_config_block(normalized)
    if not model_block:
        logger.warning("%s skipped config.yaml model sync: HERMES_MODEL is empty", context)
        return

    CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = CONFIG_FILE_PATH.read_text().splitlines() if CONFIG_FILE_PATH.exists() else []
    output: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        line = lines[i]
        if line.strip() == "model:" and line == line.lstrip():
            replaced = True
            i += 1
            existing_block = []
            while i < len(lines) and (not lines[i].strip() or lines[i] != lines[i].lstrip()):
                existing_block.append(lines[i])
                i += 1
            output.extend(merge_model_config_block(existing_block, model_block))
            continue
        output.append(line)
        i += 1

    if not replaced:
        if output and output[-1].strip():
            output.append("")
        output.extend(model_block)

    CONFIG_FILE_PATH.write_text("\n".join(output) + "\n")
    logger.info(
        "%s synced config.yaml model: default=%r, provider=%r, path=%s",
        context,
        normalized.get("HERMES_MODEL", ""),
        infer_model_provider(normalized, normalized.get("HERMES_MODEL", "")),
        CONFIG_FILE_PATH,
    )


class BasicAuthBackend(AuthenticationBackend):
    async def authenticate(self, conn):
        if "Authorization" not in conn.headers:
            return None

        auth = conn.headers["Authorization"]
        try:
            scheme, credentials = auth.split()
            if scheme.lower() != "basic":
                return None
            decoded = base64.b64decode(credentials).decode("ascii")
        except (ValueError, UnicodeDecodeError):
            raise AuthenticationError("Invalid credentials")

        username, _, password = decoded.partition(":")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            return AuthCredentials(["authenticated"]), SimpleUser(username)

        raise AuthenticationError("Invalid credentials")


def require_auth(request: Request):
    if not request.user.is_authenticated:
        return PlainTextResponse(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="hermes"'},
        )
    return None


class GatewayManager:
    def __init__(self, log_buffer: deque[str] | None = None):
        self.process: asyncio.subprocess.Process | None = None
        self.state = "stopped"
        self.logs: deque[str] = log_buffer if log_buffer is not None else deque(maxlen=500)
        self.start_time: float | None = None
        self.restart_count = 0
        self._read_tasks: list[asyncio.Task] = []

    async def start(self):
        if self.process and self.process.returncode is None:
            logger.info("Gateway start requested but process is already running (pid=%s)", self.process.pid)
            return
        self.state = "starting"
        logger.info("Starting Hermes gateway")
        try:
            env = os.environ.copy()
            env["HERMES_HOME"] = HERMES_HOME
            env_vars = normalize_model_env_vars(read_env_file(ENV_FILE_PATH))
            sync_model_config_file(env_vars, "Gateway start")
            env.update(env_vars)
            log_model_settings(env_vars, "Gateway start")
            provider_count = sum(1 for key in PROVIDER_KEYS if env_vars.get(key))
            channel_count = count_configured_channels(env_vars)
            logger.info(
                "Gateway start config summary: providers_configured=%d, channels_configured=%d",
                provider_count,
                channel_count,
            )

            self.process = await asyncio.create_subprocess_exec(
                "hermes", "gateway",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            self.state = "running"
            self.start_time = time.time()
            task = asyncio.create_task(self._read_output())
            self._read_tasks.append(task)
            logger.info("Hermes gateway started (pid=%s)", self.process.pid)
        except Exception as e:
            self.state = "error"
            self.logs.append(f"Failed to start gateway: {e}")
            logger.exception("Failed to start Hermes gateway")

    async def stop(self):
        if not self.process or self.process.returncode is not None:
            self.state = "stopped"
            logger.info("Gateway stop requested but process is already stopped")
            return
        self.state = "stopping"
        logger.info("Stopping Hermes gateway (pid=%s)", self.process.pid)
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("Gateway did not stop gracefully in time, killing process (pid=%s)", self.process.pid)
            self.process.kill()
            await self.process.wait()
        self.state = "stopped"
        self.start_time = None
        logger.info("Hermes gateway stopped")

    async def restart(self):
        logger.info("Restarting Hermes gateway")
        await self.stop()
        self.restart_count += 1
        await self.start()

    async def _read_output(self):
        try:
            while self.process and self.process.stdout:
                line = await self.process.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                cleaned = ANSI_ESCAPE.sub("", decoded)
                self.logs.append(cleaned)
        except asyncio.CancelledError:
            return
        if self.process and self.process.returncode is not None and self.state == "running":
            self.state = "error"
            self.logs.append(f"Gateway exited with code {self.process.returncode}")
            logger.error("Gateway exited unexpectedly with code %s", self.process.returncode)

    def get_status(self) -> dict:
        pid = None
        if self.process and self.process.returncode is None:
            pid = self.process.pid
        uptime = None
        if self.start_time and self.state == "running":
            uptime = int(time.time() - self.start_time)
        return {
            "state": self.state,
            "pid": pid,
            "uptime": uptime,
            "restart_count": self.restart_count,
        }


gateway = GatewayManager(log_buffer=SERVICE_LOGS)
config_lock = asyncio.Lock()


async def homepage(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    return templates.TemplateResponse(request, "index.html")


async def health(request: Request):
    return JSONResponse({"status": "ok", "gateway": gateway.state})


async def api_config_get(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    async with config_lock:
        env_vars = normalize_model_env_vars(read_env_file(ENV_FILE_PATH))
    defs = [
        {"key": key, "label": label, "category": cat, "password": is_pw}
        for key, label, cat, is_pw in ENV_VAR_DEFS
    ]
    return JSONResponse({"vars": mask_secrets(env_vars), "defs": defs})


async def api_config_put(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    try:
        restart = body.pop("_restartGateway", False)
        new_vars = body.get("vars", {})

        async with config_lock:
            existing = read_env_file(ENV_FILE_PATH)
            merged = merge_secrets(new_vars, existing)
            # Preserve any existing vars not in the UI
            for key, value in existing.items():
                if key not in merged:
                    merged[key] = value
            normalized = normalize_model_env_vars(merged)
            # Persist only canonical model key.
            normalized.pop("LLM_MODEL", None)
            write_env_file(ENV_FILE_PATH, normalized)
            sync_model_config_file(normalized, "Config save")
            provider_count = sum(1 for key in PROVIDER_KEYS if normalized.get(key))
            channel_count = count_configured_channels(normalized)
            log_model_settings(normalized, "Config save")
            logger.info(
                "Configuration saved: restart_requested=%s, providers_configured=%d, channels_configured=%d",
                restart,
                provider_count,
                channel_count,
            )

        if restart:
            asyncio.create_task(gateway.restart())

        return JSONResponse({"ok": True, "restarting": restart})
    except Exception as e:
        logger.exception("Config save error: %s: %s", type(e).__name__, e)
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_status(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err

    env_vars = read_env_file(ENV_FILE_PATH)

    providers = {}
    for key in PROVIDER_KEYS:
        label = key.replace("_API_KEY", "").replace("_TOKEN", "").replace("HF_", "HuggingFace ").replace("_", " ").title()
        providers[label] = {"configured": bool(env_vars.get(key))}

    channels = {}
    for name, key in CHANNEL_KEYS.items():
        val = env_vars.get(key, "")
        channels[name] = {"configured": bool(val) and val.lower() not in ("false", "0", "no")}

    return JSONResponse({
        "gateway": gateway.get_status(),
        "providers": providers,
        "channels": channels,
    })


async def api_logs(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    return JSONResponse({"lines": list(gateway.logs)})


async def api_gateway_start(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    logger.info("API gateway start requested by user=%s", getattr(request.user, "display_name", "unknown"))
    asyncio.create_task(gateway.start())
    return JSONResponse({"ok": True})


async def api_gateway_stop(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    logger.info("API gateway stop requested by user=%s", getattr(request.user, "display_name", "unknown"))
    asyncio.create_task(gateway.stop())
    return JSONResponse({"ok": True})


async def api_gateway_restart(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    logger.info("API gateway restart requested by user=%s", getattr(request.user, "display_name", "unknown"))
    asyncio.create_task(gateway.restart())
    return JSONResponse({"ok": True})


def _load_pairing_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_pairing_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _pairing_platforms(suffix: str) -> list[str]:
    if not PAIRING_DIR.exists():
        return []
    return [
        f.stem.rsplit(f"-{suffix}", 1)[0]
        for f in PAIRING_DIR.glob(f"*-{suffix}.json")
    ]


async def api_pairing_pending(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    now = time.time()
    results = []
    for platform in _pairing_platforms("pending"):
        pending = _load_pairing_json(PAIRING_DIR / f"{platform}-pending.json")
        for code, info in pending.items():
            age = now - info.get("created_at", now)
            if age > CODE_TTL_SECONDS:
                continue
            results.append({
                "platform": platform,
                "code": code,
                "user_id": info.get("user_id", ""),
                "user_name": info.get("user_name", ""),
                "age_minutes": int(age / 60),
            })
    return JSONResponse({"pending": results})


async def api_pairing_approve(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    platform = body.get("platform", "")
    code = body.get("code", "").upper().strip()
    if not platform or not code:
        return JSONResponse({"error": "platform and code required"}, status_code=400)

    pending_path = PAIRING_DIR / f"{platform}-pending.json"
    pending = _load_pairing_json(pending_path)
    if code not in pending:
        return JSONResponse({"error": "Code not found or expired"}, status_code=404)

    entry = pending.pop(code)
    _save_pairing_json(pending_path, pending)

    approved_path = PAIRING_DIR / f"{platform}-approved.json"
    approved = _load_pairing_json(approved_path)
    approved[entry["user_id"]] = {
        "user_name": entry.get("user_name", ""),
        "approved_at": time.time(),
    }
    _save_pairing_json(approved_path, approved)
    logger.info(
        "Pairing approved: platform=%s user_id=%s user_name=%r",
        platform,
        entry["user_id"],
        entry.get("user_name", ""),
    )

    return JSONResponse({"ok": True, "user_id": entry["user_id"], "user_name": entry.get("user_name", "")})


async def api_pairing_deny(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    platform = body.get("platform", "")
    code = body.get("code", "").upper().strip()
    if not platform or not code:
        return JSONResponse({"error": "platform and code required"}, status_code=400)

    pending_path = PAIRING_DIR / f"{platform}-pending.json"
    pending = _load_pairing_json(pending_path)
    if code in pending:
        del pending[code]
        _save_pairing_json(pending_path, pending)
        logger.info("Pairing denied: platform=%s code=%s", platform, code)

    return JSONResponse({"ok": True})


async def api_pairing_approved(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    results = []
    for platform in _pairing_platforms("approved"):
        approved = _load_pairing_json(PAIRING_DIR / f"{platform}-approved.json")
        for user_id, info in approved.items():
            results.append({
                "platform": platform,
                "user_id": user_id,
                "user_name": info.get("user_name", ""),
                "approved_at": info.get("approved_at", 0),
            })
    return JSONResponse({"approved": results})


async def api_pairing_revoke(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    platform = body.get("platform", "")
    user_id = body.get("user_id", "")
    if not platform or not user_id:
        return JSONResponse({"error": "platform and user_id required"}, status_code=400)

    approved_path = PAIRING_DIR / f"{platform}-approved.json"
    approved = _load_pairing_json(approved_path)
    if user_id in approved:
        del approved[user_id]
        _save_pairing_json(approved_path, approved)
        logger.info("Pairing revoked: platform=%s user_id=%s", platform, user_id)

    return JSONResponse({"ok": True})


async def auto_start_gateway():
    env_vars = read_env_file(ENV_FILE_PATH)
    has_provider = any(env_vars.get(key) for key in PROVIDER_KEYS)
    sync_model_config_file(env_vars, "Service startup")
    log_model_settings(env_vars, "Service startup")
    provider_count = sum(1 for key in PROVIDER_KEYS if env_vars.get(key))
    channel_count = count_configured_channels(env_vars)
    logger.info(
        "Service startup config summary: providers_configured=%d, channels_configured=%d, auto_start_gateway=%s",
        provider_count,
        channel_count,
        has_provider,
    )
    if has_provider:
        asyncio.create_task(gateway.start())


routes = [
    Route("/", homepage),
    Route("/health", health),
    Route("/api/config", api_config_get, methods=["GET"]),
    Route("/api/config", api_config_put, methods=["PUT"]),
    Route("/api/status", api_status),
    Route("/api/logs", api_logs),
    Route("/api/gateway/start", api_gateway_start, methods=["POST"]),
    Route("/api/gateway/stop", api_gateway_stop, methods=["POST"]),
    Route("/api/gateway/restart", api_gateway_restart, methods=["POST"]),
    Route("/api/pairing/pending", api_pairing_pending),
    Route("/api/pairing/approve", api_pairing_approve, methods=["POST"]),
    Route("/api/pairing/deny", api_pairing_deny, methods=["POST"]),
    Route("/api/pairing/approved", api_pairing_approved),
    Route("/api/pairing/revoke", api_pairing_revoke, methods=["POST"]),
]

@asynccontextmanager
async def lifespan(app):
    await auto_start_gateway()
    yield
    await gateway.stop()


app = Starlette(
    routes=routes,
    middleware=[Middleware(AuthenticationMiddleware, backend=BasicAuthBackend())],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    logger.info(
        "Starting Hermes Railway service: port=%s, hermes_home=%s, env_file=%s",
        port,
        HERMES_HOME,
        ENV_FILE_PATH,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)

    def handle_signal():
        loop.create_task(gateway.stop())
        server.should_exit = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    loop.run_until_complete(server.serve())
