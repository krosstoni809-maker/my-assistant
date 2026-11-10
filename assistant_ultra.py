"""
AI көмекші ULTRA (Streamlit + Anthropic API)

PRO нұсқасына қосылғандар:
  - чаттар дискіге сақталады, кейін қайта ашуға болады
  - ұзақ мерзімді жады: көмекші сен туралы маңызды нәрсені есте сақтайды
  - дауыспен жазу (Whisper) және дауыспен жауап (edge-tts), қазақша да бар
  - құпиясөз қорғанысы (интернетке шығарғанда)

Орнату:
    pip install -U streamlit anthropic
    pip install faster-whisper edge-tts      # дауыс керек болса (міндетті емес)

Іске қосу:
    streamlit run assistant_ultra.py

Айнымалылар (міндетті емес):
    ANTHROPIC_API_KEY   API кілті
    APP_PASSWORD        қойсаң, сайт құпиясөз сұрайды, Python іске қосу әдепкіде өшеді

НАЗАР: "Python іске қосу" модель жазған кодты сервердің/компьютердің өзінде
орындайды. Сайтты басқаларға ашсаң, оны қоспа.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import anthropic
import streamlit as st

WORKSPACE = Path("workspace")
DATA = Path("data")
CHATS = DATA / "chats"
MEMORY_FILE = DATA / "memory.json"
for d in (WORKSPACE, CHATS):
    d.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Sonnet 5.5 (тең, жылдам)": "claude-sonnet-5-5",
    "Opus 5.5 (ең ақылды)": "claude-opus-5-5",
    "Haiku 5.5 (өте жылдам, арзан)": "claude-haiku-5-5",
}

DEFAULT_SYSTEM = (
    "Сен күшті, сенімді көмекшісің. Қазақша сұраса, қазақша жауап бер. "
    "Күрделі тапсырманы қадамдарға бөліп, құралдарды өзің қолдан: "
    "жаңа ақпарат керек болса интернеттен іздеп, есеп немесе деректі "
    "Python арқылы тексер, файлдарды workspace папкасында сақта. "
    "Пайдаланушы өзі туралы тұрақты маңызды нәрсе айтса (аты, жұмысы, "
    "жобасы, қалауы) немесе 'есіңде сақта' десе, remember құралын қолдан. "
    "Нәтижені тексеріп алып жауап бер. Қысқа әрі нақты жаз."
)

IMAGE_TYPES = {"png", "jpg", "jpeg", "gif", "webp"}
VOICES = {
    "kk": "kk-KZ-AigulNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "en": "en-US-AriaNeural",
}

# ------------------------------------------------------------- Құпиялар


def secret(name: str):
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets.get(name)
    except Exception:
        return None


# ------------------------------------------------------------- Жады


def load_memory() -> list[str]:
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_memory(facts: list[str]) -> None:
    MEMORY_FILE.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------- Чаттар


def chat_path(chat_id: str) -> Path:
    return CHATS / f"{chat_id}.json"


def save_chat() -> None:
    if not st.session_state.display:
        return
    first = next((m["text"] for m in st.session_state.display if m["role"] == "user"), "Чат")
    data = {
        "id": st.session_state.chat_id,
        "title": first.strip().replace("\n", " ")[:45] or "Чат",
        "display": st.session_state.display,
        "api_messages": st.session_state.api_messages,
    }
    chat_path(st.session_state.chat_id).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def list_chats() -> list[tuple[str, str]]:
    items = []
    for p in sorted(CHATS.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            items.append((d["id"], d["title"]))
        except Exception:
            continue
    return items


def new_chat() -> None:
    st.session_state.chat_id = uuid.uuid4().hex[:10]
    st.session_state.display = []
    st.session_state.api_messages = []


def open_chat(chat_id: str) -> None:
    d = json.loads(chat_path(chat_id).read_text(encoding="utf-8"))
    st.session_state.chat_id = d["id"]
    st.session_state.display = d["display"]
    st.session_state.api_messages = d["api_messages"]


# ------------------------------------------------------------- Құралдар


def _safe_path(rel: str) -> Path:
    p = (WORKSPACE / rel).resolve()
    if not p.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("Тек workspace папкасының ішіне рұқсат бар.")
    return p


def run_python(code: str) -> str:
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=WORKSPACE,
        )
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        return out.strip()[:8000] or "(шығыс жоқ)"
    except subprocess.TimeoutExpired:
        return "Қате: код 30 секундта аяқталмады."


def run_tool(name: str, inp: dict) -> tuple[str, bool]:
    try:
        if name == "run_python":
            return run_python(inp["code"]), False
        if name == "read_file":
            return _safe_path(inp["path"]).read_text(encoding="utf-8", errors="replace")[:20000], False
        if name == "write_file":
            p = _safe_path(inp["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inp["content"], encoding="utf-8")
            return f"Сақталды: {inp['path']} ({len(inp['content'])} таңба)", False
        if name == "list_files":
            files = [str(p.relative_to(WORKSPACE)) for p in WORKSPACE.rglob("*") if p.is_file()]
            return "\n".join(sorted(files)) or "(workspace бос)", False
        if name == "remember":
            facts = load_memory()
            fact = inp["fact"].strip()
            if fact and fact not in facts:
                facts.append(fact)
                save_memory(facts)
            return "Жадқа сақталды.", False
        if name == "forget":
            key = inp["text"].strip().lower()
            facts = load_memory()
            kept = [f for f in facts if key not in f.lower()]
            save_memory(kept)
            return f"{len(facts) - len(kept)} жазба өшірілді.", False
        return f"Белгісіз құрал: {name}", True
    except Exception as e:
        return f"Қате: {e}", True


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props)}


CODE_TOOL = {
    "name": "run_python",
    "description": "Python кодын іске қосады (30 сек лимит). Нәтижені көру үшін print() қолдан. "
    "Жұмыс папкасы - workspace. Есептеу, деректі өңдеу, тексеру үшін қолдан.",
    "input_schema": _obj({"code": {"type": "string"}}),
}
FILE_TOOLS = [
    {"name": "read_file", "description": "workspace ішіндегі мәтіндік файлды оқиды.",
     "input_schema": _obj({"path": {"type": "string"}})},
    {"name": "write_file", "description": "workspace ішіне файл жазады (бар болса үстінен жазады).",
     "input_schema": _obj({"path": {"type": "string"}, "content": {"type": "string"}})},
    {"name": "list_files", "description": "workspace ішіндегі барлық файлды көрсетеді.",
     "input_schema": {"type": "object", "properties": {}}},
]
MEMORY_TOOLS = [
    {"name": "remember", "description": "Пайдаланушы туралы тұрақты маңызды фактіні ұзақ жадқа сақтайды. "
     "Бір қысқа, нақты сөйлем жаз.",
     "input_schema": _obj({"fact": {"type": "string"}})},
    {"name": "forget", "description": "Құрамында осы мәтін бар жад жазбаларын өшіреді.",
     "input_schema": _obj({"text": {"type": "string"}})},
]
WEB_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}

# ------------------------------------------------------------- Дауыс


@dataclass
class Prompt:
    text: str = ""
    files: list = field(default_factory=list)


@st.cache_resource(show_spinner="Дауыс моделі жүктелуде...")
def get_whisper():
    from faster_whisper import WhisperModel

    return WhisperModel("small", compute_type="int8")


def transcribe(audio_bytes: bytes, lang: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        segments, _ = get_whisper().transcribe(path, language=None if lang == "auto" else lang)
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        os.unlink(path)


def speak(text: str, lang: str) -> bytes | None:
    try:
        import edge_tts

        clean = re.sub(r"```.*?```", " код үзіндісі ", text, flags=re.S)
        clean = re.sub(r"[*_`#>|]", "", clean)[:1500]
        voice = VOICES.get(lang, VOICES["kk"])

        async def run() -> bytes:
            chunks = []
            async for ch in edge_tts.Communicate(clean, voice).stream():
                if ch["type"] == "audio":
                    chunks.append(ch["data"])
            return b"".join(chunks)

        return asyncio.run(run())
    except Exception:
        return None


# ------------------------------------------------------------- Агент


def build_user_content(prompt) -> list:
    blocks = []
    for f in prompt.files or []:
        data = f.getvalue()
        ext = f.name.rsplit(".", 1)[-1].lower()
        if ext in IMAGE_TYPES:
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": f.type or f"image/{ext}",
                "data": base64.b64encode(data).decode()}})
        elif ext == "pdf":
            blocks.append({"type": "document", "source": {
                "type": "base64", "media_type": "application/pdf",
                "data": base64.b64encode(data).decode()}})
        else:
            text = data.decode("utf-8", errors="replace")[:60000]
            blocks.append({"type": "text", "text": f"[Файл: {f.name}]\n{text}"})
    if prompt.text:
        blocks.append({"type": "text", "text": prompt.text})
    return blocks


def agent_turn(client, model, system, tools, placeholder, log) -> str:
    full_text = ""
    for _ in range(15):
        with client.messages.stream(
            model=model, max_tokens=8000, system=system, tools=tools,
            messages=st.session_state.api_messages,
        ) as stream:
            for chunk in stream.text_stream:
                full_text += chunk
                placeholder.markdown(full_text + "▌")
            final = stream.get_final_message()

        content = [b.model_dump(exclude_none=True) for b in final.content]
        st.session_state.api_messages.append({"role": "assistant", "content": content})

        for b in content:
            if b.get("type") == "server_tool_use" and b.get("name") == "web_search":
                log.append(f"🔎 Іздеу: {b['input'].get('query', '')}")

        if final.stop_reason == "tool_use":
            results = []
            for b in content:
                if b.get("type") == "tool_use":
                    log.append(f"🛠 {b['name']}: {json.dumps(b['input'], ensure_ascii=False)[:300]}")
                    out, is_err = run_tool(b["name"], b["input"])
                    log.append(f"   ↳ {out[:500]}")
                    results.append({"type": "tool_result", "tool_use_id": b["id"],
                                    "content": out, "is_error": is_err})
            st.session_state.api_messages.append({"role": "user", "content": results})
            full_text += "\n\n"
            continue
        if final.stop_reason == "pause_turn":
            continue
        break

    placeholder.markdown(full_text)
    return full_text.strip()


# ------------------------------------------------------------- Интерфейс

st.set_page_config(page_title="AI көмекші ULTRA", page_icon="⚡", layout="centered")

# Құпиясөз қорғанысы
app_password = secret("APP_PASSWORD")
if app_password and not st.session_state.get("authed"):
    st.title("⚡ AI көмекші")
    pw = st.text_input("Құпиясөз", type="password")
    if pw:
        if pw == app_password:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Құпиясөз қате.")
    st.stop()

if "chat_id" not in st.session_state:
    new_chat()

st.title("⚡ AI көмекші ULTRA")

with st.sidebar:
    api_key = secret("ANTHROPIC_API_KEY") or st.text_input("API кілті", type="password")
    st.header("Чаттар")
    if st.button("➕ Жаңа чат", use_container_width=True):
        new_chat()
        st.rerun()
    saved = list_chats()
    if saved:
        ids = [i for i, _ in saved]
        titles = dict(saved)
        cur = st.session_state.chat_id
        pick = st.selectbox(
            "Сақталған чаттар", ids,
            index=ids.index(cur) if cur in ids else None,
            format_func=lambda i: titles[i], placeholder="Таңда...",
        )
        if pick and pick != cur:
            open_chat(pick)
            st.rerun()
        if cur in ids and st.button("🗑 Осы чатты өшіру", use_container_width=True):
            chat_path(cur).unlink(missing_ok=True)
            new_chat()
            st.rerun()

    st.header("Баптау")
    model_name = st.selectbox("Модель", list(MODELS))
    use_web = st.toggle("🔎 Интернеттен іздеу", value=True)
    use_files = st.toggle("📁 Файл құралдары", value=True)
    use_code = st.toggle("🐍 Python іске қосу", value=not app_password)
    use_mem = st.toggle("🧠 Ұзақ жады", value=True)
    system_prompt = st.text_area("Жүйелік нұсқау", DEFAULT_SYSTEM, height=160)

    st.header("Дауыс")
    lang = st.selectbox("Тіл", ["kk", "ru", "en", "auto"])
    tts_on = st.toggle("🔊 Дауыспен жауап беру", value=False)

    if use_mem:
        with st.expander(f"🧠 Жад ({len(load_memory())})"):
            facts = load_memory()
            for i, f in enumerate(facts):
                c1, c2 = st.columns([6, 1])
                c1.write(f"• {f}")
                if c2.button("✕", key=f"del{i}"):
                    save_memory([x for j, x in enumerate(facts) if j != i])
                    st.rerun()
            if not facts:
                st.caption("Жад бос.")

    st.download_button(
        "💾 Чатты жүктеу (JSON)",
        json.dumps(st.session_state.display, ensure_ascii=False, indent=2),
        file_name="chat.json", mime="application/json", use_container_width=True,
    )

if not api_key:
    st.info("Сол жақ панельге API кілтін қой.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

tools = []
if use_web:
    tools.append(WEB_TOOL)
if use_files:
    tools += FILE_TOOLS
if use_code:
    tools.append(CODE_TOOL)
if use_mem:
    tools += MEMORY_TOOLS

system = f"{system_prompt}\n\nБүгінгі күн: {date.today().isoformat()}."
if use_mem and (facts_now := load_memory()):
    system += "\n\nПайдаланушы туралы жадтағы мәліметтер:\n" + "\n".join(f"- {f}" for f in facts_now)

# Тарихты көрсету
for m in st.session_state.display:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("log"):
            with st.expander("Құрал әрекеттері"):
                st.code("\n".join(m["log"]), language=None)

prompt = st.chat_input(
    "Сұрағыңды жаз немесе файл тірке...",
    accept_file="multiple",
    file_type=list(IMAGE_TYPES) + ["pdf", "txt", "md", "py", "csv", "json", "js", "html"],
)

with st.expander("🎤 Дауыспен жазу"):
    audio = st.audio_input("Басып сөйле", label_visibility="collapsed")
    if audio is not None:
        raw = audio.getvalue()
        h = hashlib.md5(raw).hexdigest()
        if st.session_state.get("last_audio") != h:
            st.session_state.last_audio = h
            try:
                text = transcribe(raw, lang)
                if text:
                    prompt = Prompt(text=text)
                else:
                    st.warning("Дауыс танылмады.")
            except ImportError:
                st.error("Дауыс үшін: pip install faster-whisper")
            except Exception as e:
                st.error(f"Дауысты тану қатесі: {e}")

if prompt:
    n0 = len(st.session_state.api_messages)
    content = build_user_content(prompt)
    shown = prompt.text or ""
    if prompt.files:
        shown += "\n\n📎 " + ", ".join(f.name for f in prompt.files)

    with st.chat_message("user"):
        st.markdown(shown)
    st.session_state.display.append({"role": "user", "text": shown})
    st.session_state.api_messages.append({"role": "user", "content": content})

    log: list[str] = []
    ok = True
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Ойланып жатыр...")
        try:
            answer = agent_turn(client, MODELS[model_name], system, tools, placeholder, log)
        except Exception as e:
            ok = False
            answer = f"Қате шықты: {e}"
            placeholder.markdown(answer)
            del st.session_state.api_messages[n0:]  # сәтсіз сұрауды тарихтан алып тастау
        if log:
            with st.expander("Құрал әрекеттері"):
                st.code("\n".join(log), language=None)
        if ok and tts_on and answer:
            mp3 = speak(answer, lang)
            if mp3:
                st.audio(mp3, format="audio/mp3", autoplay=True)
            else:
                st.caption("Дауыспен жауап үшін: pip install edge-tts (және интернет керек)")

    st.session_state.display.append({"role": "assistant", "text": answer, "log": log})
    save_chat()
