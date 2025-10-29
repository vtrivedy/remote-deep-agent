#!/usr/bin/env python3
"""
CLI (Textual) for a DeepAgent running inside a Modal Sandbox via HTTP.

Highlights
- Textual TUI with beautiful interface
  • Enter=send · Shift+Enter=newline · '@' attach (uploads to sandbox) · Ctrl+E external editor
  • Command palette: /history · /files · /settings · /clear · /editor · /logs · /jump <idx>
- Streams JSONL events from a FastAPI server running inside the Modal sandbox
- ALL state lives in the sandbox (conversation_history, files, etc.)
- Host is a thin client that only handles display and user input
- Server persists conversation to /workspace/conversation.json after each turn

Architecture
- Sandbox runs a FastAPI server with uvicorn on port 8000
- Modal provides encrypted tunnel for HTTP communication
- Agent initialized once at server startup (fast subsequent turns)
- Client streams responses via aiohttp for responsive UI

Setup
  uv pip install --system textual "textual[syntax]" modal-client aiohttp python-dotenv

  export OPENAI_API_KEY=...  # or ANTHROPIC_API_KEY if your agent uses that
  python cli.py --model openai:gpt-5-mini

Notes
- File attachments: when you '@' attach via the picker, files are uploaded directly
  to the sandbox under /workspace/uploads via sandbox.open() and an '@/uploads/<name>'
  mention is inserted into the prompt so your Deep Agent can reference them.
- The server script (SERVER_SCRIPT) is written to /tmp/server.py and started with nohup.
- You can check server logs with /logs command if there are issues.
- Model name can be specified via --model flag (default: openai:gpt-5-mini).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import modal  # modal-client
from dotenv import load_dotenv

# ---------- Textual imports ----------
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Provider, Hits, Hit
from textual.containers import Vertical
from textual.reactive import var
from textual.screen import ModalScreen
from textual.widgets import Header, Footer, TextArea, RichLog, DirectoryTree, Static, Label
from textual import events

load_dotenv()

# ---------- Constants ----------
APP_NAME = "deepagent-interactive"
SANDBOX_TIMEOUT = 3600
SANDBOX_IDLE = 600
WORKSPACE_DIR = Path.cwd()

# ASCII Art Banner
BANNER = r"""
 ██████╗  ███████╗ ███████╗ ██████╗
 ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
 ██║  ██║ █████╗   █████╗   ██████╔╝
 ██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝
 ██████╔╝ ███████╗ ███████╗ ██║
 ╚═════╝  ╚══════╝ ╚══════╝ ╚═╝

  █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗ ███████╗
 ██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝
 ███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ███████╗
 ██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
 ██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
 ╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
"""

# The FastAPI server script that will run *inside* the sandbox.
SERVER_SCRIPT = r'''#!/usr/bin/env python3
import json
import os
import sys
import traceback
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

print("Server initializing...", flush=True)

app = FastAPI()
agent = None
conversation_history = []

# Initialize agent on startup
try:
    from deepagents import create_deep_agent
    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("MODEL_NAME", "gpt-5-mini")

    # Configure the OpenAI chat model via langchain-openai
    if model_name.startswith("openai:"):
        base_model = model_name.split(":", 1)[1]
        model = ChatOpenAI(model=base_model, streaming=False)
    else:
        # Fallback: treat model_name as a raw model id
        model = ChatOpenAI(model=model_name, streaming=False)

    agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You are a helpful AI assistant running in a secure Modal sandbox.\n"
            "You may reference files under /workspace/uploads if the user attaches them.\n"
            "Be concise but thorough in your responses."
        ),
    )
    print("Agent initialized successfully", flush=True)
except Exception as e:
    print(f"Agent init error: {e}", flush=True)
    traceback.print_exc()
    agent = None

# Restore conversation if exists
if os.path.exists("/workspace/conversation.json"):
    try:
        with open("/workspace/conversation.json") as f:
            conversation_history = json.load(f)
        print(f"Restored {len(conversation_history)} messages", flush=True)
    except Exception as e:
        print(f"Failed to restore conversation: {e}", flush=True)

class ChatRequest(BaseModel):
    content: str

class JumpRequest(BaseModel):
    index: int

@app.get("/health")
def health():
    return {"status": "ok", "agent_ready": agent is not None}

@app.get("/history")
def get_history():
    return {"history": conversation_history}

@app.post("/clear")
def clear_history():
    global conversation_history
    conversation_history = []
    # Delete persisted file
    if os.path.exists("/workspace/conversation.json"):
        try:
            os.remove("/workspace/conversation.json")
        except:
            pass
    return {"status": "cleared"}

@app.post("/jump")
def jump_to(request: JumpRequest):
    global conversation_history
    idx = request.index
    if 0 <= idx < len(conversation_history):
        conversation_history = conversation_history[:idx+1]
        # Persist
        try:
            with open("/workspace/conversation.json", "w") as f:
                json.dump(conversation_history, f)
        except:
            pass
        return {"status": "jumped", "new_length": len(conversation_history)}
    return {"error": "invalid index"}

@app.post("/chat")
async def chat(request: ChatRequest):
    global conversation_history

    if agent is None:
        async def error_stream():
            yield json.dumps({"type": "error", "message": "Agent not initialized"}) + "\n"
        return StreamingResponse(error_stream(), media_type="application/x-ndjson")

    # Add user message
    conversation_history.append({
        "role": "user",
        "content": request.content
    })

    async def generate():
        final_text_parts = []
        try:
            yield json.dumps({"type": "system", "message": "Processing..."}) + "\n"

            # Stream from agent
            for _, chunk in agent.stream(
                {"messages": conversation_history},
                stream_mode="updates",
                subgraphs=True
            ):
                data = list(chunk.values())[0] if chunk else {}
                msgs = data.get("messages") or []
                if not msgs:
                    continue

                m = msgs[-1]
                mtype = getattr(m, "type", None)
                content = getattr(m, "content", None)

                if mtype == "tool":
                    tool_name = (getattr(m, "additional_kwargs", None) or {}).get("tool_name") or "tool"
                    preview = str(content or "")[:240].replace("\n", " ")
                    yield json.dumps({"type": "tool", "name": tool_name, "preview": preview}) + "\n"

                elif mtype == "ai":
                    if isinstance(content, str):
                        yield json.dumps({"type": "text", "content": content}) + "\n"
                        final_text_parts.append(content)
                    elif isinstance(content, list):
                        for item in content:
                            itype = getattr(item, "type", None)
                            if itype == "text":
                                text = getattr(item, "text", "")
                                yield json.dumps({"type": "text", "content": text}) + "\n"
                                final_text_parts.append(text)
                            elif itype == "tool_use":
                                name = getattr(item, "name", "tool")
                                inp = getattr(item, "input", {})
                                yield json.dumps({"type": "tool", "name": name, "input": str(inp)}) + "\n"

            # Add assistant response to history
            if final_text_parts:
                assistant_message = "\n".join(final_text_parts)
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })

            yield json.dumps({"type": "system", "message": "Complete"}) + "\n"

            # Persist conversation
            try:
                with open("/workspace/conversation.json", "w") as f:
                    json.dump(conversation_history, f)
            except Exception as e:
                print(f"Failed to persist conversation: {e}", flush=True)

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e), "traceback": traceback.format_exc()}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

if __name__ == "__main__":
    print("Starting uvicorn on 0.0.0.0:8000...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
'''

# ---------- Modal Manager ----------
class ModalAgentManager:
    def __init__(self, model_name: str = "openai:gpt-5-mini") -> None:
        self.model_name = model_name
        self.app: Optional[modal.App] = None
        self.image: Optional[modal.Image] = None
        self.sandbox: Optional[modal.Sandbox] = None
        self.tunnel_url: Optional[str] = None

    def start(self) -> None:
        # Ensure API keys exist
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
            raise RuntimeError("Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment.")

        self.app = modal.App.lookup(APP_NAME, create_if_missing=True)
        # Build image with required deps including FastAPI and uvicorn
        self.image = (
            modal.Image.debian_slim()
            .pip_install("deepagents", "langchain", "langchain-openai", "langgraph", "fastapi", "uvicorn")
        )
        # Create sandbox WITH encrypted ports for HTTP server
        self.sandbox = modal.Sandbox.create(
            image=self.image,
            app=self.app,
            timeout=SANDBOX_TIMEOUT,
            idle_timeout=SANDBOX_IDLE,
            encrypted_ports=[8000],  # Required for HTTP server
            env={
                "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
                "MODEL_NAME": self.model_name,
            },
            verbose=False,
        )

        # Prepare workspace directories used by attachments
        for p in ["/workspace", "/workspace/uploads"]:
            try:
                self.sandbox.exec("mkdir", "-p", p, timeout=30)
            except Exception:
                pass

        # Write server script
        with self.sandbox.open("/tmp/server.py", "w") as f:
            f.write(SERVER_SCRIPT)

        # Start server in background
        self.sandbox.exec("sh", "-c", "nohup python /tmp/server.py > /tmp/server.log 2>&1 &", timeout=10)

        # Wait for server to start
        time.sleep(5)

        # Get tunnel URL
        tunnels = self.sandbox.tunnels()
        if 8000 not in tunnels:
            # Wait a bit more and retry
            time.sleep(5)
            tunnels = self.sandbox.tunnels()
            if 8000 not in tunnels:
                raise RuntimeError("Failed to create tunnel on port 8000")

        self.tunnel_url = tunnels[8000].url

    def get_server_logs(self) -> str:
        """Get server logs for debugging"""
        try:
            proc = self.sandbox.exec("cat", "/tmp/server.log", timeout=10)  # type: ignore[union-attr]
            return proc.stdout.read()
        except Exception as e:
            return f"Failed to read logs: {e}"

    def upload_attachments(self, paths: List[Path]) -> List[Tuple[Path, str]]:
        if not paths:
            return []
        uploaded: List[Tuple[Path, str]] = []
        for p in paths:
            remote = f"/workspace/uploads/{p.name}"
            with self.sandbox.open(remote, "wb") as f:  # type: ignore[union-attr]
                f.write(Path(p).read_bytes())
            uploaded.append((p, remote))
        return uploaded

    def stop(self) -> None:
        if self.sandbox is not None:
            try:
                self.sandbox.terminate()
            except Exception:
                pass
            self.sandbox = None
            self.tunnel_url = None

# ---------- File picker modal ----------
class FilePicker(ModalScreen[List[Path]]):
    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
        Binding("enter", "confirm", "Select"),
        Binding("space", "toggle", "Toggle"),
    ]
    def __init__(self, start: Path):
        super().__init__()
        self.start = start
        self.chosen: set[Path] = set()
    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Attach files: arrows navigate · Space toggle · Enter confirm · Esc cancel"),
            DirectoryTree(str(self.start), id="picker-tree"),
        )
    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:  # type: ignore[name-defined]
        p = Path(event.path)
        if p in self.chosen:
            self.chosen.remove(p)
        else:
            self.chosen.add(p)
    def action_toggle(self) -> None:
        tree = self.query_one(DirectoryTree)
        if tree.cursor_node and tree.cursor_node.path and Path(tree.cursor_node.path).is_file():
            p = Path(tree.cursor_node.path)
            if p in self.chosen:
                self.chosen.remove(p)
            else:
                self.chosen.add(p)
    def action_confirm(self) -> None:
        self.dismiss(sorted(self.chosen))

# ---------- Command palette provider ----------
class Commands(Provider):
    async def search(self, query: str) -> Hits:  # type: ignore[override]
        q = query.strip().lower()
        items: List[Hit] = []
        def H(cmd: str, desc: str, cb):
            return Hit(cmd, desc, cb)
        if "/history".startswith(q) or q in {"/h", "history"}:
            items.append(H("/history", "Show conversation history from server", lambda: self.app.post_message_from_user("/history")))
        if "/files".startswith(q):
            items.append(H("/files", "Show workspace paths", lambda: self.app.post_message_from_user("/files")))
        if "/settings".startswith(q):
            items.append(H("/settings", "Show key bindings", lambda: self.app.post_message_from_user("/settings")))
        if "/clear".startswith(q):
            items.append(H("/clear", "Clear conversation on server & display", lambda: self.app.post_message_from_user("/clear")))
        if "/editor".startswith(q):
            items.append(H("/editor", "External editor for prompt", lambda: self.app.post_message_from_user("/editor")))
        if "/logs".startswith(q):
            items.append(H("/logs", "Show server logs", lambda: self.app.post_message_from_user("/logs")))
        if "/jump".startswith(q):
            items.append(H("/jump <idx>", "Truncate history at index", lambda: self.app.post_message_from_user("/jump")))
        return Hits(items)

# ---------- Textual App ----------
class ModalDeepAgentCLI(App):
    CSS = """
    #top {height: 1fr;}
    #log {border: solid $surface; padding: 1 2; overflow: auto;}
    #input-panel {height: 10; border: solid $surface;}
    #status {height: 1; content-align: left middle; color: $accent;}
    #helpbar {height: 1; color: $text-muted;}
    """

    BINDINGS = [
        Binding("enter", "send", "Send"),
        Binding("shift+enter", "newline", "New Line", show=False),
        Binding("/", "slash", "Slash", show=False),
        Binding("@", "attach", "Attach", show=False),
        Binding("ctrl+e", "external_editor", "Editor"),
        Binding("ctrl+p", "command_palette", "Palette"),
        Binding("ctrl+l", "clear_log", "Clear Log"),
    ]

    COMMANDS = {Commands}

    manager: ModalAgentManager

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.manager = ModalAgentManager(model_name=model_name)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Vertical(RichLog(id="log", highlight=True, markup=True), id="top"),
            Vertical(Static("[b cyan]Message Input[/]  [dim](Enter=send · Shift+Enter=newline · @ to attach files · / for commands · Ctrl+E=editor)[/]"),
                    TextArea(soft_wrap=True, id="ta"), id="input-panel"),
            Static("Status: Ready", id="status"),
            Static("[dim]Commands:[/] /history · /files · /settings · /clear · /logs · /jump <n>  [dim]|[/]  [b]@ = Attach Files[/]", id="helpbar"),
        )
        yield Footer()

    async def on_mount(self) -> None:
        # Show banner
        self._log(f"[b cyan]{BANNER}[/]")
        self._log("[b magenta]━" * 60 + "[/]\n")

        # Start modal sandbox with HTTP server
        self._log("[b yellow]⚡ Starting Modal sandbox with HTTP server…[/]")
        await asyncio.to_thread(self.manager.start)
        self._log(f"[b green]✓ Sandbox ready[/] [dim]id={self.manager.sandbox.object_id}[/]")
        self._log(f"[b blue]🔗 Server URL:[/] [dim]{self.manager.tunnel_url}[/]")

        # Health check
        self._log("[b yellow]⏳ Checking server health…[/]")
        health_ok = await self._wait_for_server_ready(max_attempts=10)
        if health_ok:
            self._log("[b green]✓ Server ready and agent initialized![/]")
            self._log("[b magenta]━" * 60 + "[/]\n")
            self._log("[b cyan]💬 Ready to chat! Type your message and press Enter to send.[/]\n")
        else:
            self._log("[b red]⚠ Warning: Server health check failed. Check logs with /logs[/]")

        self.query_one(TextArea).focus()

    async def _wait_for_server_ready(self, max_attempts: int = 10) -> bool:
        """Wait for server to be ready with exponential backoff"""
        for i in range(max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.manager.tunnel_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        data = await resp.json()
                        if data.get("agent_ready"):
                            return True
            except Exception as e:
                if i == max_attempts - 1:
                    self._log(f"[yellow]Health check error: {e}[/]")

            wait_time = min(2 ** (i // 3), 5)
            await asyncio.sleep(wait_time)

        return False

    async def on_unmount(self) -> None:
        await asyncio.to_thread(self.manager.stop)

    # Key event handler - intercept keys before TextArea gets them
    async def on_key(self, event: events.Key) -> None:
        """Handle key presses before widgets get them"""
        ta = self._ta()

        # Only intercept if TextArea is focused
        if not ta.has_focus:
            return

        # Enter = send (unless Shift is held)
        if event.key == "enter" and not event.shift:
            event.prevent_default()
            event.stop()
            await self.action_send()

        # Shift+Enter = newline (let TextArea handle it naturally)
        elif event.key == "enter" and event.shift:
            pass  # Let TextArea handle it

        # @ = attach
        elif event.key == "@":
            event.prevent_default()
            event.stop()
            await self.action_attach()

        # / = command palette
        elif event.key == "/":
            event.prevent_default()
            event.stop()
            await self.action_slash()

    # Helpers
    def _ta(self) -> TextArea:
        return self.query_one(TextArea)
    def _log(self, text: str) -> None:
        self.query_one(RichLog).write(text)
    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    # Actions
    async def action_command_palette(self) -> None:
        """Open command palette - called by Ctrl+P or /"""
        # In newer Textual, we need to trigger this differently
        # For now, show available commands in the log
        self._log("\n[b cyan]Available Commands:[/]")
        self._log("  [b]/history[/] - Show conversation history")
        self._log("  [b]/clear[/] - Clear conversation")
        self._log("  [b]/jump <n>[/] - Jump to turn n")
        self._log("  [b]/files[/] - Show workspace info")
        self._log("  [b]/logs[/] - Show server logs")
        self._log("  [b]/settings[/] - Show key bindings\n")

    async def action_slash(self) -> None:
        """Handle / key - show command palette"""
        await self.action_command_palette()

    async def action_clear_log(self) -> None:
        """Clear the display log"""
        self.query_one(RichLog).clear()

    async def action_newline(self) -> None:
        """Insert newline in TextArea"""
        self._ta().insert("\n")

    async def action_attach(self) -> None:
        """Show file picker and upload selected files"""
        self._log("[b cyan]📎 Opening file picker...[/]")
        result = await self.push_screen_wait(FilePicker(WORKSPACE_DIR))
        if not result:
            self._log("[dim]File picker cancelled[/]")
            return

        self._log(f"[b yellow]⬆ Uploading {len(result)} file(s) to sandbox...[/]")
        # Upload to sandbox and insert mentions
        uploaded = await asyncio.to_thread(self.manager.upload_attachments, result)
        for src, remote in uploaded:
            mention = f"@/uploads/{Path(remote).name}"
            self._ta().insert(mention + " ")
            self._log(f"[b green]✓ Uploaded:[/] [cyan]{src.name}[/] → [dim]{remote}[/]")

        self._log(f"[b green]✓ {len(uploaded)} file(s) ready! Mentions added to prompt.[/]\n")

    async def action_external_editor(self) -> None:
        """Open external editor for composing message"""
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
        ta = self._ta()
        text = ta.text

        self._log(f"[b cyan]📝 Opening external editor ({editor})...[/]")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tf:
            tf.write(text.encode("utf-8"))
            tf.flush()
            tf_path = Path(tf.name)
        try:
            proc = await asyncio.create_subprocess_exec(editor, str(tf_path))
            await proc.communicate()
            ta.text = tf_path.read_text("utf-8")
            self._log("[b green]✓ Editor closed; prompt updated.[/]\n")
        except Exception as e:
            self._log(f"[red]⚠ Editor failed:[/] {e}\n")
        finally:
            try:
                tf_path.unlink(missing_ok=True)
            except Exception:
                pass

    async def action_send(self) -> None:
        ta = self._ta()
        prompt = ta.text.strip()
        if not prompt:
            return
        ta.text = ""

        # Handle commands (starting with /)
        if prompt.startswith("/"):
            await self._handle_command(prompt)
            return

        # Normal chat message
        self._log(f"[b magenta]You:[/]\n{prompt}\n")
        await self._stream_chat(prompt)

    async def _handle_command(self, cmd: str) -> None:
        """Handle slash commands typed in the input"""
        parts = cmd.split()
        command = parts[0].lower()

        if command == "/history":
            await self._history()
        elif command == "/files":
            self._log(f"[b]Workspace:[/] {WORKSPACE_DIR}  ·  [b]Sandbox uploads:[/] /workspace/uploads")
        elif command == "/settings":
            self._log("\n[b cyan]Key Bindings:[/]")
            self._log("  [b]Enter[/] - Send message")
            self._log("  [b]Shift+Enter[/] - New line")
            self._log("  [b]@[/] - Attach files")
            self._log("  [b]/[/] - Show commands")
            self._log("  [b]Ctrl+E[/] - External editor")
            self._log("  [b]Ctrl+P[/] - Command palette")
            self._log("  [b]Ctrl+L[/] - Clear log\n")
        elif command == "/clear":
            await self._clear_history()
        elif command == "/logs":
            await self._show_logs()
        elif command.startswith("/jump"):
            if len(parts) == 2 and parts[1].isdigit():
                await self._jump(int(parts[1]))
            else:
                self._log("[red]Usage: /jump <index>[/]")
        else:
            self._log(f"[red]Unknown command: {command}[/]")
            await self.action_command_palette()

    async def _stream_chat(self, prompt: str) -> None:
        """Send message to server and stream response"""
        self._set_status("Thinking… (streaming from sandbox)")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.manager.tunnel_url}/chat",
                    json={"content": prompt},
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    # Stream JSONL events
                    async for line in resp.content:
                        line_str = line.decode().rstrip("\n")
                        if not line_str:
                            continue

                        try:
                            event = json.loads(line_str)
                            await self._handle_event(event)
                        except json.JSONDecodeError:
                            self._log(f"[dim]{line_str}[/]")

            self._set_status("Done.")
        except aiohttp.ClientError as e:
            self._log(f"[red]Connection error: {e}[/]")
            self._set_status("Error - connection lost")
        except Exception as e:
            self._log(f"[red]Error: {e}[/]")
            self._set_status("Error")

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        """Handle a streamed event from the server"""
        t = event.get("type")
        if t == "text":
            content = event.get("content", "")
            if content:
                self._log(f"[b cyan]Agent:[/] {content}")
        elif t == "tool":
            name = event.get("name", "tool")
            preview = event.get("preview") or event.get("input") or ""
            self._log(f"[yellow]Tool {name} →[/] {preview}")
        elif t == "system":
            self._log(f"[dim]{event.get('message')}[/]")
        elif t == "error":
            self._log(f"[red]Error:[/] {event.get('message')}")
            if "traceback" in event:
                self._log(f"[dim]{event['traceback']}[/]")

    async def _history(self) -> None:
        """Fetch and display conversation history from server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.manager.tunnel_url}/history",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    data = await resp.json()
                    history = data.get("history", [])

            self._log("[b]History:[/]")
            for i, msg in enumerate(history):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                brief = content[:120].replace("\n", " ")
                if role == "user":
                    self._log(f"[dim]{i:02d}[/] [magenta]user[/]: {brief}")
                else:
                    self._log(f"[dim]{i:02d}[/] [cyan]assistant[/]: {brief}")
            self._log("Type: /jump <index> to truncate convo at that turn.")
        except Exception as e:
            self._log(f"[red]Failed to fetch history: {e}[/]")

    async def _clear_history(self) -> None:
        """Clear conversation history on server and local display"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.manager.tunnel_url}/clear",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    await resp.json()  # Ensure request completes

            self.query_one(RichLog).clear()
            self._log("[b yellow]History cleared on server and display[/]")
        except Exception as e:
            self._log(f"[red]Failed to clear history: {e}[/]")

    async def _show_logs(self) -> None:
        """Show server logs for debugging"""
        logs = await asyncio.to_thread(self.manager.get_server_logs)
        self._log("[b]Server Logs:[/]")
        self._log(f"[dim]{logs}[/]")

    async def _jump(self, idx: int) -> None:
        """Truncate conversation at given index"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.manager.tunnel_url}/jump",
                    json={"index": idx},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    data = await resp.json()

            if "error" in data:
                self._log(f"[red]{data['error']}[/]")
            else:
                new_length = data.get("new_length", 0)
                self._log(f"[b yellow]Jumped to turn {idx}[/] (history now has {new_length} messages)")
                self._set_status("History truncated on server.")
        except Exception as e:
            self._log(f"[red]Failed to jump: {e}[/]")

# ---------- Argparse / entry ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Textual CLI for Modal DeepAgent")
    p.add_argument("--model", default="openai:gpt-5-mini", help="Model name (default: openai:gpt-5-mini)")
    return p.parse_args()

async def run_app(args: argparse.Namespace) -> None:
    app = ModalDeepAgentCLI(model_name=args.model)
    await app.run_async()

def main() -> None:
    args = parse_args()
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        print("Error: set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment.")
        sys.exit(1)
    try:
        asyncio.run(run_app(args))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
