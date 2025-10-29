#!/usr/bin/env python3
"""
Interactive DeepAgent via HTTP server in Modal Sandbox
Uses HTTP communication to avoid StreamReader issues
"""

import modal
import os
import requests
import time
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()

def main():
    console.print(Panel.fit(
        "[bold blue]🤖 Interactive DeepAgent via HTTP[/bold blue]\n"
        "Agent runs as HTTP server in Modal Sandbox",
        border_style="blue"
    ))

    # Setup
    app = modal.App.lookup("deepagent-http", create_if_missing=True)

    console.print("📦 Building container...", style="yellow")
    image = modal.Image.debian_slim().pip_install(
        "deepagents",
        "langchain-openai",
        "langgraph",
        "fastapi",
        "uvicorn"
    )

    console.print("🚀 Creating sandbox...", style="yellow")
    sandbox = modal.Sandbox.create(
        image=image,
        app=app,
        timeout=3600,
        encrypted_ports=[8000],  # Explicitly expose port 8000
        env={"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
    )

    try:
        # Create HTTP server that maintains conversation state
        server_script = '''
import sys
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

print("Starting server initialization...", flush=True)

app = FastAPI()

# Try to initialize agent
agent = None
conversation_history = []

try:
    print("Importing deepagents...", flush=True)
    from deepagents import create_deep_agent

    print("Creating agent...", flush=True)
    agent = create_deep_agent(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful AI assistant."
    )
    print("Agent created successfully!", flush=True)
except Exception as e:
    print(f"AGENT INIT ERROR: {e}", flush=True)
    print(traceback.format_exc(), flush=True)
    agent = None

class Message(BaseModel):
    content: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent_ready": agent is not None
    }

@app.post("/chat")
def chat(message: Message):
    global conversation_history

    if agent is None:
        return {"response": "ERROR: Agent not initialized"}

    try:
        # Add user message
        conversation_history.append({
            "role": "user",
            "content": message.content
        })

        # Get agent response
        print(f"Processing message: {message.content[:50]}...", flush=True)
        result = agent.invoke({"messages": conversation_history})
        print(f"Got result: {type(result)}", flush=True)

        # Extract response
        if "messages" in result:
            last_msg = result["messages"][-1]
            response_text = str(last_msg.content) if hasattr(last_msg, 'content') else str(last_msg)

            # Add to history
            conversation_history.append({
                "role": "assistant",
                "content": response_text
            })

            return {"response": response_text}

        return {"response": "Error: No messages in result"}

    except Exception as e:
        print(f"CHAT ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {"response": f"ERROR: {str(e)}"}

if __name__ == "__main__":
    print("Starting uvicorn on 0.0.0.0:8000...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
'''

        console.print("📝 Setting up HTTP server...", style="yellow")
        with sandbox.open("/tmp/server.py", "w") as f:
            f.write(server_script)

        # Start server in background
        console.print("🎬 Starting server...", style="yellow")

        # First check if Python and dependencies are available
        test_proc = sandbox.exec("python", "-c", "import fastapi, uvicorn, deepagents; print('OK')", timeout=30)
        test_output = test_proc.stdout.read()  # Already returns string in Modal
        if "OK" not in test_output:
            console.print(f"[red]Dependency check failed: {test_output}[/red]")
            return
        console.print("[dim]Dependencies OK[/dim]")

        # Start the server with output capture
        console.print("[dim]Starting uvicorn server...[/dim]")

        # Run server with nohup to keep it running in background
        sandbox.exec("sh", "-c", "nohup python /tmp/server.py > /tmp/server.log 2>&1 &", timeout=10)

        # Wait for server to start
        console.print("⏳ Waiting for server to start...", style="yellow")
        time.sleep(5)

        # Check server logs
        log_proc = sandbox.exec("cat", "/tmp/server.log", timeout=10)
        server_logs = log_proc.stdout.read()
        console.print("[dim]Server logs:[/dim]")
        console.print(f"[dim]{server_logs}[/dim]")

        # Get tunnel to the server
        console.print("🔗 Creating tunnel...", style="yellow")
        tunnels = sandbox.tunnels()

        console.print(f"[dim]Available tunnels: {list(tunnels.keys())}[/dim]")

        if 8000 not in tunnels:
            console.print("[red]Failed to create tunnel on port 8000[/red]")
            console.print(f"[yellow]Available ports: {list(tunnels.keys())}[/yellow]")
            console.print("[yellow]Server might not have started yet. Waiting longer...[/yellow]")
            time.sleep(10)
            tunnels = sandbox.tunnels()
            console.print(f"[dim]Available tunnels after wait: {list(tunnels.keys())}[/dim]")

            if 8000 not in tunnels:
                console.print("[red]Still no tunnel on port 8000. Checking server logs...[/red]")
                # Try to get any output from the process
                return

        tunnel_url = tunnels[8000].url
        console.print(f"✅ Agent ready at: [cyan]{tunnel_url}[/cyan]\n", style="green")

        # Test health endpoint with retries
        console.print("🔍 Testing health endpoint...", style="yellow")
        for attempt in range(3):
            try:
                health_response = requests.get(f"{tunnel_url}/health", timeout=10)
                health_data = health_response.json()
                console.print(f"✓ Health check: {health_data}", style="green")

                if not health_data.get("agent_ready", False):
                    console.print("[yellow]Warning: Agent not ready - check logs above[/yellow]")
                break
            except Exception as e:
                if attempt < 2:
                    console.print(f"[yellow]Health check attempt {attempt + 1} failed, retrying...[/yellow]")
                    time.sleep(3)
                else:
                    console.print(f"[red]Health check failed after 3 attempts: {e}[/red]")
                    console.print("[yellow]Server may not be responding. Check logs above.[/yellow]")

        # Interactive loop - just make HTTP requests!
        while True:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")

            if user_input.lower() in ['exit', 'quit', 'bye']:
                console.print("👋 Goodbye!", style="blue")
                break

            if not user_input.strip():
                continue

            # Send message to agent via HTTP
            console.print("\n[bold blue]🤖 Agent[/bold blue]", style="blue")

            try:
                response = requests.post(
                    f"{tunnel_url}/chat",
                    json={"content": user_input},
                    timeout=60
                )

                data = response.json()
                console.print(data.get("response", "No response"))

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    finally:
        console.print("\n🗑️  Cleaning up...", style="yellow")
        sandbox.terminate()
        console.print("✅ Done!", style="green")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY not set[/red]")
        exit(1)

    main()
