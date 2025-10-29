# Remote Deep Agent

Interactive DeepAgent running in Modal Sandbox - powered by LangChain DeepAgents + OpenAI GPT-5-mini.

## What It Does

- Runs a conversational AI agent in a secure Modal sandbox
- Agent has planning, filesystem, and tool capabilities via DeepAgents
- HTTP-based architecture with FastAPI server in sandbox
- Persistent conversation state maintained server-side

## Quick Setup

### 1. Install uv (if needed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies
```bash
uv pip install modal rich requests python-dotenv
```

### 3. Configure Modal
```bash
modal setup  # Follow prompts to authenticate
```

### 4. Set OpenAI API key
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

## Usage

### Run the agent (main.py - working version)
```bash
uv run python main.py
```

**Note:** `cli.py` is a work-in-progress Textual TUI version. Use `main.py` for ok working HTTP-based CLI.

### Example
```
🤖 Interactive DeepAgent via HTTP
📦 Building container...
🚀 Creating sandbox...
✅ Agent ready at: https://xxx.modal.run

You: What can you help me with?

🤖 Agent
I'm an AI assistant running in a Modal sandbox. I can help with...

You: exit
👋 Goodbye!
```

## Architecture

```
Local Machine              Modal Sandbox
─────────────             ──────────────
  main.py      ──HTTP──>   FastAPI Server
  (CLI)        <─JSON──    + DeepAgent
                            + Conversation State
```

1. Creates Modal sandbox with DeepAgents + FastAPI
2. Starts HTTP server in sandbox on port 8000
3. Modal provides encrypted HTTPS tunnel
4. CLI sends messages via HTTP POST
5. Server maintains conversation history
6. Agent processes with full DeepAgents capabilities

## Files

- `main.py` - Working HTTP-based interactive CLI ✅
- `cli.py` - WIP Textual TUI with file uploads, command palette (unstable) 🚧
- `README.md` - This file

## Troubleshooting

**Modal auth failed:**
```bash
modal setup
```

**API key not set:**
```bash
export OPENAI_API_KEY='sk-...'
```

**Server timeout:**
Wait 10-15 seconds for sandbox to start. Check logs if needed.

## License

MIT
