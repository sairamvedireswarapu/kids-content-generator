# Kids Content Generator 🎬

An agentic AI pipeline that automatically generates kids YouTube story videos — complete with narration, illustrations, and subtitles — using LangGraph and MCP tool calling.

---

## What It Does

Give it nothing. It figures out everything:

1. **Researches** trending topics kids are interested in (via Tavily web search)
2. **Plans** a story outline with a moral value (kindness, courage, honesty etc.)
3. **Writes** a short story for kids aged 4-10
4. **Reviews** its own work with a quality critic agent (retries if score < 8.0)
5. **Generates** YouTube metadata (title, description, hashtags, thumbnail concept)
6. **Narrates** the story as audio (Edge TTS)
7. **Illustrates** each paragraph as an AI-generated image
8. **Assembles** everything into a video with subtitles
9. **Saves** locally and uploads to Google Drive via MCP

---

## Architecture

```
Trend Researcher (Tavily Search)
        │
        ▼
Outline Agent (picks value + builds structure)
        │
        ▼
Story Writer (writes story)
        │
        ▼
Quality Critic (scores safety, educational, engagement)
        │
   route_after_critic
   ├── passed (>=8.0) ──→ Metadata Agent
   ├── failed + retry < 3 ──→ Story Writer (retry)
   └── retry >= 3 ──→ Save Content (give up gracefully)
        │
        ▼
Metadata Agent (title, description, hashtags, thumbnail)
        │
        ▼
Audio Generator (Edge TTS → audio.mp3)
        │
        ▼
Video Generator (AI images + moviepy → video.mp4)
        │
        ▼
Save Content via MCP
   ├── save_local → stories/ folder
   └── upload_to_drive → Google Drive "Kids Stories"
        │
        ▼
END
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Agent orchestration | LangGraph |
| LLM | Groq (llama-3.3-70b-versatile) |
| Web search | Tavily |
| Audio | Edge TTS (en-AU-NatashaNeural) |
| Image generation | Pollinations AI (flux-pro) |
| Video assembly | moviepy |
| Tool calling protocol | MCP (Model Context Protocol) |
| Cloud storage | Google Drive API |
| Package manager | uv |

---

## Project Structure

```
kids-content-generator/
├── main.py               # entry point
├── nodes.py              # all LangGraph agent nodes + StoryState
├── graph.py              # LangGraph wiring + routing logic
├── video_generator.py    # image generation + video assembly node
├── mcp_server.py         # MCP server with save_local + upload_to_drive tools
├── .env                  # API keys (not committed)
├── credentials.json      # Google OAuth credentials (not committed)
├── token.pickle          # Google OAuth token (not committed)
└── stories/
    └── story_YYYYMMDD_HHMMSS/
        ├── audio.mp3
        ├── frame_00.png
        ├── frame_01.png
        ├── ...
        ├── video.mp4
        └── story.json
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd kids-content-generator
uv venv
uv add langgraph langchain-groq tavily-python edge-tts moviepy mutagen pillow requests mcp google-api-python-client google-auth-httplib2 google-auth-oauthlib google-genai python-dotenv
```

### 2. Set up API keys

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
GEMINI_API_KEY=your_gemini_key   # optional, for Imagen 4
```

Get your keys:
- Groq: [console.groq.com](https://console.groq.com) — free
- Tavily: [tavily.com](https://tavily.com) — free tier
- Gemini: [aistudio.google.com](https://aistudio.google.com) — optional

### 3. Set up Google Drive (optional)

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project → Enable **Google Drive API**
3. Create OAuth credentials → Download as `credentials.json`
4. Place `credentials.json` in project root
5. Add your Gmail as a test user in OAuth consent screen
6. First run will open browser for authorization → auto-saves `token.pickle`

### 4. Run

```bash
python main.py
```

---

## Configuration

Edit these variables at the top of `nodes.py`:

```python
VALUES = ["kindness", "courage", "honesty", "friendship", "patience", "gratitude", "perseverance"]
TARGET_DURATION_SECONDS = 150   # story length (~350 words at 150s)
WORDS_PER_MINUTE = 140
VOICE = "en-AU-NatashaNeural"   # Edge TTS voice
```

---

## Output

Each run creates a folder in `stories/`:

```json
{
  "topic": "SpongeBob SquarePants",
  "value": "perseverance",
  "story": "...",
  "scores": {
    "safety": 9.0,
    "educational": 8.0,
    "engagement": 8.0
  },
  "metadata": {
    "title": "SpongeBob Never Gives Up",
    "description": "...",
    "hashtags": ["#SpongeBob", "#NeverGiveUp", "..."],
    "thumbnail_concept": "..."
  },
  "audio_path": "stories/story_.../audio.mp3",
  "video_path": "stories/story_.../video.mp4"
}
```

---

## MCP Tool Calling

This project uses the **Model Context Protocol (MCP)** for file saving and Drive upload. The LangGraph node doesn't directly write files — it calls tools via MCP:

```
LangGraph save_content node
        │
        │  calls tool by name
        ▼
MCP Server (mcp_server.py)
        ├── save_local()       → writes story.json locally
        └── upload_to_drive()  → uploads to Google Drive
```

This means you can swap Google Drive for S3, Dropbox, or any storage — just change the MCP server, not the agent.

---

## Roadmap

- [ ] Better image quality (Imagen 4 Fast)
- [ ] Upload video.mp4 to Google Drive
- [ ] Fix topic variety (Tavily query improvement)
- [ ] YouTube auto-upload via YouTube Data API
- [ ] Daily scheduling (Windows Task Scheduler / cron)
- [ ] Intro/outro branding cards

---

## License

MIT