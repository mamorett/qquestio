# ◉ QQuestio — Enterprise RAG TUI

QQuestio is an enterprise-grade, terminal-based Retrieval-Augmented Generation (RAG) user interface built with Go and the **Charmbracelet ecosystem** (`bubbletea`, `lipgloss`, `viewport`, `textinput`, `spinner`).

Designed around the **Nord color palette**, QQuestio delivers a visually stunning, low-latency, and interactive interface for semantic searching and multi-turn conversations with local or hosted LLMs, utilizing a completely non-blocking async HTTP pipelines architecture.

---

## 🌟 Key Features

- **Non-Blocking Async Architecture**: Full background execution of HTTP embeddings, vector similarity search, reranking, and SSE stream reading via structured `tea.Cmd` loops.
- **Real-time SSE Streaming**: High-performance, self-chaining Server-Sent Events parser that prints LLM responses token-by-token directly into a scrollable viewport.
- **Two-Panel TUI Layout**: Split-screen design dividing the screen into a main chat panel (2/3 width) and a scrollable side panel for retrieved document references (1/3 width) that can be focused and scrolled independently.
- **Automatic Session Recovery & Management**: Chronological timestamp-based sessions stored locally in `$HOME/config/qquestio/sessions`, with quick resume flags (`-c` to recover the latest session or `-c <sessionid>` for specific sessions), printing the active session ID upon quit.
- **Model-Agnostic Generic Reranking**: Optional, generic, and provider-agnostic reranking step that automatically expands the database candidate pool, query-scores retrieved points, and selects the top documents.
- **Full-Corpus Recall by Default**: Uses Qdrant's native exact brute-force search (`params.exact=true`) to score every single vector in the collection server-side, with sub-second latency even for million-scale corpora. An optional `/cap` (or `--search-cap` / `SEARCH_CAP`) switches to HNSW approximate search for reduced latency on extremely large collections.
- **Dynamic Slash Commands**: Modify parameters at runtime (e.g., active collection, search limits, or system prompt) or copy transcripts without restarting the TUI.
- **Skills System**: Plug-and-play local tools framework featuring a registry interface and execution dispatcher.
- **Nord Theme Aesthetics**: Sophisticated, premium color design featuring distinct status bars, responsive padding, and dynamic state transitions.

---

## 🛠️ Architecture & FSM

QQuestio is built upon a deterministic State Machine that sequences async pipeline stages:

```
                      ┌──────────────────────────────┐
                      │                              │
                      ▼                              │
   ┌──────────┐  Enter key   ┌──────────────┐        │
   │ stateIdle │────────────►│ stateEmbedding│       │
   └──────────┘              └──────┬───────┘        │
        ▲                          │                 │
        │                    embeddingMsg            │
        │                          │                 │
        │                          ▼                 │
        │                   ┌─────────────┐          │
        │                   │stateSearching│         │
        │                   └──────┬──────┘          │
        │                          │                 │
        │                   searchResultMsg          │
        │                          │                 │
        │                          ▼                 │
        │                   ┌─────────────┐          │
        │                   │stateReranking│ (optional)
        │                   └──────┬──────┘          │
        │                          │                 │
        │                    rerankResultMsg         │
        │                          │                 │
        │                          ▼                 │
        │                   ┌─────────────┐          │
        │                   │stateStreaming │─── streamChunkMsg (done=false)
        │                   └──────┬──────┘          │
        │                          │                 │
        │                streamChunkMsg (done=true)   │
        │                          │                 │
        └──────────────────────────┘                 │
                                                     │
   stateError ◄── appErrMsg (from any stage) ────────┘
```

---

## ⚙️ Configuration

QQuestio supports three configuration methods. Values are merged with the following precedence:
**Local `config.json`** (Lowest) ──► **System Environment Variables** ──► **CLI Flags** (Highest)

### Configuration Options

| Environment Variable / JSON Key / CLI Flag | Description | Required | Example |
|---|---|---|---|
| `QDRANT_URL` / `qdrant_url` | Base URL of the Qdrant REST API | Yes | `http://localhost:6333` |
| `QDRANT_API_KEY` / `qdrant_api_key` | Authentication API Key for Qdrant | Yes | `your-secret-api-key` |
| `EMBEDDING_URL` / `embedding_url` | Base URL of the embedding server | Yes | `http://localhost:8080` |
| `EMBEDDING_API_KEY` / `embedding_api_key` | Optional API Key for the embedding endpoint | No | `your-embedding-key` |
| `EMBEDDING_MODEL` / `embedding_model` | Embedding model identifier | Yes | `nomic-embed-text-v1.5` |
| `OPENAI_URL` / `openai_url` | Base URL of the OpenAI-compatible API | Yes | `http://localhost:4000` |
| `OPENAI_API_KEY` / `openai_api_key` | Optional API Key for the OpenAI-compatible endpoint | No | `your-openai-key` |
| `OPENAI_MODEL` / `openai_model` | LLM model name | Yes | `meta-llama/Llama-3-8B-Instruct` |
| `DEFAULT_COLLECTION` / `default_collection` | Starting Qdrant vector database collection | Yes | `documentation` |
| `RERANKER_URL` / `reranker_url` | Base URL of the model-agnostic rerank endpoint | No | `http://localhost:8080/rerank` |
| `RERANKER_API_KEY` / `reranker_api_key` | Optional API Key for the reranker endpoint | No | `your-reranker-key` |
| `RERANKER_MODEL` / `reranker_model` | Optional model name for the rerank endpoint | No | `bge-reranker-large` |
| `SEARCH_CAP` / `search_cap` / `--search-cap` | Optional upper bound on the Qdrant search candidate pool. `0` (default) = no cap, search the full corpus. See [Search Scope vs. Return Count](#-search-scope-vs-return-count) below. | No | `50000` |
| `CONTEXT_LIMIT` / `context_limit` | Maximum token limit (4-chars ≈ 1 token heuristic). Auto-compaction triggers at 85%. Default is `131072`. Set to `0` to disable auto-compaction. | No | `131072` |

### 1. Example `config.json`
```json
{
  "qdrant_url": "http://localhost:6333",
  "qdrant_api_key": "your-key",
  "embedding_url": "http://localhost:8080",
  "embedding_api_key": "your-embedding-key",
  "embedding_model": "nomic-embed",
  "openai_url": "http://localhost:4000",
  "openai_api_key": "your-openai-key",
  "openai_model": "llama3",
  "default_collection": "documents",
  "reranker_url": "http://localhost:8080/rerank",
  "reranker_api_key": "your-reranker-key",
  "reranker_model": "bge-reranker-large",
  "search_cap": 0,
  "context_limit": 131072
}
```

> `search_cap` is optional. `0` (or omitted) means **no cap** — QQuestio will search the entire collection before truncating to the requested number of documents.

---

## 💬 Interactive Slash Commands

Change state parameters or trigger clipboard actions at runtime from the prompt input:

- **`/collection <name>`**: Switches the active vector store collection instantly.
- **`/limit <1-100>`**: Sets the number of context documents (`docs`) to RETRIEVE into the prompt. This is the return-count side of the search; see [Search Scope vs. Return Count](#-search-scope-vs-return-count).
- **`/cap [N|off|auto|exact|local]`**: Controls the candidate pool cap and search mode. `/cap 50000` → HNSW approximate top-50k; `/cap off` → no cap, uses Qdrant native brute-force (default); `/cap auto` → let the runtime decide; `/cap exact` → always force server-side brute-force (max Qdrant CPU usage); `/cap local` → force client-side brute-force on all local CPU cores (fallback). `/cap` alone prints the current cap and mode. See [Search Scope vs. Return Count](#-search-scope-vs-return-count).
- **`/filter <key> <value>`** (or **`/filter clear`**): Filters vector search by exact metadata key-value match (e.g. `/filter file_name guide.txt`).
- **`/mode <strict|hybrid>`**: Switches between strict closed-book RAG and hybrid general-knowledge RAG modes.
- **`/rerank <on|off>`**: Enables or bypasses the optional reranker step.
- **`/cache [status|refresh|warmup|clear|dir]`**: Inspect or control the on-disk corpus cache. `/cache warmup` pre-populates the cache for offline use.
- **`/system <prompt...>`**: Re-defines the active RAG system instructions for subsequent turns.
- **`/compact [N]`**: Compacts older history to free up context space, leaving the last `N` Q&A pairs intact (default 3). Auto-triggers at 85% of `CONTEXT_LIMIT`.
- **`/clear`**: Clears the conversation history, retrieved references, and context strings (retains prompt input history).
- **`/copy`**: Copies the last assistant response (or the retrieved references if the references panel is focused) to the clipboard.
- **`/copy ref`** (or **`/copy refs`**): Copies the last retrieved references directly to the clipboard.
- **`/copy all`**: Formats and copies the entire clean conversation transcript to the clipboard.
- **`/save <file.md>`** (or **`/write <file.md>`**): Saves the last assistant response directly to a local markdown file.
- **`/save all <file.md>`** (or **`/write all <file.md>`**): Formats and writes the entire conversation transcript (in full Markdown with headers, prompts, code fences, and retrieved references) directly to a local markdown file.
- **`/help`**: Shows the help menu outlining commands and shortcut keys.
- **`/quit`**: Exits the application.

> Slash-command overrides take precedence over CLI flags, which in turn take precedence over environment variables and `config.json`. So `/cap 20000` after launch will override any `--search-cap`, `SEARCH_CAP`, or `search_cap` set at startup.

---

## 🔍 Search Scope vs. Return Count

QQuestio uses two different search strategies depending on the `cap` setting:

| Strategy | When | How it works |
|---|---|---|
| **Exact search** (default) | `cap = 0` or unset | Sends `params.exact=true` to Qdrant's `/points/query` API. Qdrant performs a brute-force scan of the **entire collection** server-side using SIMD-optimized vector math. Sub-second even for millions of vectors. |
| **HNSW search** | `cap > 0` | Sends the cap as the `limit` to Qdrant's HNSW index. Faster for very large collections, but approximate (may miss some true nearest neighbors). |

| Knob | Meaning | Default | Where to set it |
|---|---|---|---|
| **Candidate pool** (`candidateLimit`) | How many candidates Qdrant considers during HNSW search (only when capped) | Entire collection (exact search) | `search_cap` / `SEARCH_CAP` / `--search-cap` / `/cap` |
| **Return count** (`docs`) | How many of the top candidates are actually injected into the LLM prompt | `10` | `/limit` |

The header bar shows the live values, e.g. `Limit: 10  Cap: none  Mode: strict` or `Limit: 5  Cap: 50000  Mode: hybrid`.

### When to use `/cap`

- **Leave it unset (default)** for most collections. Qdrant's exact search handles millions of vectors in under a second.
- **Set `/cap 50000` (or similar)** if you have a multi-hundred-million-vector collection and need the fastest possible response.
- **Use `/cap off`** to restore full-corpus exact search after a cap was set.

---

## ⌨️ TUI Keybindings

Keyboard shortcuts are active global overlays and can be triggered without losing focus on the prompt input line:

- **`Enter`**: Submit prompt query or execute slash command. (In `ERROR` state, clears the error).
- **`Ctrl+C`**: Triggers a non-blocking quit confirmation dialog in the footer. Press again or type `Y` to confirm; press `Esc` or type `N` to cancel.
- **`Double Escape` (press `Esc` twice)**: Cancel in-flight prompt generation (embeddings, search, reranking, or streaming) and return gracefully to the idle state.
- **`Ctrl+R`**: Toggle viewport view mode between styled Glamour Markdown and raw Markdown source.
- **`Tab`**: Toggle active focus between the main chat panel and the right-hand references panel (visually marked by a highlighted border).
- **`Mouse Click`**: Click on either panel to focus it directly.
- **`Ctrl+Y`**: Copy the last response (or retrieved references if the references panel is focused) directly to your system clipboard.
- **`Ctrl+Up` / `Ctrl+Down`**: Scroll the focused viewport (chat response or references panel) up and down by single lines.
- **`PageUp` / `PageDown`**: Scroll the focused viewport up and down by half pages.
- **`Up` / `Down` arrow keys**: Navigate back and forward through your entered prompts history (when cursor is focused on the input prompt line).

---
## 💾 Session Management

QQuestio automatically tracks and serializes your conversations to keep your context saved between runs.

- **Storage Path**: `$HOME/config/qquestio/sessions/*.json`
- **Session IDs**: Chronological timestamp identifiers (e.g. `20260619-114154`).
- **Exiting TUI**: Upon exit, the active session is saved, and its ID is printed to the terminal:
  ```bash
  Session ID: 20260619-114154
  ```
- **Resuming the Last Session**: Resume your last conversation and settings:
  ```bash
  ./qquestio -c
  ```
- **Resuming a Specific Session**: Recover a specific past session by ID:
  ```bash
  ./qquestio -c 20260619-114154
  ```

---

## 🔌 Skills System (Local Agentic Tools)

QQuestio features a plug-and-play **Skills System** that allows the LLM to dynamically execute local actions on your machine and incorporate their results directly into the conversation.

### How It Works

1. **Tool Definition**: Skills implement the `Skill` interface, which defines a name, description, and an `Execute(ctx, args)` entrypoint.
2. **LLM Prompting**: When the registry has registered skills (such as the default `bash` skill), descriptions of these tools are dynamically injected into the system prompt.
3. **Execution Loop**:
   - If the LLM determines it needs a tool, it outputs a command using the syntax:
     ```text
     CALL: <tool_name> <arguments>
     ```
   - The TUI detects this output, pauses LLM streaming, executes the skill asynchronously (non-blocking), formats the execution output, and feeds it back into the model's chat history.
   - The TUI then restarts streaming, letting the LLM react to the execution results and finish its explanation.

### Default Skills

*   **`bash`**: Executes a bash command in a subprocess (`/bin/bash` or `/bin/sh`) on the client machine and returns combined stdout and stderr to the LLM.

---

## 🏗️ Development & Building

A comprehensive cross-compilation suite is configured in the `Makefile`.

### Local Build & Test
```bash
# Run test suite (includes mock HTTP test servers for embeddings, qdrant, rerank, and SSE streams)
make test

# Build local binary
make build

# Launch the app (ensure environment variables or config.json are set)
./qquestio
```

### Cross-Compilation (Pre-building for Releases)
To build for all supported targets, run:
```bash
make build-all
```
This generates binaries inside the `bin/` directory for:
- **Linux**: AMD64 and ARM64
- **Windows**: AMD64 and ARM64 (executable `.exe`)
- **macOS / Darwin**: AMD64 (Intel) and ARM64 (Apple Silicon)

```bash
# Clean up build artifacts and logs
make clean
```
