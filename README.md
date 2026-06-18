# ◉ QQuestio — Enterprise RAG TUI

QQuestio is an enterprise-grade, terminal-based Retrieval-Augmented Generation (RAG) user interface built with Go and the **Charmbracelet ecosystem** (`bubbletea`, `lipgloss`, `viewport`, `textinput`, `spinner`).

Designed around the **Nord color palette**, QQuestio delivers a visually stunning, low-latency, and interactive interface for semantic searching and multi-turn conversations with local or hosted LLMs, utilizing a completely non-blocking async HTTP pipelines architecture.

---

## 🌟 Key Features

- **Non-Blocking Async Architecture**: Full background execution of HTTP embeddings, vector similarity search, reranking, and SSE stream reading via structured `tea.Cmd` loops.
- **Real-time SSE Streaming**: High-performance, self-chaining Server-Sent Events parser that prints LLM responses token-by-token directly into a scrollable viewport.
- **Model-Agnostic Generic Reranking**: Optional, generic, and provider-agnostic reranking step that automatically expands the database candidate pool, query-scores retrieved points, and selects the top documents.
- **Full-Corpus Recall by Default**: Decouples the Qdrant search scope (the full collection) from the return count (the user-configurable `/limit`), so recall is no longer capped by the number of documents you want to retrieve. An optional `/cap` (or `--search-cap` / `SEARCH_CAP`) provides a hard upper bound for very large corpora.
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
  "search_cap": 0
}
```

> `search_cap` is optional. `0` (or omitted) means **no cap** — QQuestio will search the entire collection before truncating to the requested number of documents.

---

## 💬 Interactive Slash Commands

Change state parameters or trigger clipboard actions at runtime from the prompt input:

- **`/collection <name>`**: Switches the active vector store collection instantly.
- **`/limit <1-100>`**: Sets the number of context documents (`docs`) to RETRIEVE into the prompt. This is the return-count side of the search; see [Search Scope vs. Return Count](#-search-scope-vs-return-count).
- **`/cap [N|off]`**: Sets or clears the candidate pool cap. `/cap 50000` caps Qdrant search to the top 50k candidates before truncating to `docs`; `/cap off` (or `/cap 0`) restores full-corpus search; `/cap` alone prints the current value. See [Search Scope vs. Return Count](#-search-scope-vs-return-count).
- **`/filter <key> <value>`** (or **`/filter clear`**): Filters vector search by exact metadata key-value match (e.g. `/filter file_name guide.txt`).
- **`/mode <strict|hybrid>`**: Switches between strict closed-book RAG and hybrid general-knowledge RAG modes.
- **`/rerank <on|off>`**: Enables or bypasses the optional reranker step.
- **`/system <prompt...>`**: Re-defines the active RAG system instructions for subsequent turns.
- **`/copy`**: Copies the last assistant response text to the clipboard.
- **`/copy all`**: Formats and copies the entire clean conversation transcript to the clipboard.
- **`/save <file.md>`** (or **`/write <file.md>`**): Saves the last assistant response directly to a local markdown file.
- **`/save all <file.md>`** (or **`/write all <file.md>`**): Formats and writes the entire conversation transcript (in full Markdown with headers, prompts, code fences, and retrieved references) directly to a local markdown file.
- **`/help`**: Shows the help menu outlining commands and shortcut keys.
- **`/quit`**: Exits the application.

> Slash-command overrides take precedence over CLI flags, which in turn take precedence over environment variables and `config.json`. So `/cap 20000` after launch will override any `--search-cap`, `SEARCH_CAP`, or `search_cap` set at startup.

---

## 🔍 Search Scope vs. Return Count

Qdrant's `/points/query` API uses the request `limit` to determine the **search scope** (i.e. how many nearest neighbors Qdrant's HNSW index will compute). A small `limit` therefore destroys recall: Qdrant never even looks at the rest of the corpus.

QQuestio **decouples** these two concepts:

| Knob | Meaning | Default | Where to set it |
|---|---|---|---|
| **Candidate pool** (`candidateLimit`) | How many candidates Qdrant is asked to consider during the search | The full collection size (or 10k if info not yet fetched) | `search_cap` / `SEARCH_CAP` / `--search-cap` / `/cap` |
| **Return count** (`docs`) | How many of the top candidates are actually injected into the LLM prompt | `10` | `/limit` |

In other words: QQuestio always asks Qdrant to consider the **entire corpus** (unless explicitly capped) and then keeps only the **first `docs`** points — sorted by similarity — for the prompt. This maximizes recall and gives the reranker a meaningful pool to work with.

The header bar shows the live values, e.g. `Limit: 10  Cap: none  Mode: strict` or `Limit: 5  Cap: 50000  Mode: hybrid`.

### When to use `/cap`

- **Leave it unset (default)** if your collection is reasonably sized (≤ a few hundred thousand vectors). Qdrant's HNSW will return the true top-N quickly.
- **Set `/cap 50000` (or similar)** if you have a multi-million-vector collection and search latency or memory pressure is a concern. The cap bounds the candidate pool, at the cost of some recall.
- **Use `/cap off`** to restore full-corpus search after a cap was set.

---

## ⌨️ TUI Keybindings

Keyboard shortcuts are active global overlays and can be triggered without losing focus on the prompt input line:

- **`Enter`**: Submit prompt query or execute slash command. (In `ERROR` state, clears the error).
- **`Ctrl+C`**: Cancel in-flight request, close connections, and quit the application safely.
- **`Double Escape` (press `Esc` twice)**: Cancel in-flight prompt generation (embeddings, search, reranking, or streaming) and return gracefully to the idle state.
- **`Ctrl+R`**: Toggle viewport view mode between styled Glamour Markdown and raw Markdown source.
- **`Ctrl+Y`**: Copy the last assistant response directly to your system clipboard.
- **`Ctrl+Up` / `Ctrl+Down`**: Scroll the response viewport up and down by single lines.
- **`PageUp` / `PageDown`**: Scroll the response viewport up and down by half pages.
- **`Up` / `Down` arrow keys**: Navigate back and forward through your entered prompts history (when cursor is focused on the input prompt line).

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
