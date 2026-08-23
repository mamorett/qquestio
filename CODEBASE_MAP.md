# QQuestio — Codebase Map

> **Purpose:** This document is a complete, function-level map of the QQuestio codebase. It is written for coding agents (and humans) who need to implement features **without reading the entire codebase**. Every file, type, and function is listed with its signature, location, purpose, and implementation notes, plus "extension point" recipes for common changes.
>
> **Codebase shape:** ~10,350 lines of Go in a single module (`qquestio`), two packages: the root `main` package (Bubbletea TUI + RAG pipeline orchestration) and `internal/rag` (HTTP service clients: Qdrant, embeddings, LLM SSE streaming, reranking, corpus caching). No external Go dependencies beyond the Charmbracelet ecosystem (`bubbletea`, `bubbles`, `lipgloss`, `glamour`).
>
> **What the app is:** A terminal RAG chat client. User prompt → embedding → Qdrant vector search (with optional context expansion and reranking) → OpenAI-compatible LLM SSE streaming into a two-panel TUI, with slash commands, session persistence, a local tool-calling ("skills") system, and an on-disk corpus cache.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Layout](#directory-layout)
3. [The Pipeline State Machine](#the-pipeline-state-machine)
4. [Root Package — `main.go`, `version.go`, `messages.go`, `styles.go`](#root-package--core-files)
5. [Root Package — `config.go` (layered configuration)](#root-package--configgo-layered-configuration)
6. [Root Package — `skills.go` (agentic tool system)](#root-package--skillsgo-agentic-tool-system)
7. [Root Package — `slash.go` (runtime slash commands)](#root-package--slashgo-runtime-slash-commands)
8. [Root Package — `commands.go` (async pipeline commands)](#root-package--commandsgo-async-pipeline-commands)
9. [Root Package — `model.go` (the TUI model & FSM)](#root-package--modelgo-the-tui-model--fsm)
10. [`internal/rag` — `qdrant.go` (vector DB client)](#internal-rag--qdrantgo-vector-db-client)
11. [`internal/rag` — Support Clients (`httpclient.go`, `embeddings.go`, `litellm.go`, `rerank.go`, `cache.go`)](#internal-rag--support-clients-httpclientgo-embeddingsgo-litellmgo-rerankgo-cachego)
12. [Tests](#tests)
13. [Build & Tooling](#build--tooling)
14. [Extension Recipes](#extension-recipes)

---

## Architecture Overview

The application is a single Bubbletea (Elm-architecture) program:

```
┌──────────────────────────── root package (main) ────────────────────────────┐
│                                                                             │
│  main.go ──► LoadConfig (config.go) ──► NewModel (model.go) ──► tea.Program │
│                config.json < env < flags              │                     │
│                                     multi-profile ────┤                     │
│                                                       ▼                     │
│              ┌────────────── Model (model.go) ───────────────┐              │
│              │  Central state: phase, history, viewports,    │              │
│              │  settings (limit/cap/filter/mode/rerank/...)  │              │
│              │  Update() = FSM dispatcher   View() = render  │              │
│              └───────┬───────────────────────────┬───────────┘              │
│                      │ tea.Cmd (commands.go)    │ tea.Msg (messages.go)    │
│              ┌───────▼───────────────┐   ┌──────▼──────────────┐           │
│              │ slash.go (dispatch)   │   │ skills.go (tools)   │           │
│              └───────────────────────┘   └─────────────────────┘           │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ (only dependency boundary)
┌────────────────────────────── internal/rag ────────────────────────────────┐
│  qdrant.go (search/scroll/filter/expansion)   litellm.go (SSE LLM client)  │
│  embeddings.go  rerank.go  cache.go (corpus cache)                          │
│  httpclient.go (shared transport, private-host detection)  endpoints.go    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key architectural invariants:**

- **All network I/O is non-blocking.** Every HTTP interaction is launched inside a `tea.Cmd` (commands.go); results come back as `tea.Msg` values (messages.go) that `Model.Update` (model.go) routes by type. Nothing blocks the UI thread.
- **`Model` (model.go) is the single source of truth.** All mutable settings live in it; slash commands (slash.go) mutate it directly; renderers (View/renderHeader/renderFooter) read it.
- **`internal/rag` is pure service code.** It has no Bubbletea imports and no knowledge of the TUI. It takes a `context.Context` plus plain parameters, returns plain values/errors. The only exception to "no UI knowledge": the package-level `rag.VerboseLogging` flag and `rag.CacheDir()` used by main.go.
- **Cancellation** flows from `signal.NotifyContext` in main.go through the `Model.ctx` into every rag call; double-Esc in the TUI aborts in-flight work.
- **Config precedence:** `config.json` (lowest) → environment variables → CLI flags → slash-command runtime overrides (highest).

## Directory Layout

| Path | Role |
|---|---|
| `main.go` | Entry point: flag parsing, session resume, program lifecycle |
| `version.go` | `Version` var (ldflags-injectable) |
| `messages.go` | All `tea.Msg` types driving the FSM |
| `styles.go` | Nord palette + `Styles` (lipgloss) |
| `model.go` | `Model` struct, `Init`/`Update`/`View`, FSM, rendering, session persistence (~2,360 lines — the heart of the app) |
| `commands.go` | Every async `tea.Cmd` (pipeline stages, connection checks, clipboard, skills) |
| `slash.go` | Runtime slash-command parser + dispatcher |
| `config.go` | `Config` struct, layered loading, multi-profile support |
| `skills.go` | `Skill` interface, `SkillRegistry`, `ParseCall`, `BashSkill` |
| `internal/rag/qdrant.go` | Qdrant REST client: search modes, scroll, filters, context expansion (~2,116 lines) |
| `internal/rag/litellm.go` | OpenAI-compatible chat + SSE stream reader |
| `internal/rag/cache.go` | On-disk corpus cache (JSON), staleness, warmup |
| `internal/rag/embeddings.go` | Embedding generation + connection check |
| `internal/rag/rerank.go` | Provider-agnostic reranker client |
| `internal/rag/httpclient.go` | Shared `http.Transport`/`http.Client` factory, private-host detection |
| `internal/rag/endpoints.go` | `AppendAPIPath` URL-joining helper |
| `*_test.go` | Tests: `model_test`, `slash_test`, `config_test`, `skills_test`, `rag_test`, `qdrant_test`, `exact_phrase_test`, `qdrant_bench_test` |
| `Makefile` | `build`, `test`, `clean`, cross-compilation matrix (`build-all`) |

## The Pipeline State Machine

The RAG pipeline is sequenced by the `appState` enum in model.go. One user prompt travels through async stages, each stage's `tea.Cmd` producing the next stage's `tea.Msg`:

```
Idle ──(submit prompt)────────────► Embedding (generateEmbeddingCmd)
Embedding ──(embeddingMsg)────────► Searching (searchQdrantCmd)
Searching ──(searchResultMsg)
    ├── rerank ON ──► Reranking (rerankPointsCmd) ──(rerankResultMsg)──► Streaming
    └── rerank OFF ─────────────────────────────────────────────────────► Streaming
Streaming (startLLMStreamCmd → receiveStreamChunkCmd loop)
    ├── streamChunkMsg done=false ──► Streaming (self-chaining)
    ├── streamChunkMsg done=true  ──► Idle
    └── LLM emits "CALL: <skill> <args>" ──► (confirm, if gated) ──► executeSkillCmd
          └──(skillResultMsg)──► new search/stream cycle
Any stage ──(appErrMsg)──► Error view ──(Enter)──► Idle
Idle ──(Ctrl+C)──► ConfirmQuit ──(Y/Ctrl+C)──► exit / (Esc/N)──► Idle
Double-Esc (anywhere in-flight) ──► cancel context ──► Idle
```

Message types by phase (defined in messages.go):

| Msg | Producer (commands.go) | Meaning |
|---|---|---|
| `embeddingMsg` | `generateEmbeddingCmd` | Phase 1 result: query vector |
| `searchResultMsg` | `searchQdrantCmd` | Phase 2: retrieved context + points + expansion data |
| `rerankResultMsg` | `rerankPointsCmd` | Phase 2.5 (optional): reranked context |
| `streamChunkMsg` | `receiveStreamChunkCmd` | Phase 3: one SSE chunk (`done=true` on final) |
| `skillResultMsg` | `executeSkillCmd` | Tool execution output |
| `appErrMsg` | any stage | Error with `stage` ∈ embedding/search/stream/slash |
| `searchProgressMsg` | full-corpus search | Transient header progress (no FSM advance) |
| `cachePreloadMsg` / `warmupCacheMsg` | cache commands | Corpus-cache status events |
| `qdrantInfoMsg` / `llmInfoMsg` / `embedderInfoMsg` | connection checks | Service health info |
| `slashResultMsg` / `systemLogMsg` / `quitMsg` | slash.go | Command feedback / shutdown |

---

## Root Package — Core Files

### File Overview

| File | LOC | Role |
|------|-----|------|
| `main.go` | 134 | Entry point: manual `-c` session-resume flag pre-parsing, stdlib flag parsing, debug-log setup, config load, bubbletea program lifecycle, session save on exit |
| `version.go` | 5 | Single `Version` var, overridable at build time via ldflags |
| `messages.go` | 116 | All `tea.Msg` types that drive the FSM in `model.go` — pipeline stage results, errors, slash/skill/info feedback, quit |
| `styles.go` | 107 | Nord color palette constants and the `Styles` struct built by `DefaultStyles()` (lipgloss) |

---

### main.go

**Package-level vars — `main.go:16-20`**

| Var | Type | Default | Purpose |
|-----|------|---------|---------|
| `cliSearchCap` | `int` | `-1` | Copied from `--search-cap` flag; `-1` = no CLI override, `0` = no cap, `N` = cap Qdrant candidate pool to N |
| `cliSafe` | `bool` | `false` | Copied from `--safe` flag; requires user confirmation before executing any local skills/tools |
| `debugFlag` | `bool` | `false` | Copied from `--debug` flag; enables file-based debug logging |

These are the bridge between `flag.Parse()` results (local to `main`) and the rest of the package, since flags are parsed inside `main()`.

#### `main` — `main.go:22`
- **Signature:** `func main()`
- **Purpose:** Program entry point — parse CLI flags, optionally restore a previous session, run the bubbletea TUI, and persist the session on clean exit.
- **Implementation:**
  1. **Manual `-c` pre-parse (`main.go:27-40`):** before `flag.Parse()` (which would reject the positional session id), scans `os.Args` for `-c` or `--c`. If the next arg exists and doesn't start with `-`, it is captured as `sessionIDToLoad` and spliced out of `os.Args`; otherwise `sessionIDToLoad = "last"`. The flag itself is then removed and the loop breaks — only the first `-c` is honored.
  2. **Flag parsing (`main.go:43-49`):** `--version`/`-v` (print version, exit 0), `--debug`, `--search-cap` (default -1), `--safe`, `--conf` (config profile name passed to `LoadConfig`).
  3. **Debug logging (`main.go:62-68`):** enabled when `debugFlag` OR env `QQUESTIO_DEBUG == "1"`. Sets `rag.VerboseLogging = true` and redirects tea's log to `tea.LogToFile(getLogFilePath(), "qquestio")`. Off by default to prevent unbounded log growth.
  4. **Config (`main.go:70-74`):** `LoadConfig(*confFlag)`; on error prints `Config error:` to stderr and exits 1.
  5. **Context (`main.go:78-79`):** `signal.NotifyContext(context.Background(), os.Interrupt)` — Ctrl-C cancels `ctx`, threaded into `NewModel(ctx, cfg)` so commands can abort in-flight HTTP.
  6. **Session restore (`main.go:83-99`):** if `loadSession`, resolves `"last"` via `GetLastSessionID()`; then `m.loadSession(sessionIDToLoad)`. Failures print to stderr but do not exit; success sets `m.statusMsg`.
  7. **TUI run (`main.go:101-106`):** `tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())` then `p.Run()`; fatal error exits 1.
  8. **Exit handling (`main.go:108-121`):** type-asserts `finalModel.(*Model)`. If `!fm.hasUserPrompt()` (nothing happened), prints only a goodbye. Otherwise `fm.saveSession()`; on success prints the saved session ID plus recall hints; on failure prints a warning to stderr.
- **Calls / Called by:** `LoadConfig` (config.go), `NewModel` (model.go), `GetLastSessionID`, `(*Model).loadSession`, `(*Model).hasUserPrompt`, `(*Model).saveSession` (model.go), `getLogFilePath`, `rag.VerboseLogging`, `tea.LogToFile`.

#### `getLogFilePath` — `main.go:127`
- **Signature:** `func getLogFilePath() string`
- **Purpose:** Chooses the debug log location so logs land in the cache directory rather than polluting the CWD.
- **Implementation:** If `rag.CacheDir()` returns non-empty, returns `cacheDir + "/qquestio-debug.log"`; otherwise falls back to `"debug.log"` in the current directory.
- **Calls / Called by:** Called only by `main`; depends on `rag.CacheDir()` (cache.go).

---

### version.go

#### `Version` — `version.go:5`
- **Type:** `var Version = "1.3.0-dev"` (string)
- **Purpose:** Application version shown by `--version`/`-v`.
- **Key detail:** Overridable at build time with `-ldflags "-X main.Version=x.y.z"` (the Makefile injects `VERSION?=1.3.0`).

---

### messages.go

All types are unexported `tea.Msg` value types handled in `Model.Update` (model.go).

| Type | Line | Fields | Purpose |
|------|------|--------|---------|
| `embeddingMsg` | `messages.go:6` | `vector []float32` | Phase 1 result: embedding vector from the embedder |
| `searchResultMsg` | `messages.go:18` | `context string`; `points []rag.QdrantPoint`; `primaryPoints []rag.QdrantPoint`; `expansionMap rag.ExpansionMap`; `expand int` | Phase 2 result: retrieved context from Qdrant. `context` = concatenated text payloads (post-expansion); `points` = full expanded list with scores/metadata; `primaryPoints` = top-N pre-expansion matches kept for rerank; `expansionMap` = docID → chunk_index → point; `expand` = the expand value used, so rerank can re-apply it |
| `rerankResultMsg` | `messages.go:29` | `context string`; `points []rag.QdrantPoint`; `degraded bool` | Optional rerank step result. Carries reranked + re-expanded set. `degraded = true` means the reranker was unavailable and vector ranking was used as fallback |
| `streamChunkMsg` | `messages.go:36` | `content string`; `reasoning string`; `done bool`; `usage rag.TokenUsage`; `hasUsage bool` | Phase 3: one chunk from the LiteLLM SSE stream. `done = true` when exhausted; usage stats carried on the final chunk when `hasUsage` |
| `appErrMsg` | `messages.go:45` | `err error`; `reason string`; `stage string` | Wraps errors from any pipeline stage. `stage` ∈ `"embedding"`, `"search"`, `"stream"`, `"slash"`. Implements `error` |
| `slashResultMsg` | `messages.go:59` | `feedback string` | Result of running a slash command; feedback text shown in the header |
| `qdrantInfoMsg` | `messages.go:64` | `pointsCount int`; `vectorsCount int`; `status string`; `err error` | Qdrant collection statistics |
| `systemLogMsg` | `messages.go:72` | `content string`; `feedback string` | System events rendered into the viewport without polluting conversation history |
| `quitMsg` | `messages.go:78` | (empty struct) | Emitted when `/quit` executes; triggers shutdown |
| `searchProgressMsg` | `messages.go:84` | `status string` | Transient header status update from the long-running full-corpus scroll search (no FSM advance, no history mutation) |
| `cachePreloadMsg` | `messages.go:90` | `found bool`; `info string`; `pointCount int` | Returned by `preloadCacheInfoCmd` on startup to populate the header bar with cached corpus info |
| `warmupCacheMsg` | `messages.go:98` | (empty struct) | Signals a user-requested `/cache` warmup; starts scroll-based cache population |
| `skillResultMsg` | `messages.go:101` | `name string`; `input string`; `output string`; `err error` | Result of executing a local tool / skill (skills.go) |
| `llmInfoMsg` | `messages.go:109` | `err error` | LLM connection check result |
| `embedderInfoMsg` | `messages.go:114` | `err error` | Embedder connection check result |

Key design note: carrying both `primaryPoints` and `expansionMap` in `searchResultMsg` lets the rerank step rerank only the primaries (not the already-expanded set) and then re-apply ±`expand` around the reranked top-K via `rag.ApplyExpansionToPrimaries`, so context expansion survives the rerank path.

#### `appErrMsg.Error` — `messages.go:51`
- **Signature:** `func (e appErrMsg) Error() string`
- **Purpose:** Satisfies the `error` interface so `appErrMsg` can be wrapped/returned uniformly.
- **Implementation:** Returns `e.err.Error()` when `e.err != nil`; otherwise returns `e.reason`.

---

### styles.go

**Nord palette constants — `styles.go:6-35`** (unexported `lipgloss.Color` consts):

| Block | Consts | Role |
|-------|--------|------|
| Nord Polar Night (backgrounds) | `nord0` `#2E3440`, `nord1` `#3B4252`, `nord2` `#434C5E`, `nord3` `#4C566A` | Backgrounds (darkest → lightest bg) |
| Nord Snow Storm (foregrounds) | `nord4` `#D8DEE9`, `nord5` `#E5E9F0`, `nord6` `#ECEFF4` | Foregrounds |
| Nord Frost (accents) | `nord7` `#8FBCBB` (teal), `nord8` `#88C0D0` (light blue), `nord9` `#81A1C1` (medium blue), `nord10` `#5E81AC` (dark blue) | Accents |
| Nord Aurora (semantic) | `nord11` `#BF616A` (red/errors), `nord12` `#D08770` (orange/warnings), `nord13` `#EBCB8B` (yellow/highlights), `nord14` `#A3BE8C` (green/success), `nord15` `#B48EAD` (purple/special) | Semantic colors |

**`Styles` struct — `styles.go:37`** — constructed once by `DefaultStyles()` and held by `Model`:

| Field | Line | Styling | Used for |
|-------|------|---------|----------|
| `Header` | 38 | bg `nord1`, fg `nord8`, bold, full-width | Header bar |
| `HeaderStatus` | 39 | base; fg applied dynamically (`nord14` idle, `nord13` working, `nord11` error) | Status text in header |
| `Viewport` | 40 | bg `nord0`, fg `nord4`, padding 1 | Conversation viewport body |
| `Footer` | 41 | bg `nord1`, fg `nord5` | Footer bar |
| `InputPrompt` | 42 | fg `nord8`, bold | The `"❯ "` input prompt |
| `InputText` | 43 | fg `nord6` | Typed input text |
| `ErrorText` | 44 | fg `nord11`, italic | Error output |
| `CollectionTag` | 45 | fg `nord15`, bg `nord2`, padding 0 1 | Collection name tag |
| `MainViewportFocused` | 46 | rounded border, border-fg `nord8` | Focused main viewport |
| `MainViewportUnfocused` | 47 | rounded border, border-fg `nord3` | Unfocused main viewport |
| `RefViewportFocused` | 48 | rounded border, border-fg `nord8` | Focused references viewport |
| `RefViewportUnfocused` | 49 | rounded border, border-fg `nord3` | Unfocused references viewport |
| `SpinnerStyle` | 50 | fg `nord8` | Spinner |
| `ThinkingText` | 51 | fg `nord3`, italic, subdued | Model reasoning text (deliberately dim) |

#### `DefaultStyles` — `styles.go:54`
- **Signature:** `func DefaultStyles() Styles`
- **Purpose:** Constructs the `Styles` value with the full Nord theme; the single factory the model calls at startup.
- **Implementation:** Returns a `Styles` literal; each field built with `lipgloss.NewStyle()` chained with `Background`/`Foreground`/`Bold`/`Italic`/`Padding`/`Border(lipgloss.RoundedBorder())`/`BorderForeground`. `HeaderStatus` is built as an empty `lipgloss.NewStyle()` — its foreground color is chosen at render time by status state.
- **Calls / Called by:** Called by `NewModel` (model.go); all view-rendering code in model.go consumes the fields.

---

## Root Package — `config.go` (Layered Configuration)

### File Overview

| File | Lines | Role |
|---|---|---|
| `config.go` | 283 | Layered configuration: `config.json` (root keys + named profiles) < env vars < CLI flags; validation; profile listing for `/conf` |
| `skills.go` | 152 | `Skill` interface, `SkillRegistry`, `ParseCall` line protocol ("CALL: <name> <args>"), and the default `BashSkill` subprocess runner |

Settings come from three layers with strictly increasing precedence: a JSON file (root-level keys, optionally overridden by a named profile selected via `--conf` or the file's `default_configuration`), then environment variables, then CLI flags (which reach `LoadConfig` through package globals `cliSearchCap`/`cliSafe` set in `main.go`). The file is searched at `$HOME/.config/qquestio/config.json` with a fallback to `./config.json`. After merging, defaults are applied and seven fields are validated as required, producing a large multi-section error message on failure. `GetAvailableConfigs` exists solely to power the `/conf` slash command's profile listing.

### Types & Globals — `config.go`

#### `Config` — `config.go:12`

The fully-resolved runtime configuration, produced by `LoadConfig`. Passed by value into the Bubble Tea model (`m.cfg`) and re-assigned wholesale by `/conf`.

| Field | Type | JSON tag | Env var | CLI flag | Default | Meaning |
|---|---|---|---|---|---|---|
| `ActiveConfigName` | string | `active_config_name,omitempty` | — | — | `""` | Output-only: name of the profile that was loaded (set at `config.go:117`; not a real input) |
| `QdrantURL` | string | `qdrant_url` | `QDRANT_URL` | — | *(required)* | Qdrant server base URL |
| `QdrantAPIKey` | string | `qdrant_api_key` | `QDRANT_API_KEY` | — | *(required)* | Qdrant auth key |
| `QdrantVectorName` | string | `qdrant_vector_name,omitempty` | `QDRANT_VECTOR_NAME` | — | `""` | Named vector to use in Qdrant queries |
| `EmbeddingURL` | string | `embedding_url` | `EMBEDDING_URL` | — | *(required)* | Embedding API endpoint |
| `EmbeddingAPIKey` | string | `embedding_api_key` | `EMBEDDING_API_KEY` | — | `""` | Embedding API key (optional) |
| `EmbeddingModel` | string | `embedding_model` | `EMBEDDING_MODEL` | — | *(required)* | Embedding model name |
| `OpenAIURL` | string | `openai_url` | `OPENAI_URL` | — | *(required)* | OpenAI-compatible LLM endpoint |
| `OpenAIAPIKey` | string | `openai_api_key` | `OPENAI_API_KEY` | — | `""` | LLM API key (optional) |
| `OpenAIModel` | string | `openai_model` | `OPENAI_MODEL` | — | *(required)* | LLM model name |
| `DefaultCollection` | string | `default_collection` | `DEFAULT_COLLECTION` | — | *(required)* | Default Qdrant collection |
| `RerankerURL` | string | `reranker_url` | `RERANKER_URL` | — | `""` | Reranker endpoint; empty disables reranking |
| `RerankerAPIKey` | string | `reranker_api_key` | `RERANKER_API_KEY` | — | `""` | Reranker auth key |
| `RerankerModel` | string | `reranker_model` | `RERANKER_MODEL` | — | `""` | Reranker model name |
| `SearchCap` | int | `search_cap,omitempty` | `SEARCH_CAP` | `--search-cap` | `0` | Max Qdrant candidate pool; `0` = full corpus |
| `RerankerPool` | int | `reranker_pool,omitempty` | `RERANKER_POOL` | — | `0` | Candidates fed to the reranker; `0` = auto |
| `HTTPTimeoutSeconds` | int | `http_timeout_seconds,omitempty` | `QQUESTIO_HTTP_TIMEOUT` | — | `60` | HTTP timeout (seconds) for upstream calls |
| `SkillsRequireConfirm` | bool | `skills_require_confirm,omitempty` | `QQUESTIO_SKILLS_REQUIRE_CONFIRM=1` | `--safe` | `false` | Require y/a/n confirmation before any skill executes |
| `ContextLimit` | int | `context_limit,omitempty` | `CONTEXT_LIMIT` | — | `131072` | Token budget for conversation history; auto-compaction triggers at 85% of it |
| `QueryRewrite` | string | `query_rewrite,omitempty` | `QUERY_REWRITE` | — | `"llm"` | Query rewrite mode: `"llm"`, `"heuristic"`, or `"off"` |
| `OpenAIMaxTokens` | int | `openai_max_tokens,omitempty` | `OPENAI_MAX_TOKENS` | — | `0` | Max completion tokens for the LLM; `0` = no limit |

Quirks worth knowing:

- `SkillsRequireConfirm` is **force-true only** from env/CLI (`config.go:169-171`): `--safe` or `QQUESTIO_SKILLS_REQUIRE_CONFIRM=1` turn it on, but nothing can turn off a `true` set in JSON.
- `ContextLimit`: the struct comment says "Set to 0 to disable auto-compaction", but defaulting at `config.go:183-185` replaces `0` with `131072`. Since env parsing rejects negatives (`n >= 0`), the only way to actually disable compaction is a negative JSON value — `model.go:268` treats `ContextLimit <= 0` as a no-op.
- Integer env parsing (`config.go:139-163`) silently ignores non-numeric and negative values.
- `OpenAIMaxTokens` "default" block (`config.go:188-190`) is a literal no-op (`if == 0 { = 0 }`) — kept as a placeholder.
- The env override for `QUERY_REWRITE` sits inside the "CLI overrides" block (`config.go:172-174`), not with the other string env vars.

#### `configFile` — `config.go:39`

On-disk shape of `config.json`. Embeds `Config` (so root-level keys populate it directly) plus two profile fields:

| Field | Type | JSON tag | Meaning |
|---|---|---|---|
| `Configurations` | `map[string]json.RawMessage` | `configurations` | Named partial profiles; each value is a JSON object whose keys are a subset of `Config`'s JSON tags |
| `DefaultConfiguration` | `string` | `default_configuration` | Profile to load when `--conf` is not given |
| *(embedded `Config`)* | | root keys | Base values every profile inherits |

#### Package globals consumed by `config.go` (declared in `main.go:16-20`)

- `cliSearchCap int` — initialized `-1`; set from `--search-cap` at `main.go:51`. `LoadConfig` applies it only when `>= 0` (`config.go:166-168`), so `-1` means "no CLI override".
- `cliSafe bool` — set from `--safe` at `main.go:52`; forces `SkillsRequireConfirm = true`.

### Configuration Reference (derived from code)

| JSON key | Env var | CLI flag | Default | Meaning |
|---|---|---|---|---|
| `qdrant_url` | `QDRANT_URL` | — | required | Qdrant base URL |
| `qdrant_api_key` | `QDRANT_API_KEY` | — | required | Qdrant auth key |
| `qdrant_vector_name` | `QDRANT_VECTOR_NAME` | — | `""` | Named Qdrant vector |
| `embedding_url` | `EMBEDDING_URL` | — | required | Embedding endpoint |
| `embedding_api_key` | `EMBEDDING_API_KEY` | — | `""` | Embedding key |
| `embedding_model` | `EMBEDDING_MODEL` | — | required | Embedding model |
| `openai_url` | `OPENAI_URL` | — | required | LLM endpoint |
| `openai_api_key` | `OPENAI_API_KEY` | — | `""` | LLM key |
| `openai_model` | `OPENAI_MODEL` | — | required | LLM model |
| `default_collection` | `DEFAULT_COLLECTION` | — | required | Default collection |
| `reranker_url` | `RERANKER_URL` | — | `""` | Reranker endpoint (empty = off) |
| `reranker_api_key` | `RERANKER_API_KEY` | — | `""` | Reranker key |
| `reranker_model` | `RERANKER_MODEL` | — | `""` | Reranker model |
| `search_cap` | `SEARCH_CAP` | `--search-cap N` | `0` | Max search candidates; 0 = full corpus |
| `reranker_pool` | `RERANKER_POOL` | — | `0` | Reranker input size; 0 = auto |
| `http_timeout_seconds` | `QQUESTIO_HTTP_TIMEOUT` | — | `60` | Upstream HTTP timeout (s) |
| `skills_require_confirm` | `QQUESTIO_SKILLS_REQUIRE_CONFIRM=1` | `--safe` | `false` | Confirm before skill execution (force-on only) |
| `context_limit` | `CONTEXT_LIMIT` | — | `131072` | History token budget; 85% triggers compaction |
| `query_rewrite` | `QUERY_REWRITE` | — | `"llm"` | `llm` / `heuristic` / `off` |
| `openai_max_tokens` | `OPENAI_MAX_TOKENS` | — | `0` | Max completion tokens; 0 = unlimited |
| `configurations` / `default_configuration` | — | `--conf <name>` | — | Profile selection (also `/conf <name>` at runtime) |

Precedence, low to high: profile-inherited root JSON values < named profile JSON keys < env vars < `--search-cap`/`--safe` CLI flags. Required fields: `qdrant_url`, `qdrant_api_key`, `embedding_url`, `embedding_model`, `openai_url`, `openai_model`, `default_collection`.

### Functions — `config.go` (source order)

#### `getConfigPath` — `config.go:45`
- **Signature:** `func getConfigPath() string`
- **Purpose:** Returns the preferred config file path.
- **Implementation:** `$HOME/.config/qquestio/config.json` via `os.UserHomeDir()`; returns `"config.json"` if the home dir can't be determined.
- **Calls / Called by:** Called by `loadJSONConfigFile`.

#### `loadJSONConfigFile` — `config.go:54`
- **Signature:** `func loadJSONConfigFile() (configFile, bool)`
- **Purpose:** Reads and decodes `config.json` if it exists.
- **Implementation:** Opens `getConfigPath()`; on failure, retries `./config.json` (only when the primary path differed). Missing file → `(zero, false)` silently (env-only configuration is legal). Parse failure → stderr warning + `(zero, false)`. Success → the decoded `configFile`.
- **Calls / Called by:** Calls `getConfigPath`. Called by `LoadConfig` and `GetAvailableConfigs`.

#### `LoadConfig` — `config.go:76`
- **Signature:** `func LoadConfig(configName ...string) (Config, error)`
- **Purpose:** Builds the resolved `Config` from file + env + CLI, applies defaults, and validates required fields.
- **Implementation:** Five phases:
  1. **File + profile resolution** (`config.go:81-115`): loads the JSON file; the active profile name is `configName[0]` if non-empty, else `file.DefaultConfiguration`, else none. With a profile: looks it up in `Configurations`; a miss returns an error listing the sorted available names. The profile's raw JSON is unmarshaled **on top of** a copy of the embedded root `Config` (`config.go:106-110`), so profiles are partial overrides — absent keys inherit root values. Without a profile: root `Config` as-is. `cfg.ActiveConfigName` is set to the selected name or `""` (`config.go:117`).
  2. **Env overrides** (`config.go:120-163`): `overrideFromEnv` closure for the 13 string fields; `strconv.Atoi` blocks (value must be `>= 0`) for `SEARCH_CAP`, `RERANKER_POOL`, `CONTEXT_LIMIT`, `QQUESTIO_HTTP_TIMEOUT`, `OPENAI_MAX_TOKENS`.
  3. **CLI overrides** (`config.go:166-174`): `cliSearchCap >= 0` → `SearchCap`; `cliSafe || QQUESTIO_SKILLS_REQUIRE_CONFIRM=1` → `SkillsRequireConfirm = true`; `QUERY_REWRITE` env → `QueryRewrite`.
  4. **Defaults** (`config.go:177-195`): `HTTPTimeoutSeconds <= 0` → 60; `ContextLimit == 0` → 131072; `QueryRewrite == ""` → `"llm"`; no-op block for `OpenAIMaxTokens`.
  5. **Validation** (`config.go:198-267`): collects the 7 required fields that are empty; if any, returns a formatted multi-block error message listing the missing vars plus example env-var and `config.json` setups.
- **Calls / Called by:** Calls `loadJSONConfigFile`. Called from `main.go:70` (aborts startup on error) and `slash.go:77` (`/conf <name>` runtime switch). No session persistence — runtime switches live only in the model's memory.

#### `GetAvailableConfigs` — `config.go:272`
- **Signature:** `func GetAvailableConfigs() ([]string, string, error)`
- **Purpose:** Lists configuration profiles for the `/conf` command.
- **Implementation:** Loads the JSON file (error `"could not load config file"` if absent), returns the sorted keys of `Configurations` plus `DefaultConfiguration`. Does not consult env vars or CLI flags.
- **Calls / Called by:** Calls `loadJSONConfigFile`. Called by the `/conf` handler at `slash.go:46`. The `/conf <name>` switch at `slash.go:77` re-runs `LoadConfig(args[0])` and then copies select fields into the live model: `m.cfg`, `m.collection`, `m.searchCap`, `m.rerankerPool`, plus the `rag` package globals `rag.HTTPTimeout` and `rag.QdrantVectorName` — a new config option that must be runtime-switchable needs a line there too.

---

## Root Package — `skills.go` (Agentic Tool System)

`skills.go` defines the local tool-execution system used inside the generation loop. The LLM is told about available tools via a system-prompt fragment (`SkillRegistry.ForPrompt`, injected in `commands.go:581`) and signals a tool call by emitting a `CALL: <name> <args>` line, which `ParseCall` extracts from the completed stream output (checked at `model.go:689`). The registry dispatches to a `Skill` implementation; the only built-in is `BashSkill`, which runs a shell command with a 30-second kill timer. Confirmation gating (`y`/`a`/`n` dialog) is driven by `Config.SkillsRequireConfirm` and lives in `model.go`, not here.

#### `Skill` interface — `skills.go:15`
```go
type Skill interface {
    Name() string                                  // unique identifier
    Description() string                           // human-readable summary for LLM tool-use prompting
    Execute(ctx context.Context, args []byte) (string, error)  // run with JSON args
}
```

#### `SkillRegistry` — `skills.go:26`
- **Field:** `skills map[string]Skill`
- Registered skills, keyed by `Name()`. Held by `Model` as `m.skills` (`model.go:129`), populated in `NewModel` (`model.go:358`) via `NewSkillRegistry()`.

#### `NewSkillRegistry` — `skills.go:30`
- **Signature:** `func NewSkillRegistry() SkillRegistry`
- **Implementation:** Creates the map and registers `BashSkill{}` — the only automatic registration site. A new skill must be added here (or registered elsewhere) to appear in the app.

#### `(*SkillRegistry).Register` — `skills.go:36`
- **Signature:** `func (r *SkillRegistry) Register(s Skill)` — adds/overwrites by name.

#### `(*SkillRegistry).Get` — `skills.go:40`
- **Signature:** `func (r *SkillRegistry) Get(name string) (Skill, bool)` — lookup used by `executeSkillCmd` (`commands.go:407`).

#### `(*SkillRegistry).List` — `skills.go:45`
- **Signature:** `func (r *SkillRegistry) List() []Skill` — all skills sorted by name (map iteration order is otherwise random).

#### `(*SkillRegistry).ForPrompt` — `skills.go:57`
- **Signature:** `func (r *SkillRegistry) ForPrompt() string`
- **Purpose:** Generates the tool-use system-prompt fragment describing all registered skills.
- **Implementation:** Empty string when no skills (caller skips injection). Otherwise `"Available tools:\n"` plus one `"- <name>: <description>\n"` line per skill (sorted). The full injected block — including the strict `CALL: <tool_name> <arguments>` syntax instructions and the "Stop generating immediately after outputting the CALL block" rule — is assembled in `buildPromptMessages` (`commands.go:581-588`).

#### `ParseCall` — `skills.go:71`
- **Signature:** `func ParseCall(output string) (name, args string, ok bool)`
- **Purpose:** Parses a tool call string of the format `CALL: <name> <args>` from LLM output.
- **Implementation:** Scans output line-by-line; first line (trimmed) starting with `"CALL:"` is taken. The remainder is split with `strings.Fields`; first field is the skill name, the rest (from the name's first occurrence onward, trimmed) is the raw args string — args may contain spaces and are NOT JSON-parsed here. Returns `("", "", false)` if no CALL line found. Called on the completed stream output at `model.go:689` to trigger the skill-execution branch of the FSM.

#### `BashSkill` — `skills.go:93`
- **Signature:** `func (BashSkill) Name() string` → `"bash"`; `Description()` → `"Execute a bash command and return stdout/stderr"`
- **`Execute` — `skills.go:103`:**
  1. **Args parsing:** tries `json.Unmarshal` into `{"command": string}` first; falls back to the raw args bytes as the command string.
  2. **Empty check:** trimmed empty command → error `"empty command"`.
  3. **Shell resolution:** `/bin/bash` → `/bin/sh` → PATH lookup `"bash"` (first that `exec.LookPath` finds).
  4. **Execution:** `exec.CommandContext(ctx, shell, "-c", cmdStr)` with stdout+stderr captured separately; `ctx` cancellation kills the process (wired to the model's context, so double-Esc cancels a running skill).
  5. **Timeout:** `time.AfterFunc(30s)` kills the process; timer stopped via defer after `Wait` returns.
  6. **Output:** stdout, and if stderr is non-empty it is appended as `"Stderr: <stderr>"`. Returns `(output, err)` where err is the process exit error (non-zero exit is surfaced, but output is still returned).

**Skill result truncation:** `executeSkillCmd` (`commands.go:412-416`) caps skill output at 8 KiB (appending `"[... Output truncated to 8 KiB ...]"`) before feeding it back to the LLM.

### Extension recipe — add a new skill

1. Implement the `Skill` interface (Name/Description/Execute) — a new type in `skills.go` or a new file.
2. Register it in `NewSkillRegistry` (`skills.go:30-34`) beside `BashSkill{}`.
3. Nothing else: `ForPrompt` automatically advertises it to the LLM, `ParseCall`/`executeSkillCmd` dispatch generically, and the confirm dialog in model.go gates it like any other skill. Test in `skills_test.go` (follow `TestParseCall`/bash skill test patterns).

---

## Root Package — `slash.go` (Runtime Slash Commands)

### File Overview

| File | LOC | Role |
|---|---|---|
| `slash.go` | 575 | Single entry point `handleSlashCmd`: a giant switch that parses, validates, and directly mutates `Model` fields for every runtime slash command, returning a `tea.Msg`. |

`slash.go` contains exactly one function, `handleSlashCmd(raw string) tea.Cmd`. Input routing is in `model.go:603-605`: any input starting with `/` (typed only in `stateIdle`/`stateError`) is dispatched to it and is NOT saved to `inputHistory`. The returned closure tokenizes with `strings.Fields`, then a `switch cmd` handles each command.

**Design pattern (important):** Model mutations happen inside the returned closure, not in `Update` — the file's header comment (`slash.go:16-18`) asserts this is safe because Bubbletea dispatches messages serially. Tests in `slash_test.go` exploit this by calling `cmd()` synchronously and then inspecting `m` directly.

`slash.go` declares no types or globals. It mutates two package globals in `internal/rag` — `rag.HTTPTimeout` and `rag.QdrantVectorName` (`slash.go:90-91`, from `/conf`).

### Slash-Command Dispatch Table

All cases are inline in `handleSlashCmd` (`slash.go:32`) unless a delegate is named. Args = whitespace-split fields after the command token.

| Command | Handler (location) | Mutates / does |
|---|---|---|
| `/collection <name>` | inline `slash.go:33` | Requires exactly 1 arg. Sets `m.collection`. |
| `/conf` (no args) | inline `slash.go:45-68` | `GetAvailableConfigs()` (`config.go:272`); returns `systemLogMsg` listing active/default/available profiles. |
| `/conf <name>` | inline `slash.go:69-93` | At most 1 arg. `LoadConfig(name)` (`config.go:76`); on success sets `m.cfg`, `m.collection`, `m.searchCap`, `m.rerankerPool`, and globals `rag.HTTPTimeout`, `rag.QdrantVectorName`. Feedback contains `"Switched to configuration"` (triggers LLM/embedder re-check in `model.go:806`). Does not reset `searchMode`/`searchExpand`/`filter*`/`searchLimit`. |
| `/limit <1-100>` | inline `slash.go:95` | Exactly 1 arg, integer 1–100. Sets `m.searchLimit`. |
| `/mode <strict\|hybrid>` | inline `slash.go:114` | Exactly 1 arg, lowercased; anything else errors. Sets `m.ragMode` (selects the default system prompt in `buildPromptMessages`). |
| `/rewrite [llm\|heuristic\|off]` | inline `slash.go:132` | No args = show current. Sets `m.cfg.QueryRewrite`. Not listed in `/help`. |
| `/exact <phrase...>` | inline `slash.go:154` | Joins args; sets `m.exactPhrases`, `m.exactPhrase`, `m.forceExactPhrase=true`, `m.lastQuery=phrase`, clears `output`/`lastPoints`/`ragContext`, `m.state=stateSearching`. Invokes search and returns its Msg: `return m.searchQdrantCmd(nil)()`. |
| `/search [auto\|exact\|local]` | inline `slash.go:174` | No args = show current. Sets `m.searchMode`. Extra args beyond the first are silently ignored. |
| `/filter [clear]`, `/filter [key] <val...>` | inline `slash.go:197` | 0 args or `clear` → empties `m.filterKey`/`m.filterValue`. 1 arg → `key="*"`, `val=args[0]`. ≥2 args → key + space-joined value. Strips one matching pair of surrounding `"` or `'` from val. |
| `/cap` (no args) | inline `slash.go:225` | Shows current cap (`none` when 0) and mode. |
| `/cap <N\|off\|none\|unlimited>` | inline `slash.go:250-263` | `off\|none\|unlimited` → `m.searchCap=0`; else positive integer required → `m.searchCap=n` (HNSW candidate pool). |
| `/cap <auto\|exact\|local>` | inline `slash.go:239-249` | Mode aliases that set `m.searchMode` without touching the numeric cap (duplicate of `/search`). |
| `/cache [status]` | inline `slash.go:266` | Default subcommand. `rag.CacheInfo` + `rag.CacheDir` → `systemLogMsg`. |
| `/cache refresh` | inline `slash.go:277` | Arms `m.cacheForceRefresh=true` (next full-corpus query re-scrolls). |
| `/cache warmup` | inline `slash.go:281` | Returns `warmupCacheMsg{}` → dispatched to `warmupCacheCmd` in `model.go:863-870`. |
| `/cache clear` | inline `slash.go:284` | `rag.DeleteCorpusCache` (disk side effect); clears `m.cacheInfo`. |
| `/cache dir` | inline `slash.go:291` | Shows `rag.CacheDir()`. Unknown subcommands error. |
| `/expand` (no args) | inline `slash.go:301` | Shows current (`off` when 0). |
| `/expand <N\|off\|none\|0>` | inline `slash.go:308-332` | Integer 0–20 (`>20` errors; negatives/parse errors error). Sets `m.searchExpand` (±N adjacent chunks per match). |
| `/rerank <on\|off>` | inline `slash.go:334` | Exactly 1 arg. Sets `m.disableReranker` (true = bypass Stage 2.5; also collapses `computeSearchDocs` to plain limit). |
| `/rerankerpool` (no args) | inline `slash.go:358` | Shows current (auto when ≤0). |
| `/rerankerpool <N\|auto\|off\|none\|0>` | inline `slash.go:364-378` | `auto\|off\|none\|0` → `m.rerankerPool=0` (dynamic); else positive int → fixed pool. |
| `/system <prompt...>` | inline `slash.go:380` | ≥1 arg required. Joins and sets `m.systemPrompt` (overrides built-in strict/hybrid prompt). |
| `/copy` | inline `slash.go:415-461` | Default: finds last `assistant` turn, prepends reasoning as a fenced "Thinking" block if present, `clipboard.WriteAll`. Error if no response. |
| `/copy all` | `m.copyAllConversationCmd()()` `slash.go:393` (impl `model.go:1671`) | Whole transcript (plain-text format) to clipboard. Note the invoked `()` — the correct pattern for chaining a model cmd from a slash handler. |
| `/copy ref` / `/copy refs` | inline `slash.go:395-413` | `m.getActiveReferences()` (`model.go:1352`) → `formatReferences(refs, 80)` (`model.go:1815`) → clipboard. |
| `/copy` while ref panel focused | inline `slash.go:415-433` | Same as `/copy ref` (branch on `m.focusRef`). |
| `/save <file>`, alias `/write` | `m.saveLastResponseCmd(filename)()` `slash.go:489` (impl `model.go:1722`) | Last assistant response (with reasoning block) written via `os.WriteFile` 0644. `/write` is an undocumented alias (shared case `slash.go:463`). |
| `/save all <file>` | `m.saveAllConversationCmd(filename)()` `slash.go:487` (impl `model.go:1750`) | Full markdown transcript to file. |
| `/compact [N]` | inline `slash.go:492` | `keepPairs` default 3; only parsed if exactly 1 arg and `n>=1` (invalid values silently fall back to 3). Calls `m.compactHistory(keepPairs)` (`model.go:224`), appends a system turn recording entries removed, `m.updateViewport()`. |
| `/clear` | inline `slash.go:517` | Nils `history`/`lastPoints`/`ragContext`/`output`, appends `[ Conversation and references cleared ]` system turn, `updateViewport()`. |
| `/quit` | inline `slash.go:529` | Returns `quitMsg{}`. |
| `/help` | inline `slash.go:532` | Static `helpText` string returned as `systemLogMsg`. **Omits `/rewrite`, `/exact`, and the `/write` alias.** |
| anything else | inline `slash.go:567` | `appErrMsg` "unknown command", stage `"slash"`. |

### Extension recipe — add a new slash command

1. Add a `case "/yourcommand":` to the switch in `handleSlashCmd` (`slash.go:32`). Parse `args` (already `strings.Fields`-tokenized), validate, mutate `m.*` fields, and return `slashResultMsg{feedback: ...}` (or `systemLogMsg` for verbose output, `appErrMsg{stage: "slash"}` for errors).
2. To chain a pipeline command, **invoke** the cmd and return its Msg (the `/copy all` and `/exact` pattern, `slash.go:393`, `slash.go:173`): `return m.someCmd(args)()` — never return the `tea.Cmd` itself.
3. Add the command to the `helpText` string (`slash.go:532`) — several commands are currently missing from it.
4. Add tests in `slash_test.go` following the existing pattern: call `cmd := m.handleSlashCmd("/yourcommand ...")`, execute `msg := cmd()`, assert on the returned Msg **and** on the mutated model fields.

---

## Root Package — `commands.go` (Async Pipeline Commands)

### File Overview

| File | LOC | Role |
|---|---|---|
| `commands.go` | 649 | All `tea.Cmd` builders for the RAG pipeline (embed → search → rerank → stream), cache warmup, skill execution, connection checks, plus the prompt-assembly and query-rewrite logic they rely on. |

`commands.go` is the asynchronous side of the Bubbletea app: every function returns a `tea.Cmd` (a `func() tea.Msg`) whose closure captures the `*Model` pointer and runs on a goroutine owned by the Bubbletea runtime. The file implements the 4-stage pipeline — Stage 1 embedding (with 3-layer query rewriting), Stage 2 Qdrant search (four selectable paths: exact-phrase, capped HNSW, local full-corpus, default context-expansion), optional Stage 2.5 rerank, and Stage 3 SSE streaming — plus the system-prompt builder `buildPromptMessages` that decides what the LLM actually sees. Two pure helpers (`computeSearchDocs`, `computeRerankerPool`) encode the candidate-pool sizing policy: a wide pool for recall at search time, a narrow clamped pool (10–20) for reranker calibration accuracy.

This file declares no types, consts, or globals of its own. It **produces** pipeline message types defined in `messages.go` and calls into `internal/rag` (`GetEmbedding`, `ChatComplete`, `SearchQdrant*`, `Rerank`, `StartLiteLLMStream`, `WarmupCorpusCache`, `LoadCorpusCache`, connection checkers).

### Functions (source order)

#### `generateEmbeddingCmd` — `commands.go:17`
- **Signature:** `func (m *Model) generateEmbeddingCmd(query string) tea.Cmd`
- **Purpose:** Pipeline Stage 1 — rewrite the query for conversational context, then call the embedding endpoint.
- **Implementation:** Calls `m.condenseQueryForRetrieval(m.ctx, query)` to get the retrieval text, then `rag.GetEmbedding(ctx, cfg.EmbeddingURL, cfg.EmbeddingAPIKey, cfg.EmbeddingModel, retrievalQuery)`. On error returns `appErrMsg{stage: "embedding"}`; on success `embeddingMsg{vector}`.
- **Calls / Called by:** Called from `model.go:643` (Enter-key pipeline start). Calls `condenseQueryForRetrieval`, `rag.GetEmbedding`.

#### `condenseQueryForRetrieval` — `commands.go:33`
- **Signature:** `func (m *Model) condenseQueryForRetrieval(ctx context.Context, raw string) string`
- **Purpose:** Decide what text gets embedded, via a 3-layer fallback: (1) LLM rewrite, (2) heuristic pronoun detection, (3) raw passthrough.
- **Implementation:** Layer 3 — if `m.cfg.QueryRewrite == "off"` or `m.history` is empty, return `raw` unchanged. Layer 1 (`"llm"`, the config default) — builds `[]rag.ChatMessage`: a fixed system prompt instructing standalone-question rewriting, the last 4 history turns (only `user`/`assistant` roles, each content truncated to 500 bytes), then the raw query; calls `rag.ChatComplete(ctx, cfg.OpenAIURL, cfg.OpenAIAPIKey, cfg.OpenAIModel, messages, 128)` under a 15s `context.WithTimeout`. The rewrite is accepted **only if** err is nil, non-empty, and contains no `"\n"` — otherwise falls through to Layer 2. Layer 2 (heuristic, also the fallback for `"heuristic"` mode) — marks the query follow-up-shaped if it has ≤8 fields OR its lowercase form *contains* any of ~20 pronoun/reference keywords (`"it"`, `"this"`, `"above"`, `"and what"`, …; note substring matching, so `"it"` matches inside `"with"`); if follow-up, finds the most recent `user` turn in history and returns `prevContent + "\n" + raw`. Default: `raw`.
- **Calls / Called by:** Only called by `generateEmbeddingCmd`. Calls `rag.ChatComplete`.

#### `computeSearchDocs` — `commands.go:117`
- **Signature:** `func (m *Model) computeSearchDocs(docs, expand int) int`
- **Purpose:** Size the Qdrant candidate pool for a query given the reranker configuration and expansion setting.
- **Implementation:** Clamps `expand` to ≥0. If no reranker is configured (`cfg.RerankerURL == ""`) or `m.disableReranker` is true, returns `docs` unchanged — no second stage, so no oversized pool needed. Otherwise `basePool = m.rerankerPool` if >0, else `max(docs*5, 50)`. If `expand > 0`: `searchDocs = basePool * (expand+1)`, then capped at `max(500, m.rerankerPool)`. Final floor at `docs`.
- **Calls / Called by:** Called only by `searchQdrantCmd`.

#### `computeRerankerPool` — `commands.go:169`
- **Signature:** `func (m *Model) computeRerankerPool() int`
- **Purpose:** Number of primary candidates forwarded to the reranker API — deliberately decoupled from the search pool (small models like Qwen3-Reranker-4B degrade beyond ~20 candidates).
- **Implementation:** Returns `m.rerankerPool` if >0 (the `/rerankerpool` override); otherwise `clamp(m.searchLimit*3, 10, 20)`.
- **Calls / Called by:** Called only by `rerankPointsCmd`.

#### `searchQdrantCmd` — `commands.go:200`
- **Signature:** `func (m *Model) searchQdrantCmd(vector []float32) tea.Cmd`
- **Purpose:** Pipeline Stage 2 — similarity search in Qdrant, choosing one of four paths from `m.searchCap`/`m.searchMode`/phrase state. (The doc comment documenting these paths at `commands.go:113-115` is orphaned — it sits above `computeSearchDocs`, not `searchQdrantCmd`.)
- **Implementation:** Computes `docs = m.searchLimit`, `expand = m.searchExpand`, `searchDocs = m.computeSearchDocs(docs, expand)`, then: (1) **Exact-phrase path** — if `len(m.exactPhrases) > 0`: `rag.SearchQdrantExactPhrases(ctx, …, m.exactPhrases, searchDocs, m.filterKey, m.filterValue)` → `searchResultMsg` with `expand: 0` and empty `ExpansionMap` (works with a nil vector, which is how the `/exact` and fully-quoted-query flows call it). (2) **Capped HNSW path** — if `m.searchCap > 0 && m.exactPhrase == ""`: `candidateLimit = max(searchCap, searchDocs)`; `exact := (m.searchMode == "exact")`; `rag.SearchQdrant(ctx, …, vector, candidateLimit, searchDocs, filter, exact)`; expansion bypassed, `expand: 0`. (3) **Local path** — if `m.searchMode == "local" || m.exactPhrase != ""`: `rag.SearchQdrantFullCorpus(ctx, …, searchDocs, filter, m.qdrantPoints, m.cacheForceRefresh, nil, m.exactPhrase)`; on success mutates the Model from the closure: `m.cacheForceRefresh = false`, and if served from cache reloads `rag.LoadCorpusCache` to refresh `m.cacheInfo` and `m.cacheFilterAtWarmup`. (4) **Default no-cap path** — `exact := m.searchMode == "exact" || m.searchMode == "auto"` (auto = exact brute-force when uncapped); `rag.SearchWithContextExpansionDetailed(ctx, …, searchDocs, expand, filter, exact)` returning `res.Context / res.ExpandedPoints / res.PrimaryPoints / res.ExpansionMap`; if quoted phrases exist but exact search was not forced, boosts primaries via `rag.BoostPhraseMatches`. All errors → `appErrMsg{stage: "search"}` with path-specific reasons.
- **Calls / Called by:** Called from `model.go:631` (exact-phrase Enter flow), `model.go:652` (after `embeddingMsg`), and `slash.go:173` (`/exact` invoked via `return m.searchQdrantCmd(nil)()`).

#### `fetchQdrantInfoCmd` — `commands.go:306`
- **Signature:** `func (m *Model) fetchQdrantInfoCmd() tea.Cmd`
- **Purpose:** Fetch collection stats for the header bar.
- **Implementation:** `rag.GetCollectionInfo(ctx, cfg.QdrantURL, cfg.QdrantAPIKey, m.collection)` → `qdrantInfoMsg{pointsCount, vectorsCount, status, err}` (error carried in the msg, not as `appErrMsg`).
- **Calls / Called by:** `model.go:373` (Init batch), `model.go:807` (after every `slashResultMsg`).

#### `preloadCacheInfoCmd` — `commands.go:322`
- **Signature:** `func (m *Model) preloadCacheInfoCmd() tea.Cmd`
- **Purpose:** Check for an existing on-disk corpus cache at startup and report it.
- **Implementation:** `rag.LoadCorpusCache(cfg.QdrantURL, m.collection)`; on error/nil → `cachePreloadMsg{found: false}`; else builds `"✓ <N> pts (<age> old)"` (age truncated to seconds) → `cachePreloadMsg{found, info, pointCount}`.
- **Calls / Called by:** `model.go:374` (Init), `model.go:807` (after every `slashResultMsg`).

#### `startLLMStreamCmd` — `commands.go:335`
- **Signature:** `func (m *Model) startLLMStreamCmd() tea.Cmd`
- **Purpose:** Pipeline Stage 3 start — open the SSE stream to the LLM and bootstrap the chunk loop.
- **Implementation:** `messages := m.buildPromptMessages()`; sets `m.currentPromptEstimate = estimateChatMessageTokens(messages)` (`model.go:176`) as a fallback for providers that don't return usage. Creates a cancellable child context stored in `m.cancelRequest` (used by Esc-Esc stop and quit). `rag.StartLiteLLMStream(ctx, cfg.OpenAIURL, cfg.OpenAIAPIKey, cfg.OpenAIModel, cfg.OpenAIMaxTokens, cfg.ContextLimit, messages)`; error → `appErrMsg{reason: "LLM connection failed", stage: "stream"}`; success stores `m.streamReader` and **returns `m.receiveStreamChunkCmd()()`** — immediately invoking the first chunk read so the self-chaining loop starts without waiting for a round trip.
- **Calls / Called by:** `model.go:668` (after search), `model.go:683` (after rerank), `model.go:761` (after skill result). Calls `buildPromptMessages`, `estimateChatMessageTokens`, `rag.StartLiteLLMStream`, `receiveStreamChunkCmd`.

#### `receiveStreamChunkCmd` — `commands.go:356`
- **Signature:** `func (m *Model) receiveStreamChunkCmd() tea.Cmd`
- **Purpose:** Read one SSE chunk; the self-chaining loop node.
- **Implementation:** Nil `m.streamReader` → `streamChunkMsg{done: true}`. `m.streamReader.Next()` → `(chunk, reasoning, done, err)`; error → `appErrMsg{reason: "Stream read error"}`. Captures `usage := m.streamReader.Usage()` and sets `hasUsage` true iff any of Total/Prompt/Completion tokens > 0. `model.go:740` re-issues this cmd while `done` is false.
- **Calls / Called by:** Called by `startLLMStreamCmd` and `model.go:740`.

#### `warmupCacheCmd` — `commands.go:379`
- **Signature:** `func (m *Model) warmupCacheCmd() tea.Cmd`
- **Purpose:** Scroll the whole Qdrant collection into the on-disk cache (triggered by `/cache warmup` via `warmupCacheMsg`).
- **Implementation:** `rag.WarmupCorpusCache(ctx, …, m.collection, progressCb)` with a no-op progress callback; error → `appErrMsg{reason: "Cache warmup failed", stage: "search"}`. On success mutates `m.cacheInfo` and `m.cacheFilterAtWarmup` and returns `systemLogMsg` with completion content + short feedback (which lands the FSM back in idle via the `systemLogMsg` case at `model.go:795`).
- **Calls / Called by:** `model.go:870` only.

#### `executeSkillCmd` — `commands.go:405`
- **Signature:** `func (m *Model) executeSkillCmd(name, args string) tea.Cmd`
- **Purpose:** Run a registered local tool/skill asynchronously.
- **Implementation:** `m.skills.Get(name)`; miss → `skillResultMsg{err: "skill not found: <name>"}`. `skill.Execute(m.ctx, []byte(args))`; output truncated to 8192 bytes with a `"[... Output truncated to 8 KiB ...]"` marker to protect prompt size. Returns `skillResultMsg{name, input, output, err}`.
- **Calls / Called by:** `model.go:466` and `model.go:475` (skill-confirmation dialog), `model.go:712` (parsing a `CALL:` block emitted by the LLM per the tool prompt injected in `buildPromptMessages`).

#### `rerankPointsCmd` — `commands.go:439`
- **Signature:** `func (m *Model) rerankPointsCmd(result searchResultMsg) tea.Cmd`
- **Purpose:** Optional Stage 2.5 — rerank only the *primary* top-N (not the expanded set), then re-apply ±expand around the reranked winners.
- **Implementation:** Takes the whole `searchResultMsg` (needs `primaryPoints`, `expansionMap`, `expand`). `primaries = result.primaryPoints`, falling back to `result.points` (HNSW/local paths don't separate the two); empty → `rerankResultMsg{context: "", points: nil}`. Caps primaries at `m.computeRerankerPool()`. Builds texts via `pt.ExtractPrimaryText()` (single clean passage; empty-text candidates skipped, surviving indices tracked in `validIndices`); no valid texts → `rerankResultMsg{…, degraded: true}`. `rag.Rerank(ctx, cfg.RerankerURL, cfg.RerankerAPIKey, cfg.RerankerModel, m.lastQuery, texts)`; error → graceful degradation returning the original primaries with `degraded: true`. Maps returned scores back through `validIndices`; unranked primaries get `-999.0`; each point keeps its Qdrant score in `OriginalScore` and takes the rerank score in `Score`. Stable-sorts descending, slices to `m.searchLimit` → `rerankedTopK`. `rag.ApplyExpansionToPrimaries(rerankedTopK, result.expansionMap, result.expand)` rebuilds context+points from chunks already fetched (pure CPU, no network); if it returns nil points, defensively rebuilds context by joining `ExtractText()` with `"\n---\n"`. Returns `rerankResultMsg{context, points}`.
- **Calls / Called by:** `model.go:661` (only when a reranker is configured and enabled).

#### `buildPromptMessages` — `commands.go:557`
- **Signature:** `func (m *Model) buildPromptMessages() []rag.ChatMessage`
- **Purpose:** Assemble the full chat payload for the LLM: system prompt, history, RAG context, current query.
- **Implementation:** System prompt: uses `m.systemPrompt` if set (`/system`); otherwise a long built-in prompt selected by `m.ragMode` — `"hybrid"` blends context with general knowledge with explicit attribution (`commands.go:563-569`); anything else (strict, the default) mandates closed-book grounding, forbids hallucination, and requires `[Document: filename | Chunk X]` citations (`commands.go:571-579`). If `m.skills.ForPrompt()` is non-empty, appends the tool list plus the exact `CALL: <tool_name> <arguments>` syntax block (`commands.go:581-588`). History replay rules (`commands.go:595-609`): `user` turns replayed prefixed `"Question: "`; `assistant` turns as-is; `system` turns replayed **only** when content starts with `"[ Context compacted"` — rewritten as a `user` message `"[ Summary of earlier conversation ]\n…"` (roles matter for provider compatibility); all other system turns (slash feedback, warmup logs, skill logs) are UI-only and excluded — retrieval is turn-scoped so old results don't contaminate refined searches. RAG context: if `m.lastPoints` is non-empty, emits `"Retrieved Context Chunks from Knowledge Base:"` followed by `"--- Chunk N | Document: <name|ID> ---"` blocks (`extractDocumentName` is `model.go:2045`). Final message: `context + "Question: " + m.lastQuery` as one `user` message.
- **Calls / Called by:** Only called by `startLLMStreamCmd`.

#### `checkLLMInfoCmd` — `commands.go:636`
- **Signature:** `func (m *Model) checkLLMInfoCmd() tea.Cmd`
- **Purpose:** Connectivity check for the LLM endpoint.
- **Implementation:** `rag.CheckLLMConnection(ctx, cfg.OpenAIURL, cfg.OpenAIAPIKey, cfg.OpenAIModel)` → `llmInfoMsg{err}`; the `model.go:814` case drives `m.llmStatus` (`"ok"`/`"error"`).
- **Calls / Called by:** `model.go:375` (Init), `model.go:811` (after config switch).

#### `checkEmbedderInfoCmd` — `commands.go:644`
- **Signature:** `func (m *Model) checkEmbedderInfoCmd() tea.Cmd`
- **Purpose:** Connectivity check for the embedding endpoint.
- **Implementation:** `rag.CheckEmbeddingConnection(ctx, cfg.EmbeddingURL, cfg.EmbeddingAPIKey, cfg.EmbeddingModel)` → `embedderInfoMsg{err}`; drives `m.embedderStatus` via `model.go:825`.
- **Calls / Called by:** `model.go:376` (Init), `model.go:811` (after config switch).

---

## Root Package — `model.go` (The TUI Model & FSM)

### File Overview

`model.go` (2360 lines) is the heart of the QQuestio Bubbletea TUI: it defines the central `Model` struct (the entire app state), the `appState` FSM enum, the `Update` message-dispatch loop that drives the RAG pipeline (embed → search → rerank → stream → optional tool-call/skill loop), all rendering (`View`, header, footer, conversation viewport, references panel), context-window auto-compaction, session save/load, and clipboard/file export helpers. Pipeline commands themselves live in `commands.go`; message types (`embeddingMsg`, `searchResultMsg`, `streamChunkMsg`, …) in `messages.go`; slash-command dispatch in `slash.go`; skill registry/`ParseCall` in `skills.go`; HTTP clients in `internal/rag`.

#### Types defined here

- `appState` (model.go:27) — int enum: `stateIdle`, `stateEmbedding`, `stateSearching`, `stateReranking`, `stateStreaming`, `stateError`, `stateConfirmQuit`, `stateConfirmSkill` (model.go:30-37).
- `ConversationTurn` (model.go:40) — one transcript entry: `Role` ("user"|"assistant"|"system"), `Content`, `Reasoning` (thinking text), `References []rag.QdrantPoint`, plus render caches `RenderedContent`/`RenderedWidth`/`RenderedReferences`/`RenderedReferencesWidth` (invalidated by width change).
- `Session` (model.go:2182) — JSON persistence struct: `ID`, `Collection`, `SearchLimit`, `SearchCap`, `RerankerPool`, `SearchExpand`, `SearchMode`, `SystemPrompt`, `RAGMode`, `FilterKey`, `FilterValue`, `History []ConversationTurn`, `TokensConsumed`, `TokensAreEstimated`.
- `Model` (model.go:51) — the Bubbletea model. Every field:

| Field | Type | Meaning |
|---|---|---|
| `cfg` | `Config` | Immutable config from config.go |
| `sessionID` | `string` | Session id, timestamp format `20060102-150405` |
| `loadingMessage` | `string` | Random "Consulting the oracle..." phrase per query |
| `leakState` | `int` | Mini-FSM (0/1/2) filtering leaked ESC[... mouse sequences arriving as KeyMsgs |
| `collection` | `string` | Active Qdrant collection (init `cfg.DefaultCollection`) |
| `searchLimit` | `int` | Top-N results (init 10 — the field comment says 5, code says 10) |
| `searchCap` | `int` | Candidate-pool cap for search; 0 = full corpus |
| `rerankerPool` | `int` | Reranker primary pool size; 0 = auto |
| `searchExpand` | `int` | ±N adjacent chunks per top match (init 1; 0 = off) |
| `searchMode` | `string` | "auto" / "exact" / "local" |
| `systemPrompt` | `string` | Custom system prompt override |
| `ragMode` | `string` | "strict" or "hybrid" |
| `filterKey`/`filterValue` | `string` | Active Qdrant metadata filter |
| `disableReranker` | `bool` | Bypass reranking |
| `cacheForceRefresh` | `bool` | One-shot full-corpus re-scroll flag (set by /cache refresh) |
| `cacheInfo` | `string` | Header cache summary line |
| `cacheFilterAtWarmup` | `string` | Filter string active when cache was warmed (mismatch → header warning) |
| `state` | `appState` | Current FSM state |
| `textInput` | `textarea.Model` | The input box (a textarea despite the name; 1 line tall, grows with wrap) |
| `viewport` | `viewport.Model` | Main conversation panel (left) |
| `refViewport` | `viewport.Model` | References panel (right) |
| `focusRef` | `bool` | True = right panel has focus |
| `spinner` | `spinner.Model` | Dot spinner |
| `statusMsg` | `string` | Header status line text |
| `history` | `[]ConversationTurn` | Full transcript |
| `output` / `reasoning` | `string` | Accumulated LLM answer / thinking text for the current turn |
| `showRawSource` | `bool` | Ctrl+R toggle: raw markdown vs glamour render |
| `tokensConsumed` | `int` | Cumulative session tokens |
| `lastTurnTokens` | `int` | Tokens of last completed call |
| `tokensAreEstimated` | `bool` | Server omitted usage → "~" prefix in header |
| `currentPromptEstimate` | `int` | Fallback prompt estimate (set by commands.go before LLM call) |
| `lastQuery` | `string` | Query that started the current pipeline (also reused as tool-response text in the skill loop) |
| `forceExactPhrase` | `bool` | Set by `/exact` |
| `exactPhrase` / `exactPhrases` | `string` / `[]string` | Quoted-phrase extraction results (bypass embedding) |
| `ragContext` | `string` | Retrieved context text for current turn |
| `lastPoints` | `[]rag.QdrantPoint` | Retrieved points for current turn |
| `cancelRequest` | `context.CancelFunc` | Aborts in-flight HTTP |
| `streamReader` | `*rag.SSEReader` | Live SSE reader (closed on completion/abort/error) |
| `escCount` | `int` | Consecutive Esc presses (double-Esc aborts pipeline) |
| `stoppedByUser` | `bool` | Suppresses the appErrMsg that follows a user abort |
| `qdrantPoints`/`qdrantVectors`/`qdrantStatus`/`qdrantInfoErr` | `int`/`int`/`string`/`error` | Qdrant stats for header |
| `llmStatus`/`llmInfoErr` | `string`/`error` | "checking"/"ok"/"error" |
| `embedderStatus`/`embedderInfoErr` | `string`/`error` | Same for embedder |
| `inputHistory`/`historyIndex`/`tempInput` | `[]string`/`int`/`string` | Up/Down prompt history + draft being edited |
| `skills` | `SkillRegistry` | Loaded skills |
| `pendingSkillName`/`pendingSkillArgs` | `string` | Skill awaiting confirmation |
| `skillsAlwaysAllowed` | `bool` | Set by "A" in the skill-confirm dialog |
| `width`/`height` | `int` | Terminal size |
| `ctx` | `context.Context` | Root context from main.go |

### State Machine

States: `stateIdle` → `stateEmbedding` → `stateSearching` → `stateReranking` (optional) → `stateStreaming` → `stateIdle`; side states `stateError`, `stateConfirmQuit`, `stateConfirmSkill`.

Transitions (all inside `Update`, model.go:380):

| From | To | Trigger |
|---|---|---|
| idle/error | searching | `tea.KeyEnter` with quoted-only query or `forceExactPhrase` → `searchQdrantCmd(nil)` (model.go:622-631) |
| idle/error | embedding | `tea.KeyEnter` otherwise → `generateEmbeddingCmd(query)` (model.go:632-644) |
| embedding | searching | `embeddingMsg` → `searchQdrantCmd(msg.vector)` (model.go:649) |
| searching | reranking | `searchResultMsg` when `cfg.RerankerURL != "" && !disableReranker && exactPhrase == ""` → `rerankPointsCmd` (model.go:657-661) |
| searching | streaming | `searchResultMsg` otherwise → `startLLMStreamCmd()` (model.go:662-668) |
| reranking | streaming | `rerankResultMsg` → `startLLMStreamCmd()`; `msg.degraded` only changes statusMsg (model.go:671-683) |
| streaming | idle | `streamChunkMsg{done:true}` with no `ParseCall` match: appends user+assistant turns, resets `lastQuery`/`output`/`reasoning`, closes `streamReader` (model.go:714-735) |
| streaming | confirmSkill | `streamChunkMsg{done:true}` where `ParseCall(m.output)` detects a CALL: tool request and `cfg.SkillsRequireConfirm && !skillsAlwaysAllowed` (model.go:689-709) |
| streaming | (stay) streaming | otherwise `executeSkillCmd(name, args)` directly (model.go:711-713) |
| confirmSkill | searching | key `y` (allow once) or `a` (allow always, sets `skillsAlwaysAllowed`) → `executeSkillCmd` (model.go:459-475) |
| confirmSkill | idle | key `n` → emits synthetic `skillResultMsg{err: "denied"}` (model.go:476-489); Esc/Ctrl+C → cancel (model.go:490-496) |
| searching (skill) | streaming | `skillResultMsg` success: `lastQuery = "Tool X executed. Result:\n..."`, `output=""`, → `startLLMStreamCmd()` — this is the tool loop (model.go:743-761) |
| any | error | `appErrMsg` (unless `stoppedByUser`) — error state keeps input active (model.go:763-780) |
| any (not confirmQuit) | confirmQuit | first Ctrl+C (model.go:522-526) |
| confirmQuit | quit | second Ctrl+C or `y` (cancels `cancelRequest` first) (model.go:427-449) |
| confirmQuit | idle | Esc or `n` (model.go:432-448) |
| embedding/searching/reranking/streaming | idle | double Esc within the pipeline (cancels request, closes `streamReader`, clears `lastQuery`/`output`, sets `stoppedByUser`) (model.go:501-521) |
| idle | searching | `warmupCacheMsg` (from /cache warmup) → `warmupCacheCmd()` (model.go:863-870) |
| any | — | `quitMsg` → `cancelRequest()` + `tea.Quit` (model.go:836-840) |

Keys (idle path): Enter submit, Up/Down input history, Ctrl+R raw/rendered toggle, Ctrl+Y copy, Tab panel focus, Ctrl+Up/Down & PgUp/PgDn scroll (focus-dependent), Ctrl+C confirm-quit, Esc-Esc abort.

Non-FSM messages handled: `qdrantInfoMsg`, `llmInfoMsg`, `embedderInfoMsg`, `slashResultMsg` (re-fetches stats; "Switched to configuration" feedback re-checks LLM+embedder), `systemLogMsg`, `searchProgressMsg` (status-only), `cachePreloadMsg`, `spinner.TickMsg`, `tea.MouseMsg` (click left/right of ref panel boundary sets `focusRef`), `tea.WindowSizeMsg`.

### Functions (source order)

#### `estimateTokens` — `model.go:146`
- **Signature:** `func estimateTokens(s string) int`
- **Purpose:** Rune-aware token heuristic: ~1 token per 4 ASCII bytes, ~1 token per non-ASCII rune (CJK/emoji).
- **Calls / Called by:** `estimateContextTokens`, `estimateChatMessageTokens`, `recordTokenUsage` fallback (indirectly).

#### `estimateContextTokens` — `model.go:164`
- **Signature:** `func (m *Model) estimateContextTokens() int`
- **Purpose:** Estimate what will actually be sent to the LLM.
- **Implementation:** Sums `estimateTokens(turn.Content)` over `m.history` plus `ExtractText()` of current-turn `m.lastPoints` (history references are not replayed into prompts). Read by `maybeAutoCompact` and `renderHeader` (Ctx: % bar).
- **Calls / Called by:** `maybeAutoCompact` (model.go:274, 296), `renderHeader` (model.go:1092).

#### `estimateChatMessageTokens` — `model.go:176`
- **Signature:** `func estimateChatMessageTokens(messages []rag.ChatMessage) int`
- **Purpose:** Token estimate over a `[]rag.ChatMessage` payload (content + role).
- **Calls / Called by:** commands.go when pre-flight estimating a request; not called within model.go.

#### `recordTokenUsage` — `model.go:184`
- **Signature:** `func (m *Model) recordTokenUsage(usage rag.TokenUsage, hasUsage bool)`
- **Purpose:** Commit usage of a finished LLM call.
- **Implementation:** Uses `usage.TotalTokens` when present; otherwise falls back to `currentPromptEstimate + (len(output)+len(reasoning))/4` and sets `tokensAreEstimated = true`. Writes `lastTurnTokens` and accumulates `tokensConsumed`. Clamps negatives to 0.
- **Calls / Called by:** `Update` on `streamChunkMsg{done:true}` (model.go:687).

#### `hardWrappedLines` — `model.go:198`
- **Signature:** `func hardWrappedLines(text string, width int) []string`
- **Purpose:** Split text into display lines with ANSI-aware hard wrap (so URLs never overflow).
- **Implementation:** Uses `ansi.Hardwrap` per logical line; guards width<1 and empty results.
- **Calls / Called by:** `recalcLayout` (input height), `updateViewport` (current query).

#### `CompactionSummaryPrefix` — `model.go:218`
- Const `"[ Context compacted"` marking system turns holding compacted history.

#### `compactHistory` — `model.go:224`
- **Signature:** `func (m *Model) compactHistory(keepPairs int)`
- **Purpose:** Replace oldest Q&A pairs with one summary system turn.
- **Implementation:** Finds indices of `Role=="user"` turns; if pairs <= keepPairs, no-op. `cutoff = userIdx[len-keepPairs]`. Builds summary "Q: .../A: ..." (assistant content truncated at 300 chars + "…"), prepends `[]ConversationTurn{Role:"system", Content: summary}` before `m.history[cutoff:]`. Note: drops non-user/non-assistant turns below cutoff.
- **Calls / Called by:** `maybeAutoCompact`.

#### `maybeAutoCompact` — `model.go:267`
- **Signature:** `func (m *Model) maybeAutoCompact()`
- **Purpose:** Auto-compact when context exceeds 85% of `cfg.ContextLimit`; no-op when limit is 0.
- **Implementation:** Loops: `compactHistory(3)`; breaks if history stopped shrinking; appends an "[ Auto-compacted: N entries removed ... ]" system turn and sets `statusMsg`; exits when under 75% target.
- **Calls / Called by:** `Update` on `searchResultMsg` and `rerankResultMsg` (before starting the LLM call).

#### `NewModel` — `model.go:302`
- **Signature:** `func NewModel(ctx context.Context, cfg Config) *Model`
- **Purpose:** Construct the model with styled components.
- **Implementation:** Builds spinner (Dot, nord8), textarea (prompt `" ❯ "`, multi-line prompt func, blinking cursor, nord styles), two viewports, `sessionID` from `time.Now()`. Side effect on globals: sets `rag.HTTPTimeout` and `rag.QdrantVectorName` from cfg. Defaults: `searchLimit: 10`, `searchExpand: 1`, `searchMode: "auto"`, `state: stateIdle`, `ragMode: "strict"`, `llmStatus`/`embedderStatus: "checking"`, `skills: NewSkillRegistry()`, `statusMsg: "Ready"`.
- **Calls / Called by:** main.go program startup.

#### `Init` — `model.go:369`
- **Signature:** `func (m *Model) Init() tea.Cmd`
- **Purpose:** Bootstrap commands.
- **Implementation:** `tea.Batch(textarea.Blink, spinner.Tick, fetchQdrantInfoCmd, preloadCacheInfoCmd, checkLLMInfoCmd, checkEmbedderInfoCmd)`.

#### `Update` — `model.go:380`
- **Signature:** `func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd)`
- **Purpose:** The FSM dispatcher — all state transitions happen here.
- **Implementation:** Structure (in order):
  1. Leak filter (model.go:391-413): `leakState` machine swallows KeyMsgs forming `ESC [ <params> <letter>` CSI sequences.
  2. Confirm dialogs swallow all keys except their own: confirmQuit (Ctrl+C/y quit, Esc/n cancel; quit path calls `cancelRequest` first), confirmSkill (y/a/n, Esc/Ctrl+C).
  3. Global keys: Esc-Esc abort (pipeline states only), Ctrl+C → confirmQuit, Ctrl+Y → `copyLastResponseCmd`, Ctrl+R toggle `showRawSource`, Tab toggle `focusRef`, Ctrl+Up/Down + PgUp/PgDn scroll focused viewport, Up/Down walk `inputHistory` (with `tempInput` draft preservation), Enter submit (idle/error only; `/` prefix → `handleSlashCmd`; exact-phrase path vs embedding path as in the FSM table).
  4. Pipeline messages (see FSM table): `embeddingMsg`, `searchResultMsg`, `rerankResultMsg`, `streamChunkMsg` (chunk accumulation + `receiveStreamChunkCmd` re-arm; done path: `recordTokenUsage`, `ParseCall(m.output)` CALL: detection, `cleanLLMOutput`, history append, skill confirm or finish), `skillResultMsg`, `appErrMsg` (guarded by `stoppedByUser`; stage=="slash" also appends a system turn).
  5. Status/info messages: `qdrantInfoMsg`, `systemLogMsg` (appends system turn + sets statusMsg), `slashResultMsg` (re-fetch stats; config-switch re-check), `llmInfoMsg`, `embedderInfoMsg`, `quitMsg`, `spinner.TickMsg` (refreshes viewport during pipeline states), `searchProgressMsg` (status-only), `cachePreloadMsg`, `warmupCacheMsg`, `tea.MouseMsg` (focus by x < mainWidth), `tea.WindowSizeMsg`.
  6. Sub-model forwarding (model.go:902-946): spinner always (unless already handled); textarea gets all non-Mouse, non-WindowSize, non-Enter key msgs; viewports get non-key msgs only in streaming/idle/error, routed by `focusRef`. `recalcLayout()` on every non-WindowSize message.
  7. Leak scrub (model.go:948-963): regex `\[<\d+;\d+;\d+[mM]` (recompiled every call) strips residual SGR codes from `tempInput` and the textarea value.
  Returns `tea.Batch(cmds...)`.

#### `View` — `model.go:968`
- **Signature:** `func (m *Model) View() string`
- **Purpose:** Compose the screen.
- **Implementation:** `renderHeader()` + `lipgloss.JoinHorizontal(Top, mainView, refView)` (focused/unfocused styles from `DefaultStyles()` per `focusRef`) + `renderFooter()`.

#### `renderHeader` — `model.go:988`
- **Signature:** `func (m *Model) renderHeader() string`
- **Purpose:** 4-5 line header (status, DB stats, endpoints, reranker, cache warning).
- **Implementation:** Line 1: `[STATUS] statusMsg` colored per state + spinner + `(ragMode)` + collection tag. Quirk: the state switch (model.go:994-1013) has no cases for `stateConfirmQuit`/`stateConfirmSkill`, so statusText renders empty there. Line 2: Qdrant URL/status, collection, Limit, Expand (`off`/`±N`), Cap (`none`/int), Search mode, Cache, RAG mode, Ctx usage % (green <70 / yellow 70-84 / red >=85), Tokens (`~` prefix when estimated). Line 3: Embed + LLM models/URLs with (checking.../✓/✗) status. Line 4 (only when `cfg.RerankerURL != ""`): reranker model/URL, `enabled`/`bypassed`. Optional warning line when `cacheInfo != ""` and current filter != `cacheFilterAtWarmup`. Ends with a `─` border.
- **Calls / Called by:** `View`.

#### `renderFooter` — `model.go:1252`
- **Signature:** `func (m *Model) renderFooter() string`
- **Purpose:** Bottom bar: confirm dialogs or the text input.
- **Implementation:** `stateConfirmQuit` renders a red "Exit QQuestio? ..." bar; `stateConfirmSkill` renders a yellow bar with skill name + 30-char args preview (`[Y] Allow once [A] Allow always [N] Deny`); otherwise the textarea view, each line padded to `m.width-1`.

#### `getRenderedTurn` — `model.go:1310`
- **Signature:** `func (m *Model) getRenderedTurn(turn *ConversationTurn) string`
- **Purpose:** Cached glamour rendering of an assistant turn.
- **Implementation:** Non-assistant turns pass through raw. Cache keyed by `RenderedWidth` vs `m.viewport.Width` (min 20). Renders "Thinking..." header + italic `Reasoning` + divider, then `renderMarkdown(Content)`; writes back `RenderedContent`/`RenderedWidth`.
- **Calls / Called by:** `updateViewport`.

#### `getRenderedReferences` — `model.go:1336`
- **Signature:** `func (m *Model) getRenderedReferences(turn *ConversationTurn) string`
- **Purpose:** Same caching pattern for the references block (`formatReferences`), keyed by `refViewport.Width`. Note: currently defined but `updateRefViewport` re-renders via `formatReferences` directly each call.
- **Calls / Called by:** available to `updateRefViewport`.

#### `getActiveReferences` — `model.go:1352`
- **Signature:** `func (m *Model) getActiveReferences() []rag.QdrantPoint`
- **Purpose:** Which points the right panel should show.
- **Implementation:** `m.lastPoints` during pipeline states; otherwise newest history turn with non-empty `References`; else nil.
- **Calls / Called by:** `updateRefViewport`, `copyLastResponseCmd`.

#### `updateRefViewport` — `model.go:1365`
- **Signature:** `func (m *Model) updateRefViewport()`
- **Purpose:** Fill the references panel.
- **Implementation:** Placeholder "Retrieving references..." during embedding/searching/reranking; "No references loaded." when empty; otherwise `formatReferences(refs, m.refViewport.Width)` → `SetContent`.
- **Calls / Called by:** `updateViewport` (always synced).

#### `recalcLayout` — `model.go:1382`
- **Signature:** `func (m *Model) recalcLayout()`
- **Purpose:** Recompute all panel geometry.
- **Implementation:** `headerH` = 4, +1 if reranker configured, +1 if cache-filter-mismatch warning shown. `refWidth = m.width/3` clamped to [20, m.width/2]; `mainWidth = m.width - refWidth`. Input width = `mainWidth - 2` (min 10), input height = number of `hardWrappedLines` of value (or placeholder) so the footer grows; `footerH` = that count. Viewport heights = `m.height - headerH - footerH - 2` (min 1). Sets `viewport.Width = mainWidth-2`, `refViewport.Width = refWidth-2`.
- **Calls / Called by:** `updateViewport`, and directly in `Update` on every non-WindowSize message.

#### `updateViewport` — `model.go:1438`
- **Signature:** `func (m *Model) updateViewport()`
- **Purpose:** Rebuild the whole main conversation view.
- **Implementation:** Calls `recalcLayout`, then renders each history turn: user turns as bold `❯ You: ` label + hard-wrapped continuation lines; assistant turns via `getRenderedTurn` (or raw + `*Thinking:*` prefix when `showRawSource`) followed by a `─` divider; system turns as italic `ℹ ...` + divider. Then in-flight content: `lastQuery` as a user turn, stage checklists with spinner for embedding/searching/reranking, and streaming output (thinking block + `cleanLLMOutput` + markdown + spinner + `loadingMessage`, or the "Retrieved N documents" checklist before first token). Auto-scroll: `GotoBottom()` if previously at bottom or streaming with `len(output) < 20`. Finishes with `updateRefViewport()`.
- **Calls / Called by:** called from ~20 sites in `Update` and `loadSession`.

#### `cleanLLMOutput` — `model.go:1581`
- **Signature:** `func cleanLLMOutput(text string) string`
- **Purpose:** Strip LLM chat-template wrappers (`RT_TEXT|>`, `<|im_start|>`, `START TEXT`, `[END TEXT]`, etc. — case-insensitive prefix/suffix lists) repeatedly until stable.
- **Calls / Called by:** `Update` (streamChunkMsg done paths), `updateViewport` (live streaming text).

#### `copyLastResponseCmd` — `model.go:1633`
- **Signature:** `func (m *Model) copyLastResponseCmd() tea.Cmd`
- **Purpose:** Ctrl+Y handler.
- **Implementation:** If `focusRef`, copies `formatReferences(getActiveReferences(), 80)` via `clipboard.WriteAll`; else scans history backwards for the last assistant turn and copies its `Content`. Errors → `appErrMsg{stage:"slash"}`; success → `slashResultMsg{feedback}`.

#### `copyAllConversationCmd` — `model.go:1671`
- **Signature:** `func (m *Model) copyAllConversationCmd() tea.Cmd`
- **Purpose:** Copy the full transcript (with per-reference document blocks via `extractDocumentName`) to the clipboard. Same error/feedback pattern. Invoked from slash.go.

#### `saveLastResponseCmd` — `model.go:1722`
- **Signature:** `func (m *Model) saveLastResponseCmd(filename string) tea.Cmd`
- **Purpose:** Write last assistant response (Reasoning prepended as a fenced "Thinking" block) to a file with `os.WriteFile(..., 0644)`. Same message pattern.

#### `saveAllConversationCmd` — `model.go:1750`
- **Signature:** `func (m *Model) saveAllConversationCmd(filename string) tea.Cmd`
- **Purpose:** Markdown transcript export ("## ❯ You" / "## Assistant" / "### References" with fenced chunk text).

#### `formatReferences` — `model.go:1815`
- **Signature:** `func formatReferences(points []rag.QdrantPoint, width int) string`
- **Purpose:** Render retrieved points as document-grouped reference blocks for the right panel (and clipboard/file exports).
- **Implementation:** Groups points by `extractDocumentName(pt.Payload)` (same helper the prompt builder uses, so panel and prompt agree on titles); points without a doc name go to one fallback group rendered per-point. Within a group, sorts by chunk index read from payload keys `chunk_index|chunkIndex|position|seq|index|ord`. Header line `Document: <name>` + `Score: X · chunks lo-hi` (score = max `IsPrimary` point score, else first point's); per-chunk bullets `• chunk (chunk N, score S, db cosine O)` including `OriginalScore` when non-zero; text preview wrapped at `width-8`. Quirk: `chunks %d of %d` uses `hi` as the "total" (model.go:1971-1973), i.e. the range max, not a real total count. Dotted separators between groups.
- **Calls / Called by:** `updateRefViewport`, `getRenderedReferences`, `copyLastResponseCmd`, `copyAllConversationCmd`.

#### `extractDocumentName` — `model.go:2045`
- **Signature:** `func extractDocumentName(payload map[string]interface{}) string`
- **Purpose:** Find a human-readable document name in a Qdrant payload.
- **Implementation:** Recursive closure: (1) exact matches against `rag.DocumentIDKeys`; (2) any key containing file/name/title/source/path/url/doc, skipping keys containing "id" or "score"; (3) nested maps. Hex-hash values (`isHexHash`) are stashed in `hashCandidate` and only returned as a last resort.
- **Calls / Called by:** `formatReferences`, `copyAllConversationCmd`, `saveAllConversationCmd`, and the prompt builder (commands.go).

#### `truncateMiddle` — `model.go:2110`
- **Signature:** `func truncateMiddle(s string, maxLen int) string`
- **Purpose:** Middle-ellipsis truncation preserving prefix/suffix (e.g. file extension). Not called within model.go (used by other files).

#### `isHexHash` — `model.go:2125`
- **Signature:** `func isHexHash(s string) bool`
- **Purpose:** True for 32/40/64-char hex strings (MD5/SHA-1/SHA-256), tolerating `-`/`_`.
- **Calls / Called by:** `extractDocumentName`.

#### `formatNumber` — `model.go:2140`
- **Signature:** `func formatNumber(n int) string`
- **Purpose:** Thousands-separated integers ("1,234,567") for header/stats.

#### `renderMarkdown` — `model.go:2167`
- **Signature:** `func renderMarkdown(text string, width int) string`
- **Purpose:** Glamour ("dark" style) rendering with word wrap; silently falls back to raw text on error.
- **Calls / Called by:** `getRenderedTurn`, `updateViewport`.

#### `GetSessionsDir` — `model.go:2199`
- **Signature:** `func GetSessionsDir() (string, error)`
- **Purpose:** `~/config/qquestio/sessions`, created with `MkdirAll(0755)`.

#### `GetLastSessionID` — `model.go:2211`
- **Signature:** `func GetLastSessionID() (string, error)`
- **Purpose:** Lexicographically newest `*.json` session id (IDs sort by timestamp), or error "no sessions found".

#### `hasUserPrompt` — `model.go:2233`
- **Signature:** `func (m *Model) hasUserPrompt() bool`
- **Purpose:** True if any user turn exists — gate for saving.

#### `saveSession` — `model.go:2242`
- **Signature:** `func (m *Model) saveSession() error`
- **Purpose:** Persist session JSON.
- **Implementation:** No-op without a user prompt. Builds a `Session` with a trimmed history copy (Role/Content/References only — render caches and Reasoning are dropped). Writes `<sessionsDir>/<sessionID>.json` via `json.MarshalIndent` + `os.WriteFile(0644)`.
- **Calls / Called by:** main.go / commands.go (autosave and quit paths), not called inside model.go.

#### `loadSession` — `model.go:2286`
- **Signature:** `func (m *Model) loadSession(sessionID string) error`
- **Purpose:** Restore a session.
- **Implementation:** Reads + unmarshals the JSON; restores `sessionID`, collection, searchLimit (only if >0), searchCap, rerankerPool, searchExpand, searchMode, systemPrompt, ragMode, filterKey/Value, history, token counters. Rebuilds `inputHistory` from user turns and sets `historyIndex` to the end. Forces `updateViewport()` + `viewport.GotoBottom()`.
- **Calls / Called by:** slash.go (/resume-style command).

#### `selectRandomLoadingMessage` — `model.go:2343`
- **Signature:** `func (m *Model) selectRandomLoadingMessage()`
- **Purpose:** Pick one of 13 whimsical loading phrases using `time.Now().UnixNano() % len`. Called on each new query in `Update`.

### Extension Points

- **New state:** add the const to the `appState` iota block (model.go:29-38). Then touch every place that enumerates states: the `renderHeader` status switch (model.go:994-1013 — currently misses the two confirm states, so new states render an empty `[STATUS]`); the "pipeline active" checks that list `stateEmbedding/stateSearching/stateReranking/stateStreaming` (Esc-abort at model.go:502, spinner refresh at model.go:846, `getActiveReferences` at model.go:1353, `updateRefViewport` at model.go:1366, streaming checklist rendering in `updateViewport`); the viewport-forwarding gate at model.go:932; and `Enter` submit gating at model.go:593. Transitions are just assignments in `Update` plus a `case` for the message that advances the pipeline.
- **New keybinding:** add a `case tea.Key*:` to the main `switch msg.Type` in `Update` (model.go:500-646), after the confirmQuit/confirmSkill blocks (those swallow everything). Return the tea.Cmd (usually one of the `m.*Cmd` factories from commands.go) and set `statusMsg` + call `updateViewport()` for feedback. Update `renderFooter`/placeholder text if discoverability matters.
- **New pipeline stage:** (1) add a message type in messages.go and a `*Cmd` factory in commands.go following `rerankPointsCmd`; (2) insert a `case` in the "Pipeline chain" section of `Update` (model.go:648-741): set state, `statusMsg`, `updateViewport()`, append the next cmd — the chain is purely message-driven, no goroutines are launched from model.go itself (commands.go owns async HTTP via tea.Cmd closures); (3) add a state const + checklist rendering in `updateViewport` if the stage is visible.
- **New panel:** add a `viewport.Model` field plus a focus flag. Caveat: focus is a single `bool` (`focusRef`) in at least five places (Tab toggle model.go:542, mouse hit-test model.go:872-894, `recalcLayout` widths, `View` style selection, viewport forwarding model.go:938-944) — for more than two panels replace it with an enum/`focusedPanel` field first. Widths are computed in `recalcLayout` (refWidth = width/3 clamped); join in `View` via `lipgloss.JoinHorizontal`. Also keep the mouse hit-test and `headerH` math (model.go:873-876 and 1383-1393) in sync with any header height change, since `tea.MouseMsg` focus detection hardcodes header/footer heights.

---

## `internal/rag` — `qdrant.go` (Vector DB Client)

### File Overview (qdrant.go, 2116 lines)

`internal/rag/qdrant.go` is the Qdrant vector-DB client of the RAG pipeline. It talks to Qdrant over raw REST (no Go SDK): vector search via `/points/query` and `/points/search`, full-corpus streaming via `/points/scroll`, and collection metadata via `GET /collections/{name}`. Beyond server-side search it implements: a client-side brute-force cosine path with a parallel top-N heap (`topNByCosine`), an on-disk corpus cache integration (delegates to `cache.go`), context expansion (fetching ±N adjacent chunks of each hit's source document in parallel), and exact-phrase search using server-side text-match filters plus strict client-side substring verification. Endpoint URLs are built inline; `internal/rag/endpoints.go` only holds `AppendAPIPath(baseURL, suffix)`, a URL normalizer used by the embedding/LLM clients (appends `/v1/<suffix>` unless the base already ends in `/v1` or already contains the suffix; special-cases `embeddings` by checking for `embedding` so it doesn't double-append).

Types, consts, and globals (source order):

| Identifier | Line | Purpose |
|---|---|---|
| `QdrantMatch` | 19 | Match clause: `Value interface{} `json:"value,omitempty"``, `Text string `json:"text,omitempty"``. `Value` = exact keyword payload match; `Text` = full-text match on text-indexed fields. |
| `QdrantRange` | 27 | Numeric range clause; all `*float64` with `gte/lte/gt/lt` omitempty tags — only set fields serialize. |
| `QdrantFieldCondition` | 34 | `Key string `json:"key"`` + optional `Match`/`Range` (omitempty). |
| `QdrantFilter` | 40 | `Must`/`Should []QdrantFieldCondition` (omitempty). No `must_not`. |
| `var QdrantVectorName = ""` | 45 | **Mutable package global** — named vector to use. Auto-detected in `GetCollectionInfo`; empty = unnamed single-vector collection. Read by `SearchQdrant` (`using` field), `exactSearchWithPoints`, `QdrantPoint.UnmarshalJSON`. |
| `QdrantNamedVector` | 47 | `{name, vector}` wrapper for legacy `/points/search` `vector` param when named vectors are in use. |
| `QdrantSearchParams` | 53 | `Exact bool `json:"exact,omitempty"`` — server brute-force scoring instead of HNSW. |
| `QdrantQueryRequest` | 57 | Body for `/points/query`: `query []float32`, `using` (omitempty), `filter` (omitempty), `limit int`, `with_payload bool`, `params` (omitempty). |
| `QdrantSearchRequest` | 66 | Body for `/points/search`: `vector interface{}` (raw slice or `QdrantNamedVector`), `filter`, `limit`, `with_payload`, `params`. |
| `QdrantPoint` | 74 | `id interface{}`, `payload map[string]interface{}`, `score float32`, `vector []float32` (omitempty), `original_score` (omitempty), `is_primary` (omitempty — true on primary hits vs expansion neighbors). Custom `UnmarshalJSON` (83) also reads `"vectors": {name: [...]}` (named-vector shape) and copies `QdrantVectorName`'s entry (or first entry) into `Vector`. |
| `QdrantQueryResponse` | 106 | `result` kept as `json.RawMessage` plus decoded `Result []QdrantPoint` (`json:"-"`). Custom `UnmarshalJSON` (114) accepts flat array (Search API) and `{"points":[...]}` wrapper (Query API). |
| `QdrantCollectionInfo` | 380 | `GET /collections/{name}` response: `result.status`, `points_count`, `vectors_count`, `config.params.vectors` (raw JSON). |
| `ScrollRequest` | 395 | Body for `/points/scroll`: `limit`, `with_payload`, `with_vector`, `offset interface{}` (omitempty), `filter` (omitempty). |
| `ScrollResponse` | 404 | `result.points []QdrantPoint` + `result.next_page_offset interface{}` (nil = last page). |
| `ProgressFunc` | 438 | `func(processed, total int)` UI-progress callback; must be cheap/non-blocking. |
| `SearchQdrantFullCorpusOpts` | 441 | Options: `FilterKey`, `FilterValue`, `LivePointCount`, `ForceRefresh`, `TTL` (0 → 7-day default), `Progress`, `ExactMatch`. |
| `scoredPoint` | 935 | Internal `{point QdrantPoint, score float32}` for parallel top-N. |
| `SingleVectorConfig` | 999 | `{size, distance}` — detects flat (unnamed) vector config. |
| `var DocumentIDKeys` | 1086 | Ordered doc-identity payload keys: `file_name, filename, fileName, document_name, doc_name, document, doc, source_file, sourceFile, source, title, path, file_path, name, url`. Drives `*`/`any_file` → Should-across-all-keys filter semantics. |
| `var chunkIndexKeys` | 1094 | Ordered chunk-position keys: `chunk_index, chunkIndex, position, seq, index, ord`. |
| `docRange` | 1098 | `{docID string, lo, hi int}` — inclusive chunk_index window per document. |
| `ExpansionMap` | 1109 | `map[string]map[int]QdrantPoint` (docID → chunk_index → point); lets rerank re-apply expansion offline. |
| `ContextExpansionResult` | 1117 | `{Context string, PrimaryPoints, ExpandedPoints []QdrantPoint, ExpansionMap ExpansionMap}`. |
| `QdrantNestedFilter` | 1871 | Nested `should`/`must` filter used inside `QdrantComplexFilter.Must`. |
| `QdrantComplexFilter` | 1876 | `{Must []interface{}}` — heterogeneous must list mixing nested filters and field conditions. |
| `exactPhraseScrollRequest` | 1880 | Scroll body variant whose `Filter` is `interface{}` so any filter shape can be sent. |

Shared helpers from sibling files: `newHTTPClient(timeout)`, `HTTPTimeout`/`VerboseLogging` (httpclient.go); `LoadCorpusCache`/`SaveCorpusCache`/`CorpusCache.IsStale` (cache.go).

### Search Strategy Matrix

Dispatch lives in `commands.go` `(*Model).searchQdrantCmd` (~`commands.go:194`), driven by `m.searchMode` (`"auto"|"exact"|"local"`, set via `/search`, `slash.go:176-247`) and `m.searchCap` (`/cap <N>`; 0 = uncapped):

| Mode | Condition | Function | Endpoint & key request fields | Scoring |
|---|---|---|---|---|
| Exact phrase | `len(m.exactPhrases) > 0` | `SearchQdrantExactPhrases` | `POST /points/scroll` with `QdrantNestedFilter.Should` = `{field: {match: {text: phrase}}}` per text-indexed field, `with_vector: false`, batch 100 | Server text-match prefilter, then strict client-side substring check on all phrases |
| HNSW (capped) | `m.searchCap > 0` | `SearchQdrant` | `POST /points/query`: `limit = max(searchCap, searchDocs)` (candidate pool), `params` omitted unless mode=="exact" | Server HNSW; `params.exact=true` only when mode=="exact" (exact scoring over the capped pool) |
| Local (client brute force) | `m.searchMode == "local"` | `SearchQdrantFullCorpus` | `POST /points/scroll` batch 10000, `with_vector: true` (or on-disk cache, no network) | `topNByCosine` — parallel cosine over `runtime.NumCPU()` workers with per-worker min-heaps |
| Default/auto/exact, uncapped | fallback | `SearchWithContextExpansionDetailed` → `exactSearchWithPoints` | `POST /points/search`: `limit = docs`, `with_payload: true`, `params.exact = (mode=="exact" || mode=="auto")` | Server brute-force when exact (auto uses exact=true when uncapped, per comment at `commands.go:283`); then ±expand adjacent-chunk scroll |

Key subtlety documented on `SearchQdrant` (lines 157-179): `candidateLimit` (request `limit` = how much of the corpus Qdrant considers — determines recall) is deliberately decoupled from `docs` (client-side truncation = the user's `/limit`). `with_payload` is always true; query/search paths do not return vectors (only scroll paths do, and only `scrollAllPoints` sets `with_vector: true`).

### Functions (source order)

#### `(p *QdrantPoint) UnmarshalJSON` — `qdrant.go:83`
- **Signature:** `func (p *QdrantPoint) UnmarshalJSON(data []byte) error`
- **Purpose:** Decode a point, additionally handling the named-vectors response shape (`"vectors": {name: [...]}` instead of `"vector": [...]`).
- **Implementation:** Unmarshals into an alias of `QdrantPoint` plus `Vectors map[string][]float32`. If `p.Vector` is empty and `Vectors` is populated, copies `Vectors[QdrantVectorName]` (or first entry if the configured name is absent) into `p.Vector`. This is what makes client-side cosine scoring work on named-vector collections.
- **Calls / Called by:** Depends on global `QdrantVectorName`; invoked implicitly by `json.Unmarshal`/`Decode` wherever points are decoded.

#### `(q *QdrantQueryResponse) UnmarshalJSON` — `qdrant.go:114`
- **Signature:** `func (q *QdrantQueryResponse) UnmarshalJSON(data []byte) error`
- **Purpose:** Accept both result shapes: flat array (`/points/search`) and `{"points": [...]}` wrapper (`/points/query`).
- **Implementation:** Keeps `ResultRaw`; inspects first non-space byte — `[` → decode `[]QdrantPoint`; `{` → decode `wrapper.Points`. Empty raw → empty Result, no error. Otherwise error `unexpected json type for result`.
- **Calls / Called by:** Used by `SearchQdrant`, `exactSearchWithPoints`.

#### `SearchQdrant` — `qdrant.go:180`
- **Signature:** `func SearchQdrant(ctx context.Context, baseURL, apiKey, collection string, vector []float32, candidateLimit, docs int, filterKey, filterValue string, exact bool) (string, []QdrantPoint, error)`
- **Purpose:** The capped/HNSW search path — one server-side similarity query returning joined text + points.
- **Implementation:** `POST {baseURL}/collections/{collection}/points/query` with `QdrantQueryRequest{Query: vector, Using: QdrantVectorName, Filter, Limit: candidateLimit, WithPayload: true}`; adds `Params: {Exact: true}` only when `exact`. Filter construction is INLINED (same logic as `buildFilter` at 676, not factored out): filterKey matching any `DocumentIDKeys` (case-insensitive) or `*`/`any_file` → `Should` list across all `DocumentIDKeys`; else single `Must` with `Match.Value`. Auth via `api-key` header when non-empty. `newHTTPClient(HTTPTimeout)`; non-200 → `qdrantStatusError`. After decode: truncate `Result` to `docs`, mark all `IsPrimary = true`, extract text, join with `"\n---\n"`. No retries; cancellation via request context.
- **Calls / Called by:** Calls `pt.ExtractText`, `qdrantStatusError`, `newHTTPClient`. Called from `commands.go:231` (capped path).

#### `getPayloadKeys` — `qdrant.go:280`
- **Signature:** `func getPayloadKeys(payload map[string]interface{}) []string`
- **Purpose:** Sorted payload map keys (deterministic iteration). Nil map → nil.
- **Calls / Called by:** Helper; not on the hot paths (diagnostics/tests).

#### `(pt QdrantPoint) ExtractText` — `qdrant.go:295`
- **Signature:** `func (pt QdrantPoint) ExtractText() string`
- **Purpose:** Build maximal context text from a point's payload.
- **Implementation:** Two passes. (1) Priority keys `text, content, document, page_content, description, body, passage, chunk, context` in order. (2) Remaining keys sorted alphabetically, skipping metadata keys (`file, filename, file_name, file_path, source, title, url, path, id, score, page, author, date, created_at` and any key containing `id` or `score`) and strings ≤ 5 chars. Dedupes by trimmed content; joins with `"\n\n"`.
- **Calls / Called by:** Used by `SearchQdrant`, `extractTexts`, `applyExactMatch`, `BoostPhraseMatches`, expansion assembly, `SearchQdrantExactPhrases`, and `ExtractPrimaryText` fallback.

#### `(pt QdrantPoint) ExtractPrimaryText` — `qdrant.go:364`
- **Signature:** `func (pt QdrantPoint) ExtractPrimaryText() string`
- **Purpose:** Return only the first matched primary text key — clean single-passage input for reranker models.
- **Implementation:** Same priority keys; returns first non-empty string trimmed; falls back to full `ExtractText()` when no primary key matches.
- **Calls / Called by:** Fallback calls `ExtractText`; consumed by rerank code (rerank.go / commands.go rerank path).

#### `SearchQdrantFullCorpus` — `qdrant.go:451`
- **Signature:** `func SearchQdrantFullCorpus(ctx context.Context, baseURL, apiKey, collection string, vector []float32, docs int, filterKey, filterValue string, livePointCount int, forceRefresh bool, progress ProgressFunc, exactMatch string) (string, []QdrantPoint, bool, error)`
- **Purpose:** Backward-compatible positional-arg wrapper over the options-struct implementation.
- **Calls / Called by:** Delegates entirely to `SearchQdrantFullCorpusOptsImpl`. Called from `commands.go:250` (local mode).

#### `SearchQdrantFullCorpusOptsImpl` — `qdrant.go:473`
- **Signature:** `func SearchQdrantFullCorpusOptsImpl(ctx context.Context, baseURL, apiKey, collection string, vector []float32, docs int, opts *SearchQdrantFullCorpusOpts) (string, []QdrantPoint, bool, error)`
- **Purpose:** TRUE full-corpus search: on-disk cache first, else scroll everything and score client-side; returns `(context, topPoints, fromCache, err)`.
- **Implementation:** Four phases. (1) Cache hit: unless `ForceRefresh`, `LoadCorpusCache`; if `!cache.IsStale(len(vector), opts.LivePointCount, ttl)` and points exist → filter client-side (`applyFilter`, then `applyExactMatch` if `opts.ExactMatch != ""`), `topNByCosine`, mark `IsPrimary`, return `fromCache=true`. Default TTL `7*24h` when `opts.TTL == 0`. (2) Miss/stale/forced: `scrollAllPoints` (filterKey/Value pushed down server-side); empty → `("", nil, false, nil)`. (3) `SaveCorpusCache` best-effort (error ignored) and ONLY when no filter active (avoids caching a subset); dim from query vector or first point's vector. (4) `applyExactMatch` (server path relies on scroll-time pushdown for filterKey/Value, so only ExactMatch is re-applied), `topNByCosine` with progress, mark primaries, join with `"\n---\n"`.
- **Calls / Called by:** Calls `LoadCorpusCache`, `SaveCorpusCache`, `scrollAllPoints`, `applyFilter`, `applyExactMatch`, `topNByCosine`, `extractTexts`. Called by `SearchQdrantFullCorpus`.

#### `WarmupCorpusCache` — `qdrant.go:562`
- **Signature:** `func WarmupCorpusCache(ctx context.Context, baseURL, apiKey, collection string, progress ProgressFunc) (*CorpusCache, error)`
- **Purpose:** Pre-populate the canonical on-disk corpus cache without a query vector.
- **Implementation:** `scrollAllPoints` unfiltered; errors if collection empty or first point has empty vector (dim derivation). Saves with `filterAtWarmup := ""`; a save failure IS returned as an error. Reloads via `LoadCorpusCache`, returns cache metadata.
- **Calls / Called by:** Calls `scrollAllPoints`, `SaveCorpusCache`, `LoadCorpusCache`. Invoked by root-package cache warmup flows.

#### `scrollAllPoints` — `qdrant.go:595`
- **Signature:** `func scrollAllPoints(ctx context.Context, baseURL, apiKey, collection string, filterKey, filterValue string, progress ProgressFunc) ([]QdrantPoint, error)`
- **Purpose:** Stream every point (payload + vector) via the scroll cursor.
- **Implementation:** `POST /collections/{collection}/points/scroll` loop; `ScrollRequest{Limit: batchSize, WithPayload: true, WithVector: true, Offset, Filter}`. NOTE: doc comment says batching at 1000 but `const batchSize = 10000` (line 608). Checks `ctx.Done()` per iteration (Esc aborts). Own client, 60s timeout. Non-200 error includes 512-byte body excerpt. Accumulates points, `progress(len(all), 0)` per batch, terminates on `next_page_offset == nil`. Filter via `buildFilter`, pushed down server-side.
- **Calls / Called by:** Calls `buildFilter`, `newHTTPClient`. Called by `SearchQdrantFullCorpusOptsImpl`, `WarmupCorpusCache`.

#### `buildFilter` — `qdrant.go:676`
- **Signature:** `func buildFilter(filterKey, filterValue string) *QdrantFilter`
- **Purpose:** Canonical server-side filter construction shared by search/scroll paths.
- **Implementation:** filterKey case-insensitively in `DocumentIDKeys`, or `*`/`any_file` → `Should` condition per key with `Match.Value` (any-key OR). Else single `Must`. `SearchQdrant` duplicates this logic inline instead of calling it.
- **Calls / Called by:** Called by `scrollAllPoints`, `exactSearchWithPoints`, `SearchQdrantExactPhrases`; mirrored client-side by `applyFilter`.

#### `applyFilter` — `qdrant.go:703`
- **Signature:** `func applyFilter(points []QdrantPoint, filterKey, filterValue string) []QdrantPoint`
- **Purpose:** Client-side equivalent of `buildFilter` for cache-served points.
- **Implementation:** No-op when key or value empty. Doc-key/`*`/`any_file` → keep if ANY `DocumentIDKeys` payload value string-equals filterValue; else keep only on exact `Payload[filterKey]` string equality.
- **Calls / Called by:** Called from `SearchQdrantFullCorpusOptsImpl` cache-hit path.

#### `applyExactMatch` — `qdrant.go:743`
- **Signature:** `func applyExactMatch(points []QdrantPoint, phrase string) []QdrantPoint`
- **Purpose:** Keep only points whose extracted text contains `phrase` (case-insensitive).
- **Implementation:** Lowercases both sides; `strings.Contains` against `p.ExtractText()`.
- **Calls / Called by:** Called from `SearchQdrantFullCorpusOptsImpl` (cache and fresh paths) when `opts.ExactMatch != ""`.

#### `topNByCosine` — `qdrant.go:769`
- **Signature:** `func topNByCosine(query []float32, points []QdrantPoint, n int, progress func(processed, total int)) ([]QdrantPoint, error)`
- **Purpose:** Parallel client-side top-N by cosine similarity — the core of "local" mode.
- **Implementation:** Errors on empty/zero-norm query; empty points → nil. Precomputes `qNorm`. If `n <= 0 || n >= total` → `topNAllParallel`. Else: `workers = runtime.NumCPU()` clamped to `[1, total]`, contiguous chunk per worker, one goroutine per chunk (`sync.WaitGroup`), each keeping a local min-heap of size n (`heapifyMin` once full; replace-root + `siftDownMin` when a candidate beats the heap min). Progress throttled: only worker 0 reports, every 8192 points, via `atomic.Int64`. After `wg.Wait()`, merges per-worker heaps into one global min-heap of size n, then `sort.SliceStable` descending. Final `progress(total, total)`.
- **Calls / Called by:** Calls `scorePoint`, `heapifyMin`, `siftDownMin`, `topNAllParallel`. Called by `SearchQdrantFullCorpusOptsImpl` (both paths).

#### `topNAllParallel` — `qdrant.go:885`
- **Signature:** `func topNAllParallel(query []float32, qNorm float32, points []QdrantPoint, progress func(processed, total int)) ([]QdrantPoint, error)`
- **Purpose:** Score and return ALL points sorted descending (when n ≤ 0 or n ≥ total).
- **Implementation:** Same chunking/worker count; writes `scoredPoint` into a preallocated slice by original index (no heap). Progress once per worker (worker 0 only) plus final `(total, total)`. `sort.SliceStable` descending.
- **Calls / Called by:** Calls `scorePoint`. Called from `topNByCosine`.

#### `scorePoint` — `qdrant.go:940`
- **Signature:** `func scorePoint(query []float32, qNorm float32, p QdrantPoint, origIdx int) scoredPoint`
- **Purpose:** Cosine similarity of one point against the query.
- **Implementation:** Dimension mismatch → score -1; zero point norm → score 0; else `dot / (qNorm * sqrt(pNorm))` in one fused loop (SIMD-friendly). `origIdx` accepted but discarded (`_ = origIdx`).
- **Calls / Called by:** Called by `topNByCosine` and `topNAllParallel` workers.

#### `heapifyMin` — `qdrant.go:959`
- **Signature:** `func heapifyMin(h []scoredPoint)`
- **Purpose:** Build a min-heap in place (root = worst score of the current top-N).
- **Implementation:** Bottom-up sift-down from `len(h)/2 - 1`.
- **Calls / Called by:** Calls `siftDownMin`; used by `topNByCosine` (worker-local and merge phases).

#### `siftDownMin` — `qdrant.go:965`
- **Signature:** `func siftDownMin(h []scoredPoint, i int)`
- **Purpose:** Restore min-heap property from index `i` downward.
- **Implementation:** Iterative compare/swap with children `2i+1`/`2i+2`.
- **Calls / Called by:** Called by `heapifyMin` and directly by `topNByCosine`.

#### `extractTexts` — `qdrant.go:985`
- **Signature:** `func extractTexts(points []QdrantPoint) []string`
- **Purpose:** Map points to non-empty extracted texts (via `pt.ExtractText()`).
- **Calls / Called by:** Called by `SearchQdrantFullCorpusOptsImpl`, `SearchQdrantExactPhrases`.

#### `containsName` — `qdrant.go:1004`
- **Signature:** `func containsName(list []string, name string) bool`
- **Purpose:** Linear membership test.
- **Calls / Called by:** Used by `GetCollectionInfo` for named-vector validation.

#### `GetCollectionInfo` — `qdrant.go:1014`
- **Signature:** `func GetCollectionInfo(ctx context.Context, baseURL, apiKey, collection string) (int, int, string, error)`
- **Purpose:** Fetch points_count, vectors_count, status; auto-detect the named vector.
- **Implementation:** `GET /collections/{collection}`, 30s client, non-200 → `qdrantStatusError`. Parses `config.params.vectors` (raw JSON): decodes as `SingleVectorConfig` with `Size > 0` → unnamed collection, resets `QdrantVectorName = ""`. Else parses as a name map; if current `QdrantVectorName` is empty or not among names, picks first of `text/content/document/vector` present, else alphabetically-first name, assigns the GLOBAL, logs under `VerboseLogging`. Later searches rely on this side effect.
- **Calls / Called by:** Called by `fetchQdrantInfoCmd` (`commands.go:~296`); its `pointsCount` return feeds `SearchQdrantFullCorpus`'s staleness check (`m.qdrantPoints`).

#### `extractDocID` — `qdrant.go:1125`
- **Signature:** `func extractDocID(p QdrantPoint) string`
- **Purpose:** First non-empty string payload value across `DocumentIDKeys`, else "".
- **Calls / Called by:** Used by `SearchWithContextExpansionDetailed`, `ApplyExpansionToPrimaries`.

#### `IsFullyQuoted` — `qdrant.go:1141`
- **Signature:** `func IsFullyQuoted(s string) bool`
- **Purpose:** True when the whole trimmed string is one quoted phrase.
- **Implementation:** First/last rune pairs: `"`, `'`, `«»`, `„`+`"`/`”`, `“`+`"`/`”`, `‹›`, `《》`.
- **Calls / Called by:** Exported; used by root-package query preprocessing (exact-phrase detection).

#### `BoostPhraseMatches` — `qdrant.go:1173`
- **Signature:** `func BoostPhraseMatches(points []QdrantPoint, phrases []string) []QdrantPoint`
- **Purpose:** Stable-partition so chunks containing ALL phrases (case-insensitive) come first, preserving relative order.
- **Implementation:** Two slices (matching/non-matching) concatenated; text via `pt.ExtractText()`. No score rewrite.
- **Calls / Called by:** Called from `commands.go:289` on `res.PrimaryPoints` when quoted phrases exist but full exact search wasn't forced.

#### `extractChunkIndex` — `qdrant.go:1204`
- **Signature:** `func extractChunkIndex(p QdrantPoint) int`
- **Purpose:** Chunk's positional index from `chunkIndexKeys`, else -1.
- **Implementation:** First present key wins; accepts `float64` (JSON numbers), `int`, `int64`, strings via `fmt.Sscanf("%d")`.
- **Calls / Called by:** Used throughout the expansion pipeline.

#### `SearchWithContextExpansionDetailed` — `qdrant.go:1238`
- **Signature:** `func SearchWithContextExpansionDetailed(ctx context.Context, baseURL, apiKey, collection string, vector []float32, limit, expand int, filterKey, filterValue string, exact bool) (ContextExpansionResult, error)`
- **Purpose:** Default uncapped path: exact top-N search, then ±expand neighboring chunks per document, reassembled into coherent windows.
- **Implementation:** Validates `limit > 0`; clamps negative expand to 0. **Phase 1:** `exactSearchWithPoints` for top-`limit` primaries; seeds `ExpansionMap` (points without docID or idx < 0 skipped from map but kept in primaries). expand==0 or no primaries → `ExpandedPoints = primaryPoints`, done. **Phase 2:** group primaries by `extractDocID` into `docRange{lo: idx-expand, hi: idx+expand}` (union per doc); no usable metadata → return primaries unchanged. **Phase 3:** `scrollAdjacentChunks` (parallel scrolls per range). **Phase 4:** fold scrolled points into `ExpansionMap`; walk primaries in original order, emitting per document (once, via `seenSlices`) the full window `[r.lo, r.hi]` from the map sorted ascending by chunk index, text joined `"\n---\n"`; primaries lacking docID/chunk_index emitted standalone; dedupe by `idKey(c.ID)`. Per-chunk `--- Chunk N | Document: name ---` rewrapping deferred to `buildPromptMessages` (root package).
- **Calls / Called by:** Calls `exactSearchWithPoints`, `scrollAdjacentChunks`, `extractDocID`, `extractChunkIndex`, `idKey`. Called from `commands.go:281`.

#### `ApplyExpansionToPrimaries` — `qdrant.go:1437`
- **Signature:** `func ApplyExpansionToPrimaries(primaries []QdrantPoint, em ExpansionMap, expand int) (string, []QdrantPoint)`
- **Purpose:** Re-apply ±expand expansion to a DIFFERENT primary set (e.g. post-rerank top-K) using only the cached ExpansionMap — zero network calls.
- **Implementation:** Mirrors Phase-4 assembly: recomputes per-doc windows from the new primaries; when pulling chunk `c` from the map, if its ID is in `mutatedMap` (the new primaries, possibly carrying rerank-mutated data such as `OriginalScore`) the mutated copy is used, else `c.IsPrimary = false`. Standalone text for primaries without docID/chunk_index; dedupe via `idKey`; join `"\n---\n"`.
- **Calls / Called by:** Calls `extractDocID`, `extractChunkIndex`, `idKey`, `ExtractText`. Called by rerank-then-expand post-processing (commands.go / rerank.go flow).

#### `exactSearchWithPoints` — `qdrant.go:1591`
- **Signature:** `func exactSearchWithPoints(ctx context.Context, baseURL, apiKey, collection string, vector []float32, docs int, filterKey, filterValue string, exact bool) (string, []QdrantPoint, error)`
- **Purpose:** Primary top-N fetch for the expansion pipeline (legacy Search API variant of `SearchQdrant` that also returns points).
- **Implementation:** `POST /collections/{collection}/points/search` with `QdrantSearchRequest{Vector: vecParam, Filter, Limit: docs, WithPayload: true, Params}`. `vecParam` is the raw vector, wrapped in `QdrantNamedVector{Name: QdrantVectorName, ...}` when a named vector is configured. `params.exact=true` only when `exact`. Filter via `buildFilter`. `HTTPTimeout` client; non-200 → `qdrantStatusError`; response decoded as `QdrantQueryResponse` (flat-array shape). Marks all `IsPrimary`, extracts and joins texts.
- **Calls / Called by:** Calls `buildFilter`, `qdrantStatusError`. Called by `SearchWithContextExpansionDetailed` Phase 1.

#### `scrollAdjacentChunks` — `qdrant.go:1675`
- **Signature:** `func scrollAdjacentChunks(ctx context.Context, baseURL, apiKey, collection string, ranges []docRange) []QdrantPoint`
- **Purpose:** Fan out one scroll per (docID, chunk range) to fetch expansion neighbors.
- **Implementation:** Worker pool of `min(runtime.NumCPU(), len(ranges))` goroutines consuming a buffered `jobs` channel of `docRange`s; each job calls `scrollOneRange`. Results placed by linear search matching `docID+lo+hi` against the original slice (order preserved), then concatenated. Vectors NOT fetched (`with_vector: false`) — neighbors are text-only.
- **Calls / Called by:** Calls `scrollOneRange`. Called by `SearchWithContextExpansionDetailed` Phase 3.

#### `scrollOneRange` — `qdrant.go:1736`
- **Signature:** `func scrollOneRange(ctx context.Context, url, apiKey string, r docRange) []QdrantPoint`
- **Purpose:** Cursor-following scroll for one document's `[lo, hi]` chunk window.
- **Implementation:** `POST {url}` (points/scroll) with `ScrollRequest{Limit: 1024, WithPayload: true, WithVector: false, Offset, Filter: Must[{file_name: match docID}, {chunk_index: range gte lo, lte hi}]}` — the doc filter HARDCODES payload key `file_name` (first entry of `DocumentIDKeys`), so expansion only matches collections using that key. Own 60s client. Error-tolerant: any marshal/HTTP/decode failure or non-200 (logged with 512-byte body) returns the partial `all` accumulated so far — errors never propagate. Checks `ctx.Done()` per iteration; stops on `next_page_offset == nil`.
- **Calls / Called by:** Called by `scrollAdjacentChunks` workers.

#### `ExtractQuotedPhrases` — `qdrant.go:1815`
- **Signature:** `func ExtractQuotedPhrases(raw string) []string`
- **Purpose:** Pull every quoted substring from the raw user query.
- **Implementation:** Single-pass rune scan; openers `"`, `'`, `“`, `”`, `„`, `«` with matching closers (`„` accepts `”` or `“`); unmatched openers skipped. Returns inner text of each pair.
- **Calls / Called by:** Exported; populates `m.exactPhrases` in the root package, selecting the `SearchQdrantExactPhrases` path and feeding `BoostPhraseMatches`.

#### `GetTextIndexedFields` — `qdrant.go:1889`
- **Signature:** `func GetTextIndexedFields(ctx context.Context, baseURL, apiKey, collection string) ([]string, error)`
- **Purpose:** List payload fields carrying a Qdrant full-text index.
- **Implementation:** `GET /collections/{collection}` (10s client), generic `map[string]interface{}` decode, reads `result.payload_schema`, keeps fields whose `type` or `data_type` == `"text"`. Missing `payload_schema` → `(nil, nil)`.
- **Calls / Called by:** Called by `SearchQdrantExactPhrases`; same endpoint as `GetCollectionInfo` but generic decoding.

#### `SearchQdrantExactPhrases` — `qdrant.go:1945`
- **Signature:** `func SearchQdrantExactPhrases(ctx context.Context, baseURL, apiKey, collection string, phrases []string, docs int, filterKey, filterValue string) (string, []QdrantPoint, error)`
- **Purpose:** Quoted-phrase search: server-side text-match prefilter on the first phrase, then strict client-side verification of ALL phrases.
- **Implementation:** (1) `GetTextIndexedFields`; error/empty → fallback `["text","content","document","page_content"]`. (2) `Should` conditions `{field: Match{Text: phrases[0]}}` per text field. (3) With a metadata filter active: `QdrantComplexFilter{Must: [QdrantNestedFilter{Should: textConds}, ...metaFilter.Must, QdrantNestedFilter{Should: metaFilter.Should} (if any)]}`; without: plain `QdrantNestedFilter{Should: ...}`. (4) `scrollWithFilter` (batch 100, `with_vector: false`). (5) Client-side: keep points whose lowercased `ExtractText()` contains every lowercased phrase. (6) Truncate to `docs`, mark `IsPrimary`, join texts.
- **Calls / Called by:** Calls `GetTextIndexedFields`, `buildFilter`, `scrollWithFilter`, `extractTexts`. Called from `commands.go:206`.

#### `scrollWithFilter` — `qdrant.go:2025`
- **Signature:** `func scrollWithFilter(ctx context.Context, baseURL, apiKey, collection string, filter interface{}) ([]QdrantPoint, error)`
- **Purpose:** Generic filtered scroll using `exactPhraseScrollRequest` (filter is `interface{}` to accept any filter shape).
- **Implementation:** `POST /points/scroll`, `const batchSize = 100`, `with_payload: true`, `with_vector: false`, 30s client, ctx check per page, non-200 error includes 512-byte body excerpt, loop until `next_page_offset == nil`.
- **Calls / Called by:** Called by `SearchQdrantExactPhrases` only.

#### `idKey` — `qdrant.go:2093`
- **Signature:** `func idKey(id interface{}) interface{}`
- **Purpose:** Normalize a point ID for map dedup across JSON numeric types.
- **Implementation:** `float64/float32/int/int64` → `uint64`; `uint64`/`string` pass through; else `fmt.Sprintf("%v")`.
- **Calls / Called by:** Used by both expansion assembly functions.

#### `qdrantStatusError` — `qdrant.go:2112`
- **Signature:** `func qdrantStatusError(resp *http.Response) error`
- **Purpose:** Convert a non-200 response into an error with status + body excerpt (up to 512 bytes).
- **Implementation:** Format `HTTP status error: %d %s (body: %s)`. Does NOT close the body (callers defer/close it).
- **Calls / Called by:** Called by `SearchQdrant`, `exactSearchWithPoints`, `GetCollectionInfo`, `GetTextIndexedFields`.

### Extension points

- **Adding a new Qdrant API call:** follow the established pattern — build the URL with `fmt.Sprintf("%s/collections/%s/<path>", strings.TrimSuffix(baseURL, "/"), collection)`; define request/response structs with JSON tags near the top of the file; `http.NewRequestWithContext(ctx, http.MethodPost/Get, url, body)`; set `Content-Type: application/json` and `api-key` (only when non-empty); use `newHTTPClient(<timeout>)` (existing calls use 10s/30s/60s/`HTTPTimeout` by expected latency); map non-200 via `qdrantStatusError(resp)`; decode with `json.NewDecoder(resp.Body)`. Add cursor pagination (`offset`/`next_page_offset` loop with `select ctx.Done()` per page) for anything exceeding one page — `scrollWithFilter` is the minimal template. `QdrantPoint.UnmarshalJSON` and `QdrantQueryResponse.UnmarshalJSON` already handle named-vector and both result shapes.
- **Adding a new search mode:** modes are strings on `Model.searchMode` in the root package (`"auto"|"exact"|"local"`, set in `slash.go:176-247`). Add the mode string there, then add a branch in `(*Model).searchQdrantCmd` (`commands.go:~194`) before the default fallthrough, calling a new exported `rag` function returning the `searchResultMsg` shape (`context`, `points`, `primaryPoints`, `expansionMap`, `expand`). If the mode should compose with rerank + re-expansion, return primaries plus an `ExpansionMap` and let `ApplyExpansionToPrimaries` handle post-rerank expansion rather than expanding eagerly.
- **Adding a new filter type:** server-side and client-side must stay in sync. Extend `QdrantFieldCondition` (e.g. `GeoBoundingBox`, `DatetimeRange`) and update `buildFilter` (server pushdown) AND `applyFilter` (cache-hit filtering) with matching semantics; note `SearchQdrant` re-implements the filter logic inline (lines 183-216) rather than calling `buildFilter`, so change both or refactor it to call `buildFilter`. For filters combining with the exact-phrase path, `QdrantComplexFilter.Must` is `[]interface{}` and can already hold nested filters and field conditions side by side. `must_not` is not represented anywhere — adding it requires touching both filter structs and both apply paths.

Cross-file notes for the merge: `newHTTPClient`/`HTTPTimeout`/`VerboseLogging` live in httpclient.go; `LoadCorpusCache`/`SaveCorpusCache`/`CorpusCache.IsStale` (staleness = dimension mismatch, non-empty `FilterAtWarmup`, TTL expiry, or `PointCount != liveCount`) live in cache.go; caller-side dispatch (`searchQdrantCmd`, `computeSearchDocs`, `computeRerankerPool`, `fetchQdrantInfoCmd`, `preloadCacheInfoCmd`, warmup invocation) lives in commands.go and slash.go in the root package.

---

## `internal/rag` — Support Clients (`httpclient.go`, `embeddings.go`, `litellm.go`, `rerank.go`, `cache.go`)

### File Overview

| File | LOC | Role |
|---|---|---|
| `internal/rag/httpclient.go` | 104 | Shared `http.Transport`/`http.Client` factory: global timeout var, verbose flag, proxy bypass for local/private hosts |
| `internal/rag/embeddings.go` | 74 | OpenAI-style embeddings client (`POST .../embeddings`) + connection check |
| `internal/rag/litellm.go` | 281 | OpenAI-compatible chat client: SSE streaming reader (`SSEReader`), non-streaming `ChatComplete`, connection check |
| `internal/rag/rerank.go` | 237 | Provider-agnostic reranker client (`POST .../rerank`) with schema fallback and 3-format response parsing |
| `internal/rag/cache.go` | 412 | On-disk corpus cache for a Qdrant collection: binary v3 format, load/save/delete/status, staleness policy |

All five files are package `rag` and form the support layer beneath `qdrant.go`. Every HTTP call goes through `newHTTPClient` in `httpclient.go` and every URL is built with `AppendAPIPath` (defined in `internal/rag/endpoints.go:6`, not covered here) which normalizes the base URL and appends a suffix like `"embeddings"` or `"chat/completions"`.

**`httpclient.go`** — Centralizes HTTP plumbing for the package. Two mutable package globals (`HTTPTimeout`, `VerboseLogging`) act as the host app's tuning knobs. A single lazily-cloned `http.Transport` is shared by all clients so connection pooling works across embeddings/LLM/rerank/Qdrant calls; its `ResponseHeaderTimeout` is explicitly zeroed so the client timeout governs. A proxy wrapper bypasses corporate proxies for loopback, RFC1918, ULA, and dot-less/`.local` hostnames so local Qdrant/llama.cpp servers never hang.

**`embeddings.go`** — Thin client for llama.cpp/OpenAI-compatible embedding servers. Despite the doc comment saying "/embedding endpoint", it posts to the plural `embeddings` path. Single text per call; returns the first embedding vector.

**`litellm.go`** — The generation client. `StartLiteLLMStream` opens a `text/event-stream` POST to `chat/completions` with `stream: true` and `stream_options.include_usage`; the returned `SSEReader` yields `(content, reasoning)` pairs chunk-by-chunk via `Next()`, capturing token usage from the final chunk. `ChatComplete` is the non-streaming variant used for short utility calls (query rewriting). `CheckLLMConnection` sends a literal `"ping"` prompt.

**`rerank.go`** — One function, `Rerank`, that works across Cohere/Jina/llama.cpp/LiteLLM/HuggingFace-TEI servers by (a) retrying with `texts` instead of `documents` on 400/422, and (b) trying three response shapes: raw float array, raw object array, and `{"results": [...]}` envelope. Index recovery falls back to matching returned document text against the input texts.

**`cache.go`** — `CorpusCache` persists an entire Qdrant collection (payloads as one JSON blob + vectors as raw little-endian float32) so the first full-corpus search pays the scroll cost once. Files live in `$QQUESTIO_CACHE_DIR` or `~/.cache/qquestio`, named `<safe-collection>-<sha1(url)[:10]>.qcache`, written atomically. `IsStale` encodes the invalidation policy: dimension mismatch, filtered warmup, TTL, or live point-count drift.

---

### `httpclient.go`

#### Types & globals

- **`HTTPTimeout`** (`httpclient.go:14`) — `var HTTPTimeout = 60 * time.Second`. Package-global timeout applied to all non-streaming requests. Mutated by the host app: `slash.go:90` and `model.go:338` set it from `cfg.HTTPTimeoutSeconds` (config field at `config.go:29`, defaulted to 60 at `config.go:177-178`).
- **`VerboseLogging`** (`httpclient.go:18`) — `var VerboseLogging bool`. Off by default (logging would corrupt the TUI). Set to true in `main.go:64` behind the verbose flag; consumed at `litellm.go:88` and `qdrant.go:1071`.
- **`sharedTransport`** / **`initTransportOnce`** (`httpclient.go:21-22`) — unexported; the one `*http.Transport` (guarded by `sync.Once`) behind every client in the package.

#### `getSharedTransport` — `httpclient.go:25`
- **Signature:** `func getSharedTransport() *http.Transport`
- **Purpose:** Lazily build the process-wide transport, cloned from `http.DefaultTransport` (falling back to `&http.Transport{}` if the default isn't a `*Transport`).
- **Implementation:** `sync.Once` guard. Sets `ResponseHeaderTimeout = 0` with a comment explaining a hardcoded 30s header timeout overrides the client timeout and prematurely kills large-database queries. Wraps the transport's existing `Proxy` function (or `http.ProxyFromEnvironment`): if `isLocalOrPrivateHost(hostname)` returns true, return `nil, nil` (no proxy); otherwise delegate to the original.
- **Calls / Called by:** Called only by `newHTTPClient`. Transitively behind every HTTP call in the package.

#### `newHTTPClient` — `httpclient.go:56`
- **Signature:** `func newHTTPClient(timeout time.Duration) *http.Client`
- **Purpose:** The single constructor for all HTTP clients in the package; couples a per-call timeout with the shared pooled transport.
- **Implementation:** Returns `&http.Client{Timeout: timeout, Transport: getSharedTransport()}`. Convention across the package: `newHTTPClient(HTTPTimeout)` for request/response calls; `newHTTPClient(0)` (no timeout, lifecycle governed by `ctx`) for SSE streaming (`litellm.go:82`); `newHTTPClient(30 * time.Second)` for `CheckLLMConnection` (`litellm.go:268`).
- **Calls / Called by:** `embeddings.go:47`, `litellm.go:82/140/268`, `rerank.go:65`, `qdrant.go:243/1642`.

#### `isLocalOrPrivateHost` — `httpclient.go:63`
- **Signature:** `func isLocalOrPrivateHost(host string) bool`
- **Purpose:** Decide whether a hostname should bypass configured proxies.
- **Implementation:** True for `localhost`, `::1`, `127.*`. If `net.ParseIP` succeeds: loopback, `10.0.0.0/8` (`ip4[0]==10`), `172.16.0.0/12` (`ip4[0]==172 && ip4[1] in 16..31`), `192.168.0.0/16`, and IPv6 ULA `fc00::/7` (`(ip16[0] & 0xfe) == 0xfc`). Non-IP hosts: true if no dot (bare hostname) or suffix `.local`. Public hostnames return false.
- **Calls / Called by:** Called by the proxy wrapper in `getSharedTransport` only.

---

### `embeddings.go`

#### Types

- **`EmbeddingRequest`** (`embeddings.go:11`) — wire struct: `Model string \`json:"model"\``, `Input []string \`json:"input"\``.
- **`EmbeddingResponse`** (`embeddings.go:16`) — wire struct: `Data []struct{ Embedding []float32 \`json:"embedding"\` } \`json:"data"\``. Extra fields (e.g. OpenAI `usage`) are ignored by the decoder.

#### `GetEmbedding` — `embeddings.go:25`
- **Signature:** `func GetEmbedding(ctx context.Context, baseURL, apiKey, model, text string) ([]float32, error)`
- **Purpose:** Embed a single text via `POST {baseURL}/embeddings` (path appended by `AppendAPIPath`; the doc comment's "/embedding" is stale — the code uses the plural OpenAI path).
- **Implementation:** Marshals `EmbeddingRequest{Model, Input: []string{text}}`. Context-aware request; `Content-Type: application/json`; `Authorization: Bearer <apiKey>` only when key non-empty. Uses `newHTTPClient(HTTPTimeout)`. Non-200 → error with status code and text. Decodes JSON; empty `data` list → `"received empty embedding data list"`. Returns `Data[0].Embedding`. No retries; cancellation/errors propagate via `ctx` and `client.Do` wrapped with `%w`.
- **Calls / Called by:** Uses `AppendAPIPath`, `newHTTPClient`. Called by `commands.go:20` (retrieval query embedding) and `CheckEmbeddingConnection`.

#### `CheckEmbeddingConnection` — `embeddings.go:71`
- **Signature:** `func CheckEmbeddingConnection(ctx context.Context, baseURL, apiKey, model string) error`
- **Purpose:** Health check: embed the dummy string `"ping"` and return only the error.
- **Implementation:** Single-line delegate to `GetEmbedding`, discarding the vector.
- **Calls / Called by:** Called by `commands.go:646` (connection-check flow alongside `CheckLLMConnection` at `commands.go:638`).

---

### `litellm.go`

#### Types

- **`SSEReader`** (`litellm.go:17`) — wraps a stream: unexported fields `body io.ReadCloser`, `scanner *bufio.Scanner`, `usage TokenUsage`. Access via `Next`, `Usage`, `Close`.
- **`TokenUsage`** (`litellm.go:24`) — wire struct: `PromptTokens/CompletionTokens/TotalTokens int` with JSON tags `prompt_tokens` / `completion_tokens` / `total_tokens`.
- **`ChatMessage`** (`litellm.go:34`) — wire struct: `Role string \`json:"role"\`` (`"system" | "user" | "assistant"` per comment), `Content string \`json:"content"\``. Plain text content only — no tool/multimodal parts.
- **`LiteLLMRequest`** (`litellm.go:39`) — wire struct: `Model string \`json:"model"\``, `Messages []ChatMessage \`json:"messages"\``, `Stream bool \`json:"stream"\``, `StreamOptions map[string]interface{} \`json:"stream_options,omitempty"\``, `MaxTokens int \`json:"max_tokens,omitempty"\``, `Options map[string]interface{} \`json:"options,omitempty"\``. Note `Options` is not an OpenAI field — it carries Ollama-style extras like `num_ctx` through OpenAI-compatible gateways.

#### `Usage` — `litellm.go:31`
- **Signature:** `func (r *SSEReader) Usage() TokenUsage`
- **Purpose:** Return token counts captured from the last usage-bearing SSE chunk (servers send usage in a final chunk when requested via `stream_options.include_usage`).
- **Calls / Called by:** Getter over `r.usage`, mutated in `Next`. Consumed by stream drivers in the root package (via the reader returned at `commands.go:344`).

#### `StartLiteLLMStream` — `litellm.go:48`
- **Signature:** `func StartLiteLLMStream(ctx context.Context, baseURL, apiKey, model string, maxTokens, contextLimit int, messages []ChatMessage) (*SSEReader, error)`
- **Purpose:** Open a streaming chat completion against an OpenAI-compatible endpoint and return a ready-to-read `SSEReader`.
- **Implementation:** URL = `AppendAPIPath(baseURL, "chat/completions")`. Body: `Stream: true`, `StreamOptions: {"include_usage": true}`; `MaxTokens` set only when `> 0`; when `contextLimit > 0` sets `Options: {"num_ctx": contextLimit}`. Headers: `Content-Type`, `Accept: text/event-stream`, `Authorization: Bearer` when key present. Client timeout is **0** — the stream lifetime is bounded solely by `ctx`. On verbose (`VerboseLogging`, `litellm.go:88`) logs URL/status/content-type via `log.Printf("[LLM] ...")`. Non-200: drains body, closes it, returns error including the response body text. On success wraps `resp.Body` in a `bufio.Scanner` with an initial 64 KB buffer growing to a 4 MB max — the comment notes this avoids `bufio.ErrTooLong` on large final usage chunks.
- **Calls / Called by:** Uses `AppendAPIPath`, `newHTTPClient`. Called by `commands.go:344` with `cfg.OpenAIURL/OpenAIAPIKey/OpenAIModel/OpenAIMaxTokens/ContextLimit`.

#### `ChatComplete` — `litellm.go:112`
- **Signature:** `func ChatComplete(ctx context.Context, baseURL, apiKey, model string, messages []ChatMessage, maxTokens int) (string, error)`
- **Purpose:** One-shot non-streaming chat completion returning the trimmed assistant content.
- **Implementation:** Same endpoint, `Stream: false`, `MaxTokens` only when `> 0`. Sends **both** `Authorization: Bearer` and `api-key` headers when a key is present (the latter is the Azure convention). Uses `newHTTPClient(HTTPTimeout)`. Non-200 → error with status and body. Decodes an anonymous `choices[0].message.content` struct; errors on zero choices (`"no choices in chat complete response"`) and on empty trimmed content. No retries.
- **Calls / Called by:** Uses `AppendAPIPath`, `newHTTPClient`. Called by `commands.go:75` for query rewriting with `maxTokens = 128`.

#### `Next` — `litellm.go:179`
- **Signature:** `func (r *SSEReader) Next() (content string, reasoning string, done bool, err error)`
- **Purpose:** Read the next meaningful SSE chunk and split it into visible content and reasoning-model thinking text.
- **Implementation:** Loop over `scanner.Scan()`: trim each line; skip blank lines; skip any line not starting with `data:` (this drops `event:` metadata and comment lines). Strip the `data:` prefix and trim. `[DONE]` → `("", "", true, nil)`. Otherwise unmarshal into an anonymous chunk struct (`litellm.go:196-205`): `choices[0].delta.content`, `delta.reasoning`, `delta.reasoning_content` (two vendor spellings of thinking text; `reasoning` wins, `reasoning_content` is the fallback when `reasoning == ""`), plus optional `usage *TokenUsage` — if present, stored into `r.usage`. Chunks where both content and reasoning are empty are skipped (loop continues) — this silently discards role-only deltas, finish chunks, and keep-alives. Any JSON parse failure of a `data:` line returns an error immediately. **Tool calls are not surfaced at all**: `delta.tool_calls` is an unknown field and dropped by `json.Unmarshal`; any `CALL:`-style syntax would have to be detected downstream in the root package's text handling. After scanner exhaustion: scanner error → return it; clean EOF without `[DONE]` → treated as graceful completion (`("", "", true, nil)`).
- **Calls / Called by:** Pumps `r.scanner`; writes `r.usage`. Called in a loop by the streaming consumer started at `commands.go:344`.

#### `Close` — `litellm.go:234`
- **Signature:** `func (r *SSEReader) Close() error`
- **Purpose:** Close the underlying response body (which also terminates the HTTP stream).
- **Implementation:** Nil-safe delegate to `r.body.Close()`.
- **Calls / Called by:** Must be called by whoever owns the reader from `StartLiteLLMStream`.

#### `CheckLLMConnection` — `litellm.go:242`
- **Signature:** `func CheckLLMConnection(ctx context.Context, baseURL, apiKey, model string) error`
- **Purpose:** Verify endpoint reachability, model validity, and credentials with a minimal request.
- **Implementation:** Builds the request from a raw `map[string]interface{}` (not `LiteLLMRequest`) containing one user message `"ping"`. Bearer auth only (no `api-key` header here). Uses a dedicated `newHTTPClient(30 * time.Second)` — comment: 30s accommodates local GPU cold starts. Non-200 → error including body. Success (200) returns nil; response body is not otherwise inspected.
- **Calls / Called by:** Uses `AppendAPIPath`, `newHTTPClient`. Called by `commands.go:638`.

---

### `rerank.go`

#### Types

- **`RerankItem`** (`rerank.go:13`) — result record: `Index int`, `Score float64`. No JSON tags — this is the internal normalized form, never serialized.
- **`RerankRequest`** (`rerank.go:19`) — wire struct with dual vocabulary: `Model string \`json:"model,omitempty"\``, `Query string \`json:"query"\``, `Documents []string \`json:"documents,omitempty"\``, `Texts []string \`json:"texts,omitempty"\``. Exactly one of `documents`/`texts` is populated per attempt.
- **`GenericRerankItem`** (`rerank.go:27`) — permissive wire shape: `Index interface{} \`json:"index"\`` (some servers return strings), `Score *float64 \`json:"score"\``, `RelevanceScore *float64 \`json:"relevance_score"\``, `Document interface{} \`json:"document"\``. Pointers distinguish absent from zero.

#### `Rerank` — `rerank.go:35`
- **Signature:** `func Rerank(ctx context.Context, baseURL, apiKey, model, query string, texts []string) ([]RerankItem, error)`
- **Purpose:** Rerank `texts` against `query` on any provider-agnostic `/rerank` endpoint and return normalized `{Index, Score}` pairs.
- **Implementation:** Empty `texts` → `(nil, nil)` short-circuit. URL = `AppendAPIPath(baseURL, "rerank")`. **Attempt 1**: body with `Documents` (Cohere/Jina/llama.cpp/LiteLLM standard). Headers include both `Authorization: Bearer` and `api-key` when key present. If the server answers **400 or 422** (schema validation), the body is closed and **attempt 2** re-sends the identical request with `Texts` instead (HuggingFace TEI standard). Any other non-200 (after any retry) → error with status and body. The response is read fully; empty body → `"empty response from rerank server"`. Parsing then tries three shapes in order, first match wins:
  1. Raw JSON float array `[0.95, 0.82, ...]` — `Index` = position, `Score` = value.
  2. Raw array of `GenericRerankItem` — index from `parseIndex(item.Index)`; if that fails, `extractRerankText(item.Document)` is whitespace-trimmed and matched against the input `texts` to recover the true index; final fallback `idx = i` (position in the response). Score = `item.Score` else `item.RelevanceScore` else 0.
  3. Envelope `{"results": [ ... ]}` — same per-item logic as shape 2, accepted only if `len(envelope.Results) > 0`.
  All shapes failing → `"could not parse rerank response format: <trimmed body>"`. No retries beyond the documents→texts fallback; errors are `%w`-wrapped.
- **Calls / Called by:** Uses `AppendAPIPath`, `newHTTPClient`, `parseIndex`, `extractRerankText`. Called by `commands.go:478` to reorder retrieved chunks.

#### `parseIndex` — `rerank.go:200`
- **Signature:** `func parseIndex(val interface{}) (int, bool)`
- **Purpose:** Coerce a loosely-typed JSON index value to `int`.
- **Implementation:** Handles `nil` (false), `float64` (the default `encoding/json` number type), `int`, `int64`, and `string` via `fmt.Sscanf(n, "%d", &i)`. Anything else → `(0, false)`.
- **Calls / Called by:** Called twice inside `Rerank` (shapes 2 and 3).

#### `extractRerankText` — `rerank.go:220`
- **Signature:** `func extractRerankText(doc interface{}) string`
- **Purpose:** Pull document text out of an echo'd `document` field that may be a plain string or an object.
- **Implementation:** String passes through. `map[string]interface{}` is probed for the first string value under keys `"text"`, `"content"`, `"document"` (in that order). Otherwise empty string.
- **Calls / Called by:** Called twice inside `Rerank` for text-based index recovery.

---

### `cache.go`

#### Types & constants

- **`CorpusCache`** (`cache.go:27`) — metadata header for a cached corpus. Serialized fields: `Collection string \`json:"collection"\``, `ServerURL string \`json:"server_url"\``, `Dimension int \`json:"dimension"\``, `PointCount int \`json:"point_count"\``, `CachedAt time.Time \`json:"cached_at"\``, `FilterAtWarmup string \`json:"filter_at_warmup"\``; plus unexported `filePath string`. Note the JSON tags exist for potential external rendering, but the on-disk file is the custom binary format below — these tags are not how the cache is persisted.
- **`cacheMagic`** (`cache.go:38`) — `uint32(0x51434341)` — `"QQCA"` little-endian ("qquestio corpus cache").
- **`cacheVersion`** (`cache.go:39`) — `uint8(3)`. Version gates inside `parseCacheBytes`: v2 added the warmup-filter string; v3 added the server-URL field.

**On-disk format (v3), all integers little-endian:** `magic uint32` | `version uint8` | `cachedAt unix int64` | `nameLen uint32` + collection name | `serverURLLen uint32` + server URL (v3+) | `dim uint32` | `pointCount uint64` | `filterLen uint32` + filter string (v2+) | `payloadLen uint32` + one JSON array of ALL payloads | then per point: `idLen uint32` + JSON-encoded id (interface{} round-trip preserves int/UUID/string) | `vecLen uint32` + `dim*4` raw little-endian float32 bytes.

#### `CacheDir` — `cache.go:44`
- **Signature:** `func CacheDir() string`
- **Purpose:** Resolve the cache directory.
- **Implementation:** `$QQUESTIO_CACHE_DIR` if set; else `$HOME/.cache/qquestio`; last-resort `os.TempDir()/qquestio-cache` so it never silently writes to CWD.
- **Calls / Called by:** Used by `CachePath`, `SaveCorpusCache` (mkdir), and the root package (`main.go:129`, `slash.go:271/292` for `/cache dir`).

#### `safeCollectionName` — `cache.go:58`
- **Signature:** `func safeCollectionName(collection string) string`
- **Purpose:** Make a Qdrant collection name (hyphens, dots, unicode allowed) filename-safe.
- **Implementation:** Rune-by-rune: keep `[A-Za-z0-9_-]`, replace everything else with `_`. Empty result → `"default"`.
- **Calls / Called by:** Called by `CachePath` only.

#### `CachePath` — `cache.go:80`
- **Signature:** `func CachePath(baseURL, collection string) string`
- **Purpose:** Compute the cache file path; this is the cache-key function.
- **Implementation:** `filepath.Join(CacheDir(), safeCollectionName(collection) + "-" + first10hex(sha1(baseURL)) + ".qcache")`. Keying on both collection and a 10-hex-char SHA-1 of the server URL prevents cross-contamination when one collection name exists on multiple Qdrant servers. (The same baseURL+collection pair is also asserted inside the file at load time — see `parseCacheBytes`.)
- **Calls / Called by:** Used by `LoadCorpusCache`, `SaveCorpusCache`, `DeleteCorpusCache`, `CacheInfo`, `parseCacheBytes`.

#### `LoadCorpusCache` — `cache.go:90`
- **Signature:** `func LoadCorpusCache(baseURL, collection string) (*CorpusCache, []QdrantPoint, error)`
- **Purpose:** Load a cached corpus. Missing file is the normal cold case, not an error.
- **Implementation:** Reads the whole file at `CachePath`; `os.IsNotExist` → `(nil, nil, nil)`. Any other read error or any format error (from `parseCacheBytes`) is returned. Returned points carry `Score: 0` — scores are meaningless on a cache hit and are recomputed by the caller.
- **Calls / Called by:** Delegates to `parseCacheBytes`. Called by `qdrant.go:488` (stale-check path), `qdrant.go:585` (post-save verification), `commands.go:265` and `commands.go:324`.

#### `parseCacheBytes` — `cache.go:103`
- **Signature:** `func parseCacheBytes(baseURL, collection string, data []byte) (*CorpusCache, []QdrantPoint, error)`
- **Purpose:** Decode and validate the binary cache format.
- **Implementation:** Rejects files shorter than 13 bytes (4+1+8). Reads and checks in order, each wrapped with a `"cache: ..."` prefixed error: magic must equal `cacheMagic`; version must equal `cacheVersion` (old versions are hard errors — bumping `cacheVersion` is the invalidation mechanism); embedded collection name must equal the requested `collection`; embedded server URL must equal `baseURL` (v3 field); then `dim`, `pointCount`; then filter string if `version >= 2`; then the payload blob — JSON array of `map[string]interface{}` whose length must equal `pointCount` exactly. Per-point loop reads id (JSON-decoded into `interface{}` to preserve int/UUID/string typing), asserts `vecLen == dim*4`, decodes the vector as little-endian `[]float32`, and appends `QdrantPoint{ID, Payload: allPayloads[i], Vector, Score: 0}`. Rebuilds the `CorpusCache` with `CachedAt: time.Unix(tsUnix, 0)` and `filePath` recomputed via `CachePath`.
- **Calls / Called by:** Called by `LoadCorpusCache` and `CacheInfo`. Depends on `QdrantPoint` (defined in `qdrant.go`).

#### `SaveCorpusCache` — `cache.go:257`
- **Signature:** `func SaveCorpusCache(baseURL, collection string, dim int, points []QdrantPoint, filterAtWarmup string) error`
- **Purpose:** Persist a full corpus, replacing any existing cache for the same key.
- **Implementation:** Refuses empty `points` and `dim <= 0` (guards against caching a broken scroll). `os.MkdirAll(CacheDir(), 0o755)`. Writes the v3 header in the exact order above (timestamp = `time.Now().Unix()`), marshals all payloads as one JSON array, then per point writes the JSON-marshaled id and raw little-endian float32 vector; a point whose vector length ≠ `dim` aborts with an error (partial buffer discarded). Atomic commit: write to `path + ".tmp"` (`0o644`) then `os.Rename`; on rename failure removes the tmp file. Header writes ignore `binary.Write` errors (`_ =`) since writes to a `bytes.Buffer` cannot fail.
- **Calls / Called by:** Called by `qdrant.go:533` (best-effort, error ignored with `_ =`) and `qdrant.go:579` (warmup path, error checked, then re-loads at `qdrant.go:585` to verify).

#### `DeleteCorpusCache` — `cache.go:341`
- **Signature:** `func DeleteCorpusCache(baseURL, collection string) error`
- **Purpose:** Remove the cache file (used by `/cache clear`).
- **Implementation:** `os.Remove(CachePath(...))`; `os.IsNotExist` is treated as success (nil).
- **Calls / Called by:** Called by `slash.go:285`.

#### `CacheInfo` — `cache.go:352`
- **Signature:** `func CacheInfo(baseURL, collection string) (string, error)`
- **Purpose:** Human-readable one-line status for the `/cache status` slash command.
- **Implementation:** `os.Stat` first: missing → `("no cache", nil)`; stat error → error. Then full `LoadCorpusCache`; parse failure is **not** an error — returns `"cache file present (<size>) but failed to parse: <err>"` so the UI can show something. Success → `"collection=%s dim=%d points=%d size=%s age=%s path=%s"` where age is `time.Since(CachedAt).Truncate(time.Second)` and size is `formatBytes(fi.Size())`.
- **Calls / Called by:** Uses `LoadCorpusCache`, `formatBytes`. Called by `slash.go:267`.

#### `formatBytes` — `cache.go:371`
- **Signature:** `func formatBytes(n int64) string`
- **Purpose:** Human-readable byte size (B/KB/MB/GB, 2 decimal places above 1024).
- **Calls / Called by:** Called by `CacheInfo`.

#### `IsStale` — `cache.go:394`
- **Signature:** `func (c *CorpusCache) IsStale(queryDim int, liveCount int, ttl time.Duration) bool`
- **Purpose:** The entire invalidation policy for a corpus cache, in one predicate.
- **Implementation:** Stale if any of: (1) `c.Dimension != queryDim` — embedding model/dimension changed; (2) `c.FilterAtWarmup != ""` — cache was built from a filtered subset, never reusable; (3) `ttl > 0 && time.Since(c.CachedAt) > ttl` — TTL expiration, disabled when `ttl <= 0`; (4) `liveCount > 0 && c.PointCount != liveCount` — collection changed size (checked against the live server). Fresh otherwise.
- **Calls / Called by:** Called at `qdrant.go:491` with `opts.LivePointCount` and the configured TTL.

---

### Extension points

**Adding a new provider/endpoint** (e.g. a completions or hybrid-search call): follow the established recipe — (1) build the URL with `AppendAPIPath(baseURL, "<path>")` from `endpoints.go` so trailing-slash/base-path quirks are normalized; (2) define request/response wire structs with JSON tags in the style of `EmbeddingRequest`/`LiteLLMRequest`; (3) pick a timeout policy: `newHTTPClient(HTTPTimeout)` for request/response, `newHTTPClient(0)` for anything long-lived/streaming (rely on `ctx`), a dedicated short timeout only for connection checks (cf. `CheckLLMConnection`'s 30s); (4) set `Authorization: Bearer` when the key is non-empty, and add the `api-key` header too if Azure-style gateways must work (see `ChatComplete`/`Rerank`); (5) include response body text in non-200 errors; (6) wrap errors with `fmt.Errorf("...: %w", err)`. For a schema-flexible provider, copy the `Rerank` pattern: attempt the common wire shape, retry on 400/422 with the alternate vocabulary, then try multiple response parses in order.

**Adding a new cache behavior**: the format is version-gated, not forward-compatible. To add a field, write it unconditionally in `SaveCorpusCache`, read it in `parseCacheBytes` behind `if version >= N`, and **bump `cacheVersion`** — old files then fail the version check and are treated as a cold cache (callers regenerate). To change invalidation semantics, extend `IsStale` (the single policy point, consumed at `qdrant.go:491`) — e.g. add an embedding-model-hash check by storing it in a new versioned header field. Cache identity is fixed by `CachePath` (collection name + SHA-1 of server URL); keying on anything more (e.g. model) means changing that function, which automatically splits caches. Atomicity comes from the tmp-file-plus-rename in `SaveCorpusCache` — keep any new writer on that pattern.

**Changing the SSE parsing**: all stream semantics live in `(*SSEReader).Next` (`litellm.go:179`) and the anonymous chunk struct at `litellm.go:196-205`. To surface new delta fields (e.g. `delta.tool_calls` or finish reasons), add them to that struct and widen `Next`'s return or add a richer accessor — note the current contract returns only `(content, reasoning, done, err)` and silently skips chunks where both strings are empty, so keep-alives/role deltas are already dropped there. `CALL:`-style tool syntax is not parsed anywhere in this package; if the app adopts it, the natural seam is either a new `Next` variant or post-processing in the root-package consumer that loops over the reader. Buffer sizing for very large chunks is the 64 KB initial / 4 MB max scanner buffer set in `StartLiteLLMStream` (`litellm.go:105`). Usage accounting flows through `stream_options: {"include_usage": true}` → `r.usage` → `Usage()`; there is exactly one SSE reader implementation, so `commands.go:344` is the only call site that would feel a signature change.

---

## Tests

All tests use the standard `go test` runner with **mock HTTP test servers** (`httptest.NewServer`) standing in for Qdrant, the embedding server, the reranker, and the LLM SSE stream — no live services needed. Run with `make test` or `go test -v ./...`.

| File | Covers |
|---|---|
| `skills_test.go` | `ParseCall` line-protocol parsing, `BashSkill.Execute` (incl. exit codes/stderr), registry register/get/list |
| `config_test.go` | Config loading: missing required fields, env-only config, `config.json` config, `RERANKER_POOL` parsing, multi-profile selection with `--conf`/`default_configuration` |
| `model_test.go` | Double-Esc cancel, `formatNumber`, reference-panel document titles, `computeSearchDocs` pool sizing, split-panel focus + copy, session save/load round-trip, mouse-leak CSI filter, empty-session-not-saved, `truncateMiddle`, negative-score references, long prompt/turn wrapping |
| `slash_test.go` | Each command via the call-the-closure-then-inspect-model pattern: `/collection`, `/limit`, `/system`, `/save`, `/mode`, `/search`, `/filter`, `/rerank`, `/rerankerpool`, `/conf` |
| `internal/rag/rag_test.go` | `GetEmbedding`, `SearchQdrant` (+ filters), SSE stream reading, `Rerank` (multiple response shapes), `QdrantPoint.ExtractText`, corpus cache save/load/delete/safe-name against mock servers |
| `internal/rag/qdrant_test.go` | Context expansion (on/off/empty corpus), range serialization, `SearchWithContextExpansionDetailed`, `ApplyExpansionToPrimaries` without a map |
| `internal/rag/exact_phrase_test.go` | `ExtractQuotedPhrases` (all quote styles), `SearchQdrantExactPhrases` end-to-end |
| `internal/rag/qdrant_bench_test.go` | `BenchmarkTopNCosine` — parallel client-side scoring performance |

Test conventions worth following: slash tests execute the returned closure synchronously and assert on the returned Msg **and** mutated model fields (see slash.go section); rag tests spin up mock servers that assert on the request bodies they receive.

---

## Build & Tooling

| Command | What it does |
|---|---|
| `make build` | `go build -ldflags "-X main.Version=1.3.0" -o qquestio` |
| `make test` | `go test -v ./...` (includes mock HTTP test servers for embeddings, qdrant, rerank, SSE) |
| `make clean` | Removes `bin/`, binary, `debug.log` |
| `make build-all` | Cross-compiles: linux amd64/arm64, windows amd64/arm64 (.exe), darwin amd64/arm64 → `bin/` |

Go version: 1.24.2. Dependencies: `charmbracelet/bubbles` v1.0.0, `bubbletea` v1.3.10, `lipgloss` (pinned commit), `glamour` v1.0.0 (indirect, used for markdown rendering).

---

## Extension Recipes

Consolidated index of the per-section recipes above. Each links to the detailed recipe.

| I want to… | Steps (details in section) |
|---|---|
| **Add a new slash command** | Add a `case` in `handleSlashCmd` (`slash.go:32`); mutate `m.*`, return `slashResultMsg`/`systemLogMsg`/`appErrMsg{stage:"slash"}`. Chain pipeline cmds by invoking them (`return m.xCmd()()`), never returning the Cmd itself. Add to `helpText`. Test in `slash_test.go`. → [slash.go section](#root-package--slashgo-runtime-slash-commands) |
| **Add a new config option** | Add the field to `Config` (`config.go:12`) with JSON tag + env var handling in `LoadConfig` phases 2-4. If it must survive `/conf` runtime switches, copy it in the `/conf` handler (`slash.go:77-93`). Test in `config_test.go`. → [config.go section](#root-package--configgo-layered-configuration) |
| **Add a new skill (local tool)** | Implement the `Skill` interface; register in `NewSkillRegistry` (`skills.go:30`). Prompt injection, `CALL:` dispatch, and confirmation gating are automatic. Test in `skills_test.go`. → [skills.go section](#root-package--skillsgo-agentic-tool-system) |
| **Add a new pipeline stage** | New `tea.Msg` type in `messages.go`; new `*Cmd` factory in `commands.go`; insert the `case` in `Update`'s pipeline chain (model.go:648-741); new state const + checklist rendering in `updateViewport` if visible. → [model.go extension points](#root-package--modelgo-the-tui-model--fsm) |
| **Add a new state (FSM)** | Const in the `appState` iota (model.go:29-38), then update every enumeration site: `renderHeader` status switch, pipeline-active checks (Esc-abort model.go:502, spinner model.go:846, `getActiveReferences` model.go:1353, `updateRefViewport` model.go:1366), viewport forwarding (model.go:932), Enter gating (model.go:593). → [model.go extension points](#root-package--modelgo-the-tui-model--fsm) |
| **Add a new keybinding** | `case tea.Key*:` in `Update`'s main key switch (model.go:500-646), after the confirm-dialog blocks. Return a cmd, set `statusMsg`, call `updateViewport()`. → [model.go extension points](#root-package--modelgo-the-tui-model--fsm) |
| **Add a new search mode** | New mode string in `/search` (`slash.go:174`); branch in `searchQdrantCmd` (`commands.go:200`) before the default path; exported `rag` function returning context + points + primaries + `ExpansionMap`. Compose with rerank via `ApplyExpansionToPrimaries` rather than eager expansion. → [qdrant.go extension points](#internal-rag--qdrantgo-vector-db-client) |
| **Add a new Qdrant API call** | Wire structs near the top of `qdrant.go`; `POST/GET /collections/{c}/...` with `api-key` header; `newHTTPClient(<timeout>)`; `qdrantStatusError` for non-200; cursor-paginate via `scrollWithFilter` template. → [qdrant.go extension points](#internal-rag--qdrantgo-vector-db-client) |
| **Add a new filter type** | Extend `QdrantFieldCondition` **and keep both sides in sync**: `buildFilter` (server pushdown) + `applyFilter` (client-side cache path) — `SearchQdrant` re-implements the logic inline, so change it too (or refactor it onto `buildFilter`). `must_not` is unrepresented. → [qdrant.go extension points](#internal-rag--qdrantgo-vector-db-client) |
| **Add a new provider/endpoint (LLM-style)** | URL via `AppendAPIPath`; wire structs; `newHTTPClient(HTTPTimeout)` for request/response, `newHTTPClient(0)` for streaming; `Authorization: Bearer` (+ `api-key` header if Azure-style gateways matter); include body text in non-200 errors. For schema-flexible providers, copy the `Rerank` fallback pattern. → [rag support extension points](#internal-rag--support-clients-httpclientgo-embeddingsgo-litellmgo-rerankgo-cachego) |
| **Add a cache field / change invalidation** | Write unconditionally in `SaveCorpusCache`, read behind `if version >= N` in `parseCacheBytes`, **bump `cacheVersion`** (old files become cold caches). Invalidation semantics all live in `IsStale`. Atomicity = tmp-file + rename. → [rag support extension points](#internal-rag--support-clients-httpclientgo-embeddingsgo-litellmgo-rerankgo-cachego) |
| **Surface new SSE delta fields** | Extend the anonymous chunk struct (`litellm.go:196-205`) and `Next`'s contract; `commands.go:344` is the only call site. → [rag support extension points](#internal-rag--support-clients-httpclientgo-embeddingsgo-litellmgo-rerankgo-cachego) |
| **Add a new message type** | Unexported struct in `messages.go`; emit from a `tea.Cmd`; add the `case` in `Update`. Transient status updates follow `searchProgressMsg` (header-only), not the pipeline pattern. → [core files section](#root-package--core-files) |
| **Add a new style / CLI flag / version bump** | See [core files section](#root-package--core-files) — `Styles` field + `DefaultStyles()`; `flag.*` + package-level var; `version.go`/ldflags. |

### Known quirks & gotchas (verified against source)

1. **`/help` is incomplete**: omits `/rewrite`, `/exact`, `/write` alias.
2. **`scrollOneRange` hardcodes `file_name`** (`qdrant.go:1755`) as the doc-identity key for expansion scrolls — collections using other `DocumentIDKeys` (e.g. `source`, `title`) get no adjacent-chunk expansion.
3. **`SearchQdrant` duplicates `buildFilter` inline** (`qdrant.go:183-216`) — filter changes must touch both.
4. **`ContextLimit` "0 disables compaction" comment is wrong** (`config.go:181-185`): 0 is defaulted to 131072; only a negative value disables.
5. **`--safe` is force-on only** (`config.go:169-171`): no CLI/env way to turn confirmation off once set in JSON.
6. **`renderHeader`'s status switch misses confirm states** (model.go:994-1013): `stateConfirmQuit`/`stateConfirmSkill` render an empty `[STATUS]`.
7. **`chunks %d of %d` in references panel** (`model.go:1971-1973`) uses the range max (`hi`) as "total" — cosmetic mislabel, not a real count.
8. **`condenseQueryForRetrieval` keyword matching is substring-based** (`commands.go:90-97`): `"it"` matches inside `"with"` — follow-ups are over-detected (intentional bias toward recall).
9. **`rag.QdrantVectorName` and `rag.HTTPTimeout` are mutable package globals** set as side effects by `NewModel`, `/conf`, and `GetCollectionInfo` — tests that mutate them should reset.
10. **`saveSession` drops `Reasoning`** — thinking text is not persisted across sessions (render caches too).
11. **Integer env parsing ignores invalid values silently** (`config.go:139-163`) — a typo'd `SEARCH_CAP=abc` is quietly dropped.
12. **`loadSession` restores `searchLimit` only when > 0** (`model.go:2286+`) — a session saved with limit 0 (impossible via `/limit` validation, but possible in a hand-edited file) falls back to the config default.
13. **The skill confirm dialog only gates once per session** once "A" (allow-always) is pressed — `skillsAlwaysAllowed` is never reset, including across `/conf` profile switches.
