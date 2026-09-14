import type { Config, Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"

type SessionClient = Parameters<Plugin>[0]["client"]
import { execFile, spawn } from "node:child_process"
import { mkdirSync, writeFileSync, cpSync, existsSync, readFileSync, rmSync } from "node:fs"
import { homedir, tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { promisify } from "node:util"

/**
 * ontomem opencode plugin -- packaged, self-running, key-free.
 *
 * Everything this file adds on top of the original adapter
 * (engine/adapters/opencode/README.md's history) exists to satisfy one goal:
 * `opencode plugin <spec>` and it's running and learning from then on, with
 * no API key and no separately-managed service or cluster. Two facts make
 * that possible (verified against this fork's own source, not assumed):
 *
 *   - opencode's session API (client.session.prompt) generates text using
 *     whatever provider opencode itself is already authenticated for --
 *     including the user's own OpenCode Zen/Go subscription. This plugin
 *     never needs to see a model API key; it proxies generation calls
 *     through a hidden internal session instead (see startCallbackServer).
 *   - opencode's own Zen/Go server (packages/console/app/src/routes/zen/)
 *     exposes chat completions only -- no embeddings route exists at any
 *     tier. So embeddings run fully locally (LocalEmbedder in the engine,
 *     `local` extra / fastembed) -- no second key to ask for, and node/edge
 *     text never leaves the machine to be embedded.
 *
 * The engine itself is an ordinary standalone Python HTTP service (see
 * engine/README.md); this plugin bootstraps it on first use instead of
 * requiring the user to run `uv run python -m ontomem.service` by hand.
 */

const PLUGIN_DIR = dirname(fileURLToPath(import.meta.url))
const BUNDLED_ENGINE_DIR = join(PLUGIN_DIR, "..", "engine")
const PLUGIN_VERSION: string = JSON.parse(readFileSync(join(PLUGIN_DIR, "..", "package.json"), "utf8")).version

const ONTOMEM_HOME = join(homedir(), ".local", "share", "ontomem")
const ENGINE_RUNTIME_DIR = join(ONTOMEM_HOME, "engine")
const ENGINE_VERSION_MARKER = join(ENGINE_RUNTIME_DIR, ".plugin-version")
const ENGINE_DATA_DIR = join(ONTOMEM_HOME, "data")
const BOOTSTRAP_LOG = join(ONTOMEM_HOME, "bootstrap.log")
const WRITTEN_SESSIONS_PATH = join(ONTOMEM_HOME, "written_sessions.json")

const SERVICE_URL = process.env.ONTOMEM_URL ?? "http://127.0.0.1:8765"
const READ_TIMEOUT_MS = 4000
const HEALTH_TIMEOUT_MS = 2000
const HEALTH_POLL_ATTEMPTS = 60
const HEALTH_POLL_INTERVAL_MS = 2000

// Sequential (agentic) extraction (spec's "sequential commit" fix -- see the
// notebook this was validated against, Documentation/Technical Writeup/
// Temporal Cross Conversational Memory for Large Language Models.ipynb,
// "Everything that Broke, and learnings"): the model commits each
// node/edge to the graph the moment it decides on it, via real opencode
// tools, instead of building the whole graph in its reasoning trace and
// emitting it all at the end (which the notebook found times out on longer
// conversations and produces worse graphs, regardless of model capability).
// Opt-in for now -- pass {"agenticExtraction": true} in the plugin's config
// entry -- until the "does this survive /exit" question below is verified
// live (see runAgenticWrite's comment) and it can become the default.
const AGENTIC_EXTRACTION_AGENT = "ontomem-extract"
const AGENTIC_TOOL_NAMES = ["ontomem_add_entity", "ontomem_add_relationship", "ontomem_finish_extraction"] as const
const AGENTIC_CALL_TIMEOUT_MS = 30000

// Mirrors engine/src/ontomem/model.py's NODE_KINDS/STABILITIES/CARDINALITIES
// and extraction_schema.py's AGENTIC_TOOLS -- these are the wire-format
// source of truth; keep in sync by hand (small, stable enums).
const NODE_KINDS = ["PERSON", "ORG", "PLACE", "EVENT", "THING", "TOPIC", "PREFERENCE", "OTHER"] as const
const STABILITIES = ["immutable", "stable", "mutable", "time_bound", "ephemeral"] as const
const CARDINALITIES = ["one_to_one", "one_to_many"] as const

const execFileAsync = promisify(execFile)

function log(message: string): void {
  try {
    mkdirSync(ONTOMEM_HOME, { recursive: true })
    writeFileSync(BOOTSTRAP_LOG, `${new Date().toISOString()} ${message}\n`, { flag: "a" })
  } catch {
    // best-effort logging only
  }
}

async function isEngineHealthy(): Promise<boolean> {
  try {
    // /health is POST-only (see service.py's dispatch()) -- a plain GET
    // (the default fetch() method) always 404s there, which meant this
    // returned false unconditionally, even against a perfectly healthy
    // engine. Confirmed live: the bootstrap log showed repeated re-spawns
    // because ensureEngineRunning() never believed any of them succeeded.
    const res = await fetch(`${SERVICE_URL}/health`, {
      method: "POST",
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    })
    return res.ok
  } catch {
    return false
  }
}

// Copies the bundled engine/ (synced into this package at publish time by
// scripts/sync-engine.sh) into a per-user runtime directory the whole engine
// process actually runs from. Re-synced whenever the installed plugin
// version changes, so a plugin update ships engine fixes too.
function syncEngineSource(): void {
  const marker = existsSync(ENGINE_VERSION_MARKER) ? readFileSync(ENGINE_VERSION_MARKER, "utf8").trim() : null
  if (marker === PLUGIN_VERSION) return
  rmSync(ENGINE_RUNTIME_DIR, { recursive: true, force: true })
  mkdirSync(ENGINE_RUNTIME_DIR, { recursive: true })
  cpSync(BUNDLED_ENGINE_DIR, ENGINE_RUNTIME_DIR, { recursive: true })
  writeFileSync(ENGINE_VERSION_MARKER, PLUGIN_VERSION)
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms))
}

// Spawns the engine service pointed at the given callback URL, detached so
// it survives this opencode process (same "spawn detached, unref, let it
// outlive us" shape flushWriteInBackground already uses below), then polls
// /health until it answers or we give up. Assumes `uv` is already confirmed
// present (see ensureEngineRunning) and the source is already synced.
async function startEngineService(callbackUrl: string): Promise<boolean> {
  log(`running "uv sync --extra local" in ${ENGINE_RUNTIME_DIR} (first run downloads the local embedding model, can take a few minutes)`)
  try {
    await execFileAsync("uv", ["sync", "--extra", "local"], { cwd: ENGINE_RUNTIME_DIR })
  } catch (err) {
    // Deliberately NOT falling back to `uv sync` without the local extra
    // here. That would silently start the service on the lexical
    // HashingEmbedder instead of real semantic embeddings -- a real quality
    // downgrade with no visible signal, which is worse than a loud failure:
    // the user would just conclude memory recall "doesn't really work"
    // with no idea why. engine/pyproject.toml's `local` extra already pins
    // onnxruntime to a version with real wheels on every platform this
    // plugin supports (including macOS Intel, which needs a specific
    // version window -- see that file's comment); a failure here means an
    // actually-unexpected platform/environment problem worth surfacing,
    // not quietly working around.
    log(`uv sync --extra local failed, engine will not start: ${errorMessage(err)}`)
    return false
  }

  mkdirSync(ENGINE_DATA_DIR, { recursive: true })
  const child = spawn("uv", ["run", "python", "-m", "ontomem.service"], {
    cwd: ENGINE_RUNTIME_DIR,
    detached: true,
    stdio: "ignore",
    env: {
      ...process.env,
      ONTOMEM_DIR: ENGINE_DATA_DIR,
      ONTOMEM_HOST_CALLBACK_URL: callbackUrl,
    },
  })
  child.unref()
  log(`spawned engine service (pid ${child.pid}), waiting for /health`)

  for (let attempt = 0; attempt < HEALTH_POLL_ATTEMPTS; attempt++) {
    if (await isEngineHealthy()) {
      log("engine service is healthy")
      return true
    }
    await sleep(HEALTH_POLL_INTERVAL_MS)
  }
  log("engine service did not become healthy within the poll window")
  return false
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

// Called once at plugin load (fire-and-forget, not awaited by the plugin
// factory -- see OntomemPlugin below for why) and again, cheaply, from every
// hook that needs the engine: the health check makes this idempotent across
// opencode restarts (a still-running service from a prior session just
// passes the check immediately, no new spawn) and across concurrent calls
// within one process (bootstrapPromise below caches the in-flight attempt).
let bootstrapPromise: Promise<boolean> | null = null

function ensureEngineRunning(callbackUrl: string): Promise<boolean> {
  if (bootstrapPromise) return bootstrapPromise
  bootstrapPromise = (async () => {
    if (await isEngineHealthy()) return true

    const uvPath = Bun.which("uv")
    if (!uvPath) {
      log(
        "uv not found on PATH -- install it (https://docs.astral.sh/uv/getting-started/installation/) " +
          "then restart opencode. Memory is disabled until then.",
      )
      return false
    }

    try {
      syncEngineSource()
    } catch (err) {
      log(`failed to sync engine source into ${ENGINE_RUNTIME_DIR}: ${errorMessage(err)}`)
      return false
    }

    return startEngineService(callbackUrl)
  })()
  // A failed attempt should be retryable on the NEXT hook call (e.g. uv got
  // installed after the first failure), not permanently cached as false.
  bootstrapPromise.then((ok) => {
    if (!ok) bootstrapPromise = null
  })
  return bootstrapPromise
}

async function callService<T>(path: string, body: unknown, timeoutMs = READ_TIMEOUT_MS): Promise<T | null> {
  try {
    const res = await fetch(`${SERVICE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    })
    if (!res.ok) return null
    return (await res.json()) as T
  } catch {
    return null
  }
}

function textOf(parts: Array<{ type: string; text?: string }>): string {
  return parts
    .filter((p) => p.type === "text" && p.text)
    .map((p) => p.text)
    .join("\n")
    .trim()
}

// Tracks, per session, how many turns (of the shape dispose() builds below)
// were already dispatched for writing -- durable across process restarts
// (unlike touchedSessions, which is in-memory and resets every opencode
// launch). Exists specifically because dispose fires on every /exit, not
// once per conversation's true lifetime: exiting, reopening the same
// session just to look at it, and exiting again would otherwise re-dispatch
// the whole conversation for a second full (wasteful, non-deterministic)
// extraction pass with nothing new to extract. Turn COUNT (not a content
// hash) is enough here -- opencode sessions are append-only, so "same or
// fewer turns than last time" reliably means "nothing new since the last
// write" without needing to hash message content.
type WrittenSessions = Record<string, number>

function readWrittenSessions(): WrittenSessions {
  try {
    return JSON.parse(readFileSync(WRITTEN_SESSIONS_PATH, "utf8"))
  } catch {
    return {}
  }
}

function markSessionWritten(sessionID: string, turnCount: number): void {
  try {
    const state = readWrittenSessions()
    state[sessionID] = turnCount
    mkdirSync(dirname(WRITTEN_SESSIONS_PATH), { recursive: true })
    writeFileSync(WRITTEN_SESSIONS_PATH, JSON.stringify(state))
  } catch {
    // best-effort -- worst case a future dispose re-writes this session once more
  }
}

// Maps an opencode session id (the DEDICATED extraction session
// runAgenticWrite creates -- never the user's own coding session) to the
// engine-side agentic session id returned by /agentic/start. Looked up by
// each ontomem_* tool's execute() via its ToolContext.sessionID, both to
// find which engine-side session to apply the call to AND, as the
// defense-in-depth scoping backstop (see this file's header), to refuse to
// act at all if a call arrives from any OTHER session -- the agent-level
// tool restriction (see the config hook below) should already prevent that,
// but this doesn't rely on it being airtight.
const agenticSessionByOpencodeSession = new Map<string, string>()

async function startAgenticWrite(conversation: unknown): Promise<{ session_id: string; prompt_text: string } | null> {
  return callService("/agentic/start", { conversation }, AGENTIC_CALL_TIMEOUT_MS)
}

async function applyAgenticToolCall(
  sessionId: string,
  toolName: string,
  args: unknown,
): Promise<{ result_text: string; finished: boolean; stats?: unknown } | null> {
  return callService(
    "/agentic/tool_call",
    { session_id: sessionId, tool_name: toolName, arguments: args },
    AGENTIC_CALL_TIMEOUT_MS,
  )
}

const WRITE_FAILURE_LOG = join(tmpdir(), "ontomem-write-failures.log")

function flushWriteInBackground(conversation: unknown): void {
  const tmpFile = join(tmpdir(), `ontomem-write-${Date.now()}-${Math.random().toString(36).slice(2)}.json`)
  writeFileSync(tmpFile, JSON.stringify({ conversation }))
  const cmd =
    `if curl -fsS --retry 6 --retry-delay 15 --retry-all-errors -X POST '${SERVICE_URL}/write' ` +
    `-H 'Content-Type: application/json' --data-binary @'${tmpFile}' -o /dev/null; then ` +
    `rm -f '${tmpFile}'; else ` +
    `echo "$(date -u +%FT%TZ) write failed after retries, payload kept at ${tmpFile}" >> '${WRITE_FAILURE_LOG}'; fi`
  const child = spawn("sh", ["-c", cmd], { detached: true, stdio: "ignore" })
  child.unref()
}

// Sequential-extraction alternative to flushWriteInBackground (see the
// AGENTIC_EXTRACTION_AGENT comment above for why this exists). Two real
// HTTP round-trips happen here rather than a curl-and-forget like the
// batch path: /agentic/start is a fast, deterministic, no-LLM call (Stage 0
// context assembly only), so awaiting it briefly in dispose is fine -- the
// SAME reasoning that makes awaiting a single request there acceptable
// doesn't extend to the actual multi-minute tool-calling loop, which is why
// that part is still handed off to a detached process below.
//
// OPEN QUESTION, not yet verified against a real opencode+Zen session: does
// the server process behind a plain `opencode` (TUI) launch actually
// outlive /exit long enough for a multi-minute native tool-calling loop to
// finish, or does the whole process (server included) tear down once
// dispose() resolves? If the server does NOT outlive it, this silently
// loses the in-flight extraction the moment the terminal returns to the
// prompt. That's exactly why this mode is opt-in, not the default, until
// confirmed live.
async function runAgenticWrite(conversation: unknown, client: SessionClient, serverUrl: URL): Promise<void> {
  const started = await startAgenticWrite(conversation)
  if (!started) {
    log("agentic write: /agentic/start failed or engine unavailable, skipping")
    return
  }

  const session = await client.session.create({ body: { title: "ontomem (internal, extraction)" } }).catch(() => null)
  if (!session?.data) {
    log("agentic write: failed to create dedicated extraction session")
    return
  }
  agenticSessionByOpencodeSession.set(session.data.id, started.session_id)

  const tmpFile = join(tmpdir(), `ontomem-agentic-${Date.now()}-${Math.random().toString(36).slice(2)}.json`)
  writeFileSync(
    tmpFile,
    JSON.stringify({
      agent: AGENTIC_EXTRACTION_AGENT,
      parts: [{ type: "text", text: started.prompt_text }],
    }),
  )
  const url = new URL(`/session/${session.data.id}/message`, serverUrl).toString()
  const cmd =
    `if curl -fsS --max-time 1800 -X POST '${url}' ` +
    `-H 'Content-Type: application/json' --data-binary @'${tmpFile}' -o /dev/null; then ` +
    `rm -f '${tmpFile}'; else ` +
    `echo "$(date -u +%FT%TZ) agentic extraction request failed, payload kept at ${tmpFile}" >> '${WRITE_FAILURE_LOG}'; fi`
  const child = spawn("sh", ["-c", cmd], { detached: true, stdio: "ignore" })
  child.unref()
}

export const OntomemPlugin: Plugin = async ({ client, serverUrl }, options) => {
  // The internal session used only for ontomem's own generation calls (see
  // startCallbackServer) -- never the user's visible coding session. Created
  // lazily on first use, then reused for the process lifetime.
  let internalSessionId: string | null = null
  const modelOverride = options?.model as { providerID: string; modelID: string } | undefined
  const agenticExtractionEnabled = options?.agenticExtraction === true

  // A tiny localhost-only proxy the engine calls back into for every
  // extraction/merge/supersede LLM call (see engine/src/ontomem/inference.py's
  // HostCallbackGenerator). This is the ONLY place a model API is ever
  // touched from this plugin -- and it's not an API at all, it's opencode's
  // own session.prompt, so no key is ever handled here.
  const callbackServer = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    fetch: async (req) => {
      if (req.method !== "POST" || new URL(req.url).pathname !== "/generate") {
        return new Response("not found", { status: 404 })
      }
      try {
        const { prompt } = (await req.json()) as { prompt: string }
        if (!internalSessionId) {
          const session = await client.session.create({ body: { title: "ontomem (internal, not a chat)" } })
          if (!session.data) throw new Error("failed to create internal session")
          internalSessionId = session.data.id
        }
        const result = await client.session.prompt({
          path: { id: internalSessionId },
          body: {
            ...(modelOverride ? { model: modelOverride } : {}),
            tools: {},
            parts: [{ type: "text", text: prompt }],
          },
        })
        if (!result.data) throw new Error("session.prompt returned no data")
        const text = textOf(result.data.parts as Array<{ type: string; text?: string }>)
        if (!text) throw new Error("session.prompt returned no text content")
        return Response.json({ text })
      } catch (err) {
        log(`callback /generate failed: ${errorMessage(err)}`)
        return Response.json({ error: errorMessage(err) }, { status: 500 })
      }
    },
  })
  const callbackUrl = `http://127.0.0.1:${callbackServer.port}`

  // Fire-and-forget: the plugin factory must return quickly (this runs at
  // opencode startup), and a first-run bootstrap (uv sync can take minutes
  // to fetch the local embedding model) must not block that. Every hook
  // below degrades to a no-op if the engine isn't up yet -- same best-effort
  // contract as any other engine-unavailable case.
  ensureEngineRunning(callbackUrl).then((ok) => {
    if (ok) callService("/decay", {}) // self-gated inside the engine; safe to fire every start
  })

  // Per-session memory block computed at Read time, consumed at injection time.
  const pendingContext = new Map<string, string>()
  const touchedSessions = new Set<string>()

  return {
    // Injects the ontomem_* tools' dedicated subagent (see AGENTIC_TOOL_NAMES
    // above) and, best-effort, denies those same tool names in the user's
    // own default agents -- belt-and-suspenders alongside each tool's own
    // sessionID check (below), since there's no fully airtight native way
    // to scope a plugin-registered tool's visibility to one session. The
    // agent's tools/permission restriction is the primary control; its
    // exact unlisted-key semantics aren't independently verified yet (see
    // this file's header and the plan this shipped from) -- the sessionID
    // check is what actually enforces correctness regardless.
    config: async (input: Config) => {
      try {
        const denyOntomemTools: Record<string, boolean> = {}
        for (const name of AGENTIC_TOOL_NAMES) denyOntomemTools[name] = false
        const agent: NonNullable<Config["agent"]> = { ...input.agent }
        for (const name of ["build", "plan", "general", "explore"] as const) {
          agent[name] = { ...agent[name], tools: { ...agent[name]?.tools, ...denyOntomemTools } }
        }
        agent[AGENTIC_EXTRACTION_AGENT] = {
          mode: "subagent",
          description: "Internal use only -- ontomem's sequential graph-extraction agent, invoked by the plugin itself. Never select this manually.",
          tools: {
            // Known built-in tool names from reading opencode's own
            // built-in agent definitions -- not guaranteed exhaustive, see
            // header comment. add_entity/add_relationship/finish_extraction
            // are the only tools this agent is meant to use.
            bash: false, edit: false, write: false, patch: false,
            webfetch: false, websearch: false, read: false, grep: false,
            glob: false, list: false, todowrite: false, todoread: false,
            ...Object.fromEntries(AGENTIC_TOOL_NAMES.map((name) => [name, true])),
          },
          permission: { edit: "deny", bash: "deny", webfetch: "deny", external_directory: "deny" },
        }
        input.agent = agent
      } catch (err) {
        log(`config hook failed to register ${AGENTIC_EXTRACTION_AGENT} agent: ${errorMessage(err)}`)
      }
    },

    "chat.message": async (input, output) => {
      touchedSessions.add(input.sessionID)
      try {
        // Do NOT await ensureEngineRunning() here -- opencode awaits this
        // whole hook before the message is even rendered, and a first-run
        // bootstrap can take well over a minute (uv sync + spawn + health
        // polling). That turned every first message after install into an
        // apparent hang. Bootstrapping is already kicked off, fire-and-
        // forget, once at plugin load (see above); /read below already has
        // its own short timeout (READ_TIMEOUT_MS) and degrades to null if
        // the engine isn't up yet -- that's the correct behavior here, not
        // blocking the turn on the full bootstrap.
        const message = textOf(output.parts)
        if (!message) return
        const result = await callService<{ text: string }>("/read", { message })
        if (result?.text) pendingContext.set(input.sessionID, result.text)
      } catch {
        // best-effort memory capture — swallow and proceed without it
      }
    },

    "experimental.chat.system.transform": async (input, output) => {
      try {
        const block = input.sessionID ? pendingContext.get(input.sessionID) : undefined
        if (block) output.system.push(block)
      } catch {
        // best-effort injection — swallow and let the turn proceed unmodified
      }
    },

    tool: {
      retrieve_memory: tool({
        description:
          "Retrieve deeper long-term memory about a known entity (person, org, place, topic). " +
          "Use when the injected [MEMORY] context references something you need more detail on.",
        args: {
          node_name: tool.schema.string().describe("Entity name to look up, e.g. 'Sarah' or 'Walmart'"),
          depth: tool.schema.number().optional().describe("Traversal depth, default 1"),
        },
        execute: async (args) => {
          const result = await callService("/retrieve_memory", {
            node_name: args.node_name,
            depth: args.depth ?? 1,
          })
          if (!result) return "Memory service unavailable."
          return JSON.stringify(result)
        },
      }),

      // The three tools below exist ONLY for ontomem's own sequential
      // extraction (AGENTIC_TOOL_NAMES / runAgenticWrite above) -- schemas
      // mirror engine/src/ontomem/extraction_schema.py's AGENTIC_TOOLS
      // exactly, which is the wire-format source of truth. Each execute()
      // checks context.sessionID against the dedicated extraction session
      // runAgenticWrite creates and refuses to act for any other session --
      // the primary scoping is the config hook's agent-level tool
      // restriction above, this is the defense-in-depth backstop (see this
      // file's header comment on why both layers exist).
      ontomem_add_entity: tool({
        description:
          "INTERNAL USE ONLY (ontomem sequential graph extraction). Add ONE durable entity to the graph " +
          "as soon as you identify it. Call once per entity, not in a batch. 'type' MUST be exactly one of: " +
          NODE_KINDS.join(", ") + ".",
        args: {
          text: tool.schema.string().describe("Canonical entity name, as specific as possible."),
          type: tool.schema.enum(NODE_KINDS),
          confidence: tool.schema.number().min(0).max(1),
          properties: tool.schema
            .record(tool.schema.string(), tool.schema.unknown())
            .describe("Time-invariant facts only. Empty {} for most entities."),
          aliases: tool.schema.array(tool.schema.string()),
          candidate_merge_key: tool.schema.string().nullable().describe("Null unless you identified a likely existing node match."),
        },
        execute: async (args, context) => {
          const sessionId = agenticSessionByOpencodeSession.get(context.sessionID)
          if (!sessionId) return "error: this tool is only available during ontomem's internal extraction session"
          const result = await applyAgenticToolCall(sessionId, "add_entity", args)
          return result?.result_text ?? "error: memory service unavailable"
        },
      }),

      ontomem_add_relationship: tool({
        description:
          "INTERNAL USE ONLY (ontomem sequential graph extraction). Add ONE relationship to the graph as " +
          "soon as you identify it. Both source and target must already exist -- either added earlier via " +
          "ontomem_add_entity or already present in the existing graph context. Call once per relationship.",
        args: {
          source: tool.schema.string().describe("Canonical entity name, must already exist."),
          relation: tool.schema.string().describe("UPPER_SNAKE_CASE, verb-first, max 4 words."),
          target: tool.schema.string().describe("Canonical entity name, must already exist."),
          confidence: tool.schema.number().min(0).max(1),
          stability: tool.schema.enum(STABILITIES),
          ttl_days: tool.schema.number().int().nullable(),
          cardinality: tool.schema
            .enum(CARDINALITIES)
            .describe(
              "one_to_one: only ONE target can be true for this relation from this source at a time, even if " +
                "it changes over time (current manager, current partner, current employer) -- lets the system " +
                "replace the old value non-destructively later. one_to_many: multiple simultaneous targets are " +
                "normal and expected (friends, hobbies, places visited).",
            ),
          evidence: tool.schema.string().describe("Short verbatim phrase from the text."),
          snippet: tool.schema.string().describe("2-6 sentence verbatim excerpt, meaningful read in isolation."),
          properties: tool.schema
            .record(tool.schema.string(), tool.schema.unknown())
            .describe("Quantifiers/qualifiers about this relationship -- duration, frequency, degree."),
        },
        execute: async (args, context) => {
          const sessionId = agenticSessionByOpencodeSession.get(context.sessionID)
          if (!sessionId) return "error: this tool is only available during ontomem's internal extraction session"
          const result = await applyAgenticToolCall(sessionId, "add_relationship", args)
          return result?.result_text ?? "error: memory service unavailable"
        },
      }),

      ontomem_finish_extraction: tool({
        description:
          "INTERNAL USE ONLY (ontomem sequential graph extraction). Call exactly once, after every durable " +
          "entity and relationship in the conversation has already been added, to close out the episode.",
        args: {
          summary: tool.schema.string().describe("1-2 sentence specific summary of the conversation."),
          importance: tool.schema.number().min(0).max(1),
          tags: tool.schema.array(tool.schema.string()).describe("2-5 snake_case domain labels."),
        },
        execute: async (args, context) => {
          const sessionId = agenticSessionByOpencodeSession.get(context.sessionID)
          if (!sessionId) return "error: this tool is only available during ontomem's internal extraction session"
          const result = await applyAgenticToolCall(sessionId, "finish_extraction", args)
          agenticSessionByOpencodeSession.delete(context.sessionID)
          return result?.result_text ?? "error: memory service unavailable"
        },
      }),
    },

    dispose: async () => {
      const written = readWrittenSessions()
      for (const sessionID of touchedSessions) {
        const messages = await client.session.messages({ path: { id: sessionID } }).catch(() => null)
        if (!messages?.data) continue
        const conversation = messages.data
          .map((m, i) => ({ role: m.info.role, turn: i + 1, text: textOf(m.parts) }))
          .filter((t) => t.text)
        if (conversation.length === 0) continue
        // Nothing new since the last write for this session (e.g. reopened
        // it just to look, then exited again) -- skip the redundant
        // extraction pass entirely rather than re-writing unchanged content.
        if ((written[sessionID] ?? 0) >= conversation.length) continue
        // Mark BEFORE dispatch, not after: flushWriteInBackground hands off
        // to a detached process we don't await, so there's no reliable
        // "write actually finished" signal to hook this on. Marking here
        // means a second dispose moments later (before the first write even
        // reaches the engine) still sees this session as handled.
        markSessionWritten(sessionID, conversation.length)
        if (agenticExtractionEnabled) {
          // Awaited: /agentic/start itself is fast (no LLM call, just
          // deterministic context assembly -- see runAgenticWrite's
          // comment), it's only the actual multi-minute tool-calling loop
          // that's handed off detached.
          await runAgenticWrite(conversation, client, serverUrl).catch((err) =>
            log(`agentic write failed to start: ${errorMessage(err)}`),
          )
        } else {
          flushWriteInBackground(conversation)
        }
      }
      callbackServer.stop()
    },
  }
}

export default { id: "ontomem", server: OntomemPlugin }
