import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"
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
// Default since 0.2.0 -- pass {"agenticExtraction": false} in the plugin's
// config entry to opt back out (see spawnDetachedWrite's comment for the
// current mechanism and what it replaced).
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
// outlive us" shape spawnDetachedWrite uses below), then polls /health
// until it answers or we give up. Assumes `uv` is already confirmed present
// (see ensureEngineRunning) and the source is already synced.
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
      // Confirmed live (Trevor/Deloitte test conversation): without this,
      // the extractor reads both roles, including the assistant's own long
      // replies -- which it then has to spend reasoning tokens parsing back
      // out of, for zero benefit (assistant text was never a source of
      // durable facts to begin with; see the "acknowledged agent concepts"
      // exception in the spec, which is the only case that should matter
      // and doesn't require sending full assistant turns to get). The
      // notebook this plugin's sequential-extraction fix came from
      // independently found the same thing for reasoning-model extraction
      // specifically: parsing the agent's own turns is exactly what made
      // extraction slow. Stage 0 context assembly (engine.py's
      // assemble_context call) still sees the full conversation, both
      // roles -- only the text actually handed to the extractor is
      // user-only. Always on for the packaged plugin; not conditional on
      // agenticExtraction, since the reasoning applies equally to the
      // single-shot path.
      ONTOMEM_USER_ONLY_EXTRACTION: "1",
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
    if (await isEngineHealthy()) {
      // The engine is long-lived and outlives individual opencode
      // processes (see the README's "Does the engine run forever?"),
      // while callbackUrl is THIS process's own ephemeral Bun.serve()
      // port -- a fresh random one every launch. Without refreshing here,
      // a reused engine keeps calling back to whichever port happened to
      // be listening when it was first spawned, possibly sessions ago and
      // long since dead. Confirmed live: curl to a stale baked-in callback
      // URL from an exited session returns connection refused, silently
      // failing every subsequent generation call (extraction, merge/
      // disambiguation, supersede) with no visible error. Cheap and
      // idempotent, so it's called every time, not just on a fresh spawn.
      await refreshCallbackUrl(callbackUrl)
      return true
    }

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

    const started = await startEngineService(callbackUrl)
    // Also redundant for a fresh spawn (callbackUrl was already passed as
    // ONTOMEM_HOST_CALLBACK_URL at startup) -- kept for one code path
    // instead of special-casing "did we just spawn it" here.
    if (started) await refreshCallbackUrl(callbackUrl)
    return started
  })()
  // A failed attempt should be retryable on the NEXT hook call (e.g. uv got
  // installed after the first failure), not permanently cached as false.
  bootstrapPromise.then((ok) => {
    if (!ok) bootstrapPromise = null
  })
  return bootstrapPromise
}

async function refreshCallbackUrl(callbackUrl: string): Promise<void> {
  await callService("/refresh_callback", { url: callbackUrl })
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

// Set only inside a dedicated, single-purpose `opencode serve` process
// spawned by scripts/run-detached-write.ts in agentic mode (see that file's
// header for why the write has to run there, detached, instead of in-
// process). The extraction session is the ONLY session that will ever
// exist on that process, so its presence is treated as "this whole process
// is scoped to the one extraction session" -- no per-opencode-session
// bookkeeping needed; every ontomem_* tool call arriving anywhere in this
// process applies to it.
const DETACHED_EXTRACTION_SESSION_ID = process.env.ONTOMEM_DETACHED_EXTRACTION_SESSION_ID ?? null

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

const DETACHED_WRITE_SCRIPT = join(PLUGIN_DIR, "..", "scripts", "run-detached-write.ts")

// Both write modes (plain single-shot and sequential/agentic) end up here.
// What this went through before landing on ONE shared path for both, for
// whoever reads this next:
//   1. Plain mode used to be a bare detached curl straight to the engine's
//      /write, relying on THIS process's own callback server for
//      generation. Confirmed live (by direct code-path tracing, not a
//      guess): dispose() stops that callback server essentially
//      immediately after dispatching the detached curl, before the
//      request could plausibly reach the engine and trigger a callback --
//      so the plain path likely never completed real generation, ever.
//   2. Agentic mode used to directly await client.session.prompt() inside
//      dispose() -- confirmed live via granular step logging that this
//      fails, root-caused to opencode's TUI shutdown path
//      (packages/opencode/src/cli/cmd/tui.ts) wrapping the whole dispose()
//      call in `withTimeout(client.call("shutdown"), 5000)` then calling
//      `worker.terminate()` unconditionally after -- a hard 5-second
//      ceiling neither a multi-minute tool-calling loop nor even a single
//      slow generation call can reliably finish inside.
//   3. An earlier attempt at (2) spawned a detached curl straight at
//      PluginInput.serverUrl -- confirmed live every path there 404s; the
//      default (no --port) TUI launch has no real listening HTTP server to
//      begin with (see tui.ts's `external` flag).
//   4. An even earlier attempt used a custom "ontomem-extract" agent
//      registered via the `config` hook -- confirmed by tracing the source
//      that this fork unconditionally discards a plugin's `config` hook
//      return value, so that agent never existed.
//
// Both failure modes share one root cause: real generation work routinely
// takes longer than the 5-second window opencode gives dispose() before
// force-killing the process. scripts/run-detached-write.ts is the fix for
// both -- a fully detached process (survives dispose() returning and the
// parent's death) that drives its own independent `opencode serve`
// instance AND its own callback proxy for generation, immune to the
// parent's shutdown timeout because it isn't the parent.
function spawnDetachedWrite(
  mode: "agentic" | "plain",
  conversation: unknown,
  directory: string,
  modelOverride?: { providerID: string; modelID: string },
): void {
  // Deliberately NOT process.execPath here. Confirmed live: the real
  // installed opencode CLI (`~/.nvm/.../bin/opencode`, the actual binary the
  // user runs) launches under Node, not Bun -- so process.execPath inside a
  // running plugin can resolve to a `node` binary. run-detached-write.ts
  // uses Bun-only APIs (Bun.which) and raw TypeScript, which a bare `node
  // <script>.ts` invocation fails on almost immediately -- and with
  // stdio: "ignore" that failure was completely silent (confirmed live: the
  // spawned process left zero log output and never even reached its first
  // log() call). This plugin file itself only runs at all because SOMETHING
  // in the opencode process is a real Bun runtime (Bun.serve/Bun.which both
  // work here, elsewhere in this file) -- so resolve `bun` explicitly via
  // PATH instead of trusting how the parent process itself was launched.
  const bunBin = Bun.which("bun")
  if (!bunBin) {
    log(`dispose: 'bun' not found on PATH, cannot spawn detached ${mode} write -- skipping`)
    return
  }
  const payloadFile = join(tmpdir(), `ontomem-write-${Date.now()}-${Math.random().toString(36).slice(2)}.json`)
  writeFileSync(
    payloadFile,
    JSON.stringify({ mode, conversation, directory, modelOverride, serviceUrl: SERVICE_URL, bootstrapLog: BOOTSTRAP_LOG }),
  )
  log(`dispose: spawning detached ${mode} write via ${bunBin} (payload ${payloadFile})`)
  const child = spawn(bunBin, [DETACHED_WRITE_SCRIPT, payloadFile], {
    detached: true,
    stdio: "ignore",
  })
  // Without this, a spawn-level failure (e.g. bunBin resolved but somehow
  // unusable) throws an unhandled 'error' event asynchronously, after
  // dispose() has already returned -- invisible, same failure mode as the
  // process.execPath bug above. Cheap insurance against the next version of
  // that same class of bug.
  child.on("error", (err) => log(`dispose: detached ${mode} write spawn failed: ${errorMessage(err)}`))
  child.unref()
}

export const OntomemPlugin: Plugin = async ({ client, directory }, options) => {
  // The internal session used only for ontomem's own generation calls (see
  // startCallbackServer) -- never the user's visible coding session. Created
  // lazily on first use, then reused for the process lifetime.
  let internalSessionId: string | null = null
  const modelOverride = options?.model as { providerID: string; modelID: string } | undefined
  // Default as of this release -- sequential extraction (see
  // spawnDetachedWrite's comment) has now been verified end-to-end live
  // multiple times: per-call commits land immediately,
  // survive the detached server's own lifetime, tool scoping holds, and
  // cross-session recall works. Still escapable via {"agenticExtraction":
  // false} if it ever needs to be turned off for a specific install.
  const agenticExtractionEnabled = options?.agenticExtraction !== false

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

  // Registered ONLY inside the dedicated extraction server (see
  // spawnDetachedWrite/scripts/run-detached-write.ts, agentic mode) --
  // DETACHED_EXTRACTION_SESSION_ID is read once from process.env at module
  // load, so this is a per-process decision, not per-session, which lines
  // up exactly with reality: a normal opencode launch never has this env
  // var set and never needs these tools, while the dedicated server always
  // has it set and never handles any OTHER kind of session. Confirmed live
  // that just BLOCKING execution (tool.execute.before below) wasn't enough
  // on its own -- the model could still see ontomem_add_relationship in a
  // completely normal chat, try to call it, and get a visible rejection
  // (red tool-call line in the TUI) for no reason a real user should ever
  // see. Not registering the tool at all in a normal session's process
  // removes it from the model's tool list entirely, not just from what it's
  // allowed to execute. tool.execute.before stays as defense-in-depth.
  // Explicit annotation, not inferred: the two ternary branches below have
  // different shapes ({} vs three named tools), and TypeScript infers a
  // union of object types across them rather than an index signature --
  // which then fails to satisfy Hooks["tool"]'s `{[key: string]: ToolDef}`
  // because the union's "missing" keys type as `undefined`, not "absent".
  const agenticExtractionTools: Record<string, ReturnType<typeof tool>> =
    DETACHED_EXTRACTION_SESSION_ID !== null
      ? {
          // Schemas mirror engine/src/ontomem/extraction_schema.py's
          // AGENTIC_TOOLS exactly, which is the wire-format source of truth.
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
              candidate_merge_key: tool.schema
                .string()
                .nullable()
                .describe("Null unless you identified a likely existing node match."),
            },
            execute: async (args) => {
              const result = await applyAgenticToolCall(DETACHED_EXTRACTION_SESSION_ID, "add_entity", args)
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
            execute: async (args) => {
              const result = await applyAgenticToolCall(DETACHED_EXTRACTION_SESSION_ID, "add_relationship", args)
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
            execute: async (args) => {
              const result = await applyAgenticToolCall(DETACHED_EXTRACTION_SESSION_ID, "finish_extraction", args)
              return result?.result_text ?? "error: memory service unavailable"
            },
          }),
        }
      : {}

  return {
    // Defense-in-depth backstop, bidirectional: the PRIMARY scoping is now
    // agenticExtractionTools above simply not registering the ontomem_*
    // tools at all outside the dedicated extraction server, so a normal
    // chat session's model never even sees them in its tool list. This hook
    // catches the two cases that leaves: the (should-be-impossible) case of
    // an ontomem_* tool call reaching a process where it isn't registered,
    // and the extraction session calling anything OTHER than an ontomem_*
    // tool -- verified (by reading packages/opencode/src/session/tools.ts's
    // real execution path, not just types) to run for every plugin-
    // registered tool call, with no Effect.ignore wrapping it, so a thrown
    // error here genuinely aborts execution before the tool's own execute()
    // runs. Two earlier mechanisms were tried and confirmed NOT to work,
    // for two different concrete reasons -- worth recording so this isn't
    // re-attempted:
    //   1. A custom restricted "ontomem-extract" agent injected via the
    //      `config` hook. Confirmed by tracing the source: this fork
    //      unconditionally discards a plugin's `config` hook return value
    //      (`Effect.ignore` in packages/opencode/src/plugin/index.ts), and
    //      the real agent permission registry only ever reads frozen
    //      filesystem config -- so that agent never actually existed.
    //   2. The `permission.ask` hook, overriding the outcome of a
    //      permission check. Confirmed by tracing the source: plugin-
    //      registered tools (session/tools.ts's `registry.tools()` loop)
    //      call `item.execute(args, ctx)` DIRECTLY, with no `ctx.ask(...)`
    //      permission check at all -- unlike MCP tools, which do go through
    //      one. permission.ask never had anything to intercept.
    "tool.execute.before": async (input) => {
      const isOntomemTool = (AGENTIC_TOOL_NAMES as readonly string[]).includes(input.tool)
      const isExtractionSession = DETACHED_EXTRACTION_SESSION_ID !== null
      if (isOntomemTool !== isExtractionSession) {
        throw new Error(
          isOntomemTool
            ? "ontomem_* tools are only available during ontomem's own internal extraction session"
            : "ontomem's internal extraction session can only use ontomem_* tools",
        )
      }
    },

    "chat.message": async (input, output) => {
      // Skip ontomem's own internal sessions entirely -- both the
      // generation-callback session (startCallbackServer's /generate,
      // internalSessionId) and, inside the dedicated detached extraction
      // server, the one extraction session (DETACHED_EXTRACTION_SESSION_ID
      // is set for the whole process there). Confirmed live: without this,
      // the extraction prompt itself -- a ~130k-char internal instruction
      // block, not anything a real user typed -- got sent through /read and
      // cached as "last read," which the viewer's Graph Inspector surfaces
      // as "what OpenAgent just injected." Same bug would also apply to the
      // merge/disambiguate /generate calls on the parent process. This also
      // stops these sessions from ever landing in touchedSessions, which
      // would otherwise make dispose() try to run extraction AGAIN on the
      // extraction/merge session's own transcript.
      if (input.sessionID === internalSessionId || DETACHED_EXTRACTION_SESSION_ID !== null) return
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

      ...agenticExtractionTools,
    },

    dispose: async () => {
      log(`dispose: called, ${touchedSessions.size} touched session(s)`)
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
        // Mark BEFORE dispatch, not after: spawnDetachedWrite hands off to
        // a detached process we don't await, so there's no reliable "write
        // actually finished" signal to hook this on. Marking here means a
        // second dispose moments later (before the first write even
        // reaches the engine) still sees this session as handled.
        markSessionWritten(sessionID, conversation.length)
        // Detached, not awaited, for either mode -- see spawnDetachedWrite's
        // comment for why an in-process await/curl can never survive
        // opencode's own 5-second shutdown ceiling.
        const mode = agenticExtractionEnabled ? "agentic" : "plain"
        log(`dispose: dispatching ${mode} write for session ${sessionID} (${conversation.length} turns)`)
        spawnDetachedWrite(mode, conversation, directory, modelOverride)
      }
      log("dispose: loop finished, stopping callback server")
      callbackServer.stop()
    },
  }
}

export default { id: "ontomem", server: OntomemPlugin }
