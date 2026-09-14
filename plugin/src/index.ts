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

const SERVICE_URL = process.env.ONTOMEM_URL ?? "http://127.0.0.1:8765"
const READ_TIMEOUT_MS = 4000
const HEALTH_TIMEOUT_MS = 2000
const HEALTH_POLL_ATTEMPTS = 60
const HEALTH_POLL_INTERVAL_MS = 2000

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
    const res = await fetch(`${SERVICE_URL}/health`, { signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) })
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

export const OntomemPlugin: Plugin = async ({ client }, options) => {
  // The internal session used only for ontomem's own generation calls (see
  // startCallbackServer) -- never the user's visible coding session. Created
  // lazily on first use, then reused for the process lifetime.
  let internalSessionId: string | null = null
  const modelOverride = options?.model as { providerID: string; modelID: string } | undefined

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
    "chat.message": async (input, output) => {
      touchedSessions.add(input.sessionID)
      try {
        await ensureEngineRunning(callbackUrl)
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
    },

    dispose: async () => {
      for (const sessionID of touchedSessions) {
        const messages = await client.session.messages({ path: { id: sessionID } }).catch(() => null)
        if (!messages?.data) continue
        const conversation = messages.data
          .map((m, i) => ({ role: m.info.role, turn: i + 1, text: textOf(m.parts) }))
          .filter((t) => t.text)
        if (conversation.length === 0) continue
        flushWriteInBackground(conversation)
      }
      callbackServer.stop()
    },
  }
}

export default { id: "ontomem", server: OntomemPlugin }
