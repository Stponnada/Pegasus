#!/usr/bin/env bun
// Detached companion to plugin/src/index.ts's dispose() hook, for BOTH
// write modes (mode: "agentic" | "plain" in the payload).
//
// Why this exists at all: opencode's own TUI shutdown path
// (packages/opencode/src/cli/cmd/tui.ts) wraps the plugin dispose() call in
// `withTimeout(client.call("shutdown", undefined), 5000)`, then calls
// `worker.terminate()` unconditionally -- hard-killing the worker process
// whether or not dispose() (and whatever it awaited) actually finished. A
// real write -- agentic's many tool calls, or even a single plain
// extraction call, both routinely multi-second-to-minutes -- never gets
// anywhere close to finishing inside that 5-second window. So this script
// is spawned detached (outlives the parent opencode process entirely, same
// "spawn detached + unref" shape used elsewhere) and drives its OWN,
// independent `opencode serve` instance for the whole write. Nothing here
// depends on the original opencode process, or its 5-second grace period,
// staying alive.
//
// Why plain mode ALSO needs this, not just a bare curl to /write: the
// engine's generation calls (extraction itself in plain mode; merge/
// disambiguation/supersede decisions in agentic mode) are proxied back
// through a Bun.serve() callback the ORIGINAL opencode process's plugin
// instance runs -- and dispose() stops that callback server (and the
// process exits) essentially immediately after dispatch, before a detached
// curl's request could ever reach the engine and trigger a callback. This
// script sets up its OWN callback proxy on the dedicated server instead
// (see setupOwnCallbackProxy below) and tells the engine to use it via
// /refresh_callback -- see engine.py's Engine.refresh_callback_url for the
// fuller story on why the engine's callback target has to be refreshable
// at all (it's long-lived; the callback target is not).
//
// Invoked as: bun run-detached-write.ts <payload-file>
// payload-file is JSON matching the Payload type below, deleted once read.

import { spawn, type ChildProcess } from "node:child_process"
import { appendFileSync, readFileSync, unlinkSync } from "node:fs"
import { createOpencodeClient } from "@opencode-ai/sdk"

type Payload = {
  mode: "agentic" | "plain"
  conversation: unknown
  directory: string
  modelOverride?: { providerID: string; modelID: string }
  serviceUrl: string
  bootstrapLog: string
}

const READY_POLL_ATTEMPTS = 30
const READY_POLL_INTERVAL_MS = 1000
// Hard safety cap so a stuck write can't leak a zombie `opencode serve`
// forever -- independent of whatever step/tool-call caps the engine and
// opencode's own session runner already enforce.
const WRITE_TIMEOUT_MS = 15 * 60 * 1000

function log(logPath: string, message: string): void {
  try {
    appendFileSync(logPath, `${new Date().toISOString()} [detached-write] ${message}\n`)
  } catch {
    // best-effort logging only
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

function textOf(parts: Array<{ type: string; text?: string }>): string {
  return parts
    .filter((p) => p.type === "text" && p.text)
    .map((p) => p.text)
    .join("\n")
    .trim()
}

async function waitUntilReady(client: ReturnType<typeof createOpencodeClient>): Promise<boolean> {
  for (let attempt = 0; attempt < READY_POLL_ATTEMPTS; attempt++) {
    const res = await client.app.agents().catch(() => null)
    if (res && !res.error) return true
    await sleep(READY_POLL_INTERVAL_MS)
  }
  return false
}

// Mirrors index.ts's startCallbackServer /generate handler exactly, just
// running standalone here instead of inside the main plugin process --
// same contract (POST {prompt} -> {text}), same "create one internal
// session, reuse it for every call" shape.
function setupOwnCallbackProxy(
  client: ReturnType<typeof createOpencodeClient>,
  modelOverride: Payload["modelOverride"],
  bootstrapLog: string,
) {
  let internalSessionId: string | null = null
  const server = Bun.serve({
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
        log(bootstrapLog, `detached callback /generate failed: ${errorMessage(err)}`)
        return Response.json({ error: errorMessage(err) }, { status: 500 })
      }
    },
  })
  return { url: `http://127.0.0.1:${server.port}`, stop: () => server.stop() }
}

async function runPlain(serviceUrl: string, conversation: unknown, bootstrapLog: string): Promise<void> {
  log(bootstrapLog, "calling /write")
  const res = await fetch(`${serviceUrl}/write`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation }),
  })
  if (!res.ok) throw new Error(`/write failed: HTTP ${res.status} ${await res.text().catch(() => "")}`)
  const stats = await res.json()
  log(bootstrapLog, `write finished: ${JSON.stringify(stats)}`)
}

async function runAgentic(
  serviceUrl: string,
  sessionId: string,
  promptText: string,
  client: ReturnType<typeof createOpencodeClient>,
  modelOverride: Payload["modelOverride"],
  directory: string,
  bootstrapLog: string,
): Promise<void> {
  const session = await client.session.create({ body: { title: "ontomem (internal, extraction)" } }).catch(() => null)
  if (!session?.data) throw new Error("failed to create session on dedicated server")

  log(bootstrapLog, `calling session.prompt (session ${session.data.id}, can take minutes)`)
  const result = await client.session.prompt({
    path: { id: session.data.id },
    body: {
      ...(modelOverride ? { model: modelOverride } : {}),
      parts: [{ type: "text", text: promptText }],
    },
  })
  if (!result.data) throw new Error(`session.prompt returned no data: ${JSON.stringify(result.error ?? result)}`)
  log(bootstrapLog, `finished (${JSON.stringify(result.data.parts).length} chars of final response)`)
}

async function main(): Promise<void> {
  const payloadFile = process.argv[2]
  if (!payloadFile) {
    console.error("usage: run-detached-write.ts <payload-file>")
    process.exit(1)
  }
  const payload = JSON.parse(readFileSync(payloadFile, "utf8")) as Payload
  const { mode, conversation, directory, modelOverride, serviceUrl, bootstrapLog } = payload
  unlinkSync(payloadFile)

  // Deliberately requires a real `opencode` on PATH rather than trying to
  // reconstruct "however the parent process was launched" (e.g. from
  // process.argv) -- every real install of this plugin already has one,
  // since that's how a user runs opencode in the first place. Not falling
  // back to a guess here for the same reason the hashing-embedder fallback
  // was rejected elsewhere in this plugin: a silent, worse path is worse
  // than a loud, visible failure.
  const opencodeBin = Bun.which("opencode")
  if (!opencodeBin) {
    log(bootstrapLog, "'opencode' not found on PATH, cannot start a dedicated server -- aborting")
    process.exit(1)
  }

  // Agentic mode needs the engine-side session id BEFORE the dedicated
  // server spawns, since it's passed as an env var fixed at spawn time
  // (see index.ts's ONTOMEM_DETACHED_EXTRACTION_SESSION_ID comment) --
  // /agentic/start itself needs no generation (Stage 0 context assembly is
  // deterministic), so calling it before the callback proxy exists is fine.
  let agenticSessionId: string | null = null
  let agenticPromptText: string | null = null
  if (mode === "agentic") {
    log(bootstrapLog, "calling /agentic/start")
    const startRes = await fetch(`${serviceUrl}/agentic/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation }),
    }).catch(() => null)
    const started =
      startRes && startRes.ok ? ((await startRes.json()) as { session_id: string; prompt_text: string }) : null
    if (!started) {
      log(bootstrapLog, "/agentic/start failed, aborting")
      process.exit(1)
    }
    agenticSessionId = started.session_id
    agenticPromptText = started.prompt_text
    log(bootstrapLog, `/agentic/start ok, session_id=${agenticSessionId}`)
  }

  const port = 20000 + Math.floor(Math.random() * 20000)
  log(bootstrapLog, `launching dedicated 'opencode serve' on port ${port} (mode=${mode})`)
  let server: ChildProcess | null = spawn(opencodeBin, ["serve", "--port", String(port), "--hostname", "127.0.0.1"], {
    cwd: directory,
    stdio: "ignore",
    env: {
      ...process.env,
      // Tells the plugin instance loaded INTO this dedicated server that
      // this whole process is a single-purpose extraction server -- see
      // index.ts's "tool.execute.before" and DETACHED_EXTRACTION_SESSION_ID.
      // Absent (undefined) for plain mode, which registers no ontomem_*
      // tools and needs none of that scoping.
      ...(agenticSessionId ? { ONTOMEM_DETACHED_EXTRACTION_SESSION_ID: agenticSessionId } : {}),
    },
  })
  let callbackProxy: { url: string; stop: () => void } | null = null
  const killAll = () => {
    server?.kill()
    server = null
    callbackProxy?.stop()
    callbackProxy = null
  }

  const client = createOpencodeClient({ baseUrl: `http://127.0.0.1:${port}`, directory })

  const ready = await waitUntilReady(client)
  if (!ready) {
    log(bootstrapLog, "dedicated server never became ready, aborting")
    killAll()
    process.exit(1)
  }
  log(bootstrapLog, "dedicated server ready")

  // Own callback proxy + explicit /refresh_callback, rather than relying on
  // the dedicated server's own plugin instance to bootstrap and refresh the
  // engine's callback on its own schedule (it does, via ensureEngineRunning
  // -- but that's fire-and-forget from this script's perspective, racing
  // against whatever we do next). This makes the ordering deterministic
  // instead of depending on one bootstrap being reliably faster than the
  // other -- confirmed live that "opencode serve" readiness alone (several
  // seconds) is not a reliable proxy for "the engine's callback URL has
  // definitely already been refreshed."
  callbackProxy = setupOwnCallbackProxy(client, modelOverride, bootstrapLog)
  log(bootstrapLog, `callback proxy ready at ${callbackProxy.url}, refreshing engine`)
  await fetch(`${serviceUrl}/refresh_callback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: callbackProxy.url }),
  }).catch((err) => log(bootstrapLog, `refresh_callback failed: ${errorMessage(err)}`))

  const timeout = setTimeout(() => {
    log(bootstrapLog, "hit hard safety timeout, killing dedicated server")
    killAll()
    process.exit(1)
  }, WRITE_TIMEOUT_MS)
  timeout.unref?.()

  try {
    if (mode === "plain") {
      await runPlain(serviceUrl, conversation, bootstrapLog)
    } else {
      await runAgentic(serviceUrl, agenticSessionId!, agenticPromptText!, client, modelOverride, directory, bootstrapLog)
    }
  } catch (err) {
    log(bootstrapLog, `write failed: ${errorMessage(err)}`)
  } finally {
    clearTimeout(timeout)
    killAll()
  }
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err)
    process.exit(1)
  })
