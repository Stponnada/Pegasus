#!/usr/bin/env bun
// Detached companion to plugin/src/index.ts's dispose() hook, for sequential
// (agentic) extraction only.
//
// Why this exists: opencode's own TUI shutdown path
// (packages/opencode/src/cli/cmd/tui.ts) wraps the plugin dispose() call in
// `withTimeout(client.call("shutdown", undefined), 5000)`, then calls
// `worker.terminate()` unconditionally -- hard-killing the worker process
// whether or not dispose() (and whatever it awaited) actually finished.
// Confirmed live: a real sequential extraction (many tool calls, minutes of
// wall-clock time) never gets anywhere close to finishing inside that
// 5-second window, no matter what dispose() itself does -- the process is
// gone before client.session.prompt() can resolve. See index.ts's dispose()
// for the fuller trail (this replaced an earlier version that just awaited
// session.prompt() directly inside dispose(), which this finding ruled out).
//
// So this script is spawned detached (outlives the parent opencode process
// entirely, same "spawn detached + unref" shape flushWriteInBackground uses
// for the non-agentic write path) and drives its OWN, independent
// `opencode serve` instance for the whole extraction. Nothing here depends
// on the original opencode process -- or its 5-second grace period --
// staying alive.
//
// Invoked as: bun run-agentic-extraction.ts <payload-file>
// payload-file is JSON matching the Payload type below, deleted once read.

import { spawn, type ChildProcess } from "node:child_process"
import { appendFileSync, readFileSync, unlinkSync } from "node:fs"
import { createOpencodeClient } from "@opencode-ai/sdk"

type Payload = {
  conversation: unknown
  directory: string
  modelOverride?: { providerID: string; modelID: string }
  serviceUrl: string
  bootstrapLog: string
}

const READY_POLL_ATTEMPTS = 30
const READY_POLL_INTERVAL_MS = 1000
// Hard safety cap so a stuck extraction can't leak a zombie `opencode serve`
// forever -- independent of whatever step/tool-call caps the engine and
// opencode's own session runner already enforce.
const EXTRACTION_TIMEOUT_MS = 15 * 60 * 1000

function log(logPath: string, message: string): void {
  try {
    appendFileSync(logPath, `${new Date().toISOString()} [detached-extraction] ${message}\n`)
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

async function waitUntilReady(client: ReturnType<typeof createOpencodeClient>): Promise<boolean> {
  for (let attempt = 0; attempt < READY_POLL_ATTEMPTS; attempt++) {
    const res = await client.app.agents().catch(() => null)
    if (res && !res.error) return true
    await sleep(READY_POLL_INTERVAL_MS)
  }
  return false
}

async function main(): Promise<void> {
  const payloadFile = process.argv[2]
  if (!payloadFile) {
    console.error("usage: run-agentic-extraction.ts <payload-file>")
    process.exit(1)
  }
  const payload = JSON.parse(readFileSync(payloadFile, "utf8")) as Payload
  const { conversation, directory, modelOverride, serviceUrl, bootstrapLog } = payload
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
    log(bootstrapLog, "detached extraction: 'opencode' not found on PATH, cannot start a dedicated server -- aborting")
    process.exit(1)
  }

  log(bootstrapLog, "detached extraction: calling /agentic/start")
  const startRes = await fetch(`${serviceUrl}/agentic/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation }),
  }).catch(() => null)
  const started =
    startRes && startRes.ok ? ((await startRes.json()) as { session_id: string; prompt_text: string }) : null
  if (!started) {
    log(bootstrapLog, "detached extraction: /agentic/start failed, aborting")
    process.exit(1)
  }
  log(bootstrapLog, `detached extraction: /agentic/start ok, session_id=${started.session_id}`)

  const port = 20000 + Math.floor(Math.random() * 20000)
  log(bootstrapLog, `detached extraction: launching dedicated 'opencode serve' on port ${port}`)
  let server: ChildProcess | null = spawn(opencodeBin, ["serve", "--port", String(port), "--hostname", "127.0.0.1"], {
    cwd: directory,
    stdio: "ignore",
    env: {
      ...process.env,
      // Tells the plugin instance loaded INTO this dedicated server that
      // this whole process is a single-purpose extraction server -- every
      // session on it is the one this script itself creates below, so the
      // ontomem_* tools should treat it as already-scoped (see index.ts's
      // "tool.execute.before" and each tool's execute()) rather than
      // relying on agenticSessionByOpencodeSession, which is only populated
      // by the (never-called-here) old in-process trigger path.
      ONTOMEM_DETACHED_EXTRACTION_SESSION_ID: started.session_id,
    },
  })
  const killServer = () => {
    server?.kill()
    server = null
  }

  const client = createOpencodeClient({ baseUrl: `http://127.0.0.1:${port}`, directory })

  const ready = await waitUntilReady(client)
  if (!ready) {
    log(bootstrapLog, "detached extraction: dedicated server never became ready, aborting")
    killServer()
    process.exit(1)
  }
  log(bootstrapLog, "detached extraction: dedicated server ready, creating session")

  const session = await client.session.create({ body: { title: "ontomem (internal, extraction)" } }).catch(() => null)
  if (!session?.data) {
    log(bootstrapLog, "detached extraction: failed to create session on dedicated server, aborting")
    killServer()
    process.exit(1)
  }

  log(bootstrapLog, `detached extraction: calling session.prompt (session ${session.data.id}, can take minutes)`)
  const timeout = setTimeout(() => {
    log(bootstrapLog, "detached extraction: hit hard safety timeout, killing dedicated server")
    killServer()
    process.exit(1)
  }, EXTRACTION_TIMEOUT_MS)
  timeout.unref?.()

  try {
    const result = await client.session.prompt({
      path: { id: session.data.id },
      body: {
        ...(modelOverride ? { model: modelOverride } : {}),
        parts: [{ type: "text", text: started.prompt_text }],
      },
    })
    if (!result.data) throw new Error(`session.prompt returned no data: ${JSON.stringify(result.error ?? result)}`)
    log(
      bootstrapLog,
      `detached extraction: finished (${JSON.stringify(result.data.parts).length} chars of final response)`,
    )
  } catch (err) {
    log(bootstrapLog, `detached extraction: session.prompt failed: ${errorMessage(err)}`)
  } finally {
    clearTimeout(timeout)
    killServer()
  }
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err)
    process.exit(1)
  })
