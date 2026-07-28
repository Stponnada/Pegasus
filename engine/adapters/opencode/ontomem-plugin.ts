import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"
import { spawn } from "node:child_process"
import { writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

/**
 * Thin OpenCode adapter for the standalone ontomem service.
 *
 * Copy this file into an OpenCode workspace before loading it; keeping the
 * runtime copy inside that workspace lets Bun resolve @opencode-ai/plugin.
 * All memory-engine communication uses the provider-neutral JSON HTTP API.
 */

const SERVICE_URL = process.env.ONTOMEM_URL ?? "http://127.0.0.1:8765"
const READ_TIMEOUT_MS = 4000
const WRITE_FAILURE_LOG = join(tmpdir(), "ontomem-write-failures.log")

async function callService<T>(path: string, body: unknown, timeoutMs = READ_TIMEOUT_MS): Promise<T | null> {
  try {
    const response = await fetch(`${SERVICE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    })
    if (!response.ok) return null
    return (await response.json()) as T
  } catch {
    return null
  }
}

function textOf(parts: Array<{ type: string; text?: string }>): string {
  return parts
    .filter((part) => part.type === "text" && part.text)
    .map((part) => part.text)
    .join("\n")
    .trim()
}

/**
 * Persist asynchronously so exiting OpenCode never waits for extraction.
 * curl retries transient failures; an exhausted payload remains in /tmp for
 * inspection and replay instead of being silently discarded.
 */
function flushWriteInBackground(conversation: unknown): void {
  const payload = join(tmpdir(), `ontomem-write-${Date.now()}-${Math.random().toString(36).slice(2)}.json`)
  writeFileSync(payload, JSON.stringify({ conversation }))
  const command =
    `if curl -fsS --retry 6 --retry-delay 15 --retry-all-errors -X POST '${SERVICE_URL}/write' ` +
    `-H 'Content-Type: application/json' --data-binary @'${payload}' -o /dev/null; then ` +
    `rm -f '${payload}'; else ` +
    `echo "$(date -u +%FT%TZ) write failed after retries, payload kept at ${payload}" >> '${WRITE_FAILURE_LOG}'; fi`
  const child = spawn("sh", ["-c", command], { detached: true, stdio: "ignore" })
  child.unref()
}

export const OntomemPlugin: Plugin = async ({ client }) => {
  const pendingContext = new Map<string, string>()
  const touchedSessions = new Set<string>()

  return {
    "chat.message": async (input, output) => {
      touchedSessions.add(input.sessionID)
      try {
        const message = textOf(output.parts)
        if (!message) return
        const result = await callService<{ text: string }>("/read", { message })
        if (result?.text) pendingContext.set(input.sessionID, result.text)
      } catch {
        // Memory is best-effort and must never block the host conversation.
      }
    },

    "experimental.chat.system.transform": async (input, output) => {
      try {
        const context = input.sessionID ? pendingContext.get(input.sessionID) : undefined
        if (context) output.system.push(context)
      } catch {
        // Continue the turn without memory if injection fails.
      }
    },

    tool: {
      retrieve_memory: tool({
        description:
          "Retrieve deeper long-term memory about a known entity (person, org, place, topic). " +
          "Use when the injected memory context is too thin.",
        args: {
          node_name: tool.schema.string().describe("Entity name to look up, such as 'Sarah' or 'Walmart'"),
          depth: tool.schema.number().optional().describe("Traversal depth; defaults to 1"),
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
          .map((message, index) => ({
            role: message.info.role,
            turn: index + 1,
            text: textOf(message.parts),
          }))
          .filter((turn) => turn.text)
        if (conversation.length > 0) flushWriteInBackground(conversation)
      }
    },
  }
}

export default { id: "ontomem", server: OntomemPlugin }
