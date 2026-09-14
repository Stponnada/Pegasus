import type { TuiPlugin, TuiPluginModule } from "@opencode-ai/plugin/tui"

/**
 * Adds a "Memory" entry to the command palette (and `/memory`) that opens
 * the live ontomem graph viewer in the browser. Pure side effect, no LLM
 * call -- this is the TUI plugin surface (api.keymap.registerLayer), a
 * separate plugin type from the server Hooks plugin in ./index.ts, since
 * only the TUI surface can add a palette command with an arbitrary onSelect
 * action. Both are exported from this one package (package.json's
 * exports["./server"] and exports["./tui"]) so a single `opencode plugin`
 * install registers both -- opencode's installer detects each target from
 * the package's own exports and patches the right config file for each
 * (opencode.json for server, tui.json for tui).
 *
 * The viewer itself is always being served by the engine's HTTP service at
 * /viewer the whole time it's running (see engine/src/ontomem/service.py) --
 * this command doesn't turn anything "on", it just opens that already-live
 * page in the system browser.
 */

const VIEWER_URL = process.env.ONTOMEM_VIEWER_URL ?? "http://127.0.0.1:8765/viewer"

function openInBrowser(url: string): Promise<boolean> {
  const command = process.platform === "darwin" ? "open" : process.platform === "win32" ? "start" : "xdg-open"
  return new Promise((resolve) => {
    const proc = Bun.spawn([command, url], { stdout: "ignore", stderr: "ignore" })
    proc.exited.then((code) => resolve(code === 0)).catch(() => resolve(false))
  })
}

const tui: TuiPlugin = async (api) => {
  api.keymap.registerLayer({
    commands: [
      {
        name: "ontomem_open_memory",
        title: "Open Memory Graph",
        category: "Plugin",
        namespace: "palette",
        slashName: "memory",
        async run() {
          const ok = await openInBrowser(VIEWER_URL)
          api.ui.toast(
            ok
              ? { variant: "info", title: "Memory", message: `Opened ${VIEWER_URL}`, duration: 2500 }
              : {
                  variant: "error",
                  title: "Memory",
                  message: `Could not open browser. Is the engine running? Visit ${VIEWER_URL} manually.`,
                  duration: 5000,
                },
          )
        },
      },
    ],
    bindings: [],
  })
}

const plugin: TuiPluginModule & { id: string } = {
  id: "ontomem-memory-viewer",
  tui,
}

export default plugin
