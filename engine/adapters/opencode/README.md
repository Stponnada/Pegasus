# ontomem ⇄ opencode adapter

The distributable adapter is the **`plugin/`** package at the parent repo's
root (sibling to `Documentation/`, `engine/`, `opencode/`). That is now the
single source of truth for the opencode integration — this directory exists
only as a pointer, kept because the design history below (why the adapter
had to live inside the opencode workspace) is still the reason `plugin/`
is shaped as a real npm package rather than a loose `.ts` file.

## Installing it (primary path)

From inside any opencode project:

```bash
opencode plugin pegasus-opencode   # once published to npm
# or, before a publish / for this repo's own dev checkout:
opencode plugin file:///absolute/path/to/plugin
```

That's the whole install. `opencode plugin` (`packages/opencode/src/cli/cmd/plug.ts`
in the opencode source) resolves the package, detects its `exports["./server"]`
target, and patches `.opencode/opencode.json(c)` for you — no manual file
copying, no hand-edited config. From then on:

- No API key: the plugin proxies every extraction/merge/supersede LLM call
  through a hidden opencode session (`client.session.prompt`), so it uses
  whatever provider opencode itself is already authenticated for (e.g. the
  user's own OpenCode Zen/Go subscription). The plugin never holds a model key.
- No separate service to start: on first use the plugin checks for `uv` on
  `PATH`, syncs the bundled Python engine into
  `~/.local/share/ontomem/engine`, runs `uv sync --extra local` once, and
  spawns `uv run python -m ontomem.service` detached. Later opencode
  launches just health-check and reuse it.
- No embeddings API either: embeddings run fully locally (`LocalEmbedder` /
  `fastembed` in the engine) — verified against this fork's own Zen server
  source that no embeddings route exists on the subscription at any tier, so
  local was the only way to avoid a second key. Node/edge text never leaves
  the machine to be embedded.
- Decay runs itself: the plugin fires `POST /decay` once per opencode start;
  the engine gates it to a real no-op if less than 24h have elapsed, so no
  cron is required for this path (a dev/dogfood cluster setup can still run
  one explicitly against `/decay?force`-equivalent — see `engine/README.md`).

See `plugin/README.md` for the package's own details (bootstrap sequence,
config options, generated `plugin/engine/` copy).

## Why the adapter is a real npm package, not a loose file

Bun/Node module resolution for a bare specifier like `@opencode-ai/plugin`
walks up from the *importing file's own directory* looking for
`node_modules`. A loose file living outside any workspace with that
dependency (this directory's old approach: hand-copying a `.ts` file into
`opencode/.opencode/plugins/` specifically so it could see
`opencode/node_modules`) never resolves correctly on its own — and
opencode's plugin loader reports import failures via an in-chat
`Session.Event.Error`, not the file log, so a misplaced plugin can look
exactly like "loaded fine, nothing to recall."

A real npm package sidesteps this permanently: `plugin/package.json`
declares `@opencode-ai/plugin` as its own dependency, so resolution works
from wherever `opencode plugin` installs it — no host-workspace placement
trick needed. Confirmed by loading it directly (`bun -e "await
import('<path>')"`) and by running the actual `opencode plugin
file://...` installer end-to-end against a scratch project.

Related, separately-discovered gotcha, still relevant: the default export
shape for a **server** plugin must be `{ id?, server: Plugin }`, not a bare
exported function (fails `isRecord()` in the loader's validation and is
dropped with no error). This differs from a **TUI** plugin's shape (`{ id?,
tui: TuiPlugin }`, its own `tui.json`, e.g. `opencode/.opencode/plugins/memory-viewer.tsx`)
— the two plugin systems are independent, with different config files and
validation, and any `.ts`/`.js` file under `.opencode/plugins/` is
auto-discovered as a *server* plugin regardless of `tui.json`, so a TUI-only
plugin must use `.tsx` to dodge that glob.

## Developing the plugin locally

Working on `plugin/src/index.ts` itself:

```bash
cd plugin && bun install
bun run scripts/sync-engine.sh   # refresh the bundled engine/ copy from ../engine
bunx tsc --noEmit                # typecheck against the real @opencode-ai/plugin types
```

To exercise it against a real opencode checkout without publishing, register
it by absolute path in that checkout's `.opencode/opencode.json(c)`:

```jsonc
{ "plugin": ["file:///absolute/path/to/plugin"] }
```

(or run `opencode plugin file:///absolute/path/to/plugin` from inside that
project, which does the same edit for you). `plugin/engine/` is generated
(gitignored) — never edit it directly; edit `../engine` and rerun
`sync-engine.sh`, or just re-run `opencode plugin ... --force` after a publish,
since `prepack` reruns the sync automatically.

`bun run scripts/sync-engine.sh` and `bunx tsc --noEmit` need
`plugin/node_modules` installed (`@opencode-ai/plugin`) — that's a real
dependency of the package now, not something checked from the Python engine
package.
