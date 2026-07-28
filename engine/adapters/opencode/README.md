# ontomem ⇄ opencode adapter

This directory contains the distributable adapter source. At runtime, copy
`ontomem-plugin.ts` into the target OpenCode workspace:

```bash
mkdir -p opencode/.opencode/plugins
cp engine/adapters/opencode/ontomem-plugin.ts \
  opencode/.opencode/plugins/ontomem-plugin.ts
```

The local `opencode/` checkout is a separate, ignored Git repository, so its
runtime copy is intentionally not tracked by this parent repository.

## Why the runtime copy belongs inside OpenCode

Bun/Node module resolution for a bare specifier like `@opencode-ai/plugin`
walks up from the *importing file's own directory* looking for `node_modules`.
Starting from this directory (a sibling of `opencode/`, outside its workspace)
it never finds one. The import fails with
`Cannot find module '@opencode-ai/plugin'`, and opencode's plugin loader
reports that failure via an in-chat `Session.Event.Error`, not the file log —
so a misplaced plugin may look as though it loaded but did nothing.

Plugin files placed under `opencode/.opencode/plugins/` (or anywhere else
inside the `opencode/` workspace) resolve fine, because they can walk up to
`opencode/node_modules/@opencode-ai/plugin`. Confirmed by reproducing the
error directly: `bun -e "await import('<path>')"` from each location.

The adapter wires `chat.message`, `experimental.chat.system.transform`,
`tool.retrieve_memory`, and `dispose`. Writes happen once per session at app
exit through a detached process, so `/exit` never blocks on extraction.

## Running

1. Start the engine service (from `engine/`):

   ```bash
   GEMINI_API_KEY=... ONTOMEM_DIR=~/.ontomem uv run python -m ontomem.service
   ```

   The service listens on `http://127.0.0.1:8765` by default
   (`ONTOMEM_HOST` / `ONTOMEM_PORT` to change; `ONTOMEM_URL` on the plugin side).

2. Register the copied plugin in `opencode/.opencode/opencode.jsonc`:

   ```json
   { "plugin": ["./plugins/ontomem-plugin.ts"] }
   ```

3. Run a daily decay cron against the engine:

   ```bash
   curl -X POST http://127.0.0.1:8765/decay
   ```

## Notes

- `bun typecheck` against the plugin file needs the opencode workspace
  (`@opencode-ai/plugin`) installed — it is type-checked there, not in the
  Python engine package. This is also exactly why it must live inside the
  opencode tree at runtime, not just at typecheck time.
