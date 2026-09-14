# pegasus-opencode

Ontology-based long-term memory for [opencode](https://opencode.ai). Install
it, and it takes over from there — no API keys, no separate service to run,
no cluster.

## Install

```bash
opencode plugin pegasus-opencode
```

That's it. Have a conversation, `/exit`, start a new one later — opencode
will remember.

## How "no keys" works

- **Generation**: every extraction/merge/supersede call the memory engine
  needs is proxied through a hidden opencode session
  (`client.session.prompt`), so it runs on whatever model/provider opencode
  itself is already authenticated for — your own OpenCode Zen/Go
  subscription, or any other provider you've connected. This plugin never
  holds or asks for a model API key.
- **Embeddings**: run fully locally via `fastembed` (ONNX, no torch) inside
  the Python engine — no embeddings endpoint exists on the Zen/Go
  subscription at any tier, so this was the only way to get semantic
  recall without a second key. Node/edge text never leaves your machine to
  be embedded.
- **The engine service**: a standalone Python process
  (`engine/` in the parent [ontomem](https://github.com/) repo), bundled
  into this package and bootstrapped automatically on first use — see
  "First run" below.

## Requirements

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) on `PATH`
  (a single small binary; it provisions Python itself if needed) — runs the
  engine service.
- `bun` on `PATH` — every write (extraction) is handed off to a detached
  script that needs to run as Bun, regardless of how opencode itself was
  installed (see below for why).
- `opencode` itself on `PATH` — real generation always goes through an
  actual opencode session (see "How 'no keys' works" above), so the write
  script spins up its own dedicated `opencode serve` instance rather than
  depending on the process that triggered it, which opencode's own shutdown
  path can kill within 5 seconds. If you can run `opencode` to begin with,
  this is already satisfied.

If either `bun` or `opencode` isn't found, memory writes fail loudly (in
`~/.local/share/ontomem/bootstrap.log`) rather than silently producing
nothing — but nothing gets remembered from that conversation. Both are
normally already true for anyone who has opencode installed and working;
these aren't extra installs so much as things worth knowing if something
seems to not be sticking.

Local embeddings work on **Apple Silicon, Intel Mac, Linux, and Windows**.
Intel Mac needs a specific pin to get there: `onnxruntime` (fastembed's own
dependency) doesn't consistently ship macOS x86_64 wheels release-to-release
— it dropped them after 1.16.3, brought them back for 1.23.0-1.23.2, then
dropped them again from 1.24 onward. `engine/pyproject.toml`'s `local` extra
pins to that 1.23.x window specifically on that platform (every other
platform is left unconstrained to get the actual latest). If a future
platform genuinely has no working `onnxruntime` at all, the plugin fails
loudly (logged, engine doesn't start) rather than silently degrading to a
worse embedder — memory quality shouldn't be a surprise.

## First run

On first activation the plugin:

1. Checks for `uv`. If missing, memory is disabled with a one-line pointer
   in `~/.local/share/ontomem/bootstrap.log` — install `uv` and restart
   opencode.
2. Copies the bundled engine into `~/.local/share/ontomem/engine` and runs
   `uv sync --extra local` once (downloads the local embedding model, ~90MB
   the first time — can take a couple of minutes on a slow connection).
3. Spawns the engine service in the background and waits for it to report
   healthy.

This happens in the background — opencode itself never waits on it, and the
current conversation just won't have memory context injected until it's
ready. Every later opencode launch just health-checks the already-running
service and reuses it.

## The `/memory` command

Installing this plugin also registers an **"Open Memory Graph"** entry in
opencode's command palette (Ctrl+P) and the `/memory` slash command. It
doesn't turn anything on — the graph viewer ("Pegasus") is already being
served continuously at `http://127.0.0.1:8765/viewer` the whole time the
engine service is running — `/memory` just opens that page in your browser.

## Does the engine run forever?

No. It's spawned detached so it survives you quitting opencode (otherwise
every restart would pay the startup cost again), but it's **not** a system
service — nothing restarts it at boot or keeps it alive indefinitely. It
watches its own request traffic and exits itself after
`ONTOMEM_IDLE_SHUTDOWN_MINUTES` (default **60**) with no activity, freeing
the memory/CPU it was holding. The next time you open opencode and it's
needed, the plugin's own health-check notices it's gone and restarts it
(just the fast "spawn + wait for health" path — the one-time `uv sync` isn't
redone). Set `ONTOMEM_IDLE_SHUTDOWN_MINUTES=0` in your environment before
opencode starts if you'd rather it stayed up permanently (e.g. you're
running a decay cron against it independently).

## Configuration

Optional, in `.opencode/opencode.json(c)`:

```jsonc
{
  "plugin": [
    ["pegasus-opencode", { "model": { "providerID": "opencode", "modelID": "<zen-model-id>" } }]
  ]
}
```

`model` pins ontomem's own internal generation calls to a specific
provider/model instead of inheriting whatever your default coding model is
(useful if you want a cheaper/faster model doing extraction than the one
you're chatting with).

### Sequential extraction (default)

A prior version of this plugin asked the model for the whole graph in one
structured response. A documented failure mode of that approach on longer
conversations: a reasoning model tries to hold the entire graph in its
reasoning trace, re-stating it on every internal revision, and either times
out (15+ minutes observed) or produces a worse graph. Sequential extraction
fixes this by committing each node/edge to the graph the moment the model
decides on it, via real opencode tool calls — the same fix already used by
this project's self-hosted cluster path, now the default in the packaged
plugin too.

Mechanically: on `/exit`, the plugin hands the extraction off to a fully
detached process (survives independently of the opencode session that
triggered it) which spins up its own dedicated `opencode serve` instance —
this is deliberate, not incidental: opencode's own shutdown path only gives
plugin `dispose()` hooks a few seconds before force-killing the process, far
too short for a real multi-step extraction, so the work has to happen
somewhere that isn't racing that timeout. The three extraction tools
(`ontomem_add_entity`/`ontomem_add_relationship`/`ontomem_finish_extraction`)
are only ever registered inside that dedicated process — they don't exist in
your normal coding sessions' tool list at all, not just blocked from
executing there.

Setting `agenticExtraction: false` doesn't revert to some simpler mechanism
— the single-shot extraction call needs the exact same detached dedicated
server (a prior version tried a plain background `curl`, which turned out
to never complete real generation for the same "process dies too soon"
reason). Both modes go through `scripts/run-detached-write.ts`; only what
happens once the dedicated server is ready differs.

```jsonc
{ "plugin": [["pegasus-opencode", { "agenticExtraction": false }]] }
```

Set `agenticExtraction: false` to opt back out to the single-shot path if
you'd rather trade the reliability fix for a shorter (but riskier on long
conversations) extraction window.

## Troubleshooting

- `~/.local/share/ontomem/bootstrap.log` — every step of the first-run
  bootstrap (or why it didn't start) is logged here.
- `curl http://127.0.0.1:8765/health` — confirm the engine service is up.
- `curl http://127.0.0.1:8765/viewer` (or open it in a browser) — the
  "Pegasus" graph viewer: see exactly what's been remembered.

## Uninstalling

Removing the plugin entry from `.opencode/opencode.json(c)` stops new
activity; `~/.local/share/ontomem/` holds all persisted memory (graph
snapshot, journal, embeddings) and the synced engine runtime if you want to
delete it entirely.
