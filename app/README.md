# r3con app

The app's run panel. Served by the local server below, it runs r3con for real: keys go to the server, and a run streams its events into the panel. Once a run has ended, ask (Enter) answers a question over what it built, as another answer node with its own lit path, and new run (⌘/Ctrl+Enter) builds again. Opened as a file (`src/r3con_app/index.html`), it is the prototype: it replays one recorded run on the quickstart's five memos, or simulates one, and answers a few recorded asks over that run; any other ask is simulated. The design notes are on the project's Notion page, under App: the run panel.

`node checks/check.mjs` checks the page in headless Chromium with Playwright: one line per check, and a non-zero exit if any fails. Its live area serves the page with `checks/fake_server.py`, a scripted model behind the real server, so it needs `uv`.

It finds Playwright through `NODE_PATH`, as in Claude's cloud containers. Elsewhere, run `npm i -g playwright && npx playwright install chromium` once, then `NODE_PATH="$(npm root -g)" node checks/check.mjs`.

## The local server

`uv run --project app r3con-app`, from the repo root, serves the page on 127.0.0.1 and opens it with this launch's token in the URL, which every API call needs; `--no-browser` only prints the URL, and `--port` and `--logs-dir` set where it listens and where run folders go. The endpoints are listed at the top of `src/r3con_app/server.py`. A run or an ask can be cancelled there. Stopping the server cancels any still going, so none sends a new model call, and waits up to a minute for what is already under way to finish and write its run folder; Ctrl-C twice quits at once. The page has no stop button yet.

Its checks: `uv run --project app pytest app/tests` and `uv run --project app pyright -p app`; the root's ruff covers `app/`.
