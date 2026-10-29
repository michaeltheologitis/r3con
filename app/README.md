# r3con app

A prototype of the app's run panel, to iterate on its look and feel before it is connected to r3con. It replays one recorded run, on the quickstart's five memos.

Open `src/r3con_app/index.html` in a browser. The design notes are on the project's Notion page, under App: the run panel.

`node checks/check.mjs` checks the page in headless Chromium with Playwright: one line per check, and a non-zero exit if any fails.

It finds Playwright through `NODE_PATH`, as in Claude's cloud containers. Elsewhere, run `npm i -g playwright && npx playwright install chromium` once, then `NODE_PATH="$(npm root -g)" node checks/check.mjs`.

## The local server

`uv run --project app r3con-app`, from the repo root, serves the page on 127.0.0.1 and opens it with this launch's token in the URL, which every API call needs; `--no-browser` only prints the URL, and `--port` and `--logs-dir` set where it listens and where run folders go. The endpoints are listed at the top of `src/r3con_app/server.py`.

Its checks: `uv run --project app pytest app/tests` and `uv run --project app pyright -p app`; the root's ruff covers `app/`.
