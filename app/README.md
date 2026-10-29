# r3con app

A prototype of the app's run panel, to iterate on its look and feel before it is connected to r3con. It replays one recorded run, on the quickstart's five memos.

Open `web/index.html` in a browser. The design notes are on the project's Notion page, under App: the run panel.

`node checks/check.mjs` checks the page in headless Chromium with Playwright: one line per check, and a non-zero exit if any fails.

It finds Playwright through `NODE_PATH`, as in Claude's cloud containers. Elsewhere, run `npm i -g playwright && npx playwright install chromium` once, then `NODE_PATH="$(npm root -g)" node checks/check.mjs`.
