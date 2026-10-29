// Checks the run panel in app/src/r3con_app/index.html in headless Chromium: node app/checks/check.mjs
// The live area serves it with app/checks/fake_server.py, through uv.
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { isDeepStrictEqual } from 'node:util';

// An ESM import ignores NODE_PATH, where a global Playwright is found; require honours it.
const require = createRequire(import.meta.url);

const PAGE_FILE = fileURLToPath(new URL('../src/r3con_app/index.html', import.meta.url));
const PAGE_URL = pathToFileURL(PAGE_FILE).href;
const LAYOUTS = {
  desktop: { name: '1440 light', options: { viewport: { width: 1440, height: 900 }, colorScheme: 'light' } },
  phone: {
    name: '400 dark',
    options: {
      viewport: { width: 400, height: 800 },
      colorScheme: 'dark',
      isMobile: true,
      hasTouch: true,
      deviceScaleFactor: 2,
    },
  },
};
const ACTION_TIMEOUT_MS = 10_000;
const SETTLE_TIMEOUT_MS = 2_000;

const SAMPLE_DOCS = 5;
const DEFAULT_MODEL = 'openai/gpt-6-luna';
const COUNTED = {
  nodes: '.node',
  edges: '.edge',
  lit: '.edge.lit',
  dotted: '.edge.self',
  faint: '.edge.context',
  cited: '.edge.cite',
  flow: '.edge.flow',
  onPath: '.node.on-path',
};
const SAMPLE_COUNTS = { nodes: 22, edges: 79, lit: 26, dotted: 10, faint: 45, cited: 14, flow: 10, onPath: 17 };
const SAMPLE_STATUS = '● done · recorded run · gpt-6-luna · 29,490 tokens · $0.0038';
const POPUP_ROWS = {
  'doc-0': ['summary'],
  'r1-0': ['read'],
  'r2-3': ['read', 'drew on'],
  schema: ['why', 'fills'],
  'parsed-0': ['read', 'drew on', 'filled from'],
  answer: ['asked', 'drew on summaries', 'drew on parses'],
};
const MODEL_KEY = 'r3con-proto-model';
const RETIRED_SETTINGS_KEY = 'r3con-proto-settings';
const TEST_KEY = 'sk-check-not-a-real-key-7f3a';
const TEST_ENDPOINT = 'http://10.0.0.7:8000/v1';
const LISTED_MODEL = 'gemini/gemini-3.8-flash';
const OTHER_MODEL = 'hosted_vllm/qwen3-32b';
const NEW_QUESTION = 'Which memo mentions the storm?';
const RECORDED_MODEL = 'gpt-6-luna';
const FIRST_TYPED_MODEL = 'hosted_vllm/foo';
const ready = (kind, model) => `${SAMPLE_DOCS} documents ready · ${kind} · ${model}`;
const FIELD_NAMES = {
  OpenAI: 'key-openai',
  Anthropic: 'key-anthropic',
  Google: 'key-gemini',
  'Other: server URL': 'endpoint-other',
  'Other: key': 'key-other',
};
const LARGE_RUN_DOCS = 40;
const LARGE_RUN_COUNTS = { nodes: 162, edges: 3440 };
// The page's GRID.rowGap: answer nodes stack this far apart.
const ROW_GAP = 112;
const SIMULATED_LABEL = 'simulated — not connected to r3con yet';
const ASK_LABEL = 'Which sites does… · gpt-6-luna';
const RECORDED_ASK = 'recorded ask over this run';
const SIMULATED_ASK = 'simulated ask over this run';
const OTHER_RECORDED_MODEL = 'openai/gpt-6.1-terra';
const NOT_IN_RUN = "1 new document isn't in this run · new run to include it";
const FILE_ADDED_AFTER_RUN = '1 file added. A new run would cover 1 document; ask stays on this run.';
const TYPE_A_MODEL = 'type a model first';
// Its faint lines: each of the 5 round-2 summaries, and the 4 parses with records.
const SAMPLE_ANSWER_INPUTS = 9;
const ANSWERED_RUN_COUNTS = { nodes: SAMPLE_COUNTS.nodes, answers: 1 };
const SHORT_DESKTOP = { width: 1440, height: 450 };
// A model whose name makes an answer's label far wider than the run's own.
const LONG_MODEL = 'hosted_vllm/qwen3-32b-instruct-awq';

const REPO_DIR = fileURLToPath(new URL('../..', import.meta.url));
const FAKE_SERVER = ['run', '--project', 'app', 'python', 'app/checks/fake_server.py'];
const KEY_VARIABLES = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY'];
// A first uv run may have to build the app's environment.
const SERVER_START_MS = 120_000;
const LIVE_RUN_MS = 30_000;
// What fake_server.py's model does: the markers it obeys, its answer, and what it cites.
const FAILS = '(the model is down)';
const HANGS = '(the model hangs)';
const QUIET = '(nothing relevant here)';
const SPLIT_MEMO = '06_split_memo';
const SCRIPTED_ANSWER = "The check's scripted model answers this.";
const SCRIPTED_CITES = {
  r2: Array.from({ length: SAMPLE_DOCS }, (_, i) => ({ provenance: [i === 0 ? 1 : 0] })),
  parsed: Array.from({ length: SAMPLE_DOCS }, (_, i) => ({ provenance: [i] })),
  answer: { provenance: { summary_ids: [0, 1], parse_ids: [0, 1] } },
};
const LIVE_DONE = /^● done · openai\/gpt-6-luna · [\d,]+ tokens · \$[\d.]+$/;
const LIVE_FAILED = "● failed in relevance · RuntimeError: the check's model is down";
const STALE_TOKEN = "this page's launch token is missing or stale; reopen it from the r3con-app terminal";
const LOST_CONNECTION = '● lost the connection to r3con-app';
const UNAUTHORIZED_LOG = 'Failed to load resource: the server responded with a status of 401';
const CUT_STREAM_LOG = 'Failed to load resource: net::ERR_INCOMPLETE_CHUNKED_ENCODING';
const REFUSED_LOG = 'Failed to load resource: the server responded with a status of 422';
const ABORTED_LOG = 'Failed to load resource: net::ERR_FAILED';
const SERVER_ERROR_LOG = 'Failed to load resource: the server responded with a status of 500';
const ASK_WHEN_DONE = 'this run is still going; ask once it has finished';
const RUN_STILL_GOING = "a run is still going; it can't be stopped yet";
const ASK_STILL_GOING = "an ask is still going; it can't be stopped yet";
const LIVE_ASK = 'live ask over this run';
// Long enough that a second click lands while the first save is still in flight.
const SLOW_SAVE_MS = 400;
// fake_server.py's summary_of: a text's first line that is not quiet.
const scriptedSummary = (text) => text.split('\n').find((line) => line.trim() && !line.includes(QUIET)) ?? '';

let browser;
let failures = 0;

function check(area, name, pass, detail = '') {
  if (!pass) failures += 1;
  console.log(`${pass ? 'ok  ' : 'FAIL'}  ${area} · ${name}${detail ? `  (${detail})` : ''}`);
}

const firstLine = (error) => String(error?.message ?? error).split('\n')[0];

async function withPage(area, { layout = LAYOUTS.desktop, url = PAGE_URL, sample = true, expectedErrors = [], before }, body) {
  const messages = [];
  let context;
  try {
    context = await browser.newContext({ reducedMotion: 'reduce', ...layout.options });
    context.setDefaultTimeout(ACTION_TIMEOUT_MS);
    const page = await context.newPage();
    page.on('console', (m) => messages.push({ type: m.type(), text: m.text() }));
    page.on('pageerror', (e) => messages.push({ type: 'error', text: e.message }));
    await before?.(page);
    await page.goto(url);
    if (sample) await runSample(page);
    await body(page, messages);
  } catch (error) {
    check(area, firstLine(error), false);
  } finally {
    await context?.close();
  }
  const errors = messages.filter((m) => ['error', 'warning'].includes(m.type)).map((m) => m.text);
  const unexpected = errors.filter((text) => !expectedErrors.some((start) => text.startsWith(start)));
  const missing = expectedErrors.filter((start) => !errors.some((text) => text.startsWith(start)));
  const beyond = expectedErrors.length ? ` beyond: ${expectedErrors.join('; ')}` : '';
  const detail = [...unexpected.slice(0, 3), ...missing.map((start) => `missing: ${start}`)].join(' | ');
  check(area, `${layout.name}: no console errors or warnings${beyond}`, !unexpected.length && !missing.length, detail);
}

// Clicks save and waits for the settings to say "saved"; false if they never do.
async function saveKeys(page) {
  await page.click('#settings-save');
  return becomes(page, () => document.getElementById('settings-saved').textContent === 'saved');
}

async function runSample(page) {
  await page.click('#run-btn');
  await page.waitForSelector('#panel.is-lit');
}

// Waits for a condition in the page; false if it never holds.
async function becomes(page, predicate, arg, timeout = SETTLE_TIMEOUT_MS) {
  try {
    await page.waitForFunction(predicate, arg, { timeout });
    return true;
  } catch (error) {
    if (error.name !== 'TimeoutError') throw error;
    return false;
  }
}

const nodeSel = (id) => `.node[data-id="${id}"]`;
const count = (page, selector) => page.$$eval(selector, (els) => els.length);
const litEdges = (page) => page.$$eval('.edge.lit', (els) => els.map((e) => e.dataset.key).sort());
const litAnswers = (page) => page.$$eval('.node.answer.on-path', (els) => els.map((el) => el.dataset.id));
const answerIds = (page) => page.$$eval('.node.answer', (els) => els.map((el) => el.dataset.id));
const docIdsOf = (page) => page.$$eval('.node.doc', (els) => els.map((el) => el.dataset.id));
const rect = (page, selector) => page.$eval(selector, (el) => el.getBoundingClientRect().toJSON());
const transform = (page) => page.$eval('#world', (el) => el.style.transform);
const sameSpot = (a, b) => Boolean(a && b) && a.left === b.left && a.top === b.top;

function positionOf(page, id) {
  return page.$eval(nodeSel(id), (el) => ({ x: parseFloat(el.style.left), y: parseFloat(el.style.top) }));
}

async function centre(page, id) {
  const r = await rect(page, `${nodeSel(id)} .icon`);
  return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
}

async function drag(page, from, to) {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 8 });
  await page.mouse.up();
}

async function dragNode(page, id, dx, dy) {
  const from = await centre(page, id);
  await drag(page, from, { x: from.x + dx, y: from.y + dy });
}

async function clickNode(page, id) {
  const c = await centre(page, id);
  await page.mouse.click(c.x, c.y);
}

// Opens node `id`'s details, from a closed popup, and returns the text of `selector` in them.
async function inspect(page, id, selector = '#inspector') {
  await page.keyboard.press('Escape');
  await page.click(nodeSel(id));
  return page.textContent(selector);
}

async function rowsOf(page, id) {
  await inspect(page, id);
  return page.$$eval('#inspector .row', (rows) => rows.map((r) => r.firstChild.textContent));
}

function popup(page) {
  return page.evaluate(() => {
    const el = document.getElementById('inspector');
    if (el.hidden) return null;
    const r = el.getBoundingClientRect();
    const [left, top, right, bottom] = [r.left, r.top, r.right, r.bottom].map(Math.round);
    return { left, top, right, bottom, title: el.querySelector('.insp-title').textContent };
  });
}

function recordedSample() {
  const match = readFileSync(PAGE_FILE, 'utf8').match(/const SAMPLE = (\{[\s\S]*?\n\});/);
  return match ? JSON.parse(match[1]) : null;
}

function recordedAsks() {
  const match = readFileSync(PAGE_FILE, 'utf8').match(/const RECORDED_ASKS = (\[[\s\S]*?\n\]);/);
  return match ? JSON.parse(match[1]) : [];
}

// The edges an answer's lit path holds: the run's own answer by default, or `answerId` with `provenance`.
function lineage(sample, docIds, answerId = 'answer', provenance = sample.answer.provenance) {
  const { summary_ids, parse_ids } = provenance;
  const cites = (id) => {
    const [kind, i] = id.split('-');
    if (kind === 'answer') return [...summary_ids.map((d) => `r2-${d}`), ...parse_ids.map((d) => `parsed-${d}`)];
    if (kind === 'parsed') return sample.parsed[i].provenance.map((d) => `r2-${d}`);
    if (kind === 'r2') return sample.r2[i].provenance.map((d) => `r1-${d}`);
    return [];
  };
  const ownDocument = (id) => {
    const [kind, i] = id.split('-');
    return ['r1', 'r2', 'parsed'].includes(kind) ? docIds[i] : null;
  };
  const nodes = new Set([answerId]);
  const edges = new Set();
  for (const id of nodes) {
    for (const from of [ownDocument(id), ...cites(id)].filter(Boolean)) {
      nodes.add(from);
      edges.add(`${from}>${id}`);
    }
  }
  return [...edges].sort();
}

// In-page functions, passed to page.evaluate and page.waitForFunction.

function dottedLineHits() {
  const world = document.getElementById('world');
  const k = parseFloat(world.style.getPropertyValue('--k'));
  const origin = world.getBoundingClientRect();
  const toWorld = (r) => ({
    x0: (r.left - origin.left) / k,
    x1: (r.right - origin.left) / k,
    y0: (r.top - origin.top) / k,
    y1: (r.bottom - origin.top) / k,
  });
  const boxOf = (id, what) => (part) => ({ id, what, box: toWorld(part.getBoundingClientRect()) });
  const boxes = [...document.querySelectorAll('.node')].flatMap((el) => [
    boxOf(el.dataset.id, 'icon')(el.querySelector('.icon')),
    ...[...el.querySelectorAll('.label')].map(boxOf(el.dataset.id, 'label')),
  ]);
  const inside = (p, box) => p.x > box.x0 && p.x < box.x1 && p.y > box.y0 && p.y < box.y1;
  const hits = [];
  for (const edge of document.querySelectorAll('.edge.self')) {
    const ends = edge.dataset.key.split('>');
    const length = edge.getTotalLength();
    const points = Array.from({ length: Math.ceil(length) + 1 }, (_, i) => edge.getPointAtLength(Math.min(length, i)));
    const ownEnd = ({ id, what }) => what === 'icon' && ends.includes(id);
    const crossed = boxes.filter((b) => !ownEnd(b) && points.some((p) => inside(p, b.box)));
    hits.push(...crossed.map(({ id, what }) => `${edge.dataset.key} through ${id} ${what}`));
  }
  return { k, dotted: document.querySelectorAll('.edge.self').length, hits };
}

function countAnimationFrames() {
  window.animationFrames = 0;
  const request = window.requestAnimationFrame;
  window.requestAnimationFrame = (callback) => {
    window.animationFrames += 1;
    return request(callback);
  };
}

function nodeIsAt([selector, x, y]) {
  const el = document.querySelector(selector);
  return parseFloat(el.style.left) === x && parseFloat(el.style.top) === y;
}

function popupIsAt([left, top]) {
  const el = document.getElementById('inspector');
  const r = el.getBoundingClientRect();
  return !el.hidden && Math.round(r.left) === left && Math.round(r.top) === top;
}

function viewChanged(previous) {
  return document.getElementById('world').style.transform !== previous;
}

function popupInsidePanel() {
  const el = document.getElementById('inspector');
  const r = el.getBoundingClientRect();
  const p = document.getElementById('panel').getBoundingClientRect();
  return !el.hidden && r.left >= p.left && r.top >= p.top && r.right <= p.right && r.bottom <= p.bottom;
}

// The stand-in saveKey adds the provider to a Set, so refusing that one add acts as a server refusing the key.
function refuseKeyOnce(provider) {
  const add = Set.prototype.add;
  Set.prototype.add = function (value) {
    if (value !== provider) return add.call(this, value);
    Set.prototype.add = add;
    throw new Error('the server refused the key');
  };
}

function countStatusWrites() {
  window.statusWrites = 0;
  const observer = new MutationObserver((records) => (window.statusWrites += records.length));
  observer.observe(document.getElementById('run-status'), { childList: true, characterData: true, subtree: true });
}

function placesHolding(key) {
  const fields = [...document.querySelectorAll('input, textarea')].filter((el) => el.value.includes(key));
  const holding = (store) => Object.keys(store).filter((k) => store.getItem(k).includes(key));
  const stores = [localStorage, sessionStorage].flatMap(holding);
  const markup = document.documentElement.outerHTML.includes(key) ? ['markup'] : [];
  const address = location.href.includes(key) ? ['location'] : [];
  return [...fields.map((el) => `#${el.id}`), ...stores, ...markup, ...address];
}

function pickerState() {
  const select = document.getElementById('model-pick');
  return {
    value: select.value,
    groups: [...select.querySelectorAll('optgroup')].map((g) => g.label),
    models: [...select.options].map((o) => o.value),
    hint: !document.getElementById('model-hint').hidden,
  };
}

function keyStates() {
  return [...document.querySelectorAll('[id^="state-"]')].filter((el) => el.textContent).map((el) => el.id);
}

function statusIs(text) {
  return document.getElementById('run-status').textContent === text;
}

function statusMatches(source) {
  return new RegExp(source).test(document.getElementById('run-status').textContent);
}

function lastTurnLines() {
  const turn = [...document.querySelectorAll('.turn')].at(-1);
  return turn ? [...turn.querySelectorAll('.line .t')].map((el) => el.textContent) : [];
}

function lastTurnSays(starts) {
  const turn = [...document.querySelectorAll('.turn')].at(-1);
  const lines = turn ? [...turn.querySelectorAll('.line .t')].map((el) => el.textContent) : [];
  return starts.every((start) => lines.some((text) => text.startsWith(start)));
}

// Cuts every event stream the page reads into chunks of `size` bytes.
function cutEventStreams(size) {
  const realFetch = window.fetch;
  window.fetch = async (...args) => {
    const response = await realFetch(...args);
    if (!response.headers.get('content-type')?.startsWith('text/event-stream')) return response;
    const reader = response.body.getReader();
    const body = new ReadableStream({
      async pull(controller) {
        const { value, done } = await reader.read();
        if (done) return controller.close();
        for (let i = 0; i < value.length; i += size) controller.enqueue(value.slice(i, i + size));
      },
    });
    return new Response(body, { status: response.status, headers: response.headers });
  };
}

// Holds back each event stream from its `type` frame on, until the page sets window.release.
function holdFrom(type) {
  const realFetch = window.fetch;
  const wait = () => new Promise((resolve) => setTimeout(resolve, 50));
  window.fetch = async (...args) => {
    const response = await realFetch(...args);
    if (!response.headers.get('content-type')?.startsWith('text/event-stream')) return response;
    const reader = response.body.getReader();
    const body = new ReadableStream({
      async pull(controller) {
        const { value, done } = await reader.read();
        if (done) return controller.close();
        const text = new TextDecoder().decode(value);
        if (!text.includes(`"${type}"`)) return controller.enqueue(value);
        const end = text.lastIndexOf('data: ', text.indexOf(`"${type}"`));
        controller.enqueue(new TextEncoder().encode(text.slice(0, end)));
        while (!window.release) await wait();
        controller.enqueue(new TextEncoder().encode(text.slice(end)));
      },
    });
    return new Response(body, { status: response.status, headers: response.headers });
  };
}

function answerNodes() {
  return [...document.querySelectorAll('.node.answer')].map((el) => ({
    id: el.dataset.id,
    x: parseFloat(el.style.left),
    y: parseFloat(el.style.top),
    box: el.getBoundingClientRect().toJSON(),
    icon: el.querySelector('.icon').getBoundingClientRect().toJSON(),
  }));
}

function panelBox() {
  return document.getElementById('panel').getBoundingClientRect().toJSON();
}

function composer() {
  const note = document.getElementById('ask-note');
  return {
    primary: document.getElementById('run-btn').textContent,
    newRun: !document.getElementById('new-run-btn').hidden,
    note: note.hidden ? null : note.textContent,
  };
}

// The answer node each line into an answer ends at: faint lines from what it was shown, cite lines from what it cites.
function answerLines() {
  const ends = (kind) => [...document.querySelectorAll(`.edge.${kind}`)].map((e) => e.dataset.key.split('>')[1]).filter((to) => to.startsWith('answer'));
  return { faint: ends('context'), cited: ends('cite') };
}

function pressedAnswers() {
  return [...document.querySelectorAll('.node.answer')].map((el) => [el.dataset.id, el.getAttribute('aria-pressed')]);
}

function recordStatuses() {
  window.statuses = [];
  const status = document.getElementById('run-status');
  new MutationObserver(() => window.statuses.push(status.textContent)).observe(status, { childList: true, characterData: true, subtree: true });
}

function landedNode(id) {
  const el = document.querySelector(`.node[data-id="${id}"]`);
  return Boolean(el) && !el.classList.contains('working') && !el.classList.contains('failed');
}

function storedEntries() {
  return Object.fromEntries(Object.entries(localStorage));
}

function hasFilledIcons(els) {
  return els.map((el) => Boolean(el.querySelector('.fill-orange')));
}

function layoutSnapshot() {
  const nodes = [...document.querySelectorAll('.node')].map((el) => `${el.style.left},${el.style.top}`);
  return [document.getElementById('world').style.transform, ...nodes].join(' ');
}

function fitOfRun() {
  const panel = document.getElementById('panel').getBoundingClientRect();
  const isOutside = (el) => {
    const r = el.getBoundingClientRect();
    return r.right < panel.left || r.left > panel.right || r.bottom < panel.top || r.top > panel.bottom;
  };
  const outside = [...document.querySelectorAll('.node .icon')].filter(isOutside).length;
  return { outside, zoom: document.getElementById('zoom-level').textContent };
}

// ---------- areas ----------

async function load() {
  for (const layout of Object.values(LAYOUTS)) {
    await withPage('load', { layout, sample: false }, async (page) => {
      const [scrollWidth, clientWidth] = await page.evaluate(() => {
        const { scrollWidth: scroll, clientWidth: client } = document.documentElement;
        return [scroll, client];
      });
      const noScroll = scrollWidth <= clientWidth;
      check('load', `${layout.name}: the page has no horizontal scroll`, noScroll, `${scrollWidth} ≤ ${clientWidth}`);
    });
  }
}

async function beforeRun() {
  await withPage('before a run', { sample: false }, async (page) => {
    const docs = await count(page, '.node.doc');
    const onlyDocs = docs === SAMPLE_DOCS && (await count(page, '.node')) === docs;
    check('before a run', `the ${SAMPLE_DOCS} sample documents are the only nodes`, onlyDocs);
    const hint = await page.textContent('#panel-hint');
    const hintText = `${SAMPLE_DOCS} documents · ready to run`;
    check('before a run', 'the hint says the documents are ready to run', hint === hintText, hint);
    const status = await page.textContent('#run-status');
    const named = status === ready('recorded run', RECORDED_MODEL);
    check('before a run', "the status names a recorded run and the recording's model", named, status);
    const offered = await page.evaluate(composer);
    const onlyRun = isDeepStrictEqual(offered, { primary: 'run ⏎', newRun: false, note: null });
    check('before a run', 'only run is offered: no new run, and no footer note', onlyRun, JSON.stringify(offered));
    await page.evaluate(countStatusWrites);
    await page.focus('#question');
    await page.keyboard.type(' today');
    const writes = await page.evaluate(() => window.statusWrites);
    const once = writes === 1 && (await page.textContent('#run-status')) === ready('simulated run', DEFAULT_MODEL);
    check('before a run', 'typing a question rewrites the status once, not on every key', once, `${writes} writes`);
  });
}

async function sampleReplay() {
  await withPage('sample replay', {}, async (page) => {
    const counts = {};
    for (const [name, selector] of Object.entries(COUNTED)) counts[name] = await count(page, selector);
    const { nodes, edges: edgeCount, lit: litCount } = SAMPLE_COUNTS;
    const drawn = `the sample draws ${nodes} nodes and ${edgeCount} edges, ${litCount} of them lit`;
    check('sample replay', drawn, isDeepStrictEqual(counts, SAMPLE_COUNTS), JSON.stringify(counts));

    const edges = await page.$$eval('.edge', (els) => {
      return Object.fromEntries(els.map((e) => [e.dataset.key, [...e.classList]]));
    });
    const docIds = await docIdsOf(page);
    const sample = recordedSample();
    check('sample replay', 'the page holds a recorded sample', Boolean(sample));
    const lit = Object.keys(edges).filter((k) => edges[k].includes('lit')).sort();
    const expected = sample ? lineage(sample, docIds) : [];
    const litName = 'the lit edges are exactly the lineage computed from the recorded sample';
    const litDetail = `${lit.length} lit, ${expected.length} computed`;
    check('sample replay', litName, Boolean(sample) && isDeepStrictEqual(lit, expected), litDetail);

    const dotted = Object.keys(edges).filter((k) => edges[k].includes('self')).sort();
    const ownLines = docIds.flatMap((doc, i) => [`${doc}>r2-${i}`, `${doc}>parsed-${i}`]).sort();
    const dottedName = 'every dotted line runs from a document to its own round 2 and parse';
    check('sample replay', dottedName, isDeepStrictEqual(dotted, ownLines));
    const intoStructure = Object.keys(edges).filter((k) => k.endsWith('>schema'));
    const allFaint = intoStructure.length === SAMPLE_DOCS && intoStructure.every((k) => edges[k].includes('context'));
    const faintName = `the structure gets a faint line from each of the ${SAMPLE_DOCS} round-2 summaries`;
    check('sample replay', faintName, allFaint, intoStructure.join(' '));
    const structureOnPath = await page.$eval(nodeSel('schema'), (el) => el.classList.contains('on-path'));
    check('sample replay', 'the structure is not on the lit path', !structureOnPath);
    const filled = await page.$$eval('.node.parsed', hasFilledIcons);
    const filledName = "the parses with records have filled icons and the cafeteria's has none";
    const recordsAt = [true, true, true, true, false];
    check('sample replay', filledName, isDeepStrictEqual(filled, recordsAt), JSON.stringify(filled));
    const status = await page.textContent('#run-status');
    check('sample replay', "the status gives the run's tokens and cost", status === SAMPLE_STATUS, status);

    const geometry = await page.evaluate(dottedLineHits);
    const hitsNote = geometry.hits.length ? `; ${geometry.hits.slice(0, 3).join(', ')}` : '';
    const clear = geometry.k > 0 && geometry.dotted > 0 && geometry.hits.length === 0;
    const geometryDetail = `${geometry.dotted} dotted lines at zoom ${geometry.k?.toFixed(2)}${hitsNote}`;
    check('geometry', 'no dotted line passes through an icon or a label', clear, geometryDetail);
  });
}

async function arrange() {
  await withPage('arrange', {}, async (page) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.evaluate(countAnimationFrames);
    await page.click('#arrange-btn');
    const framesOnTidy = await page.evaluate(() => window.animationFrames);
    check('arrange', 'arrange on a tidy layout starts no animation', framesOnTidy === 0);
    const home = await positionOf(page, 'r2-1');
    await dragNode(page, 'r2-1', 60, 40);
    const dragged = await positionOf(page, 'r2-1');
    await page.click('#arrange-btn');
    const back = await becomes(page, nodeIsAt, [nodeSel('r2-1'), home.x, home.y]);
    const frames = await page.evaluate(() => window.animationFrames);
    const moved = dragged.x !== home.x || dragged.y !== home.y;
    const [dx, dy] = [dragged.x, dragged.y].map(Math.round);
    const detail = `${home.x},${home.y} → ${dx},${dy}, back in ${frames} frames`;
    check('arrange', 'arrange slides a dragged node back to where it was', moved && back && frames > 0, detail);
  });
}

async function files() {
  await withPage('files', {}, async (page) => {
    const memo = { name: '06_extra_memo.txt', mimeType: 'text/plain', buffer: Buffer.from('An extra memo.\n') };
    await page.setInputFiles('#file-input', memo);
    await page.waitForSelector('.node.pending');
    const labels = await page.$$eval('.node.pending .label', (els) => els.map((el) => el.textContent));
    const docs = await count(page, '.node.doc');
    const shown = isDeepStrictEqual(labels, ['06_extra_memo', 'not run']) && docs === SAMPLE_DOCS + 1;
    check('files', 'a file added after a run shows as not run', shown, labels.join(' · '));
    const hint = await page.textContent('#panel-hint');
    check('files', 'the hint counts it as not run yet', hint === '1 new document · not run yet', hint);
    await page.click('#tab-files');
    await page.click('#sample-btn');
    const removed = await becomes(page, () => !document.querySelector('.node.pending'));
    const hintHidden = await page.$eval('#panel-hint', (el) => el.hidden);
    const sampleOnly = removed && hintHidden && (await count(page, '.node.doc')) === SAMPLE_DOCS;
    check('files', 'using the sample memos removes it', sampleOnly);
  });
}

async function settings() {
  for (const layout of Object.values(LAYOUTS)) {
    const checkAt = (name, pass, detail) => check('settings', `${layout.name}: ${name}`, pass, detail);
    const expectedErrors = ["couldn't save the keys"];
    await withPage('settings', { layout, sample: false, expectedErrors }, async (page, messages) => {
      await keys(page, checkAt);
      await picker(page, checkAt);
      const leaks = messages.filter((m) => m.text.includes(TEST_KEY)).length;
      checkAt('no console message holds the key', leaks === 0, `${messages.length} messages`);
    });
  }
}

async function keys(page, checkAt) {
  const states = () => page.evaluate(keyStates);
  const saved = () => page.textContent('#settings-saved');
  const blocked = async () => (await saved()) === '' && (await states()).length === 0;
  await page.evaluate((key) => localStorage.setItem(key, '{"model":"hosted_vllm/old"}'), RETIRED_SETTINGS_KEY);
  await page.reload();
  const retired = await page.evaluate((key) => localStorage.getItem(key), RETIRED_SETTINGS_KEY);
  checkAt('loading removes the old settings entry', retired === null, retired);
  const fresh = await page.evaluate(pickerState);
  const defaultOnly = { value: DEFAULT_MODEL, groups: [], models: [DEFAULT_MODEL, ''], hint: true };
  const freshName = 'with no keys the picker shows the default model, other model… and the hint';
  checkAt(freshName, isDeepStrictEqual(fresh, defaultOnly), JSON.stringify(fresh));

  await page.click(nodeSel('doc-0'));
  await page.click('#settings-btn');
  const named = [];
  for (const [name, id] of Object.entries(FIELD_NAMES)) {
    const ids = await page.getByRole('textbox', { name, exact: true }).evaluateAll((els) => els.map((el) => el.id));
    if (isDeepStrictEqual(ids, [id])) named.push(name);
  }
  const allNamed = named.length === Object.keys(FIELD_NAMES).length;
  checkAt('each settings field is named for what it holds', allNamed, named.join(', '));
  await page.fill('#endpoint-other', 'localhost:8000/v1');
  await page.click('#settings-save');
  checkAt('an endpoint without http(s) blocks the save', await blocked());
  await page.fill('#endpoint-other', '');
  await page.fill('#key-other', TEST_KEY);
  await page.click('#settings-save');
  checkAt('an Other key without an endpoint blocks the save', await blocked());
  await page.fill('#key-other', '');
  await page.fill('#key-openai', TEST_KEY);
  await page.evaluate(refuseKeyOnce, 'openai');
  await page.click('#settings-save');
  const refused = await becomes(page, () => document.getElementById('settings-saved').textContent === "OpenAI · couldn't save");
  const keptKey = (await page.inputValue('#key-openai')) === TEST_KEY && (await states()).length === 0;
  checkAt('a refused save says so and keeps the typed key', refused && keptKey, await saved());

  await page.fill('#endpoint-other', TEST_ENDPOINT);
  await page.fill('#key-gemini', TEST_KEY);
  const savedKeys = await saveKeys(page);
  const set = await states();
  const marked = savedKeys && isDeepStrictEqual(set, ['state-openai', 'state-gemini', 'state-other']);
  checkAt('a save marks OpenAI, Google and Other as set', marked, set.join(', '));
  const keyed = await page.evaluate(pickerState);
  const noAnthropic = !keyed.models.some((model) => model.startsWith('anthropic/'));
  const onlyKeyed = isDeepStrictEqual(keyed.groups, ['OpenAI', 'Google']) && noAnthropic && !keyed.hint;
  checkAt('the picker offers only the providers with a key', onlyKeyed, keyed.groups.join(', '));
  await page.click('#settings-save');
  const nothing = await saved();
  const unchanged = isDeepStrictEqual(await states(), set);
  checkAt('a save with no field filled says so', nothing === 'nothing to save' && unchanged, nothing);
  await page.fill('#key-anthropic', 'x');
  checkAt('an edit clears the last save message', (await saved()) === '', await saved());

  await page.keyboard.press('Escape');
  const settingsOpen = await page.$eval('#settings', (el) => el.matches(':popover-open'));
  checkAt('Escape closes the settings but not the popup', !settingsOpen && Boolean(await popup(page)));
  const holding = (value) => page.evaluate(placesHolding, value);
  const holders = [...(await holding(TEST_KEY)), ...(await holding(TEST_ENDPOINT))];
  const notKept = 'once closed, no field, storage, address or markup holds a key or the endpoint';
  checkAt(notKept, holders.length === 0, holders.join(', '));
  await page.click('#settings-btn');
  const filled = await page.$$eval('#settings input', (els) => els.filter((el) => el.value).map((el) => el.id));
  const reopened = filled.length === 0 && isDeepStrictEqual(await states(), set);
  checkAt('reopened, every field is empty and the providers are still set', reopened, filled.join(', '));
  await page.keyboard.press('Escape');
}

async function picker(page, checkAt) {
  const status = () => page.textContent('#run-status');
  const stored = () => page.evaluate(storedEntries);
  const leaveField = () => page.$eval('#model-other', (el) => el.blur());
  const recordedReady = ready('recorded run', RECORDED_MODEL);
  await page.selectOption('#model-pick', LISTED_MODEL);
  const recordedCue = await status();
  const recordedName = "before a recorded run the status names the recording's model, not the picked one";
  checkAt(recordedName, recordedCue === recordedReady, recordedCue);

  await page.selectOption('#model-pick', '');
  await page.fill('#model-other', FIRST_TYPED_MODEL);
  await leaveField();
  await page.click('#settings-btn');
  await page.fill('#key-anthropic', TEST_KEY);
  const savedKeys = await saveKeys(page);
  await page.keyboard.press('Escape');
  const afterSave = await page.evaluate(pickerState);
  const field = await page.$eval('#model-other', (el) => (el.hidden ? null : el.value));
  const shownOnce = field === FIRST_TYPED_MODEL && !afterSave.models.includes(FIRST_TYPED_MODEL);
  const stays = savedKeys && afterSave.value === '' && shownOnce && afterSave.groups.includes('Anthropic');
  checkAt('other model… stays chosen through a key save, its model shown once', stays, JSON.stringify(afterSave));

  await page.fill('#model-other', '');
  await leaveField();
  await page.fill('#question', NEW_QUESTION);
  await page.click('#run-btn');
  const asked = await status();
  const focused = await page.evaluate(() => document.activeElement.id);
  const questionKept = (await page.inputValue('#question')) === NEW_QUESTION && (await count(page, '.turn')) === 0;
  const kept = await stored();
  const lastChoice = isDeepStrictEqual(kept, { [MODEL_KEY]: FIRST_TYPED_MODEL });
  const waits = asked === TYPE_A_MODEL && focused === 'model-other' && questionKept && lastChoice;
  const waitsName = 'a simulated run with no model asks for one, keeps the question and the last choice';
  checkAt(waitsName, waits, `${asked}; focus on #${focused}`);
  await page.keyboard.type('abc');
  await page.keyboard.press('Backspace');
  const whileTyping = await stored();
  await page.fill('#model-other', OTHER_MODEL);
  await leaveField();
  const typed = await stored();
  const onLeave = isDeepStrictEqual(whileTyping, kept) && isDeepStrictEqual(typed, { [MODEL_KEY]: OTHER_MODEL });
  const onLeaveName = 'a typed model is stored once the field is left, not while typing, and alone';
  checkAt(onLeaveName, onLeave, `${JSON.stringify(whileTyping)} → ${JSON.stringify(typed)}`);

  const simulatedCue = await status();
  const cues = recordedCue === recordedReady && simulatedCue === ready('simulated run', OTHER_MODEL);
  const cueName = 'the status says whether a run replays the recording or is simulated';
  checkAt(cueName, cues, `${recordedCue} → ${simulatedCue}`);
  await page.click('#run-btn');
  const done = `● done · simulated run · ${OTHER_MODEL} · nothing was sent`;
  const namesModel = await becomes(page, (text) => document.getElementById('run-status').textContent === text, done);
  checkAt("a simulated run's status names the typed model", namesModel, await status());

  await page.reload();
  const after = await page.evaluate(pickerState);
  const keysGone = after.groups.length === 0 && after.hint && (await page.evaluate(keyStates)).length === 0;
  const survives = after.value === OTHER_MODEL && keysGone && (await status()) === recordedReady;
  checkAt('after a reload the choice is kept and the keys are not', survives, JSON.stringify(after));
  await page.selectOption('#model-pick', '');
  await page.click('#run-btn');
  const statusIs = (text) => document.getElementById('run-status').textContent === text;
  const replayed = await becomes(page, statusIs, SAMPLE_STATUS);
  checkAt('with other model… left empty the recorded replay still starts', replayed, await status());
}

async function popupArea() {
  await withPage('popup', {}, async (page) => {
    for (const [id, expected] of Object.entries(POPUP_ROWS)) {
      const rows = await rowsOf(page, id);
      check('popup', `${id} shows ${expected.join(', ')}`, isDeepStrictEqual(rows, expected), rows.join(', '));
    }
    await page.keyboard.press('Escape');

    const panel = await rect(page, '#panel');
    await page.click(nodeSel('schema'));
    const before = await popup(page);
    const head = await rect(page, '#inspector .insp-head');
    const width = before.right - before.left;
    const target = { left: Math.round(panel.right - width - 10), top: Math.round(panel.top + 10) };
    const grab = { x: head.left + 40, y: head.top + head.height / 2 };
    await drag(page, grab, { x: grab.x + target.left - before.left, y: grab.y + target.top - before.top });
    const dropped = await popup(page);
    const moved = `${before.left},${before.top} → ${dropped.left},${dropped.top}`;
    check('popup', 'dragging its header moves it', sameSpot(dropped, target), moved);

    await clickNode(page, 'r1-1');
    const first = await popup(page);
    await clickNode(page, 'doc-3');
    const second = await popup(page);
    const titles = [first.title, second.title];
    const selected = isDeepStrictEqual(titles, ['● 02_eastfield · round 1', '● 04_contractor_registry.txt']);
    const stayed = selected && sameSpot(first, dropped) && sameSpot(second, dropped);
    check('popup', 'it stays where it was dropped while other nodes are selected', stayed, titles.join(' · '));

    const beforePan = await transform(page);
    await drag(page, { x: panel.left + 60, y: panel.top + 60 }, { x: panel.left + 160, y: panel.top + 120 });
    const afterPan = await transform(page);
    const panName = 'a pan from outside it moves the view and leaves it where it was';
    check('popup', panName, afterPan !== beforePan && sameSpot(await popup(page), dropped), afterPan);
    await page.mouse.move(panel.left + 100, panel.top + 100);
    await page.keyboard.down('Control');
    await page.mouse.wheel(0, -300);
    await page.keyboard.up('Control');
    const zoomed = await becomes(page, viewChanged, afterPan);
    const zoomName = 'a zoom from outside it moves the view and leaves it where it was';
    check('popup', zoomName, zoomed && sameSpot(await popup(page), dropped), await page.textContent('#zoom-level'));

    for (const [width, height] of [[1000, 650], [700, 600]]) {
      await page.setViewportSize({ width, height });
      const clamped = await becomes(page, popupInsidePanel);
      check('popup', `it stays inside the panel at ${width}×${height}`, clamped, JSON.stringify(await popup(page)));
    }
    await page.setViewportSize(LAYOUTS.desktop.options.viewport);
    const returned = await becomes(page, popupIsAt, [dropped.left, dropped.top]);
    check('popup', 'back at 1440 it returns to where it was dropped', returned);
    await page.keyboard.press('Escape');
    check('popup', 'Escape closes it', (await popup(page)) === null);
  });
}

async function gestures() {
  await withPage('gestures', {}, async (page) => {
    const panel = await rect(page, '#panel');
    await dragNode(page, 'r2-1', 60, 40);
    check('gestures', 'dragging a node opens no popup', (await popup(page)) === null);
    await page.$eval(nodeSel('r1-2'), (el) => el.click());
    const clicked = await popup(page);
    const opensNode = clicked?.title === '● 03_riverside · round 1';
    check('gestures', 'a click with no pointer after a drag still opens the node', opensNode, clicked?.title);
    await page.keyboard.press('Escape');

    const before = await page.evaluate(layoutSnapshot);
    const press = await centre(page, 'r1-0');
    const side = await rect(page, '#side');
    // Leave the panel in one move and release outside it, so the panel sees neither the move nor the release.
    await page.mouse.move(press.x, press.y);
    await page.mouse.down();
    await page.mouse.move(side.left + side.width / 2, side.top + side.height / 2);
    await page.mouse.up();
    await page.mouse.move(panel.left + 300, panel.top + 300, { steps: 12 });
    await page.mouse.move(panel.left + 500, panel.top + 500, { steps: 12 });
    const stuckName = 'a press released outside the panel ends: coming back with no button neither pans nor drags';
    check('gestures', stuckName, (await page.evaluate(layoutSnapshot)) === before);
  });
  await withPage('gestures', { layout: LAYOUTS.phone }, async (page) => {
    const tap = await centre(page, 'r2-3');
    await page.touchscreen.tap(tap.x, tap.y);
    await becomes(page, () => !document.getElementById('inspector').hidden);
    const opened = await popup(page);
    const fileView = await page.evaluate(() => {
      return ['files-pane', 'file-view'].some((id) => !document.getElementById(id).hidden);
    });
    const tapName = 'phone: a tap opens the tapped node, not a link that appears under the finger';
    check('gestures', tapName, opened?.title === '● 04_contractor_registry · round 2' && !fileView, opened?.title);
  });
}

async function scale() {
  const memos = Array.from({ length: LARGE_RUN_DOCS }, (_, i) => ({
    name: `doc_${String(i + 1).padStart(2, '0')}.txt`,
    mimeType: 'text/plain',
    buffer: Buffer.from(`Memo ${i + 1}. Site ${i + 1} logged ${i % 7} equipment incidents in Q3.\n`),
  }));
  for (const layout of Object.values(LAYOUTS)) {
    await withPage('scale', { layout, sample: false }, async (page) => {
      await page.setInputFiles('#file-input', memos);
      await page.waitForFunction((n) => document.querySelectorAll('.node.doc').length === n, LARGE_RUN_DOCS);
      await page.fill('#question', 'Which site logged the most incidents?');
      await page.click('#run-btn');
      await page.waitForSelector('#panel.is-lit');
      const nodes = await count(page, '.node');
      const edges = await count(page, '.edge');
      const fit = await page.evaluate(fitOfRun);
      const fits = nodes === LARGE_RUN_COUNTS.nodes && edges === LARGE_RUN_COUNTS.edges && fit.outside === 0;
      const name = `${layout.name}: a ${LARGE_RUN_DOCS}-document run fits with no node outside the panel`;
      check('scale', name, fits, `${nodes} nodes, ${edges} edges, ${fit.outside} outside, zoom ${fit.zoom}`);
    });
  }
}

// ---------- answers: asks over the run on the panel ----------

const offerLabel = (ask) => `${ask.question} · ${ask.model.split('/').pop()}`;
const offers = (page) => page.$$eval('[data-offer] .ref', (els) => els.map((el) => !el.disabled));
const overlap = (a, b) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
const inside = (box, frame) => box.left >= frame.left && box.right <= frame.right && box.top >= frame.top && box.bottom <= frame.bottom;
const tally = (ids) => ids.reduce((counts, id) => ({ ...counts, [id]: (counts[id] ?? 0) + 1 }), {});

// Types `question` into the composer and sends it with `keys`: Enter asks, Control+Enter starts a new run.
async function send(page, question, keys = 'Enter') {
  await page.fill('#question', question);
  await page.press('#question', keys);
}

async function pickModel(page, model) {
  await page.selectOption('#model-pick', '');
  await page.fill('#model-other', model);
  await page.$eval('#model-other', (el) => el.blur());
}

async function answers() {
  const sample = recordedSample();
  const asks = recordedAsks();
  const twoModels = asks.length === 3 && asks[0].question === asks[1].question && asks[0].model !== asks[1].model;
  check('answers', 'the page holds recorded asks: one question by two models, and another question', twoModels, `${asks.length} asks`);
  await answersOnTheRun(sample, asks);
  await answersAfterFiles(asks);
  await answersInView(asks);
  await answersWhileReplaying();
  await withPage('answers', { layout: LAYOUTS.phone }, async (page) => {
    await send(page, asks[0].question);
    await send(page, NEW_QUESTION);
    const newest = isDeepStrictEqual(await litAnswers(page), ['answer-1']);
    const tap = await centre(page, 'answer');
    await page.touchscreen.tap(tap.x, tap.y);
    const selected = await becomes(page, () => document.querySelector('.node.answer.on-path')?.dataset.id === 'answer');
    check('answers', '400 dark: the newest ask is lit, and a tap on the first answer lights it again', newest && selected);
  });
}

async function answersOnTheRun(sample, asks) {
  await withPage('answers', {}, async (page) => {
    const docIds = await docIdsOf(page);
    const runPath = lineage(sample, docIds);
    const askPath = (k) => lineage(sample, docIds, `answer-${k}`, asks[k].provenance);
    const offered = await page.evaluate(composer);
    const offersBoth = isDeepStrictEqual(offered, { primary: 'ask ⏎', newRun: true, note: null });
    check('answers', 'after a run the composer offers ask ⏎ and new run', offersBoth, JSON.stringify(offered));

    const first = await positionOf(page, 'answer');
    const nodes = await count(page, '.node');
    await page.evaluate(countStatusWrites);
    await send(page, asks[0].question);
    const added = (await count(page, '.node')) - nodes;
    const lit = await litEdges(page);
    const oneNode = added === 1 && isDeepStrictEqual(await answerIds(page), ['answer', 'answer-0']);
    const onlyItsPath = isDeepStrictEqual(lit, askPath(0)) && isDeepStrictEqual(await litAnswers(page), ['answer-0']);
    check('answers', 'a recorded ask adds exactly one answer node and lights only its path', oneNode && onlyItsPath, `${added} added; ${lit.length} lit`);
    const status = await page.textContent('#run-status');
    const turns = await count(page, '.turn');
    check('answers', 'Enter asks over the run on the panel: no new run starts', turns === 2 && status === SAMPLE_STATUS, `${turns} turns; ${status}`);
    const writes = await page.evaluate(() => window.statusWrites);
    check('answers', 'an ask leaves the status alone: its live region is not rewritten', writes === 0, `${writes} writes`);
    const linesWhileAskLit = await page.evaluate(answerLines);

    const lines = await page.evaluate(lastTurnLines);
    const { summary_ids, parse_ids } = asks[0].provenance;
    const cited = new Set([...summary_ids, ...parse_ids]).size;
    const summary = `${RECORDED_ASK} · ${asks[0].model} · ${asks[0].totals.tokens.toLocaleString('en-US')} tokens · $`;
    const told = lines.includes(asks[0].text) && lines.filter((text) => text.startsWith('drew on ')).length === cited && lines.some((text) => text.startsWith(summary));
    const shown = await inspect(page, 'answer-0');
    const rows = await page.$$eval('#inspector .row', (els) => els.map((r) => r.firstChild.textContent));
    const inPopup = shown.includes(asks[0].text) && shown.includes(summary) && isDeepStrictEqual(rows, POPUP_ROWS.answer);
    check('answers', "a recorded ask's chat and popup give its real answer, citations and source", told && inPopup, lines.slice(1).join(' | '));
    const label = await page.textContent(`${nodeSel('answer-0')} .label`);
    check('answers', 'an answer node is labelled with a short form of its question, and its model', label === ASK_LABEL, label);
    const kept = await positionOf(page, 'answer');
    const below = await positionOf(page, 'answer-0');
    const stacked = isDeepStrictEqual(kept, first) && below.x === first.x && below.y === first.y + ROW_GAP;
    check('answers', 'the first answer keeps its place, and the ask stacks one row gap below it', stacked, `${first.x},${first.y} → ${below.x},${below.y}`);

    await page.keyboard.press('Escape');
    await clickNode(page, 'answer');
    const byNode = isDeepStrictEqual(await litEdges(page), runPath) && isDeepStrictEqual(await litAnswers(page), ['answer']);
    check('answers', 'selecting the first answer by its node lights its path again', byNode);
    const pressed = await page.evaluate(pressedAnswers);
    const saysLit = isDeepStrictEqual(pressed, [['answer', 'true'], ['answer-0', 'false']]);
    check('answers', 'each answer node says with aria-pressed whether it is the lit one', saysLit, JSON.stringify(pressed));
    const linesWhileRunLit = await page.evaluate(answerLines);
    const runCites = sample.answer.provenance.summary_ids.length + sample.answer.provenance.parse_ids.length;
    const askCites = summary_ids.length + parse_ids.length;
    const citeLines = { answer: runCites, 'answer-0': askCites };
    const faintFollows =
      isDeepStrictEqual(tally(linesWhileAskLit.faint), { 'answer-0': SAMPLE_ANSWER_INPUTS - askCites }) &&
      isDeepStrictEqual(tally(linesWhileRunLit.faint), { answer: SAMPLE_ANSWER_INPUTS - runCites }) &&
      [linesWhileAskLit, linesWhileRunLit].every(({ cited: ends }) => isDeepStrictEqual(tally(ends), citeLines));
    const faintName = 'only the lit answer has faint lines from what it was shown, and every answer keeps its cite lines';
    check('answers', faintName, faintFollows, `${JSON.stringify(tally(linesWhileAskLit.faint))} → ${JSON.stringify(tally(linesWhileRunLit.faint))}`);
    await page.keyboard.press('Escape');
    const name = await page.getAttribute(nodeSel('answer-0'), 'aria-label');
    await page.focus(nodeSel('answer-0'));
    await page.keyboard.press('Enter');
    const byKey = isDeepStrictEqual(await litAnswers(page), ['answer-0']) && (await popup(page))?.title === `● ${ASK_LABEL}`;
    const named = name === `answer to ${asks[0].question} · ${asks[0].model}`;
    check('answers', 'an answer node is named for its question and model, takes focus, and Enter on it lights it', named && byKey, name);
    await page.keyboard.press('Escape');
    await page.locator('.turn').first().locator('.line.answer').click();
    const byLine = isDeepStrictEqual(await litEdges(page), runPath) && isDeepStrictEqual(await litAnswers(page), ['answer']);
    check('answers', 'selecting the first answer by its chat line lights its path again', byLine);

    const openTitle = (await popup(page))?.title;
    await send(page, NEW_QUESTION);
    const leftAlone = isDeepStrictEqual(await litAnswers(page), ['answer']) && (await popup(page))?.title === openTitle;
    check('answers', "a landing ask leaves an open answer's popup, and its lit path, alone", leftAlone, openTitle);
    const simulated = await page.evaluate(lastTurnLines);
    const flaggedInChat = simulated.includes(SIMULATED_LABEL) && simulated.some((text) => text.startsWith(`${SIMULATED_ASK} · ${DEFAULT_MODEL} · nothing was sent`));
    const flaggedInPopup = (await inspect(page, 'answer-1')).includes(SIMULATED_LABEL);
    check('answers', 'an unrecorded ask is labelled simulated, in the chat and its popup', flaggedInChat && flaggedInPopup, simulated.slice(1, 3).join(' | '));

    await page.keyboard.press('Escape');
    await page.getByRole('button', { name: offerLabel(asks[1]), exact: true }).click();
    const filled = await page.evaluate(() => [document.getElementById('question').value, document.getElementById('model-other').value]);
    await page.press('#question', 'Enter');
    const other = await page.evaluate(lastTurnLines);
    const fromOffer = isDeepStrictEqual(filled, [asks[1].question, OTHER_RECORDED_MODEL]) && asks[1].model === OTHER_RECORDED_MODEL;
    const otherRecorded = other.includes(asks[1].text) && other.some((text) => text.startsWith(`${RECORDED_ASK} · ${OTHER_RECORDED_MODEL} · `));
    const offeredName = 'an offered recorded ask fills its question and model, and is answered as recorded';
    check('answers', offeredName, fromOffer && otherRecorded, `${filled.join(' · ')}; ${other.slice(1, 2).join('')}`);

    await page.getByRole('button', { name: offerLabel(asks[2]), exact: true }).click();
    await page.press('#question', 'Enter');
    for (const question of ['Which memo is the registry?', 'Who manages Eastfield?']) await send(page, question);
    await dragNode(page, 'answer-1', 80, 50);
    await page.click('#arrange-btn');
    const placed = await page.evaluate(answerNodes);
    const offsets = placed.map((n) => (n.y - first.y) / ROW_GAP).sort((a, b) => a - b);
    const inColumn = placed.every((n) => n.x === first.x) && isDeepStrictEqual(offsets, [-3, -2, -1, 0, 1, 2, 3]);
    const apart = placed.every((a, i) => placed.slice(i + 1).every((b) => !overlap(a.box, b.box)));
    const arrangeName = 'arrange places every answer node in the answer column, one row gap apart, none overlapping';
    check('answers', arrangeName, inColumn && apart, `rows ${offsets.join(',')}; apart ${apart}`);

    // Short enough that height bounds the fit: answers stacked past the document rows show only if fit counts them.
    await page.setViewportSize(SHORT_DESKTOP);
    const frame = await page.evaluate(panelBox);
    await page.mouse.move(frame.left + frame.width / 2, frame.top + frame.height / 2);
    await page.mouse.wheel(1500, 1500);
    const away = (await page.evaluate(answerNodes)).filter((n) => !inside(n.icon, frame)).length;
    await page.click('#fit-btn');
    const shownAfter = (await page.evaluate(answerNodes)).filter((n) => inside(n.icon, frame)).length;
    await page.setViewportSize(LAYOUTS.desktop.options.viewport);
    check('answers', 'fit shows every answer node', away > 0 && shownAfter === placed.length, `${away} out of view, then ${shownAfter} of ${placed.length} in view`);

    const litBefore = [await litAnswers(page), await litEdges(page)];
    const idsBefore = (await answerIds(page)).sort();
    await page.click('#replay-btn');
    await page.waitForSelector('#panel.is-lit');
    const redrawn = isDeepStrictEqual((await answerIds(page)).sort(), idsBefore) && (await count(page, '.node.answer.working')) === 0;
    const stillLit = isDeepStrictEqual([await litAnswers(page), await litEdges(page)], litBefore);
    check('answers', "replay draws the run's asks again, and keeps the answer that was lit", redrawn && stillLit, `${idsBefore.length} answers; lit ${litBefore[0]}`);

    await page.fill('#question', sample.question);
    await page.click('#new-run-btn');
    await page.waitForSelector('#panel.is-lit');
    const after = { nodes: await count(page, '.node'), answers: (await answerIds(page)).length };
    const fresh = isDeepStrictEqual(after, ANSWERED_RUN_COUNTS) && isDeepStrictEqual(await litEdges(page), runPath);
    const replacedName = 'new run replaces the panel: the asks go, and the new run\'s answer is the one lit';
    check('answers', replacedName, fresh && (await page.textContent('#run-status')) === SAMPLE_STATUS, JSON.stringify(after));
    const offeredOnce = await offers(page);
    const onceName = 'a new recorded replay offers the recorded asks once, not again beside the old offers';
    check('answers', onceName, isDeepStrictEqual(offeredOnce, asks.map(() => true)), JSON.stringify(offeredOnce));

    await send(page, asks[0].question);
    await clickNode(page, 'answer');
    await page.keyboard.press('Escape');
    const earlier = page.locator('.turn').nth(1).locator('.line.answer');
    const inert = await earlier.isDisabled();
    await earlier.click({ force: true });
    const untouched = isDeepStrictEqual(await litAnswers(page), ['answer']) && (await popup(page)) === null;
    const inertName = "after a new run, an earlier run's ask lines are inert: they open nothing on the new panel";
    check('answers', inertName, inert && untouched, `disabled ${inert}; lit ${await litAnswers(page)}`);
  });
}

async function answersAfterFiles(asks) {
  await withPage('answers', {}, async (page) => {
    await page.setInputFiles('#file-input', memo('06_extra_memo.txt', 'An extra memo.\n'));
    await page.waitForSelector('.node.pending');
    const footer = await page.evaluate(composer);
    const says = isDeepStrictEqual(footer, { primary: 'ask ⏎', newRun: true, note: NOT_IN_RUN });
    check('answers', "after a file is added, the footer says it isn't in this run, and ask stays", says, JSON.stringify(footer));
    const added = await page.$eval('#transcript > .line:last-child .t', (el) => el.textContent);
    check('answers', 'after a run, the file-added line says a new run would cover the files and ask stays on this run', added === FILE_ADDED_AFTER_RUN, added);
    await send(page, asks[0].question);
    const pending = await count(page, '.node.pending');
    const into = await page.$$eval('.edge', (els) => els.map((e) => e.dataset.key).filter((key) => key.endsWith('>answer-0')));
    const fromRun = pending === 1 && into.length === SAMPLE_ANSWER_INPUTS && into.every((key) => /^(r2|parsed)-\d>/.test(key));
    const recorded = (await page.evaluate(lastTurnLines)).some((text) => text.startsWith(RECORDED_ASK));
    check('answers', "an ask after a file is added still answers over the run's documents", fromRun && recorded, into.join(' '));
    await send(page, NEW_QUESTION, 'Control+Enter');
    const status = await page.textContent('#run-status');
    const ids = await answerIds(page);
    const ran = isDeepStrictEqual(ids, ['answer']) && status === `● done · simulated run · ${DEFAULT_MODEL} · nothing was sent`;
    const newRunName = 'Ctrl+Enter starts a new run over the files now loaded, instead of asking';
    check('answers', newRunName, ran && (await count(page, '.node.pending')) === 0, `${status}; ${ids.join(' ')}`);
  });
}

async function answersInView(asks) {
  await withPage('answers', { sample: false }, async (page) => {
    await pickModel(page, LONG_MODEL);
    await send(page, NEW_QUESTION);
    const frame = await page.evaluate(panelBox);
    const outside = (await page.$$eval('.node', (els) => els.map((el) => [el.dataset.id, el.getBoundingClientRect().toJSON()])))
      .filter(([, box]) => !inside(box, frame))
      .map(([id]) => id);
    check('answers', "a run whose model has a long name is fitted with its answer's whole label in view", outside.length === 0, outside.join(' ') || 'all in view');
  });
  await withPage('answers', {}, async (page) => {
    await dragNode(page, 'answer', 60, 50);
    const moved = await positionOf(page, 'answer');
    await pickModel(page, LONG_MODEL);
    await send(page, asks[0].question);
    const placed = await positionOf(page, 'answer-0');
    const aroundIt = placed.x === moved.x && placed.y === moved.y + ROW_GAP;
    check('answers', 'an ask stacks around the first answer where it is now, after a drag', aroundIt, `${moved.x},${moved.y} → ${placed.x},${placed.y}`);
    const frame = await page.evaluate(panelBox);
    const outside = (await page.evaluate(answerNodes)).filter((n) => !inside(n.box, frame)).map((n) => n.id);
    check('answers', 'a new answer node is fitted into view, its whole label too', outside.length === 0, outside.join(' ') || 'all in view');
  });
}

// With motion on, a replay takes seconds, so a question can arrive while it is still going.
async function answersWhileReplaying() {
  await withPage('answers', { sample: false }, async (page) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.evaluate(recordStatuses);
    await page.click('#run-btn');
    await send(page, NEW_QUESTION);
    const kept = await page.inputValue('#question');
    const turns = await count(page, '.turn');
    const said = await page.evaluate(() => window.statuses);
    const waits = kept === NEW_QUESTION && turns === 1 && said.includes(ASK_WHEN_DONE);
    const waitsName = "a question while the prototype's run replays waits, as live does: ask once it has finished";
    check('answers', waitsName, waits, `${turns} turns; ${said.at(-1)}`);
  });
}

// ---------- live: served by r3con-app, with a scripted model ----------

// Starts fake_server.py with no provider key in its environment, in a process group of its own.
function startFakeServer() {
  const logsDir = mkdtempSync(join(tmpdir(), 'r3con-checks-'));
  const env = { ...process.env };
  for (const variable of KEY_VARIABLES) delete env[variable];
  const server = spawn('uv', [...FAKE_SERVER, logsDir], { cwd: REPO_DIR, env, detached: true, stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  server.stderr.on('data', (chunk) => (stderr += chunk));
  const exited = new Promise((resolve) => {
    server.once('exit', resolve);
    server.once('error', resolve);
  });
  const url = new Promise((resolve, reject) => {
    createInterface({ input: server.stdout }).once('line', resolve);
    server.once('error', reject);
    exited.then((code) => reject(new Error(`the fake server exited (${code}): ${stderr.trim().split('\n').at(-1)}`)));
    setTimeout(() => reject(new Error('the fake server printed no URL in time')), SERVER_START_MS).unref();
  });
  const stop = async () => {
    try {
      process.kill(-server.pid, 'SIGKILL');
    } catch {
      // Its process group is already gone, or never started.
    }
    await exited;
    rmSync(logsDir, { recursive: true, force: true });
  };
  return { url, stop };
}

async function withFakeServer(body) {
  const server = startFakeServer();
  try {
    await body(await server.url, server);
  } catch (error) {
    check('live', firstLine(error), false);
  } finally {
    await server.stop();
  }
}

const tokenOf = (url) => new URL(url).hash.replace('#token=', '');
const memo = (name, text) => ({ name, mimeType: 'text/plain', buffer: Buffer.from(text) });
const sampleMemos = () => recordedSample().docs.map((d) => memo(d.name, d.text));

// Long enough that fake_server.py refuses it whole, with its first half quiet, so its first part summarizes to nothing.
function splitMemo() {
  const quiet = Array.from({ length: 32 }, (_, i) => `${QUIET} Filler paragraph ${i + 1}, which the scripted model skips.`);
  const relevant = Array.from({ length: 16 }, (_, i) => `Westbrook site, note ${i + 1}: an equipment incident was logged in Q3.`);
  return memo(`${SPLIT_MEMO}.txt`, [...quiet, ...relevant].join('\n\n'));
}

async function keysOnServer(url) {
  const response = await fetch(new URL('/api/keys', url), { headers: { Authorization: `Bearer ${tokenOf(url)}` } });
  return Object.entries(await response.json()).filter(([, set]) => set).map(([provider]) => `state-${provider}`);
}

function postsOf(page) {
  const posts = [];
  page.on('request', (request) => request.method() === 'POST' && posts.push(new URL(request.url()).pathname));
  return posts;
}

// Saves a key on its server.
async function liveSession(url) {
  const token = tokenOf(url);
  await withPage('live', { url, sample: false }, async (page, messages) => {
    const address = await page.evaluate(() => location.href);
    check('live', 'the launch token leaves the address bar at once', !address.includes(token) && !address.includes('#'), address);
    const ready = await page.textContent('#run-status');
    const readyName = 'before a run the status names the picked model, not a recording';
    check('live', readyName, ready === `${SAMPLE_DOCS} documents ready · ${DEFAULT_MODEL}`, ready);

    await page.click('#settings-btn');
    await page.fill('#key-openai', TEST_KEY);
    const marked = (await saveKeys(page)) && isDeepStrictEqual(await page.evaluate(keyStates), ['state-openai']);
    await page.keyboard.press('Escape');
    const fresh = await page.context().newPage();
    await fresh.goto(url);
    const kept = await becomes(fresh, () => document.getElementById('state-openai').textContent === 'set');
    await fresh.close();
    const keyName = 'a saved key marks OpenAI as set, and a fresh page of the launch reads it from the server';
    check('live', keyName, marked && kept, `marked ${marked}, fresh page ${kept}`);

    await page.click('#run-btn');
    const done = await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    check('live', 'a run ends with the done status: its model, total tokens and cost', done, await page.textContent('#run-status'));
    const nodes = await count(page, '.node');
    const working = await count(page, '.node.working');
    const drawn = nodes === SAMPLE_COUNTS.nodes && working === 0;
    check('live', `a run draws all ${SAMPLE_COUNTS.nodes} nodes from its events, none left working`, drawn, `${nodes} nodes, ${working} working`);

    const summaries = [];
    for (let i = 0; i < SAMPLE_DOCS; i++) summaries.push(await inspect(page, `r1-${i}`, '#inspector .insp-text'));
    const scripted = recordedSample().docs.map((d) => scriptedSummary(d.text));
    const ownName = "each summary lands on its own document's node: the run's ids map to the files by order";
    check('live', ownName, isDeepStrictEqual(summaries, scripted), summaries[0]);
    const lit = await litEdges(page);
    const docIds = await docIdsOf(page);
    const expected = lineage(SCRIPTED_CITES, docIds);
    const litName = 'the lit edges are exactly the lineage of the provenance the events carried';
    check('live', litName, isDeepStrictEqual(lit, expected), `${lit.length} lit, ${expected.length} expected`);
    const costed = [];
    for (const id of ['r1-0', 'r2-0', 'schema', 'parsed-0']) {
      if ((await inspect(page, id)).includes('tokens')) costed.push(id);
    }
    check('live', 'no node shows per-call tokens or cost, which events do not carry', costed.length === 0, costed.join(', '));

    await page.keyboard.press('Escape');
    await page.fill('#question', 'Which memo is the registry?');
    await page.click('#run-btn');
    const answered = await becomes(page, lastTurnSays, [SCRIPTED_ANSWER, LIVE_ASK], LIVE_RUN_MS);
    const lines = await page.evaluate(lastTurnLines);
    const cited = lines.filter((text) => text.startsWith('drew on ')).length === 2;
    const named = lines.some((text) => text.startsWith(`${LIVE_ASK} · ${DEFAULT_MODEL} · `));
    const askName = 'a follow-up runs an ask and shows its answer, provenance and model in the chat';
    check('live', askName, answered && cited && named, lines.slice(1).join(' | '));

    const holders = [...(await page.evaluate(placesHolding, token)), ...(await page.evaluate(placesHolding, TEST_KEY))];
    const logged = messages.filter((m) => m.text.includes(token) || m.text.includes(TEST_KEY)).length;
    const keptName = 'no storage, field, address, markup or console message holds the token or the key';
    check('live', keptName, holders.length === 0 && logged === 0, [...holders, `${logged} messages`].join(', '));
  });
}

async function liveCutStream(url) {
  const before = (page) => page.addInitScript(cutEventStreams, 1);
  await withPage('live', { url, sample: false, before }, async (page) => {
    await page.click('#run-btn');
    const done = await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const nodes = await count(page, '.node');
    const summary = nodes === SAMPLE_COUNTS.nodes ? await inspect(page, 'r1-0', '#inspector .insp-text') : null;
    const whole = done && summary === scriptedSummary(recordedSample().docs[0].text);
    const cutName = 'a stream read one byte at a time, every frame split across reads, draws the same run';
    check('live', cutName, whole, `${nodes} nodes; ${await page.textContent('#run-status')}`);
  });
}

async function liveParts(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    await page.setInputFiles('#file-input', [...sampleMemos(), splitMemo()]);
    await page.waitForFunction((n) => document.querySelectorAll('.node.doc').length === n, SAMPLE_DOCS + 1);
    await page.click('#run-btn');
    const done = await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const lines = await page.evaluate(lastTurnLines);
    const expected = [
      `round 1 · ${SPLIT_MEMO} — summary ready`,
      `round 2 · ${SPLIT_MEMO} — drew on 01_northgate`,
      `round 2 · ${SAMPLE_DOCS + 1} summaries cite another document`,
      `${SPLIT_MEMO} — 2 records`,
    ];
    const missing = expected.filter((start) => !lines.some((text) => text.startsWith(start)));
    const partsName = 'a document read in two parts, the first with nothing relevant, is narrated from both';
    check('live', partsName, done && missing.length === 0, missing.join(' | ') || 'all four lines');
  });
}

// Holds the ask's last event until its popup is open.
async function liveAnswers(url) {
  const before = (page) => page.addInitScript(holdFrom, 'ask_finished');
  await withPage('live', { url, sample: false, before }, async (page) => {
    const posts = postsOf(page);
    await page.click('#run-btn');
    await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    await send(page, 'Which memo is the registry?');
    const landed = await becomes(page, landedNode, 'answer-0', LIVE_RUN_MS);
    const docIds = await docIdsOf(page);
    const expected = lineage(SCRIPTED_CITES, docIds, 'answer-0', SCRIPTED_CITES.answer.provenance);
    const lit = await litEdges(page);
    const ownPath = landed && isDeepStrictEqual(lit, expected) && isDeepStrictEqual(await litAnswers(page), ['answer-0']);
    const beforeEnd = await inspect(page, 'answer-0');
    const live = beforeEnd.includes(`${LIVE_ASK} · ${DEFAULT_MODEL}`);
    const askName = 'an ask adds an answer node, lit along its own path, and its popup says it is live';
    check('live', askName, ownPath && live, `${lit.length} lit, ${expected.length} expected`);
    await page.evaluate(() => (window.release = true));
    const costed = await becomes(page, (start) => document.getElementById('inspector').textContent.includes(start), `${LIVE_ASK} · ${DEFAULT_MODEL} · `, LIVE_RUN_MS);
    const costName = "a live ask's open popup shows its tokens and cost once the ask finishes";
    check('live', costName, !beforeEnd.includes(' tokens') && costed, await page.textContent('#inspector .k:last-child'));

    await page.keyboard.press('Escape');
    const offered = (await page.evaluate(composer)).newRun;
    await page.fill('#question', 'Which memo is the registry, once more?');
    await page.focus('#new-run-btn');
    await page.keyboard.press('Enter');
    const ended = (source) => !document.querySelector('.node[data-id="answer-0"]') && new RegExp(source).test(document.getElementById('run-status').textContent);
    const replaced = await becomes(page, ended, LIVE_DONE.source, LIVE_RUN_MS);
    const runs = posts.filter((path) => path === '/api/runs').length;
    const nodes = await count(page, '.node');
    const newRunName = 'after an ask, new run is offered and starts a run that replaces the panel';
    check('live', newRunName, offered && replaced && runs === 2 && nodes === SAMPLE_COUNTS.nodes, `${runs} runs, ${nodes} nodes`);
    const focused = await page.evaluate(() => document.activeElement.id);
    check('live', 'new run from the keyboard keeps the focus, in the question box', focused === 'question', `focus on #${focused}`);
  });
}

async function liveRunEnd(url) {
  const before = (page) => page.addInitScript(holdFrom, 'run_finished');
  await withPage('live', { url, sample: false, before }, async (page) => {
    await page.click('#run-btn');
    const answered = await becomes(page, lastTurnSays, [SCRIPTED_ANSWER], LIVE_RUN_MS);
    const whileOpen = await page.textContent('#run-btn');
    await page.evaluate(() => (window.release = true));
    const done = await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const once = await page.textContent('#run-btn');
    const endName = 'the button offers to ask only once the run has ended, not when its answer lands';
    check('live', endName, answered && done && whileOpen === 'run ⏎' && once === 'ask ⏎', `${whileOpen} → ${once}`);
  });
}

// Leaves an ask hanging on its server.
async function liveAskGoing(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    const posts = postsOf(page);
    await page.click('#run-btn');
    await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    await page.fill('#question', `Which site is quietest? ${HANGS}`);
    await page.click('#run-btn');
    const asking = await becomes(page, lastTurnSays, [`asking · ${DEFAULT_MODEL}`]);
    await page.setInputFiles('#file-input', memo('06_extra.txt', 'An extra memo.\n'));
    await page.waitForSelector('.node.pending');
    await page.fill('#question', 'Which memo is the extra one?');
    await page.click('#new-run-btn');
    const refused = await page.textContent('#run-status');
    const sent = posts.filter((path) => path === '/api/runs').length;
    const stillAsking = await page.evaluate(lastTurnLines);
    const askName = "while an ask is going, a new run is refused: it can't be stopped yet";
    check('live', askName, asking && refused === ASK_STILL_GOING && sent === 1 && stillAsking.includes(`asking · ${DEFAULT_MODEL}`), `${refused}; ${sent} runs`);
  });
}

async function liveFailure(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    await page.fill('#question', `Which site logged the most incidents? ${FAILS}`);
    await page.click('#run-btn');
    const failed = await becomes(page, statusIs, LIVE_FAILED, LIVE_RUN_MS);
    const working = await count(page, '.node.working');
    const failName = "a failed run's status gives its stage and error, and no node is left working";
    check('live', failName, failed && working === 0, `${await page.textContent('#run-status')}; ${working} working`);
    const stopped = await page.$$eval('.node.failed', (els) => els.map((el) => [el.classList.contains('void'), el.getAttribute('aria-label')]));
    const readsStopped = stopped.length > 0 && stopped.every(([empty, name]) => !empty && name.endsWith(', stopped'));
    const stoppedName = "a failed run's unfinished nodes look and read as stopped, not as empty";
    check('live', stoppedName, readsStopped, stopped.map(([, name]) => name).join(' | '));
    const label = await page.textContent('#run-btn');
    check('live', 'after a failed run the next question starts a new run', label === 'run ⏎', label);
  });
}

async function liveUnreadable(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    const posts = postsOf(page);
    await page.setInputFiles('#file-input', [memo('01_memo.txt', 'A memo.\n'), memo('02_blank.txt', '  \n')]);
    await page.waitForFunction(() => document.querySelectorAll('.node.doc').length === 2);
    await page.click('#run-btn');
    const status = await page.textContent('#run-status');
    const refused = status === 'no readable text in 02_blank.txt; nothing was sent' && posts.length === 0;
    check('live', 'a file with no readable text is named, and nothing is sent', refused, `${status}; ${posts.length} posts`);
  });
}

// Saves keys on its server.
async function liveKeySaves(url) {
  const saves = [];
  const refuseGemini = async (route) => {
    const request = route.request();
    if (request.method() !== 'POST') return route.continue();
    const { provider } = request.postDataJSON();
    saves.push(provider);
    if (provider === 'gemini') return route.fulfill({ status: 422, contentType: 'application/json', body: JSON.stringify({ error: 'refused by the check' }) });
    await new Promise((resolve) => setTimeout(resolve, SLOW_SAVE_MS));
    return route.continue();
  };
  const before = (page) => page.route('**/api/keys', refuseGemini);
  await withPage('live', { url, sample: false, before, expectedErrors: [REFUSED_LOG] }, async (page) => {
    await page.click('#settings-btn');
    await page.fill('#key-anthropic', TEST_KEY);
    await page.fill('#key-gemini', TEST_KEY);
    await page.dblclick('#settings-save');
    const named = await becomes(page, () => document.getElementById('settings-saved').textContent !== '');
    const said = await page.textContent('#settings-saved');
    const marked = (await page.evaluate(keyStates)).sort();
    const onServer = (await keysOnServer(url)).sort();
    const keysName = 'a key save one provider refuses names it, and the set markers match the server';
    const matches = named && said === 'Google · refused by the check' && isDeepStrictEqual(marked, onServer);
    check('live', keysName, matches, `${said}; page ${marked.join(' ')}; server ${onServer.join(' ')}`);
    const fields = [await page.inputValue('#key-anthropic'), await page.inputValue('#key-gemini')];
    const clearedName = 'a save stopped partway clears each key it saved and keeps the rest';
    check('live', clearedName, isDeepStrictEqual(fields, ['', TEST_KEY]), fields.map((value) => (value ? 'kept' : 'cleared')).join(', '));
    check('live', 'a second click while a save is in flight sends nothing more', isDeepStrictEqual(saves, ['anthropic', 'gemini']), saves.join(' '));
  });
}

// Saves a key on its server.
async function liveLoadProblem(url) {
  const before = (page) => page.route('**/api/keys', (route) => route.abort());
  const failing = (route) => route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ error: 'still unwell' }) });
  await withPage('live', { url, sample: false, before, expectedErrors: [ABORTED_LOG, SERVER_ERROR_LOG] }, async (page) => {
    const atLoad = await becomes(page, statusIs, LOST_CONNECTION);
    await page.fill('#question', 'Which memo mentions the storm?');
    const typing = (await page.textContent('#run-status')) === LOST_CONNECTION;
    await page.unroute('**/api/keys');
    await page.route('**/api/keys', failing);
    await page.click('#settings-btn');
    await page.fill('#key-anthropic', TEST_KEY);
    await page.click('#settings-save');
    await becomes(page, () => document.getElementById('settings-saved').textContent !== '');
    await page.fill('#question', 'Which memo mentions the storm, again?');
    const unwell = (await page.textContent('#run-status')) === LOST_CONNECTION;
    await page.unroute('**/api/keys');
    const saved = await saveKeys(page);
    await page.keyboard.press('Escape');
    const status = await page.textContent('#run-status');
    const cleared = saved && status === `${SAMPLE_DOCS} documents ready · ${DEFAULT_MODEL}`;
    const problemName = 'a connection lost at load stays in the status, through typing and a 500, until a call to r3con-app succeeds';
    check('live', problemName, atLoad && typing && unwell && cleared, `typing ${typing}; 500 ${unwell}; then ${status}`);
  });
}

async function liveStaleToken(url) {
  const wrong = url.replace(/#token=.*/, '#token=not-this-launch');
  await withPage('live', { url: wrong, sample: false, expectedErrors: [UNAUTHORIZED_LOG] }, async (page) => {
    const atLoad = await becomes(page, statusIs, `● ${STALE_TOKEN}`);
    await page.click('#run-btn');
    const inChat = await becomes(page, lastTurnSays, [STALE_TOKEN]);
    const staleName = 'a wrong token shows the stale-token message, at load and when a run is tried';
    check('live', staleName, atLoad && inChat, await page.textContent('#run-status'));
  });
}

// Stops the server it is given: its own.
async function liveGoingThenLost(url, server) {
  await withPage('live', { url, sample: false, expectedErrors: [CUT_STREAM_LOG] }, async (page) => {
    const posts = postsOf(page);
    await page.fill('#question', `Which site logged the most incidents? ${HANGS}`);
    await page.click('#run-btn');
    const held = await becomes(page, () => document.querySelector('.node.r1.working') !== null, null, LIVE_RUN_MS);
    const question = 'Which site logged the fewest?';
    await page.fill('#question', question);
    await page.click('#run-btn');
    const asked = [await page.textContent('#run-status'), await page.inputValue('#question')];
    const askName = 'while a run is going, a question stays in the box, unsent, and the status says to ask once it has finished';
    check('live', askName, held && isDeepStrictEqual(asked, [ASK_WHEN_DONE, question]) && posts.length === 1, asked[0]);
    await page.setInputFiles('#file-input', memo('06_extra.txt', 'An extra memo.\n'));
    await page.waitForSelector('.node.pending');
    await page.press('#question', 'Control+Enter');
    const refused = await page.textContent('#run-status');
    const runName = "while a run is going, a new run is refused: it can't be stopped yet";
    check('live', runName, refused === RUN_STILL_GOING && isDeepStrictEqual(posts, ['/api/runs']), `${refused}; ${posts.join(' ')}`);

    await server.stop();
    const lost = await becomes(page, statusIs, LOST_CONNECTION, LIVE_RUN_MS);
    const lostName = 'a run whose server goes away mid-stream says the connection was lost';
    check('live', lostName, lost, await page.textContent('#run-status'));
  });
}

// A copy of the page served from another host must not take a token or call any API.
async function liveElsewhere() {
  const elsewhere = 'http://r3con.example/';
  const calls = [];
  const before = (page) => {
    page.on('request', (request) => request.url().includes('/api/') && calls.push(request.url()));
    return page.route(`${elsewhere}**`, (route) => route.fulfill({ contentType: 'text/html', body: readFileSync(PAGE_FILE, 'utf8') }));
  };
  await withPage('live', { url: `${elsewhere}#token=from-elsewhere`, sample: false, before }, async (page) => {
    const status = await page.textContent('#run-status');
    await page.click('#run-btn');
    const replayed = await becomes(page, statusIs, SAMPLE_STATUS);
    const name = 'a copy of the page served from another host stays the prototype and calls no API';
    check('live', name, status === ready('recorded run', RECORDED_MODEL) && replayed && calls.length === 0, `${status}; ${calls.length} calls`);
  });
}

// A check that leaves state on its server, a key or a hanging call, gets a server of its own.
async function live() {
  await withFakeServer(async (url) => {
    await liveCutStream(url);
    await liveAnswers(url);
    await liveRunEnd(url);
    await liveParts(url);
    await liveFailure(url);
    await liveUnreadable(url);
    await liveStaleToken(url);
  });
  for (const stateful of [liveSession, liveKeySaves, liveLoadProblem, liveAskGoing, liveGoingThenLost]) await withFakeServer(stateful);
  await liveElsewhere();
}

try {
  browser = await require('playwright').chromium.launch();
  const areas = [load, beforeRun, sampleReplay, arrange, files, answers, settings, popupArea, gestures, scale, live];
  for (const area of areas) await area();
} catch (error) {
  check('setup', firstLine(error), false);
} finally {
  await browser?.close();
}
console.log(failures ? `${failures} failed` : 'all checks passed');
process.exit(failures ? 1 : 0);
