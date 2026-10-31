// Checks the run panel in src/r3con/app/ (index.html, app.css, app.js) in headless Chromium: node tests/app/check.mjs
// The live area serves it with tests/app/fake_server.py, through uv.
import { execFileSync, spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, extname, join } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';

// An ESM import ignores NODE_PATH, where a global Playwright is found; require honours it.
const require = createRequire(import.meta.url);

const APP_DIR = new URL('../../src/r3con/app/', import.meta.url);
const PAGE_URL = new URL('index.html', APP_DIR).href;
const SCRIPT_FILE = fileURLToPath(new URL('app.js', APP_DIR));
const PAGE_FILES = { '/': 'index.html', '/app.css': 'app.css', '/app.js': 'app.js' };
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
  'doc-0': ['relevance snippet'],
  'r1-0': ['read'],
  'r2-3': ['read', 'provenance'],
  schema: ['why', 'fills'],
  'parsed-0': ['read', 'provenance', 'filled from'],
  'answer-0': ['asked', 'provenance'],
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
  'OpenAI API key': 'key-openai',
  'Anthropic API key': 'key-anthropic',
  'Google API key': 'key-gemini',
  'Other: server URL': 'endpoint-other',
  'Other: API key': 'key-other',
};
const PROTOTYPE_KEYS_NOTE = 'This prototype sends nothing.';
const LARGE_RUN_DOCS = 40;
const LARGE_RUN_COUNTS = { nodes: 162, edges: 3440 };
// The page's GRID.rowGap: answer nodes stack this far apart.
const ROW_GAP = 112;
const SIMULATED_LABEL = 'simulated — not connected to r3con yet';
// A simulated run's note, from the page's SUMMARY_CHARS.
const SIMULATED_NOTE = "simulated · each relevance snippet is the file's first 140 characters, and the citations are made up";
// A relevance round's line for a document with a snippet.
const SNIPPET_READY = 'snippet ready';
// A parsed document's line, and the popup of one with nothing in its parse.
const STRUCTURE_READY = 'structure ready';
const NOTHING_PARSED = 'nothing parsed';
const ASK_LABEL = 'Which sites does… · gpt-6-luna';
const RECORDED_ASK = 'recorded ask over this run';
const SAMPLE_READY = 'ready · gpt-6-luna · $0.0038';
// Each document's line, linked to its node, under each step that makes one per document.
const perDocument = (kind) => Array(SAMPLE_DOCS).fill(kind).join(' ');
// A run's log opened, as stepsOf gives it: each step, how many lines are nested under it, and the kinds of node they link to.
const RUN_STEPS = [
  ['round 1 · extracting relevance', SAMPLE_DOCS, perDocument('r1')],
  ['round 2 · extracting relevance', SAMPLE_DOCS, perDocument('r2')],
  ['proposing a structure', 1, 'schema'],
  [`parsing ${SAMPLE_DOCS} documents`, SAMPLE_DOCS, perDocument('parsed')],
  ['reasoning', 0, ''],
];
// A replay first reads its documents, its read line nested under that.
const REPLAY_STEPS = [[`reading ${SAMPLE_DOCS} documents`, 1, ''], ...RUN_STEPS];
// What the chat no longer says: lines that only describe how r3con works, and the citations and source under an answer.
const COMMENTARY = ['one summary per document', 'beside the other summaries', 'cite another document', 'drew on', 'ask over this run'];
const PROTOTYPE_NOTE = 'prototype · replays one recorded run';
// The chat's line height: 13.5px at 1.55.
const LINE_PX = 21;
const OTHER_RECORDED_MODEL = 'openai/gpt-6.1-terra';
// WCAG 2.2's minimum target size, and its minimum contrast for text.
const MIN_TARGET_PX = 24;
const MIN_TEXT_CONTRAST = 4.5;
const LONG_NAME = 'quarterly_maintenance_incident_report_for_the_northgate_and_riverside_sites.txt';
const ADD_FILES_FIRST = 'Add files in the files tab first, then ask again.';
const ALL_SAMPLE_MEMOS = 'use all 5 sample memos';
// The page's intro, live and in the prototype, for n documents.
const loadedLine = (n) => (n === 1 ? 'Loaded 1 document.' : `Loaded ${n} documents.`);
// The shown tab's chat: each tab has its own, and the others are hidden.
const CHAT = '.transcript:not([hidden])';
const INTRO = `${CHAT} > .line:first-child .t`;
const ASKS_HEADER = 'recorded asks over this run · any other question or model is simulated';
// Fewer memos than the shipped sample has, for a page shipped with another sample: one, to be counted in the singular.
const FEWER_MEMOS = 1;
const TYPE_A_MODEL = 'type a model first';
// Its faint lines: each of the 5 round-2 summaries, and the 4 parses with records.
const SAMPLE_ANSWER_INPUTS = 9;
const ANSWERED_RUN_COUNTS = { nodes: SAMPLE_COUNTS.nodes, answers: 1 };
const SHORT_DESKTOP = { width: 1440, height: 450 };
// The page's TIMING.litDelay and ARRANGE_MS: a replay lights its answer this long after it lands, and arrange slides this long.
const LIT_DELAY_MS = 650;
const ARRANGE_MS = 420;
// Long enough for a whole replay with motion on.
const REPLAY_MS = 20_000;
// The page's MARK and SPIN_FRAMES: its mark at rest, and the frames a working node's spinner cycles through.
const MARK = '❀';
const SPIN_FRAMES = ['·', '✾', '✿', '❀', '❁', '❀', '✿', '✾'];
const TAB_LABEL = '1 Which maintenance… · gpt-6-luna';
const STARTING = 'running · starting';
// A model whose name makes an answer's label far wider than the run's own.
const LONG_MODEL = 'hosted_vllm/qwen3-32b-instruct-awq';
// Longer than the page's GLIDE_MS, so a recording of the view holds a whole glide.
const GLIDE_RECORD_MS = 600;
// How far a frame of the view, as [zoom, x, y], may stray past a glide's ends: rounding, no more.
const VIEW_SLACK = [0.005, 1, 1];
// A new tab's view before its first fit, which a run's start must never show.
const NEW_TAB_VIEW = [1, 0, 0];

const REPO_DIR = fileURLToPath(new URL('../..', import.meta.url));
const FAKE_SERVER = ['run', 'python', 'tests/app/fake_server.py'];
const KEY_VARIABLES = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY'];
// A first uv run may have to build the environment.
const SERVER_START_MS = 120_000;
const LIVE_RUN_MS = 30_000;
// What fake_server.py's model does: the markers it obeys, its answer, and what it cites.
const FAILS = '(the model is down)';
const HANGS = '(the model hangs)';
const STALLS = '(the model stalls)';
const OFF_SCHEMA = '(the model answers off schema)';
// fake_server.py's RELEASE: once it is in the logs dir, stalled calls answer.
const RELEASE_FILE = 'release';
const QUIET = '(nothing relevant here)';
const SPLIT_MEMO = '06_split_memo';
// The relevance rounds r3con's default config runs, and the page draws.
const ROUNDS = 2;
const SCRIPTED_ANSWER = "The check's scripted model answers this.";
const SCRIPTED_CITES = {
  r2: Array.from({ length: SAMPLE_DOCS }, (_, i) => ({ provenance: [i === 0 ? 1 : 0] })),
  parsed: Array.from({ length: SAMPLE_DOCS }, (_, i) => ({ provenance: [i] })),
  answer: { provenance: { summary_ids: [0, 1], parse_ids: [0, 1] } },
};
const LIVE_DONE = /^● done · openai\/gpt-6-luna · [\d,]+ tokens · \$[\d.]+$/;
const LIVE_READY = /^ready · openai\/gpt-6-luna · \$[\d.]+$/;
const LIVE_ERROR = "RuntimeError: the check's model is down";
const LIVE_FAILED = `● failed in relevance · ${LIVE_ERROR}`;
const STALE_TOKEN = "this page's launch token is missing or stale; reopen it from the r3con app terminal";
const LOST_CONNECTION = '● lost the connection to r3con app';
const UNAUTHORIZED_LOG = 'Failed to load resource: the server responded with a status of 401';
const CUT_STREAM_LOG = 'Failed to load resource: net::ERR_INCOMPLETE_CHUNKED_ENCODING';
const REFUSED_LOG = 'Failed to load resource: the server responded with a status of 422';
const ABORTED_LOG = 'Failed to load resource: net::ERR_FAILED';
const SERVER_ERROR_LOG = 'Failed to load resource: the server responded with a status of 500';
const ASK_WHEN_DONE = 'this run is still going; ask once it has finished';
const STOPPING = 'stopping · the call in flight finishes first';
const STOPPED_IN_RELEVANCE = '● stopped in relevance';
const STOPPED_IN_PARSING = '● stopped in parsing';
const STOPPED_WHILE_READING = '● stopped while reading';
const STOPPED_ASK = '● ask stopped in reasoning';
const STOPPED = '● stopped';
// A stopped ask's folded line.
const STOPPED_ASK_LINE = 'stopped in reasoning';
const CANCEL_REFUSED = 'no run with that id';
const NOT_FOUND_LOG = 'Failed to load resource: the server responded with a status of 404';
const LIVE_ASK = 'live ask over this run';
const PDF_TEXT = 'Incidents this quarter: 7.';
const PADDED_TEXT = 'A memo padded with a stream that no page uses.';
// The page's PREVIEW_CHARS: a file's preview shows this much of its text, and says so when there is more.
const PREVIEW_CHARS = 20000;
const PREVIEW_NOTE = 'first 20,000 characters shown';
// The page's PREVIEW_LINES: a parse's popup shows this many lines of its JSON, and says so when there are more.
const PREVIEW_LINES = 5000;
const PREVIEW_LINES_NOTE = 'first 5,000 lines shown';
// The page's JSON_INDENT: a parse's JSON is indented this many spaces a level.
const JSON_INDENT = 2;
// A stream this long, which no page uses, makes a PDF of more bytes than its preview has characters.
const PDF_PADDING = PREVIEW_CHARS + 5000;
// The Markdown r3con makes of report.docx below.
const WORD_TEXT = 'Incidents by site:\n\n| Site | Incidents |\n| --- | --- |\n| Halloran | 7 |\n| Merrow | 3 |';
const ZIPPED_TEXT = 'A memo that came in a zip.\n';
// What memos.zip holds, all in one folder: a memo, a blank one, a .py the walk leaves out by its suffix, and hidden,
// macOS and lock files the page leaves out unsaid.
const ZIPPED_ENTRIES = {
  'memos/': '',
  'memos/01_zipped.txt': ZIPPED_TEXT,
  'memos/02_blank.txt': '',
  'memos/.DS_Store': 'x',
  '__MACOSX/memos/._01_zipped.txt': 'x',
  'memos/notes.py': "print('not a document')\n",
  'memos/~$draft.docx': 'x',
};
// Every document the checks add, by name: the arguments its writer in tests/documents.py takes for it, so
// that the checks and the tests make them one way. A PDF with no text makes a scan.
const DOCUMENT_SPECS = {
  '01_memo.pdf': { text: PDF_TEXT },
  '02_scan.pdf': {},
  '03_padded.pdf': { text: PADDED_TEXT, padding: PDF_PADDING },
  'empty.docx': {},
  'report.docx': { blocks: ['Incidents by site:', [['Site', 'Incidents'], ['Halloran', '7'], ['Merrow', '3']]] },
  'memos.zip': { entries: ZIPPED_ENTRIES },
  'photos.zip': { entries: { 'photo.jpg': 'x' } },
};
const MIME_TYPES = {
  '.pdf': 'application/pdf',
  '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  '.zip': 'application/zip',
};
const UTF16LE_BOM = [0xff, 0xfe];
const UTF32LE_BOM = [0xff, 0xfe, 0x00, 0x00];
// More bytes in UTF-16 than PREVIEW_CHARS, but fewer characters.
const LONG_UTF16_CHARS = PREVIEW_CHARS * 0.75;
const WRITE_DOCUMENTS =
  'import json, sys; from pathlib import Path; from tests.documents import WRITERS\n' +
  'for name, spec in json.loads(sys.argv[2]).items(): WRITERS[Path(name).suffix](Path(sys.argv[1], name), **spec)';
// An old Office file is an OLE2 compound file, whose header holds NUL bytes.
const OLE2_START = [0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1, 0x00, 0x00];
const NOT_TEXT = 'r3con reads text, PDF and Word (.docx) files: save as one of those, then add again.';
const OLD_WORD = "r3con reads Word's .docx, not the older .doc: save as .docx, PDF or text, then add again.";
const NEEDS_APP = 'PDF, Word and zip files need r3con app; this prototype reads only text files.';
const UNREADABLE_ZIP = "Couldn't unzip it: it's corrupt, or not a zip.";
const NO_DOCUMENTS = 'No text, PDF or Word (.docx) files in it';
const BROWSER_COULD_NOT_READ = "The browser couldn't read it.";
const TOO_MANY_FILES = 'A tab takes 40 files at most.';
// The page's MAX_FILES.
const MAX_FILES = 40;
const NO_TEXT_LAYER = "No text layer: a scanned PDF needs OCR first, or it isn't a readable PDF.";
const NO_TEXT = "No text: it's empty or only images, or r3con can't read it.";
const NEW_TAB = 'new tab';
const FILES_LOCKED = "These are this tab's files. Press + for a new tab.";
const ADD_FILES = 'no files yet · add them in the files tab';
const LOCKED_DROP = 'This tab has run, so its files are locked: press + for a new tab.';
const RAN_FIRST = "The tab's run started first, so its files are locked.";
const DROP_ADDS = 'drop to add files';
const DROP_LOCKED = 'this tab has run · press + for a new tab';
const FIRST_TAB_BEFORE_RUN = '1 new · 5 files';
const DRAG_STEPS = 8;
// Long enough that a second click lands while the first save is still in flight.
const SLOW_SAVE_MS = 400;
// fake_server.py's summary_of: a text's first line that is not quiet.
const scriptedSummary = (text) => text.split('\n').find((line) => line.trim() && !line.includes(QUIET)) ?? '';
// fake_server.py's parse of a document read whole: its first line as a memo's title, its other lines nested under it.
function scriptedParse(text) {
  const [title, ...lines] = text.split('\n').filter((line) => line.trim());
  return { memos: [{ title, lines: lines.map((line) => ({ text: line })) }] };
}

let browser;
let failures = 0;
let documents;

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

// The drawing has no box of its own, so it is never visible: wait for it to be in the page.
const waitLit = (page, timeout = ACTION_TIMEOUT_MS) => page.waitForSelector('#world .is-lit', { state: 'attached', timeout });

async function runSample(page) {
  await page.click('#run-btn');
  await waitLit(page);
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
const focusedId = (page) => page.evaluate(() => document.activeElement.id);
const shownTab = async (page) => (await page.evaluate(tabRow)).findIndex((tab) => tab.selected);

async function zoomAt(page, { x, y }, deltaY) {
  await page.mouse.move(x, y);
  await page.keyboard.down('Control');
  await page.mouse.wheel(0, deltaY);
  await page.keyboard.up('Control');
}

function positionOf(page, id) {
  return page.$eval(nodeSel(id), (el) => ({ x: parseFloat(el.style.left), y: parseFloat(el.style.top) }));
}

async function boxCentre(page, selector) {
  const r = await rect(page, selector);
  return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
}

function centre(page, id) {
  return boxCentre(page, `${nodeSel(id)} .icon`);
}

// Presses at `from`, moves to `to` in `steps` and releases; returns what `whileHeld` reads just before the release.
async function drag(page, from, to, { whileHeld, steps = DRAG_STEPS } = {}) {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps });
  const held = await whileHeld?.();
  await page.mouse.up();
  return held;
}

// drag by touch, lifted with `end`: touchEnd, or touchCancel. Playwright's touchscreen only taps, so this goes through Chromium's own touch input.
async function touchDrag(page, from, to, { whileHeld, end = 'touchEnd' } = {}) {
  const cdp = await page.context().newCDPSession(page);
  const touchAt = (type, p) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: p ? [p] : [] });
  await touchAt('touchStart', from);
  for (let i = 1; i <= DRAG_STEPS; i += 1) await touchAt('touchMove', { x: from.x + ((to.x - from.x) * i) / DRAG_STEPS, y: from.y + ((to.y - from.y) * i) / DRAG_STEPS });
  const held = await whileHeld?.();
  await touchAt(end);
  await cdp.detach();
  return held;
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
  const match = readFileSync(SCRIPT_FILE, 'utf8').match(/const SAMPLE = (\{[\s\S]*?\n\});/);
  return match ? JSON.parse(match[1]) : null;
}

// Each sample memo's name without its suffix, as the page labels it.
const SAMPLE_STEMS = recordedSample().docs.map((d) => d.name.replace(/\.txt$/, ''));
const stem = (i) => SAMPLE_STEMS[i];

// The fields of a parse that hold records, as its popup shows them.
const filledFields = (parse) => Object.fromEntries(Object.entries(parse).filter(([, records]) => records.length > 0));

function recordedAsks() {
  const match = readFileSync(SCRIPT_FILE, 'utf8').match(/const RECORDED_ASKS = (\[[\s\S]*?\n\]);/);
  return match ? JSON.parse(match[1]) : [];
}

// app.js as it would be shipped with a sample of fewer memos, or with no recorded asks.
function withFewerMemos(script) {
  const sample = recordedSample();
  if (!sample) throw new Error('app.js has no SAMPLE literal for the checks to cut down');
  const fewer = `const SAMPLE = ${JSON.stringify({ ...sample, docs: sample.docs.slice(0, FEWER_MEMOS) })};`;
  return script.replace(/const SAMPLE = \{[\s\S]*?\n\};/, () => fewer);
}
const withNoAsks = (script) => script.replace(/const RECORDED_ASKS = \[[\s\S]*?\n\];/, 'const RECORDED_ASKS = [];');

// app.js with the sample's first parse holding its first record over and over, past PREVIEW_LINES lines of JSON.
function withLongParse(script) {
  const sample = recordedSample();
  const [record] = sample.parsed[0].parse.site_incident_reports;
  const parsed = [{ ...sample.parsed[0], parse: { site_incident_reports: Array(PREVIEW_LINES).fill(record) } }, ...sample.parsed.slice(1)];
  return script.replace(/const SAMPLE = \{[\s\S]*?\n\};/, () => `const SAMPLE = ${JSON.stringify({ ...sample, parsed })};`);
}

// What `read` finds on the page as shipped, then on the page with app.js passed through `edit`.
async function shippedAndEdited(area, options, edit, read) {
  const serveEdited = (page) => {
    const body = edit(readFileSync(SCRIPT_FILE, 'utf8'));
    return page.route('**/app.js', (route) => route.fulfill({ contentType: 'text/javascript', body }));
  };
  const found = [];
  for (const before of [undefined, serveEdited]) await withPage(area, { ...options, before }, async (page) => found.push(await read(page)));
  return found;
}

async function introCounts(area, options) {
  const intros = await shippedAndEdited(area, options, withFewerMemos, (page) => page.textContent(INTRO));
  const name = `the intro counts the sample memos: ${SAMPLE_DOCS} as shipped, ${FEWER_MEMOS} in a page shipped with ${FEWER_MEMOS}`;
  check(area, name, isDeepStrictEqual(intros, [SAMPLE_DOCS, FEWER_MEMOS].map(loadedLine)), intros.join(' | '));
}

// The edges an answer's lit path holds: the run's own answer by default, or `answerId` with `provenance`.
function lineage(sample, docIds, answerId = 'answer-0', provenance = sample.answer.provenance) {
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

// A log's steps as RUN_STEPS lists them.
const stepsOf = (log) => log.steps.map(({ step, nested, links }) => [step, nested.length, links.map((id) => id.split('-')[0]).join(' ')]);
// The lines nested under a log's parsing step, sorted.
const parsedLines = (log) => [...(log.steps.find(({ step }) => step === `parsing ${SAMPLE_DOCS} documents`)?.nested ?? [])].sort();
// The parse an open parsed node's popup shows.
const shownParse = (page) => page.$eval('#inspector pre.code', (el) => JSON.parse(el.textContent));

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

// The shown tab's last turn's lines.
function lastTurnLines() {
  const turn = [...document.querySelectorAll('.transcript:not([hidden]) .turn')].at(-1);
  return turn ? [...turn.querySelectorAll('.line .t')].map((el) => el.textContent) : [];
}

function lastTurnSays(starts) {
  const turn = [...document.querySelectorAll('.transcript:not([hidden]) .turn')].at(-1);
  const lines = turn ? [...turn.querySelectorAll('.line .t')].map((el) => el.textContent) : [];
  return starts.every((start) => lines.some((text) => text.startsWith(start)));
}

// The shown tab's last turn's log: its button's name, from what labels it, whether the button controls it and has it
// open, its folded line and the mark before it, each step in view with the lines nested under it and the nodes they
// link to, and the lines in view outside the log.
function logOf() {
  const turn = [...document.querySelectorAll('.transcript:not([hidden]) .turn')].at(-1);
  const log = turn.querySelector('.log');
  const toggle = log.querySelector('.log-toggle');
  const lines = log.querySelector('.log-lines');
  const textOf = (el) => el.querySelector('.t').textContent;
  const steps = [];
  for (const el of [...lines.querySelectorAll('.line')].filter((line) => line.checkVisibility())) {
    if (el.classList.contains('main')) steps.push({ step: textOf(el), nested: [], links: [] });
    else steps.at(-1).nested.push(textOf(el));
    if (el.dataset.node) steps.at(-1).links.push(el.dataset.node);
  }
  const outside = [...turn.querySelector('.turn-body').children].filter((el) => el !== log && el.checkVisibility());
  return {
    logs: turn.querySelectorAll('.log').length,
    name: toggle.getAttribute('aria-labelledby').split(' ').map((id) => document.getElementById(id).textContent).join(' '),
    question: textOf(turn.querySelector('.line.user')),
    controls: toggle.getAttribute('aria-controls') === lines.id,
    open: toggle.getAttribute('aria-expanded') === 'true',
    folded: textOf(log.querySelector('.log-head')),
    mark: log.querySelector('.log-head > span[aria-hidden]').className,
    steps,
    after: outside.map(textOf),
  };
}

// Each log toggle's name, from what labels it.
function logNames() {
  const textOf = (id) => document.getElementById(id).textContent;
  return [...document.querySelectorAll('.log-toggle')].map((toggle) => toggle.getAttribute('aria-labelledby').split(' ').map(textOf).join(' '));
}

// The note under the composer, if shown, and how far the composer's foot is from the chat column's.
function composerPlace() {
  const note = document.getElementById('mode-note');
  const foot = (id) => document.getElementById(id).getBoundingClientRect().bottom;
  return { note: note.checkVisibility() ? note.textContent : null, gap: Math.round(foot('side') - foot('ask-form')) };
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
  return {
    primary: document.getElementById('run-btn').textContent,
    stop: !document.getElementById('stop-btn').hidden,
    newRun: !document.getElementById('new-run-btn').hidden,
  };
}

// The answer node each line into an answer ends at: faint lines from what it was shown, cite lines from what it cites.
function answerLines() {
  const ends = (kind) => [...document.querySelectorAll(`.edge.${kind}`)].map((e) => e.dataset.key.split('>')[1]).filter((to) => to.startsWith('answer'));
  return { faint: ends('context'), cited: ends('cite') };
}

// Each tab in the row, with the × after it: shown while there are two tabs or more, and enabled while the tab is idle.
function tabRow() {
  return [...document.querySelectorAll('#run-tabs .run-tab')].map((tab) => {
    const close = tab.nextElementSibling;
    return {
      id: tab.id,
      role: tab.getAttribute('role'),
      label: tab.textContent,
      selected: tab.getAttribute('aria-selected') === 'true',
      focusable: tab.tabIndex === 0,
      keys: tab.getAttribute('aria-keyshortcuts'),
      hasClose: !close.hidden,
      closable: !close.hidden && !close.disabled,
      closeForMouseOnly: close.tabIndex === -1 && close.getAttribute('aria-hidden') === 'true',
    };
  });
}

// The tab row's colors as painted, each an element's own background or else the first behind it: the bar's, each
// tab's and its ×'s, with whether the × meets its tab and where the pair starts and ends, and the text in the row,
// + included, with the lowest contrast against what is behind it.
function tabColors() {
  const painted = (el) => {
    for (; el; el = el.parentElement) {
      const color = getComputedStyle(el).backgroundColor;
      if (color !== 'rgba(0, 0, 0, 0)') return color;
    }
    return null;
  };
  // A pixel painted in the color reads back as sRGB bytes, whatever syntax the color was computed in.
  const pixel = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  const luminance = (color) => {
    pixel.fillStyle = color;
    pixel.fillRect(0, 0, 1, 1);
    const [r, g, b] = [...pixel.getImageData(0, 0, 1, 1).data.slice(0, 3)].map((byte) => {
      const c = byte / 255;
      return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const contrast = (el) => {
    const [high, low] = [luminance(getComputedStyle(el).color), luminance(painted(el))].sort((a, b) => b - a);
    return { ratio: (high + 0.05) / (low + 0.05), text: el.textContent };
  };
  const list = document.getElementById('run-tabs');
  const texts = [...list.querySelectorAll('.run-tab, .num, .tab-close:not([hidden])'), document.getElementById('new-tab-btn')];
  const lowest = texts.map(contrast).reduce((a, b) => (b.ratio < a.ratio ? b : a));
  const tabs = [...list.querySelectorAll('.run-tab')].map((tab) => {
    const box = tab.getBoundingClientRect();
    const close = tab.nextElementSibling.getBoundingClientRect();
    const selected = tab.getAttribute('aria-selected') === 'true';
    return { selected, color: painted(tab), closeColor: painted(tab.nextElementSibling), joined: Math.abs(close.left - box.right) < 0.5, left: box.left, right: close.right };
  });
  return { bar: painted(list), tabs, lowest };
}

// The focused tab shows its focus ring, and the ring lies inside the tab row, which would clip it.
function focusRingFits() {
  const tab = document.activeElement;
  const style = getComputedStyle(tab);
  const ring = parseFloat(style.outlineOffset) + parseFloat(style.outlineWidth);
  const box = tab.getBoundingClientRect();
  const row = document.getElementById('run-tabs').getBoundingClientRect();
  const fits = box.left - ring >= row.left && box.right + ring <= row.right && box.top - ring >= row.top && box.bottom + ring <= row.bottom;
  return tab.matches('.run-tab:focus-visible') && style.outlineStyle !== 'none' && ring > 0 && fits;
}

// The shown tab as drawn, so it can be compared with how it was left: the view, every node's place, every
// line's path, whether the drawing is lit and along which lines, and the status line.
function panelSnapshot() {
  const world = document.getElementById('world');
  const nodes = [...world.querySelectorAll('.node')].map((el) => `${el.dataset.id} ${el.style.left},${el.style.top} ${el.className}`);
  const edges = [...world.querySelectorAll('.edge')].map((el) => `${el.dataset.key} ${el.getAttribute('d')}`);
  return {
    view: world.style.transform,
    nodes: nodes.sort(),
    edges: edges.sort(),
    isLit: world.querySelector('.drawing.is-lit') !== null,
    lit: [...world.querySelectorAll('.edge.lit')].map((el) => el.dataset.key).sort(),
    status: document.getElementById('run-status').textContent,
  };
}

function noHorizontalScroll() {
  return document.documentElement.scrollWidth <= document.documentElement.clientWidth;
}

// Only the animations a node or line plays as it appears count, not transitions such as the view's glide.
function panelStill() {
  const world = document.getElementById('world');
  return !document.getAnimations().some((a) => a.playState === 'running' && world.contains(a.effect?.target) && ['node-in', 'draw', 'fade'].includes(a.animationName));
}

// Lines playing an animation that do not end at answer `id`, whose own lines draw and fade in as it lands.
function linesReplayed(id) {
  const replays = (a) => a.playState === 'running' && a.effect?.target?.matches?.('#world .edge') && !a.effect.target.dataset.key.endsWith(`>${id}`);
  return document.getAnimations().filter(replays).length;
}

function answerWorking() {
  return document.querySelector('#world .node.answer.working') !== null;
}

function roundOneWorking() {
  return document.querySelector('#world .node.r1.working') !== null;
}

// The shown tab's run has ended with its status matching `source`, and it has no asks: a new run's tab.
function newRunDone(source) {
  return !document.querySelector('.node[data-id="answer-1"]') && new RegExp(source).test(document.getElementById('run-status').textContent);
}

function pressedAnswers() {
  return [...document.querySelectorAll('.node.answer')].map((el) => [el.dataset.id, el.getAttribute('aria-pressed')]);
}

function recordStatuses() {
  window.statuses = [];
  const status = document.getElementById('run-status');
  new MutationObserver(() => window.statuses.push(status.textContent)).observe(status, { childList: true, characterData: true, subtree: true });
}

// Records each frame the structure's spinner is given while it works, not the mark it is drawn with.
function recordSpins() {
  window.spins = [];
  new MutationObserver((records) => {
    for (const { target } of records) if (target.matches?.('.node.schema.working .spin')) window.spins.push(target.textContent);
  }).observe(document.getElementById('world'), { childList: true, subtree: true });
}

function failedNode(id) {
  return Boolean(document.querySelector(`.node[data-id="${id}"]`)?.classList.contains('failed'));
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

function fitOfRun() {
  const panel = document.getElementById('panel').getBoundingClientRect();
  const isOutside = (el) => {
    const r = el.getBoundingClientRect();
    return r.right < panel.left || r.left > panel.right || r.bottom < panel.top || r.top > panel.bottom;
  };
  const outside = [...document.querySelectorAll('.node .icon')].filter(isOutside).length;
  return { outside, zoom: document.getElementById('zoom-level').textContent };
}

// Records the view as drawn, mid-glide included, as [zoom, x, y]: now, then every frame until tab `tab`'s
// run starts and for `ms` after, or for `timeout` if it never does.
function recordViews([tab, ms, timeout]) {
  const world = document.getElementById('world');
  const views = [];
  const start = performance.now();
  let opened = null;
  window.recordedViews = new Promise((resolve) => {
    const frame = (now) => {
      const { a, e, f } = new DOMMatrix(getComputedStyle(world).transform);
      views.push([a, e, f]);
      const started = document.getElementById(`run-tab-${tab}`)?.getAttribute('aria-selected') === 'true' && document.querySelector('.transcript:not([hidden]) .turn');
      if (opened === null && started) opened = now;
      if (now < (opened === null ? start + timeout : opened + ms)) requestAnimationFrame(frame);
      else resolve({ views, opened: opened !== null });
    };
    frame(start);
  });
}

// ---------- areas ----------

async function load() {
  for (const layout of Object.values(LAYOUTS)) {
    await withPage('load', { layout, sample: false }, async (page) => {
      check('load', `${layout.name}: the page has no horizontal scroll`, await page.evaluate(noHorizontalScroll));
    });
  }
}

async function beforeRun() {
  await withPage('before a run', { sample: false }, async (page) => {
    const names = { title: await page.title(), brand: await page.textContent('.brand') };
    check('before a run', "the browser tab is titled with the page's brand, ❀ r3con", names.title === '❀ r3con' && names.title === names.brand, JSON.stringify(names));
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
    const onlyRun = isDeepStrictEqual(offered, { primary: 'run ⏎', stop: false, newRun: false });
    check('before a run', 'only run is offered: no stop and no new run', onlyRun, JSON.stringify(offered));
    await page.evaluate(countStatusWrites);
    await page.focus('#question');
    await page.keyboard.type(' today');
    const writes = await page.evaluate(() => window.statusWrites);
    const once = writes === 1 && (await page.textContent('#run-status')) === ready('simulated run', DEFAULT_MODEL);
    check('before a run', 'typing a question rewrites the status once, not on every key', once, `${writes} writes`);
  });
  await introCounts('before a run', { sample: false });
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

  const asks = recordedAsks();
  const offers = [ASKS_HEADER, ...asks.map(offerLabel)];
  const [shipped, none] = await shippedAndEdited('sample replay', {}, withNoAsks, (page) => page.evaluate(lastTurnLines));
  const offered = shipped.slice(-offers.length);
  check('sample replay', `the replay ends by offering the ${asks.length} recorded asks under their header`, isDeepStrictEqual(offered, offers), offered.join(' | '));
  const withoutName = 'a page shipped with no recorded asks ends the replay with neither offers nor their header';
  check('sample replay', withoutName, isDeepStrictEqual(none, shipped.slice(0, -offers.length)), none.at(-1));

  // With motion on, the structure works 1.4 s: more than one 1.12 s cycle of its spinner.
  await withPage('sample replay', { sample: false }, async (page) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.evaluate(recordSpins);
    await page.click('#run-btn');
    const landed = await becomes(page, landedNode, 'schema', REPLAY_MS);
    const spins = await page.evaluate(() => window.spins);
    const after = await page.textContent(`${nodeSel('schema')} .spin`);
    const cycled = SPIN_FRAMES.every((frame) => spins.includes(frame)) && SPIN_FRAMES.join('').repeat(spins.length).includes(spins.join(''));
    const spinName = `while a node works its spinner cycles through ${SPIN_FRAMES.join('')}, and once it settles it shows ${MARK}`;
    check('sample replay', spinName, landed && cycled && after === MARK, `${spins.join('')}; then ${after}`);
  });
}

// ---------- chat log: a run or an ask told as a log, folded under one line ----------

async function chatLog() {
  await withPage('chat log', {}, async (page) => {
    const sample = recordedSample();
    const folded = await page.evaluate(logOf);
    const oneLog = folded.logs === 1 && folded.controls && folded.name === `${folded.question} ${folded.folded}`;
    check('chat log', "a run's turn holds one log, behind a button that controls it, named for its question and its folded line", oneLog, folded.name);
    const quiet = !folded.open && folded.folded === SAMPLE_READY && folded.mark === 'ok' && folded.steps.length === 0;
    check('chat log', 'once the replay ends its log is folded under one line: ready, the model and the cost', quiet, folded.folded);
    const offers = [ASKS_HEADER, ...recordedAsks().map(offerLabel)];
    const underName = 'outside the log the turn shows only the answer, then the recorded asks on offer: nothing about the answer';
    check('chat log', underName, isDeepStrictEqual(folded.after, [sample.answer.text, ...offers]), folded.after.slice(0, 2).join(' | '));

    await page.click('.log-toggle');
    const opened = await page.evaluate(logOf);
    const openName = "its chevron opens the log: each step in turn, each document's line under its own step and linked to its node, and no line repeats ready";
    check('chat log', openName, opened.open && isDeepStrictEqual(stepsOf(opened), REPLAY_STEPS), JSON.stringify(stepsOf(opened)));
    const roundLines = opened.steps.slice(1, 1 + ROUNDS).flatMap(({ nested }) => nested).sort();
    const settled = ['r1', 'r2'].flatMap((kind) => sample[kind].map(({ summary }, i) => `${stem(i)} — ${summary ? SNIPPET_READY : 'nothing relevant'}`)).sort();
    check('chat log', "each round's line says its document's snippet is ready, or that nothing was relevant", isDeepStrictEqual(roundLines, settled), roundLines[0]);
    const parseLines = parsedLines(opened);
    const filled = sample.parsed.map(({ parse }, i) => `${stem(i)} — ${Object.keys(filledFields(parse)).length ? STRUCTURE_READY : NOTHING_PARSED}`).sort();
    check('chat log', "each parsed document's line says its structure is ready, or that nothing was parsed", isDeepStrictEqual(parseLines, filled), parseLines.join(' | '));
    await page.click('.log-lines .line[data-node="r2-3"]');
    const title = (await popup(page))?.title;
    check('chat log', 'a nested line still opens its node on the panel', title === '● 04_contractor_registry · round 2', title);
    await page.keyboard.press('Escape');
    await page.click('.log-toggle');
    const byClick = !(await page.evaluate(logOf)).open;
    await page.focus('.log-toggle');
    await page.keyboard.press('Enter');
    const byEnter = (await page.evaluate(logOf)).open;
    await page.keyboard.press('Space');
    const bySpace = !(await page.evaluate(logOf)).open;
    check('chat log', 'a click on the chevron folds the log again, and from the keyboard Enter opens it and Space folds it', byClick && byEnter && bySpace, `click ${byClick}, Enter ${byEnter}, Space ${bySpace}`);

    const [first, second] = recordedAsks();
    for (const ask of [first, second]) {
      await page.getByRole('button', { name: offerLabel(ask), exact: true }).click();
      await page.press('#question', 'Enter');
    }
    await send(page, NEW_QUESTION);
    const said = await page.textContent(CHAT);
    const commentary = COMMENTARY.filter((text) => said.includes(text));
    check('chat log', "no chat line only describes the method, and none gives an answer's citations or source", commentary.length === 0, commentary.join(', '));
    const names = await page.evaluate(logNames);
    const found = [];
    for (const name of names) found.push(await page.getByRole('button', { name, exact: true }).count());
    const distinct = names.length === 4 && new Set(names).size === names.length && found.every((n) => n === 1);
    check('chat log', 'each log button has a name of its own, even for one question asked of two models', distinct, names.join(' | '));
  });

  await withPage('chat log', { sample: false }, async (page) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.click('#run-btn');
    await becomes(page, statusStarts, 'running · round 2', REPLAY_MS);
    const second = await page.evaluate(logOf);
    await becomes(page, statusStarts, 'running · parsing', REPLAY_MS);
    const parsing = await page.evaluate(logOf);
    const going = [second, parsing].every((log) => log.mark === 'log-spin' && !log.open) && second.folded === RUN_STEPS[1][0] && parsing.folded === RUN_STEPS[3][0];
    check('chat log', 'while a replay goes its log stays folded, its line the step it is on, with a spinner', going, `${second.folded} → ${parsing.folded}`);
    await pressStop(page);
    const stopped = await page.evaluate(logOf);
    const ended = !stopped.open && stopped.mark === 'err' && stopped.folded === 'stopped in parsing' && stopped.after.length === 0;
    check('chat log', "a stopped replay's log ends on where it stopped, in its folded line, its spinner gone", ended, `${stopped.folded}; ${stopped.after.join(' | ')}`);
  });

  await withPage('chat log', { layout: LAYOUTS.phone, sample: false }, async (page) => {
    await send(page, NEW_QUESTION);
    await waitLit(page);
    const simulated = await page.evaluate(logOf);
    const flags = simulated.after.filter((text) => text.startsWith('simulated'));
    const flagged = !simulated.open && flags.length === 2 && simulated.mark === 'flag' && simulated.folded === `ready · ${DEFAULT_MODEL} · nothing was sent`;
    check('chat log', "400 dark: a simulated run's two warnings stay in view with its log folded, which says nothing was sent", flagged, flags.join(' | '));
    check('chat log', "a simulated run's note says each relevance snippet is the file's start and the citations are made up", flags[0] === SIMULATED_NOTE, flags[0]);
    const target = await rect(page, '.log-toggle');
    const fits = target.width >= MIN_TARGET_PX && target.height >= MIN_TARGET_PX && (await page.evaluate(noHorizontalScroll));
    check('chat log', `400 dark: the chevron is a ${MIN_TARGET_PX}px target, and the page has no horizontal scroll`, fits, `${target.width}×${target.height}`);
    const place = await page.evaluate(composerPlace);
    check('chat log', 'the prototype keeps its note under the composer, since its runs are replays', place.note === PROTOTYPE_NOTE, JSON.stringify(place));
  });

  // Two asks under the run make its chat long enough to scroll while the run replays.
  await withPage('chat log', {}, async (page) => {
    await page.setViewportSize(SHORT_DESKTOP);
    for (const question of [NEW_QUESTION, 'Which memo is the registry?']) await send(page, question);
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.click('#replay-btn');
    await becomes(page, statusStarts, 'running · reading', REPLAY_MS);
    await scrollChatTo(page, 0);
    await becomes(page, statusStarts, 'running · parsing', REPLAY_MS);
    const { top, scrolls } = await page.$eval(CHAT, (el) => ({ top: el.scrollTop, scrolls: el.scrollHeight > el.clientHeight }));
    const stayed = scrolls && top === 0;
    await becomes(page, statusIs, SAMPLE_STATUS, REPLAY_MS);
    const landed = await becomes(page, (selector) => document.querySelector(selector).scrollTop > 0, CHAT);
    check('chat log', 'steps logged into a folded log leave the chat where it was scrolled, and the answer still scrolls it', stayed && landed, `scrolls ${scrolls}; stayed ${stayed}; scrolled on the answer ${landed}`);
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

// Records the view while `start` starts the run of tab `tab`. It passes if that run started, the view moved, and
// every frame lies on a glide from where the view was to where it ends, none of them at a new tab's view.
async function viewOnStart(page, tab, start) {
  await page.evaluate(recordViews, [tab, GLIDE_RECORD_MS, ACTION_TIMEOUT_MS]);
  await start();
  const { views, opened } = await page.evaluate(() => window.recordedViews);
  const [from, to] = [views[0], views.at(-1)];
  const near = (view, other) => view.every((v, i) => Math.abs(v - other[i]) <= VIEW_SLACK[i]);
  const between = (view) => view.every((v, i) => v >= Math.min(from[i], to[i]) - VIEW_SLACK[i] && v <= Math.max(from[i], to[i]) + VIEW_SLACK[i]);
  const strays = views.filter((view) => !between(view));
  const snaps = near(from, NEW_TAB_VIEW) ? [] : views.filter((view) => near(view, NEW_TAB_VIEW));
  const shown = ([k, x, y]) => `${Math.round(k * 100)}% at ${Math.round(x)},${Math.round(y)}`;
  const detail = `tab ${tab} ${opened ? 'started' : 'never started'}; ${shown(from)} → ${shown(to)} over ${views.length} frames; ${strays.length} off that glide, ${snaps.length} at 100% at 0,0`;
  return { pass: opened && !near(from, to) && !strays.length && !snaps.length, detail };
}

// With motion on, a run's view glides to its fit from where the panel was; a new tab's own starting view never shows.
async function runView() {
  await withPage('run view', { sample: false }, async (page) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    // The panel's first resize observation refits the view a frame after load.
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const first = await viewOnStart(page, 1, () => page.click('#run-btn'));
    check('run view', "the first run glides the view from where it was to the run's fit, never through 100% at the top left", first.pass, first.detail);
    await waitLit(page, REPLAY_MS);
    const fitted = await transform(page);
    const frame = await page.evaluate(panelBox);
    await zoomAt(page, { x: frame.left + frame.width / 2, y: frame.top + frame.height / 2 }, 200);
    await becomes(page, viewChanged, fitted);
    const second = await viewOnStart(page, 2, () => send(page, NEW_QUESTION, 'Control+Enter'));
    const secondName = "a new run, started once the first has finished and been zoomed out, glides likewise from that zoom to its own tab's fit";
    check('run view', secondName, second.pass, second.detail);
  });
}

// Writes every document in `specs` with one uv run, and returns each by name, as setInputFiles takes it.
function writeDocuments(specs) {
  const dir = mkdtempSync(join(tmpdir(), 'r3con-documents-'));
  try {
    execFileSync('uv', ['run', 'python', '-c', WRITE_DOCUMENTS, dir, JSON.stringify(specs)], { cwd: REPO_DIR });
    const written = (name) => ({ name, mimeType: MIME_TYPES[extname(name)], buffer: readFileSync(join(dir, name)) });
    return Object.fromEntries(Object.keys(specs).map((name) => [name, written(name)]));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

const oldWord = () => ({ name: 'old.doc', mimeType: 'application/msword', buffer: Buffer.from(OLE2_START) });
const oldPowerPoint = () => ({ name: 'old.ppt', mimeType: 'application/vnd.ms-powerpoint', buffer: Buffer.from(OLE2_START) });
const filesError = (page) => page.$eval('#files-error', (el) => (el.hidden ? null : el.textContent));

// The prototype has no r3con app to extract a PDF's or a Word file's text.
async function filesNotText() {
  await withPage('files', { sample: false }, async (page) => {
    await page.click('#tab-files');
    const longUtf16 = utf16('utf16_long.txt', 'A Notepad note, saved as Unicode. '.repeat(LONG_UTF16_CHARS).slice(0, LONG_UTF16_CHARS), UTF16LE_BOM);
    const longText = memo('long.txt', 'A long memo. '.repeat(PREVIEW_CHARS).slice(0, PREVIEW_CHARS + 1));
    const texts = [memo('notes.log', 'A log line.\n'), longUtf16, longText];
    await page.setInputFiles('#file-input', [documents['01_memo.pdf'], documents['report.docx'], documents['memos.zip'], oldWord(), ...texts]);
    await removeButton(page, 'notes.log').waitFor();
    const listed = await removeNames(page);
    const error = await filesError(page);
    const expected = `Not added: 01_memo.pdf, memos.zip, report.docx. ${NEEDS_APP} Not added: old.doc. ${OLD_WORD}`;
    const name = 'the prototype refuses a PDF, a .docx and a zip, saying they need r3con app, and an old .doc, saying to save it as .docx, PDF or text, and adds a .log';
    const added = isDeepStrictEqual(listed, ['remove long.txt', 'remove notes.log', 'remove utf16_long.txt']);
    check('files', name, added && error === expected, `${listed.join(', ')}; ${error}`);

    const shown = await openFileView(page, 'utf16_long.txt');
    const whole = shown.meta === `${fmtKb(longUtf16.buffer.length)} · your file` && shown.text.length === LONG_UTF16_CHARS;
    const wholeName = "a UTF-16 file of more bytes than the preview's characters, but fewer characters, shows its whole text, with no note that only its start is shown";
    check('files', wholeName, longUtf16.buffer.length > PREVIEW_CHARS && whole, `${shown.meta}; ${shown.text.length} characters`);
    await page.click('#file-back');
    const cut = await openFileView(page, 'long.txt');
    const noted = cut.meta === `${fmtKb(longText.buffer.length)} · your file · ${PREVIEW_NOTE}` && cut.text.length === PREVIEW_CHARS;
    check('files', `a file whose text is longer than the preview shows its first ${PREVIEW_CHARS} characters, and says so`, noted, `${cut.meta}; ${cut.text.length} characters`);
  });
}

// With the files list taken out of the page, applying a batch throws; put back, the next batch is added.
async function filesAddFails() {
  await withPage('files', { sample: false, expectedErrors: ["couldn't add files"] }, async (page) => {
    await page.click('#tab-files');
    await page.evaluate(() => {
      const list = document.getElementById('file-list');
      const { parentNode, nextSibling } = list;
      window.putBack = () => parentNode.insertBefore(list, nextSibling);
      list.remove();
    });
    await page.setInputFiles('#file-input', memo('first.txt', 'A first memo.\n'));
    const failed = "Couldn't add those files.";
    const said = await becomes(page, (text) => document.getElementById('files-error').textContent === text, failed);
    const shown = [await page.textContent('#files-error'), await page.textContent('#files-said')];
    await page.evaluate(() => window.putBack());
    await page.setInputFiles('#file-input', memo('second.txt', 'A second memo.\n'));
    const next = await becomes(page, () => document.getElementById('files-said').textContent === '1 file added.');
    const listed = await removeNames(page);
    const name = 'a batch whose adding fails says so in the files pane, without the reading line, and the next batch is still added';
    const recovered = said && shown[1] === '' && next && listed.includes('remove second.txt') && (await filesError(page)) === null;
    check('files', name, recovered, `${shown.join(' | ')}; then ${listed.join(', ')}`);
  });
}

// ---------- remove: one file at a time, from the files tab ----------

const removeButton = (page, name) => page.getByRole('button', { name: `remove ${name}`, exact: true });
const focusedName = (page) => page.evaluate(() => document.activeElement.getAttribute('aria-label') ?? document.activeElement.id);
const removeNames = (page) => page.$$eval('#file-list .remove', (els) => els.map((el) => el.getAttribute('aria-label')));
const docNames = (page) => page.$$eval('#world .node.doc', (els) => els.map((el) => el.getAttribute('aria-label').split(',')[0]));

// Each button in the files list, to open a file or remove it, is at least `px` wide and tall.
function targetsFit(px) {
  return [...document.querySelectorAll('#file-list button')].every((button) => {
    const box = button.getBoundingClientRect();
    return box.width >= px && box.height >= px;
  });
}

// The files pane scrolls on its own, so the page's width says nothing about it.
function listFits() {
  const view = document.getElementById('file-list-view');
  return view.scrollWidth <= view.clientWidth;
}

async function removeArea() {
  const names = recordedSample().docs.map((d) => d.name);
  for (const layout of Object.values(LAYOUTS)) {
    await withPage('remove', { layout, sample: false }, async (page) => {
      await page.click('#tab-files');
      const buttons = await removeNames(page);
      const named = isDeepStrictEqual(buttons, names.map((name) => `remove ${name}`));
      const big = await page.evaluate(targetsFit, MIN_TARGET_PX);
      check('remove', `${layout.name}: each file has a remove button named for it, and each button in the list is at least ${MIN_TARGET_PX} px each way`, named && big, buttons.join(', '));
      await page.setInputFiles('#file-input', memo(LONG_NAME, 'A memo with a long name.\n'));
      await removeButton(page, LONG_NAME).waitFor();
      check('remove', `${layout.name}: with a long file name, the files list has no horizontal scroll`, await page.evaluate(listFits));
    });
  }

  await withPage('remove', { sample: false }, async (page) => {
    await page.click('#tab-files');
    await removeButton(page, names[1]).click();
    const left = names.filter((_, i) => i !== 1);
    const after = {
      docs: await docNames(page),
      count: await page.textContent('#files-count'),
      hint: await page.textContent('#panel-hint'),
      status: await page.textContent('#run-status'),
      head: (await page.textContent('#files-head')).split(' · ').at(-1),
      said: await page.textContent('#files-said'),
      live: await page.$eval('#files-said', (el) => el.getAttribute('aria-live')),
      focus: await focusedName(page),
    };
    const expected = {
      docs: left,
      count: String(left.length),
      hint: `${left.length} documents · ready to run`,
      status: `${left.length} documents ready · simulated run · ${DEFAULT_MODEL}`,
      head: 'sample memos',
      said: `Removed ${names[1]}.`,
      live: 'polite',
      focus: `remove ${names[2]}`,
    };
    const beforeName = "removing a file before a run drops its node, recounts, still calls the memos left the sample, says so in the files pane's live line, and the focus moves to the next file's remove";
    check('remove', beforeName, isDeepStrictEqual(after, expected), JSON.stringify(after));

    const offer = await page.$eval('#sample-btn', (button) => (button.hidden ? null : button.textContent));
    check('remove', 'with a sample memo removed, the sample button offers all of them back', offer === ALL_SAMPLE_MEMOS, offer);

    await page.setInputFiles('#file-input', memo('06_extra_memo.txt', 'An extra memo.\n'));
    await removeButton(page, '06_extra_memo.txt').waitFor();
    const listed = await removeNames(page);
    const head = await page.textContent('#files-head');
    const said = [await page.textContent('#files-said'), await lastChatLine(page)];
    const inPlace = `1 file added, in place of the ${left.length} sample memos`;
    const saysSo = said[0] === `${inPlace}.` && said[1].startsWith(`${inPlace}:`);
    const replaced = isDeepStrictEqual(listed, ['remove 06_extra_memo.txt']) && head.endsWith('your files') && saysSo;
    check('remove', 'files added over the sample memos left take their place, and the files pane and the chat say which they replaced', replaced, `${listed.join(', ')}; ${head}; ${said.join(' | ')}`);

    await page.click('#sample-btn');
    const back = { files: (await removeNames(page)).length, hidden: await page.$eval('#sample-btn', (button) => button.hidden) };
    check('remove', 'the sample button brings back all the sample memos, then hides', isDeepStrictEqual(back, { files: names.length, hidden: true }), JSON.stringify(back));

    await removeButton(page, names[1]).click();
    await page.click('#tab-chat');
    await page.click('#run-btn');
    const simulated = await page.textContent('#run-status');
    check('remove', 'with a sample memo removed, the next run is simulated, not the recording', simulated.startsWith('● done · simulated run'), simulated);
    const stillSaid = await page.textContent('#files-said');
    check('remove', "after a removal and then a run, the files pane's line says only what happened, nothing of the next run", stillSaid === `Removed ${names[1]}.`, stillSaid);
  });

  await withPage('remove', { sample: false }, async (page) => {
    await page.click('#tab-files');
    await removeButton(page, names.at(-1)).click();
    const previous = await focusedName(page);
    for (const name of names.slice(0, -1)) await removeButton(page, name).click();
    const none = await focusedName(page);
    const moves = previous === `remove ${names.at(-2)}` && none === 'file-input';
    check('remove', "removing the last file in the list moves the focus to the previous one's remove, and with none left to adding files", moves, `${previous}; ${none}`);
    await page.click('#tab-chat');
    await send(page, NEW_QUESTION);
    const said = await page.evaluate(lastTurnLines);
    check('remove', 'with every file removed, run refuses with the line it gives for no files', said.includes(ADD_FILES_FIRST) && (await count(page, '#world .node')) === 0, said.join(' | '));
  });
}

// ---------- trash: a document node dropped on the panel's can has its file removed ----------

const docNodeId = (page, name) => page.$eval(`#world .node.doc[aria-label^="${name},"]`, (el) => el.dataset.id);
// Read after a frame: Chromium hands a touch move to the page with the next one.
const canLit = (page) => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => resolve(document.getElementById('trash').classList.contains('over')))));
const canShown = (page) => page.isVisible('#trash');
// Drags by mouse, or by touch, and returns whether the can was lit just before the release.
function dragOver(page, from, to, { touch = false, ...options } = {}) {
  const whileHeld = () => canLit(page);
  return touch ? touchDrag(page, from, to, { whileHeld, ...options }) : drag(page, from, to, { whileHeld, ...options });
}

async function trashArea() {
  const names = recordedSample().docs.map((d) => d.name);
  const left = names.filter((_, i) => i !== 1);
  for (const layout of Object.values(LAYOUTS)) {
    const touch = layout === LAYOUTS.phone;
    const by = touch ? 'by touch' : 'by mouse';
    await withPage('trash', { layout, sample: false }, async (page) => {
      const shown = await canShown(page);
      const unspoken = (await page.getAttribute('#trash', 'aria-hidden')) === 'true';
      const [box, hint, panel] = await Promise.all(['#trash', '#panel-hint', '#panel'].map((selector) => rect(page, selector)));
      const corner = box.left - panel.left < box.width && box.top - panel.top < box.height;
      const clear = box.right <= hint.left || box.bottom <= hint.top;
      const placeName = `${layout.name}: before a run the panel has a can at its top left, hidden from assistive tech, clear of the hint beside it`;
      check('trash', placeName, shown && unspoken && corner && clear, JSON.stringify({ shown, unspoken, box, hint }));

      await page.click('#tab-files');
      const id = await docNodeId(page, names[1]);
      const was = await positionOf(page, id);
      const litElsewhere = await dragOver(page, await centre(page, id), await boxCentre(page, '#panel'), { touch });
      const kept = { docs: await docNames(page), pane: (await page.evaluate(filesPane)).names, moved: !isDeepStrictEqual(await positionOf(page, id), was) };
      const keptName = `${layout.name}: a document node dropped ${by} anywhere but the can only moves there, and the can stays unlit`;
      check('trash', keptName, !litElsewhere && isDeepStrictEqual(kept, { docs: names, pane: names, moved: true }), JSON.stringify({ litElsewhere, ...kept }));

      if (touch) {
        const litCancelled = await dragOver(page, await centre(page, await docNodeId(page, names[2])), await boxCentre(page, '#trash'), { touch, end: 'touchCancel' });
        const cancelled = { lit: litCancelled, docs: await docNames(page), unlit: !(await canLit(page)) };
        const cancelName = `${layout.name}: a touch drag cancelled over the lit can removes nothing, and unlights it`;
        check('trash', cancelName, isDeepStrictEqual(cancelled, { lit: true, docs: names, unlit: true }), JSON.stringify(cancelled));
      }

      const focusBefore = await focusedId(page);
      const lit = await dragOver(page, await centre(page, id), await boxCentre(page, '#trash'), { touch });
      const after = {
        lit,
        docs: await docNames(page),
        pane: (await page.evaluate(filesPane)).names,
        count: await page.textContent('#files-count'),
        tab: (await tabLabels(page))[0][0],
        said: await page.textContent('#files-said'),
        chat: await lastChatLine(page),
        status: await page.textContent('#run-status'),
        unlit: !(await canLit(page)),
        focus: await focusedId(page),
      };
      const expected = {
        lit: true,
        docs: left,
        pane: left,
        count: String(left.length),
        tab: `1 new · ${left.length} files`,
        said: `Removed ${names[1]}.`,
        chat: `Removed ${names[1]}: the next question runs over ${left.length} documents.`,
        status: `${left.length} documents ready · simulated run · ${DEFAULT_MODEL}`,
        unlit: true,
        // A press by mouse focuses the node it grabs, and the focus goes to the panel with it; a touch moves none.
        focus: touch ? focusBefore : 'panel',
      };
      const focusSays = touch ? 'the focus staying where it was' : 'the focus on the panel';
      const dropName = `${layout.name}: a document node dragged ${by} onto the can lights it, and released there its file leaves the tab, the panel and the files pane, said as the files pane's remove says it, with ${focusSays}`;
      check('trash', dropName, isDeepStrictEqual(after, expected), JSON.stringify(after));
    });
  }

  await withPage('trash', { sample: false }, async (page) => {
    const spot = await boxCentre(page, '#trash');
    const id = await docNodeId(page, names[0]);
    const node = await centre(page, id);
    const panel = await rect(page, '#panel');
    const blank = { x: panel.left + panel.width * 0.75, y: panel.top + panel.height / 2 };
    await drag(page, blank, { x: blank.x + spot.x - node.x, y: blank.y + spot.y - node.y });
    const view = await transform(page);
    const was = await positionOf(page, id);
    await drag(page, spot, await boxCentre(page, '#panel'));
    const grabbed = { moved: !isDeepStrictEqual(await positionOf(page, id), was), panned: (await transform(page)) !== view, docs: await docNames(page) };
    check('trash', 'a node panned under the can is still grabbed there, not the view', isDeepStrictEqual(grabbed, { moved: true, panned: false, docs: names }), JSON.stringify(grabbed));

    await page.click('#new-tab-btn');
    const empty = await canShown(page);
    const second = ['a_memo.txt', 'b_memo.txt'];
    await page.setInputFiles('#file-input', second.map((name) => memo(name, `A memo, ${name}, for the second tab.\n`)));
    await removeButton(page, second[1]).waitFor();
    const withFiles = await canShown(page);
    const flicked = await dragOver(page, await centre(page, await docNodeId(page, second[0])), spot, { steps: 1 });
    const plus = { empty, withFiles, flicked, docs: await docNames(page), pane: (await page.evaluate(filesPane)).names };
    await showRunTab(page, 0);
    const first = { tab: (await tabLabels(page))[0][0], pane: (await page.evaluate(filesPane)).names };
    const plusName =
      "a + tab shows the can once it has files, not before, and a node flicked onto it in one move lights it and is removed from that tab only: tab 1, not run, keeps all its files";
    const expected = { empty: false, withFiles: true, flicked: true, docs: second.slice(1), pane: second.slice(1) };
    const firstKept = isDeepStrictEqual(first, { tab: FIRST_TAB_BEFORE_RUN, pane: names });
    check('trash', plusName, isDeepStrictEqual(plus, expected) && firstKept, JSON.stringify({ ...plus, first }));

    const chatBefore = await count(page, '.transcript .line');
    const switchTab = () => page.evaluate(() => document.querySelectorAll('.run-tab')[1].click());
    await drag(page, await centre(page, await docNodeId(page, names[1])), spot, { whileHeld: switchTab });
    const switched = { shown: await shownTab(page), pane: (await page.evaluate(filesPane)).names, chat: await count(page, '.transcript .line') };
    await showRunTab(page, 0);
    const firstAfter = (await page.evaluate(filesPane)).names;
    const switchName = "a node of tab 1 released over the can after a switch to tab 2 mid-drag removes nothing, in either tab, and says nothing";
    const unmoved = isDeepStrictEqual(switched, { shown: 1, pane: second.slice(1), chat: chatBefore }) && isDeepStrictEqual(firstAfter, names);
    check('trash', switchName, unmoved, JSON.stringify({ ...switched, firstAfter }));

    await page.click('#tab-chat');
    await runSample(page);
    const lit = await dragOver(page, await centre(page, await docNodeId(page, names[0])), spot);
    const ran = { shown: await canShown(page), lit, docs: await docNames(page), pane: (await paneOf(page)).names };
    const ranName = 'a tab that has run shows no can, and a document node dropped where it was keeps its file';
    check('trash', ranName, isDeepStrictEqual(ran, { shown: false, lit: false, docs: names, pane: names }), JSON.stringify(ran));
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
  const note = await page.$eval('#keys-note', (el) => (el.hidden ? null : el.textContent));
  checkAt('the note under the keys says only that the prototype sends nothing', note === PROTOTYPE_KEYS_NOTE, note);
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
  const focused = await focusedId(page);
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
    const sample = recordedSample();
    const { summary_ids, parse_ids } = sample.answer.provenance;
    const roundNames = (await page.$$eval('.node.r1, .node.r2', (els) => els.map((el) => el.getAttribute('aria-label')))).sort();
    const snippetNames = [1, 2].flatMap((round) => sample.docs.map((_, i) => `${stem(i)}, round ${round} relevance snippet`)).sort();
    check('popup', "each round's node is named for its document and its relevance snippet", isDeepStrictEqual(roundNames, snippetNames), roundNames[0]);
    const cited = [...summary_ids.map((i) => `${stem(i)} · round 2`), ...parse_ids.map((i) => `${stem(i)} · parsed`)];
    await inspect(page, 'answer-0');
    const listed = await page.$$eval('#inspector .row', (rows) => {
      const provenance = rows.find((r) => r.firstChild.textContent === 'provenance');
      return [...(provenance?.querySelectorAll('.ref') ?? [])].map((el) => el.textContent);
    });
    check('popup', "the answer's provenance lists the round-2 nodes it drew on, then the parsed ones", isDeepStrictEqual(listed, cited), listed.join(' · '));
    await inspect(page, 'parsed-0');
    const recorded = await shownParse(page);
    const block = await page.$eval('#inspector pre.code', (el) => {
      getSelection().selectAllChildren(el);
      const heights = [...el.children].map((line) => line.getBoundingClientRect().height);
      return { text: el.textContent, copied: getSelection().toString(), oneLineEach: heights.every((height) => height === heights[0]) };
    });
    const parsedWidth = (await rect(page, '#inspector')).width;
    await inspect(page, 'schema');
    const schemaWidth = (await rect(page, '#inspector')).width;
    const empty = await inspect(page, 'parsed-4', '#inspector .insp-text');
    const filled = filledFields(sample.parsed[0].parse);
    const parseName = "a parsed node's popup shows its document's parse as recorded, without its empty fields, and an empty one says nothing was parsed";
    check('popup', parseName, isDeepStrictEqual(recorded, filled) && empty === NOTHING_PARSED, empty);
    const json = JSON.stringify(filled, null, JSON_INDENT);
    const textName = "the parse's text, and a copy of it, is its JSON, indented, each line one line high";
    check('popup', textName, block.text === json && block.copied === json && block.oneLineEach, JSON.stringify(block.copied.slice(0, 40)));
    check('popup', "a parsed node's popup is as wide as the structure's", parsedWidth === schemaWidth, `${parsedWidth} and ${schemaWidth}`);
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
    await zoomAt(page, { x: panel.left + 100, y: panel.top + 100 }, -300);
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

  const body = withLongParse(readFileSync(SCRIPT_FILE, 'utf8'));
  const before = (page) => page.route('**/app.js', (route) => route.fulfill({ contentType: 'text/javascript', body }));
  await withPage('popup', { before }, async (page) => {
    await inspect(page, 'parsed-0');
    const lines = await count(page, '#inspector pre.code > span');
    const notes = await page.$$eval('#inspector > .k', (els) => els.map((el) => el.textContent));
    const capName = `a parse of more than ${PREVIEW_LINES} lines shows its first ${PREVIEW_LINES} and says so`;
    check('popup', capName, lines === PREVIEW_LINES && notes.includes(PREVIEW_LINES_NOTE), `${lines} lines; ${notes.join(' | ')}`);
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

    const before = await page.evaluate(panelSnapshot);
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
    check('gestures', stuckName, isDeepStrictEqual(await page.evaluate(panelSnapshot), before));
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
      await waitLit(page);
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
  await answersInView(asks);
  await answersWhileReplaying();
  await withPage('answers', { layout: LAYOUTS.phone }, async (page) => {
    await send(page, asks[0].question);
    await send(page, NEW_QUESTION);
    const newest = isDeepStrictEqual(await litAnswers(page), ['answer-2']);
    const tap = await centre(page, 'answer-0');
    await page.touchscreen.tap(tap.x, tap.y);
    const selected = await becomes(page, () => document.querySelector('.node.answer.on-path')?.dataset.id === 'answer-0');
    check('answers', '400 dark: the newest ask is lit, and a tap on the first answer lights it again', newest && selected);
  });
}

async function answersOnTheRun(sample, asks) {
  await withPage('answers', {}, async (page) => {
    const docIds = await docIdsOf(page);
    const runPath = lineage(sample, docIds);
    const askPath = (k) => lineage(sample, docIds, `answer-${k + 1}`, asks[k].provenance);
    const offered = await page.evaluate(composer);
    const offersBoth = isDeepStrictEqual(offered, { primary: 'ask ⏎', stop: false, newRun: true });
    check('answers', 'after a run the composer offers ask ⏎ and new run, and no stop', offersBoth, JSON.stringify(offered));

    const first = await positionOf(page, 'answer-0');
    const nodes = await count(page, '.node');
    await page.evaluate(countStatusWrites);
    await send(page, asks[0].question);
    const added = (await count(page, '.node')) - nodes;
    const lit = await litEdges(page);
    const oneNode = added === 1 && isDeepStrictEqual(await answerIds(page), ['answer-0', 'answer-1']);
    const onlyItsPath = isDeepStrictEqual(lit, askPath(0)) && isDeepStrictEqual(await litAnswers(page), ['answer-1']);
    check('answers', 'a recorded ask adds exactly one answer node and lights only its path', oneNode && onlyItsPath, `${added} added; ${lit.length} lit`);
    const status = await page.textContent('#run-status');
    const turns = await count(page, '.turn');
    check('answers', 'Enter asks over the run on the panel: no new run starts', turns === 2 && status === SAMPLE_STATUS, `${turns} turns; ${status}`);
    const writes = await page.evaluate(() => window.statusWrites);
    check('answers', 'an ask leaves the status alone: its live region is not rewritten', writes === 0, `${writes} writes`);
    const linesWhileAskLit = await page.evaluate(answerLines);

    const askLog = await page.evaluate(logOf);
    const { summary_ids, parse_ids } = asks[0].provenance;
    const summary = `${RECORDED_ASK} · ${asks[0].model} · ${asks[0].totals.tokens.toLocaleString('en-US')} tokens · $`;
    const told = isDeepStrictEqual(askLog.after, [asks[0].text]) && !askLog.open && askLog.folded.startsWith(`ready · ${asks[0].model} · $`);
    const shown = await inspect(page, 'answer-1');
    const rows = await page.$$eval('#inspector .row', (els) => els.map((r) => r.firstChild.textContent));
    const inPopup = shown.includes(asks[0].text) && shown.includes(summary) && isDeepStrictEqual(rows, POPUP_ROWS['answer-0']);
    const toldName = "a recorded ask's chat gives its real answer with nothing under it, its folded log its model and cost, and its popup its citations and source";
    check('answers', toldName, told && inPopup, `${askLog.folded}; ${askLog.after.join(' | ')}`);
    const label = await page.textContent(`${nodeSel('answer-1')} .label`);
    check('answers', 'an answer node is labelled with a short form of its question, and its model', label === ASK_LABEL, label);
    const kept = await positionOf(page, 'answer-0');
    const below = await positionOf(page, 'answer-1');
    const stacked = isDeepStrictEqual(kept, first) && below.x === first.x && below.y === first.y + ROW_GAP;
    check('answers', 'the first answer keeps its place, and the ask stacks one row gap below it', stacked, `${first.x},${first.y} → ${below.x},${below.y}`);

    await page.keyboard.press('Escape');
    await clickNode(page, 'answer-0');
    const byNode = isDeepStrictEqual(await litEdges(page), runPath) && isDeepStrictEqual(await litAnswers(page), ['answer-0']);
    check('answers', 'selecting the first answer by its node lights its path again', byNode);
    const pressed = await page.evaluate(pressedAnswers);
    const saysLit = isDeepStrictEqual(pressed, [['answer-0', 'true'], ['answer-1', 'false']]);
    check('answers', 'each answer node says with aria-pressed whether it is the lit one', saysLit, JSON.stringify(pressed));
    const linesWhileRunLit = await page.evaluate(answerLines);
    const runCites = sample.answer.provenance.summary_ids.length + sample.answer.provenance.parse_ids.length;
    const askCites = summary_ids.length + parse_ids.length;
    const citeLines = { 'answer-0': runCites, 'answer-1': askCites };
    const faintFollows =
      isDeepStrictEqual(tally(linesWhileAskLit.faint), { 'answer-1': SAMPLE_ANSWER_INPUTS - askCites }) &&
      isDeepStrictEqual(tally(linesWhileRunLit.faint), { 'answer-0': SAMPLE_ANSWER_INPUTS - runCites }) &&
      [linesWhileAskLit, linesWhileRunLit].every(({ cited: ends }) => isDeepStrictEqual(tally(ends), citeLines));
    const faintName = 'only the lit answer has faint lines from what it was shown, and every answer keeps its cite lines';
    check('answers', faintName, faintFollows, `${JSON.stringify(tally(linesWhileAskLit.faint))} → ${JSON.stringify(tally(linesWhileRunLit.faint))}`);
    await page.keyboard.press('Escape');
    const name = await page.getAttribute(nodeSel('answer-1'), 'aria-label');
    await page.focus(nodeSel('answer-1'));
    await page.keyboard.press('Enter');
    const byKey = isDeepStrictEqual(await litAnswers(page), ['answer-1']) && (await popup(page))?.title === `● ${ASK_LABEL}`;
    const named = name === `answer to ${asks[0].question} · ${asks[0].model}`;
    check('answers', 'an answer node is named for its question and model, takes focus, and Enter on it lights it', named && byKey, name);
    await page.keyboard.press('Escape');
    await page.locator('.turn').first().locator('.line.answer').click();
    const byLine = isDeepStrictEqual(await litEdges(page), runPath) && isDeepStrictEqual(await litAnswers(page), ['answer-0']);
    check('answers', 'selecting the first answer by its chat line lights its path again', byLine);

    const openTitle = (await popup(page))?.title;
    await send(page, NEW_QUESTION);
    const leftAlone = isDeepStrictEqual(await litAnswers(page), ['answer-0']) && (await popup(page))?.title === openTitle;
    check('answers', "a landing ask leaves an open answer's popup, and its lit path, alone", leftAlone, openTitle);
    const simulated = await page.evaluate(logOf);
    const flaggedInChat = simulated.after.includes(SIMULATED_LABEL) && simulated.folded === `ready · ${DEFAULT_MODEL} · nothing was sent`;
    const flaggedInPopup = (await inspect(page, 'answer-2')).includes(SIMULATED_LABEL);
    check('answers', 'an unrecorded ask is labelled simulated, in the chat, in view, and its popup', flaggedInChat && flaggedInPopup, `${simulated.folded}; ${simulated.after[0]}`);

    await page.keyboard.press('Escape');
    await page.getByRole('button', { name: offerLabel(asks[1]), exact: true }).click();
    const filled = await page.evaluate(() => [document.getElementById('question').value, document.getElementById('model-other').value]);
    await page.press('#question', 'Enter');
    const other = await page.evaluate(lastTurnLines);
    const fromOffer = isDeepStrictEqual(filled, [asks[1].question, OTHER_RECORDED_MODEL]) && asks[1].model === OTHER_RECORDED_MODEL;
    const otherRecorded = other.includes(asks[1].text) && other.some((text) => text.startsWith(`ready · ${OTHER_RECORDED_MODEL} · $`));
    const offeredName = 'an offered recorded ask fills its question and model, and is answered as recorded';
    check('answers', offeredName, fromOffer && otherRecorded, `${filled.join(' · ')}; ${other.slice(1, 2).join('')}`);

    await page.getByRole('button', { name: offerLabel(asks[2]), exact: true }).click();
    await page.press('#question', 'Enter');
    for (const question of ['Which memo is the registry?', 'Who manages Eastfield?']) await send(page, question);
    await dragNode(page, 'answer-2', 80, 50);
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
    await waitLit(page);
    const redrawn = isDeepStrictEqual((await answerIds(page)).sort(), idsBefore) && (await count(page, '.node.answer.working')) === 0;
    const stillLit = isDeepStrictEqual([await litAnswers(page), await litEdges(page)], litBefore);
    check('answers', "replay draws the run's asks again, and keeps the answer that was lit", redrawn && stillLit, `${idsBefore.length} answers; lit ${litBefore[0]}`);

    await page.fill('#question', sample.question);
    await page.click('#new-run-btn');
    const newTabLit = await becomes(page, () => document.querySelector('#world .is-lit') !== null);
    const after = { nodes: await count(page, '.node'), answers: (await answerIds(page)).length };
    const fresh = newTabLit && isDeepStrictEqual(after, ANSWERED_RUN_COUNTS) && isDeepStrictEqual(await litEdges(page), runPath);
    const opened = (await shownTab(page)) === 1;
    const openedName = 'new run opens a second tab and shows it, with only its own answer, lit';
    check('answers', openedName, fresh && opened && (await page.textContent('#run-status')) === SAMPLE_STATUS, JSON.stringify(after));

    const offerOf = page.getByRole('button', { name: offerLabel(asks[0]), exact: true });
    const inSecond = await offerOf.count();
    await offerOf.click();
    const fills = (await shownTab(page)) === 1 && (await page.inputValue('#question')) === asks[0].question;
    await showRunTab(page, 0);
    const inFirst = await offerOf.count();
    const offersName = "each recorded tab's chat offers the recorded asks, and an offer fills the question in the tab shown";
    check('answers', offersName, inSecond === 1 && inFirst === 1 && fills, `${inSecond} offers of the first ask in tab 2, ${inFirst} in tab 1`);

    await page.locator(CHAT).locator('.turn').nth(1).locator('.line.answer').click();
    const opensThere = (await shownTab(page)) === 0 && isDeepStrictEqual(await litAnswers(page), ['answer-1']) && (await popup(page))?.title === `● ${ASK_LABEL}`;
    check('answers', "back on tab 1, its ask's answer line opens that answer there, lit", opensThere, (await popup(page))?.title);
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
    await dragNode(page, 'answer-0', 60, 50);
    const moved = await positionOf(page, 'answer-0');
    await pickModel(page, LONG_MODEL);
    await send(page, asks[0].question);
    const placed = await positionOf(page, 'answer-1');
    const aroundIt = placed.x === moved.x && placed.y === moved.y + ROW_GAP;
    check('answers', 'an ask stacks around the first answer where it is now, after a drag', aroundIt, `${moved.x},${moved.y} → ${placed.x},${placed.y}`);
    const frame = await page.evaluate(panelBox);
    const outside = (await page.evaluate(answerNodes)).filter((n) => !inside(n.box, frame)).map((n) => n.id);
    check('answers', 'a new answer node is fitted into view, its whole label too', outside.length === 0, outside.join(' ') || 'all in view');
  });
  await withPage('answers', {}, async (page) => {
    await send(page, asks[0].question);
    const zoom = await page.$eval('#world', (el) => parseFloat(el.style.getPropertyValue('--k')));
    await dragNode(page, 'answer-0', 0, -ROW_GAP * zoom);
    for (const question of [NEW_QUESTION, 'Who manages Eastfield?']) await send(page, question);
    const placed = await page.evaluate(answerNodes);
    const apart = placed.length === 4 && placed.every((a, i) => placed.slice(i + 1).every((b) => !overlap(a.box, b.box)));
    const dragName = 'after the first answer is dragged a row up, new asks never land on an existing answer node';
    check('answers', dragName, apart, placed.map((n) => `${n.id} ${Math.round(n.y)}`).join(', '));
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
    await page.click('#tab-files');
    const pane = await page.evaluate(filesPane);
    await page.click('#tab-chat');
    const lockedName = "a tab's files lock as its run starts: while it is still going, its files pane offers no add and no remove, and says why";
    check('answers', lockedName, pane.removes === 0 && !pane.adds && pane.locked === FILES_LOCKED, JSON.stringify(pane));
    const offered = await page.evaluate(composer);
    await page.click('#run-btn');
    const clickWaits = (await count(page, '.turn')) === 1 && (await page.inputValue('#question')) === NEW_QUESTION;
    await page.$eval('#new-run-btn', (button) => button.click());
    const opened = (await page.evaluate(tabRow)).map((tab) => tab.selected);
    const offerName = "while the prototype's run replays, the composer offers ask ⏎, which waits as Enter does, and new run, which opens a tab of its own";
    check('answers', offerName, offered.primary === 'ask ⏎' && offered.newRun && clickWaits && isDeepStrictEqual(opened, [false, true]), JSON.stringify(offered));
  });
}

// ---------- tabs: one per run ----------

// Counts the document nodes added to the panel from now on, in window.docNodesAdded.
function countDocNodes() {
  window.docNodesAdded = 0;
  const added = (records) => records.flatMap((record) => [...record.addedNodes]).filter((el) => el.matches?.('.node.doc')).length;
  new MutationObserver((records) => (window.docNodesAdded += added(records))).observe(document.getElementById('world'), { childList: true, subtree: true });
}

// The text of each line in tab i's chat, shown or hidden.
const chatLines = (page, i) => page.locator('.transcript').nth(i).locator('.line .t').allTextContents();

// Scrolls the shown chat `share` of the way down in the next frame, after any scroll the page has queued, and returns
// where it then is.
function scrollChatTo(page, share) {
  const scroll = ([selector, part]) => new Promise((resolve) => requestAnimationFrame(() => {
    const el = document.querySelector(selector);
    el.scrollTop = Math.round((el.scrollHeight - el.clientHeight) * part);
    resolve(el.scrollTop);
  }));
  return page.evaluate(scroll, [CHAT, share]);
}

// The shown chat's scroll position and how far it can scroll, in the next frame, once the page has scrolled it.
function chatState(page) {
  const read = (selector) => new Promise((resolve) => requestAnimationFrame(() => {
    const el = document.querySelector(selector);
    resolve({ top: el.scrollTop, end: el.scrollHeight - el.clientHeight });
  }));
  return page.evaluate(read, CHAT);
}

// Tab i's × enables once its run has ended.
function tabEnded(i) {
  return !document.querySelectorAll('.tab-close')[i].disabled;
}

async function tabs() {
  const asks = recordedAsks();
  await withPage('tabs', {}, async (page) => {
    const one = await page.evaluate(tabRow);
    const firstName = 'the first run runs in the first tab, which it names by its number, question and model; the last tab offers no × and no Delete';
    check('tabs', firstName, one.length === 1 && one[0].label === TAB_LABEL && !one[0].hasClose && one[0].keys === null, JSON.stringify(one));
    await send(page, asks[0].question);
    await dragNode(page, 'r2-1', 60, 40);
    const frame = await page.evaluate(panelBox);
    await zoomAt(page, { x: frame.left + frame.width / 2, y: frame.top + frame.height / 2 }, -200);
    const left = await page.evaluate(panelSnapshot);
    await page.evaluate(countDocNodes);
    await send(page, NEW_QUESTION, 'Control+Enter');
    const other = await page.evaluate(panelSnapshot);
    const added = await page.evaluate(() => window.docNodesAdded);
    check('tabs', 'a new run draws each of its document nodes once', added === SAMPLE_DOCS, `${added} drawn`);
    await page.locator('.run-tab').first().click();
    const back = await page.evaluate(panelSnapshot);
    const restoreName = 'switching back to the first tab restores its nodes and lines after a drag, its zoom, its lit answer and its status';
    check('tabs', restoreName, !isDeepStrictEqual(other, left) && isDeepStrictEqual(back, left), `${back.nodes.length} nodes; ${back.view}`);

    const row = await page.evaluate(tabRow);
    const pattern = await page.evaluate(() => {
      const panel = document.getElementById('panel');
      return { list: document.getElementById('run-tabs').getAttribute('role'), panel: panel.getAttribute('role'), labelledBy: panel.getAttribute('aria-labelledby') };
    });
    const owned = (await page.locator('#run-tabs').ariaSnapshot()).split('\n').slice(1);
    const selected = row.filter((tab) => tab.selected);
    const tabsOk = row.every((tab) => tab.role === 'tab' && tab.focusable === tab.selected && tab.keys === 'Delete' && tab.closeForMouseOnly);
    const ariaOk =
      pattern.list === 'tablist' && pattern.panel === 'tabpanel' && selected.length === 1 && pattern.labelledBy === selected[0].id &&
      tabsOk && owned.length === row.length && owned.every((entry) => entry.trim().startsWith('- tab '));
    const ariaName = 'the tab row is an ARIA tablist that owns only tabs, each closing with Delete; × is for the mouse alone';
    check('tabs', ariaName, ariaOk, JSON.stringify({ pattern, owned }));
    await page.locator('.run-tab').first().focus();
    await page.keyboard.press('ArrowRight');
    const right = [await shownTab(page), await focusedId(page)];
    await page.keyboard.press('ArrowLeft');
    const leftAgain = [await shownTab(page), await focusedId(page)];
    const keysMove = isDeepStrictEqual(right, [1, row[1].id]) && isDeepStrictEqual(leftAgain, [0, row[0].id]);
    check('tabs', 'arrow keys move between tabs, showing each and keeping the focus on it', keysMove, `${right} → ${leftAgain}`);
    check('tabs', '1440 light: with two tabs the page has no horizontal scroll', await page.evaluate(noHorizontalScroll));

    const firstChat = await chatLines(page, 0);
    await page.locator('.tab-close').nth(1).click();
    const kept = await page.evaluate(panelSnapshot);
    const chats = { count: await count(page, '.transcript'), first: await chatLines(page, 0) };
    const remaining = await page.evaluate(tabRow);
    const closesOne = isDeepStrictEqual(kept, left) && remaining.length === 1 && !remaining[0].hasClose && isDeepStrictEqual(chats, { count: 1, first: firstChat });
    const closesName = 'closing a tab removes its chat with it, and leaves the other tabs as they were, their chats too; the last tab has no ×';
    check('tabs', closesName, closesOne, `${chats.count} chats`);

    for (const question of [NEW_QUESTION, 'Which memo is the registry?']) await send(page, question, 'Control+Enter');
    const fourth = await page.evaluate(panelSnapshot);
    await page.locator('.run-tab').nth(1).click();
    await page.locator('.tab-close').nth(1).click();
    const two = await page.evaluate(tabRow);
    const focused = await focusedId(page);
    const showsNext = isDeepStrictEqual(two.map((tab) => [tab.id, tab.selected]), [['run-tab-1', false], ['run-tab-4', true]]) && focused === 'run-tab-4' && isDeepStrictEqual(await page.evaluate(panelSnapshot), fourth);
    check('tabs', 'closing the shown middle tab of three shows the one after it, with the focus on it', showsNext, `${two.map((tab) => tab.id)}; focus on #${focused}`);
    await page.keyboard.press('Delete');
    const afterDelete = (await page.evaluate(tabRow)).map((tab) => tab.id);
    await page.keyboard.press('Delete');
    const lastKept = (await page.evaluate(tabRow)).map((tab) => tab.id);
    const deletes = isDeepStrictEqual(afterDelete, ['run-tab-1']) && isDeepStrictEqual(lastKept, ['run-tab-1']);
    check('tabs', 'Delete closes the focused tab, but not the last one', deletes, `${afterDelete} then ${lastKept}`);
  });

  // Each tab has a chat of its own. The page is short enough that tab 1's, after its run and two asks, scrolls.
  await withPage('tabs', {}, async (page) => {
    await page.setViewportSize(SHORT_DESKTOP);
    for (const question of [asks[0].question, NEW_QUESTION]) await send(page, question);
    const firstChat = await chatLines(page, 0);
    const left = await scrollChatTo(page, 1 / 2);
    await page.click('#new-tab-btn');
    await page.click('#tab-chat');
    const plus = { lines: await chatLines(page, 1), shown: await page.$$eval('.transcript', (els) => els.map((el) => el.checkVisibility())) };
    const plusName = "a + tab's chat starts empty, and tab 1's is hidden: none of its lines are in the + tab";
    check('tabs', plusName, isDeepStrictEqual(plus, { lines: [], shown: [false, true] }), JSON.stringify(plus));

    await showRunTab(page, 0);
    const back = { lines: await chatLines(page, 0), ...(await chatState(page)) };
    const backName = 'switching back to tab 1 shows its chat as it was, scrolled where it was left';
    check('tabs', backName, left > 0 && isDeepStrictEqual(back.lines, firstChat) && back.top === left, `left at ${left}, back at ${back.top}`);
    const live = await page.$$eval('.transcript', (els) => els.map((el) => el.getAttribute('aria-live')));
    check('tabs', "every tab's chat is a polite live region, shown or hidden", isDeepStrictEqual(live, ['polite', 'polite']), JSON.stringify(live));

    const question = 'Which memo is longest?';
    await send(page, question, 'Control+Enter');
    const opened = { shown: await shownTab(page), parts: await page.$$eval(`${CHAT} > *`, (els) => els.map((el) => el.className)), first: (await chatLines(page, 2)).slice(0, 2) };
    const openedName = `a new run's tab starts its chat with "${loadedLine(SAMPLE_DOCS)}" for the files it copied, then its own run, and nothing else`;
    check('tabs', openedName, isDeepStrictEqual(opened, { shown: 2, parts: ['line main dim', 'turn'], first: [loadedLine(SAMPLE_DOCS), question] }), JSON.stringify(opened));
    await page.locator('.tab-close').nth(2).click();
    const closed = await page.$$eval('.transcript', (els, asked) => [els.length, els.some((el) => el.textContent.includes(asked))], question);
    check('tabs', 'closing the shown tab removes its chat too', isDeepStrictEqual(closed, [2, false]), JSON.stringify(closed));
  });

  // The files pane hides the chat. Short enough that tab 1's chat scrolls once its run has ended.
  await withPage('tabs', { sample: false }, async (page) => {
    await page.setViewportSize(SHORT_DESKTOP);
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.click('#run-btn');
    await page.click('#tab-files');
    const ended = await becomes(page, statusIs, SAMPLE_STATUS, REPLAY_MS);
    await page.click('#tab-chat');
    const landed = await chatState(page);
    const landedName = 'a run that ends while the files pane is shown leaves its chat at its newest line once the chat is shown';
    check('tabs', landedName, ended && landed.end > 0 && landed.top === landed.end, `ended ${ended}; at ${landed.top} of ${landed.end}`);

    await page.emulateMedia({ reducedMotion: 'reduce' });
    const left = await scrollChatTo(page, 1 / 2);
    await page.click('#tab-files');
    await page.click('#new-tab-btn');
    await page.click('#tab-chat');
    await showRunTab(page, 0);
    const back = (await chatState(page)).top;
    const backName = "switching tabs while the files pane is shown leaves tab 1's chat where it was scrolled, back in the chat";
    check('tabs', backName, left > 0 && back === left, `left at ${left}, back at ${back}`);
  });

  // With motion on, a replay takes seconds, so a tab can still be running while another is shown.
  await withPage('tabs', {}, async (page) => {
    const first = await page.evaluate(panelSnapshot);
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await send(page, NEW_QUESTION, 'Control+Enter');
    const starting = await page.textContent('#run-status');
    check('tabs', "a new tab's status line is its own from the start, not the last tab's", starting === STARTING, starting);
    const going = await page.evaluate(tabRow);
    await page.locator('.tab-close').nth(1).click({ force: true });
    const stays = (await page.evaluate(tabRow)).length === 2;
    const busyName = "a tab whose run is going can't be closed: its × is disabled while the other tab's isn't";
    check('tabs', busyName, isDeepStrictEqual(going.map((tab) => tab.closable), [true, false]) && stays, JSON.stringify(going.map((tab) => tab.closable)));
    await page.evaluate(recordStatuses);
    await page.locator('.run-tab').nth(1).focus();
    await page.keyboard.press('Delete');
    const said = await page.evaluate(() => window.statuses);
    const keeps = (await page.evaluate(tabRow)).length === 2 && said.includes('tab 2 is still going; close it once it has finished');
    const whyName = 'Delete is offered only on a tab it closes; on a tab whose run is going it keeps the tab, and the status says why';
    check('tabs', whyName, isDeepStrictEqual(going.map((tab) => tab.keys), ['Delete', null]) && keeps, `${JSON.stringify(going.map((tab) => tab.keys))}; ${said.join(' | ')}`);
    await page.locator('.run-tab').first().click();
    await page.locator('.tab-close').nth(1).hover();
    const busy = (await page.evaluate(tabColors)).tabs[1];
    const stillGoing = !(await page.evaluate(tabRow))[1].closable;
    const unlit = stillGoing && busy.closeColor === busy.color;
    check('tabs', "a tab not shown whose run is going keeps its disabled × unlit under the pointer", unlit, `going ${stillGoing}; × ${busy.closeColor}, tab ${busy.color}`);
    const ended = await becomes(page, tabEnded, 1, REPLAY_MS);
    await page.waitForTimeout(LIT_DELAY_MS);
    const untouched = ended && isDeepStrictEqual(await page.evaluate(panelSnapshot), first);
    check('tabs', 'a replay going on in another tab never draws on the shown one, nor changes its status line', untouched, `ended ${ended}`);
    // Short enough that tab 2's chat scrolls.
    await page.setViewportSize(SHORT_DESKTOP);
    await page.locator('.run-tab').nth(1).click();
    const pending = await count(page, '#world .enter, #world .draw, #world .fade');
    const still = await page.evaluate(panelStill);
    check('tabs', 'a tab drawn in the background has no node or line animations to play when first shown', pending === 0 && still, `${pending} pending`);
    const shown = await chatState(page);
    const endName = "a line that would scroll a hidden tab's chat, as its answer does, takes the chat to its newest line once the tab is shown";
    check('tabs', endName, shown.end > 0 && shown.top === shown.end, `at ${shown.top} of ${shown.end}`);

    await page.locator('.run-tab').first().click();
    await page.click('#replay-btn');
    const replaying = (await page.evaluate(tabRow)).map((tab) => [tab.selected, tab.closable]);
    const status = await page.textContent('#run-status');
    const replaysShown = isDeepStrictEqual(replaying, [[true, false], [false, true]]) && status === STARTING;
    check('tabs', 'replay replays the shown tab alone: its status starts over and its × disables, while the other tab stays idle', replaysShown, `${JSON.stringify(replaying)}; ${status}`);
  });

  // With motion on, nodes and lines animate in and arrange slides, so a switch can come while either is going.
  await withPage('tabs', { sample: false }, async (page) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.click('#run-btn');
    await waitLit(page, REPLAY_MS);
    await becomes(page, panelStill);
    await send(page, asks[0].question);
    const replayed = await page.evaluate(linesReplayed, 'answer-1');
    check('tabs', 'lighting a new answer moves lines between layers without playing how they were drawn again', replayed === 0, `${replayed} replayed`);

    await send(page, NEW_QUESTION, 'Control+Enter');
    await page.waitForSelector('#world .edge.draw', { state: 'attached' });
    await page.locator('.run-tab').first().click();
    await becomes(page, tabEnded, 1, REPLAY_MS);
    await page.waitForTimeout(LIT_DELAY_MS);
    await page.locator('.run-tab').nth(1).click();
    check('tabs', 'a tab left mid-replay and shown again plays none of the animations it cut off', await page.evaluate(panelStill));
    await page.locator('.run-tab').first().click();

    const frame = await page.evaluate(panelBox);
    await zoomAt(page, { x: frame.left + frame.width / 2, y: frame.top + frame.height / 2 }, -200);
    const zoomed = await page.evaluate(panelSnapshot);
    await page.locator('.run-tab').nth(1).click();
    await dragNode(page, 'r2-1', 60, 40);
    await page.click('#arrange-btn');
    await page.locator('.run-tab').first().click();
    await page.waitForTimeout(2 * ARRANGE_MS);
    const leftAlone = isDeepStrictEqual(await page.evaluate(panelSnapshot), zoomed);
    check('tabs', "an arrange still sliding when its tab is switched away from leaves the tab then shown as it was", leftAlone);

    await dragNode(page, 'r2-1', 60, 40);
    await page.click('#arrange-btn');
    await send(page, 'Which memo is the registry?', 'Control+Enter');
    await page.locator('.run-tab').first().click();
    const snapped = await page.evaluate(panelSnapshot);
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.click('#arrange-btn');
    const final = await page.evaluate(panelSnapshot);
    const atFinal = isDeepStrictEqual([snapped.nodes, snapped.edges], [final.nodes, final.edges]);
    check('tabs', "an arrange still sliding when a new run starts leaves the old tab's nodes and lines at their final spots", atFinal);
  });

  await withPage('tabs', { layout: LAYOUTS.phone }, async (page) => {
    for (const question of [NEW_QUESTION, 'Which memo is the registry?']) await send(page, question, 'Control+Enter');
    const inView = inside(await rect(page, '.run-tab[aria-selected="true"]'), await rect(page, '#run-tabs'));
    check('tabs', "400 dark: the shown tab's button is scrolled into view in the tab row", inView);
    const rowScrolls = await page.evaluate(() => {
      const row = document.getElementById('run-tabs');
      row.scrollLeft = 0;
      row.scrollLeft = row.scrollWidth;
      return row.scrollLeft > 0;
    });
    const noScroll = await page.evaluate(noHorizontalScroll);
    check('tabs', '400 dark: with three tabs the tab row scrolls within itself, and the page has no horizontal scroll', rowScrolls && noScroll, `row scrolls ${rowScrolls}`);
  });

  for (const layout of [LAYOUTS.desktop, LAYOUTS.phone]) {
    await withPage('tabs', { layout }, async (page) => {
      for (let i = 0; i < 2; i += 1) await page.click('#new-tab-btn');
      await page.locator('.run-tab').first().click();
      await page.mouse.move(0, 0);
      const rest = await page.evaluate(tabColors);
      const { bar, tabs: row } = rest;
      const others = row.filter((tab) => !tab.selected);
      const shown = row.find((tab) => tab.selected);
      const blocks = row.every((tab) => tab.joined && tab.closeColor === tab.color) && row.slice(1).every((tab, i) => tab.left > row[i].right);
      const colors = others.every((tab) => tab.color === others[0].color && tab.color !== bar) && ![bar, others[0].color].includes(shown.color);
      const blocksName = `${layout.name}: with three tabs each tab and its × are one block of color set apart from the next; the tabs not shown are tinted off the bar, and the shown tab differs from both`;
      check('tabs', blocksName, blocks && colors, JSON.stringify({ bar, shown: shown.color, other: others[0].color }));

      const pointed = [];
      for (const target of [page.locator('.run-tab').nth(1), page.locator('.tab-close').nth(1), page.locator('.run-tab').first(), page.locator('#new-tab-btn')]) {
        await target.hover();
        pointed.push(await page.evaluate(tabColors));
      }
      await page.mouse.move(0, 0);
      const [overTab, overClose] = pointed.map((paint) => paint.tabs[1]);
      const tabAlone = overTab.color !== row[1].color && overTab.closeColor === row[1].closeColor;
      const closeAlone = overClose.closeColor !== row[1].closeColor && overClose.color === row[1].color;
      const close = await page.locator('.tab-close').nth(1).boundingBox();
      const big = close.width >= MIN_TARGET_PX && close.height >= MIN_TARGET_PX;
      const hoverName = `${layout.name}: under the pointer a tab not shown lights alone, and so does its ×, which is at least ${MIN_TARGET_PX} px each way`;
      check('tabs', hoverName, tabAlone && closeAlone && big, `tab alone ${tabAlone}; × alone ${closeAlone}; × ${Math.round(close.width)}×${Math.round(close.height)}`);
      const lowest = [rest, ...pointed].map((paint) => paint.lowest).reduce((a, b) => (b.ratio < a.ratio ? b : a));
      const contrastName = `${layout.name}: all text in the tab row, at rest and under the pointer, has at least ${MIN_TEXT_CONTRAST}:1 contrast with what is behind it`;
      check('tabs', contrastName, lowest.ratio >= MIN_TEXT_CONTRAST, `lowest ${lowest.ratio.toFixed(2)} on ${lowest.text}`);

      await page.locator('.run-tab').first().focus();
      for (const key of ['ArrowRight', 'ArrowLeft']) await page.keyboard.press(key);
      check('tabs', `${layout.name}: a tab focused from the keyboard shows its focus ring whole inside the tab row`, await page.evaluate(focusRingFits));
    });
  }
}

// ---------- tab files: each tab's own, locked once it has run ----------

// The shown tab's files as the files pane offers them: their names, how many remove buttons, whether files
// can be added, the lock line if shown, and the count on the pane's tab.
function filesPane() {
  const shown = (id) => document.getElementById(id)?.hidden === false;
  return {
    names: [...document.querySelectorAll('#file-list .file .name')].map((el) => el.textContent),
    removes: document.querySelectorAll('#file-list .remove').length,
    adds: shown('file-actions') && shown('drop-hint'),
    locked: shown('files-locked') ? document.getElementById('files-locked').textContent : null,
    count: document.getElementById('files-count').textContent,
  };
}

// Drops `files`, each [name, text], on the side column, and returns what its overlay said while they were over it.
function dropFiles(files) {
  const transfer = new DataTransfer();
  for (const [name, text] of files) transfer.items.add(new File([text], name, { type: 'text/plain' }));
  const side = document.getElementById('side');
  const drag = (type) => side.dispatchEvent(new DragEvent(type, { dataTransfer: transfer, bubbles: true, cancelable: true }));
  drag('dragenter');
  drag('dragover');
  const overlay = document.getElementById('drop-overlay');
  const said = getComputedStyle(overlay).display === 'none' ? null : overlay.textContent;
  drag('drop');
  return said;
}

const tabLabels = async (page) => (await page.evaluate(tabRow)).map((tab) => [tab.label, tab.selected]);
const lastChatLine = (page) => page.$eval(`${CHAT} > .line:last-child .t`, (el) => el.textContent);
const showRunTab = (page, i) => page.locator('.run-tab').nth(i).click();

async function paneOf(page) {
  await page.click('#tab-files');
  return page.evaluate(filesPane);
}

async function tabFiles() {
  const names = recordedSample().docs.map((d) => d.name);
  const second = ['a_memo.txt', 'b_memo.txt'];
  await withPage('tab files', { sample: false }, async (page) => {
    const before = await tabLabels(page);
    check('tab files', 'before any run, the first tab is listed, named by its number and its files', isDeepStrictEqual(before, [[FIRST_TAB_BEFORE_RUN, true]]), JSON.stringify(before));

    const preloaded = await page.inputValue('#question');
    const plus = page.getByRole('button', { name: NEW_TAB, exact: true });
    const outside = (await plus.count()) === 1 && (await plus.evaluate((el) => el.id === 'new-tab-btn' && !el.closest('[role="tablist"]')));
    await page.locator('.run-tab').first().focus();
    await page.keyboard.press('Tab');
    const reached = await focusedId(page);
    await page.keyboard.press('Enter');
    const opened = {
      tabs: await tabLabels(page),
      focus: await focusedId(page),
      filesShown: await page.$eval('#files-pane', (el) => !el.hidden),
      pane: await page.evaluate(filesPane),
      nodes: await count(page, '#world .node'),
      hint: await page.textContent('#panel-hint'),
      status: await page.textContent('#run-status'),
    };
    const plusName = `+ is a button named ${NEW_TAB}, outside the tab list, the next stop after the shown tab, and Enter on it opens an empty tab, shows it with the focus on it, and shows its empty files pane`;
    const expected = {
      tabs: [[FIRST_TAB_BEFORE_RUN, false], ['2 new', true]],
      focus: 'run-tab-2',
      filesShown: true,
      pane: { names: [], removes: 0, adds: true, locked: null, count: '0' },
      nodes: 0,
      hint: ADD_FILES,
      status: ADD_FILES,
    };
    check('tab files', plusName, outside && reached === 'new-tab-btn' && isDeepStrictEqual(opened, expected), `${reached}; ${JSON.stringify(opened)}`);
    const inPlusTab = await page.inputValue('#question');

    await page.setInputFiles('#file-input', second.map((name) => memo(name, `A memo, ${name}, for the second tab.\n`)));
    await removeButton(page, second[1]).waitFor();
    const inSecond = { names: (await page.evaluate(filesPane)).names, docs: await docNames(page), tabs: await tabLabels(page), chat: await lastChatLine(page) };
    await showRunTab(page, 0);
    const backOnFirst = await page.inputValue('#question');
    const inFirst = { names: (await page.evaluate(filesPane)).names, docs: await docNames(page) };
    await removeButton(page, names[0]).click();
    await showRunTab(page, 1);
    const secondAgain = (await page.evaluate(filesPane)).names;
    await showRunTab(page, 0);
    const firstAgain = (await page.evaluate(filesPane)).names;
    const apart =
      isDeepStrictEqual(inSecond, { names: second, docs: second, tabs: [[FIRST_TAB_BEFORE_RUN, false], ['2 new · 2 files', true]], chat: '2 files added: the next question runs over 2 documents.' }) &&
      isDeepStrictEqual(inFirst, { names, docs: names }) &&
      isDeepStrictEqual(secondAgain, second) &&
      isDeepStrictEqual(firstAgain, names.slice(1));
    const apartName = "files added in tab 2 are its own: tab 1 keeps the sample memos, and a memo removed in tab 1 stays in tab 2";
    check('tab files', apartName, apart, JSON.stringify({ inSecond, inFirst, secondAgain, firstAgain }));
    const questions = [preloaded, inPlusTab, backOnFirst];
    const sampleQuestion = recordedSample().question;
    const questionsName = "the sample's question is in the box on the first tab, before its run, and a + tab starts with an empty box";
    check('tab files', questionsName, isDeepStrictEqual(questions, [sampleQuestion, '', sampleQuestion]), JSON.stringify(questions));

    await showRunTab(page, 1);
    await page.click('#tab-chat');
    await send(page, NEW_QUESTION);
    const ran = {
      tabs: (await tabLabels(page)).map(([label, selected]) => [label.split(' ')[0], selected]),
      label: (await page.evaluate(tabRow))[1].label,
      turns: await count(page, `${CHAT} .turn`),
      status: await page.textContent('#run-status'),
      docs: await docNames(page),
    };
    const inPlace =
      isDeepStrictEqual(ran.tabs, [['1', false], ['2', true]]) && ran.label.startsWith('2 Which memo') && ran.label.endsWith(' · gpt-6-luna') &&
      ran.turns === 1 && ran.status === `● done · simulated run · ${DEFAULT_MODEL} · nothing was sent` && isDeepStrictEqual(ran.docs, second);
    check('tab files', "the first question in a + tab runs there, in place, over that tab's files: no third tab opens, and the tab takes its run's name", inPlace, JSON.stringify(ran));

    const locked = await paneOf(page);
    const lockedName = "once a tab has run its files are locked: the pane lists them with no remove, offers no add, and says why";
    check('tab files', lockedName, isDeepStrictEqual(locked, { names: second, removes: 0, adds: false, locked: FILES_LOCKED, count: '2' }), JSON.stringify(locked));

    await page.click('#tab-chat');
    await send(page, 'Which memo is the registry?');
    const asked = { tabs: (await page.evaluate(tabRow)).length, answers: await answerIds(page), status: await page.textContent('#run-status') };
    check('tab files', 'Enter in a tab that has run asks over its run', isDeepStrictEqual(asked, { tabs: 2, answers: ['answer-0', 'answer-1'], status: ran.status }), JSON.stringify(asked));

    await send(page, 'Which memo is longest?', 'Control+Enter');
    const third = { shown: await shownTab(page), label: (await page.evaluate(tabRow))[2].label, answers: await answerIds(page), docs: await docNames(page), pane: await paneOf(page) };
    await showRunTab(page, 1);
    const kept = { answers: await answerIds(page), pane: await page.evaluate(filesPane) };
    const copied =
      third.shown === 2 && third.label.startsWith('3 Which memo is longest') && isDeepStrictEqual(third.answers, ['answer-0']) && isDeepStrictEqual(third.docs, second) &&
      isDeepStrictEqual(third.pane, locked) && isDeepStrictEqual(kept, { answers: ['answer-0', 'answer-1'], pane: locked });
    const copiedName = "new run opens a third tab with a copy of tab 2's files and runs there, its files locked too; tab 2 keeps its files and its asks";
    check('tab files', copiedName, copied, JSON.stringify({ third, kept }));

    const owned = (await page.locator('#run-tabs').ariaSnapshot()).split('\n').slice(1);
    const onlyTabs = owned.length === 3 && owned.every((entry) => entry.trim().startsWith('- tab '));
    check('tab files', 'with + beside it, the tab list still owns only tabs', onlyTabs, owned.join(' | '));

    await showRunTab(page, 2);
    const overLocked = await page.evaluate(dropFiles, [['dropped.pdf', '%PDF-1.7']]);
    const firstRefusal = await filesError(page);
    await page.evaluate(dropFiles, [['.DS_Store', 'x'], ['~$dropped.docx', 'x'], ['dropped.txt', 'A dropped memo.\n']]);
    const secondRefusal = await filesError(page);
    const unread = { tabs: (await page.evaluate(tabRow)).length, shown: await shownTab(page), names: (await page.evaluate(filesPane)).names, said: await page.textContent('#files-said') };
    const refusedName =
      "files dropped on a tab that has run are refused at once, as the overlay says while they are over it: none is read and no tab opens; a second drop's refusal takes the place of the first's, and names only what a pick would take";
    const atOnce =
      overLocked === DROP_LOCKED && firstRefusal === `Not added: dropped.pdf. ${LOCKED_DROP}` && secondRefusal === `Not added: dropped.txt. ${LOCKED_DROP}` &&
      isDeepStrictEqual(unread, { tabs: 3, shown: 2, names: second, said: '' });
    check('tab files', refusedName, atOnce, `${overLocked}; ${firstRefusal}; ${secondRefusal}; ${JSON.stringify(unread)}`);
    await page.evaluate(dropFiles, [['.DS_Store', 'x']]);
    const nothingLeft = await filesError(page);
    check('tab files', 'a drop on a tab that has run of only files a pick leaves out clears the line and names none', nothingLeft === null, nothingLeft);

    await page.click('#new-tab-btn');
    const overOpen = await page.evaluate(dropFiles, [['dropped_too.txt', 'Another dropped memo.\n']]);
    await removeButton(page, 'dropped_too.txt').waitFor();
    const added = { tabs: (await page.evaluate(tabRow)).length, names: (await page.evaluate(filesPane)).names };
    const addsName = 'files dropped on a tab that has not run are added to it, as the overlay says';
    check('tab files', addsName, overOpen === DROP_ADDS && isDeepStrictEqual(added, { tabs: 4, names: ['dropped_too.txt'] }), `${overOpen}; ${JSON.stringify(added)}`);
    check('tab files', '1440 light: with four tabs and + the page has no horizontal scroll', await page.evaluate(noHorizontalScroll));
  });

  await withPage('tab files', { sample: false }, async (page) => {
    await page.click('#new-tab-btn');
    await page.keyboard.press('Delete');
    const closedEmpty = { tabs: await tabLabels(page), focus: await focusedId(page) };
    await page.keyboard.press('Delete');
    const lastKept = await tabLabels(page);
    await page.click('#new-tab-btn');
    await page.setInputFiles('#file-input', memo('c_memo.txt', 'A memo.\n'));
    await removeButton(page, 'c_memo.txt').waitFor();
    await page.locator('.tab-close').nth(1).click();
    const closedWithFiles = await tabLabels(page);
    const one = [[FIRST_TAB_BEFORE_RUN, true]];
    const closes = isDeepStrictEqual(closedEmpty, { tabs: one, focus: 'run-tab-1' }) && isDeepStrictEqual(lastKept, one) && isDeepStrictEqual(closedWithFiles, one);
    check('tab files', "a + tab closes, empty with Delete or with files by its ×, and the focus goes to the tab shown; the last tab doesn't close", closes, JSON.stringify({ closedEmpty, lastKept, closedWithFiles }));
    await page.click('#tab-chat');
    await page.fill('#question', NEW_QUESTION);
    await page.click('#new-tab-btn');
    const inNewTab = await page.inputValue('#question');
    await showRunTab(page, 0);
    const typed = [inNewTab, await page.inputValue('#question')];
    const ownName = 'a question typed in a tab stays with it: a + tab opens with an empty box, and back on the first tab the question is there';
    check('tab files', ownName, isDeepStrictEqual(typed, ['', NEW_QUESTION]), JSON.stringify(typed));

    await page.click('#tab-chat');
    await page.press('#question', 'Enter');
    await waitLit(page);
    await send(page, 'Which memo is longest?', 'Control+Enter');
    await showRunTab(page, 0);
    const left = await page.inputValue('#question');
    check('tab files', 'back on a tab that new run left, the box is empty: no question from before its run', left === '', left);
  });

  await withPage('tab files', { layout: LAYOUTS.phone }, async (page) => {
    for (let i = 0; i < 2; i += 1) await page.click('#new-tab-btn');
    await page.setInputFiles('#file-input', memo('d_memo.txt', 'A memo that makes its tab named longer.\n'));
    await removeButton(page, 'd_memo.txt').waitFor();
    const plus = await rect(page, '#new-tab-btn');
    const row = await rect(page, '.tab-row');
    const viewport = LAYOUTS.phone.options.viewport;
    const fits = plus.width >= MIN_TARGET_PX && plus.height >= MIN_TARGET_PX && plus.left >= row.left && plus.right <= row.right && plus.right <= viewport.width;
    const shownInView = inside(await rect(page, '.run-tab[aria-selected="true"]'), await rect(page, '#run-tabs'));
    const rowScrolls = await page.$eval('#run-tabs', (el) => el.scrollWidth > el.clientWidth);
    const noScroll = await page.evaluate(noHorizontalScroll);
    const phoneName = `400 dark: with three tabs the tab list scrolls within the row while + stays in view at its end, at least ${MIN_TARGET_PX} px each way, the new tab stays in view as its files lengthen its name, and the page has no horizontal scroll`;
    check('tab files', phoneName, fits && shownInView && rowScrolls && noScroll, `+ ${Math.round(plus.width)}×${Math.round(plus.height)} at ${Math.round(plus.left)}–${Math.round(plus.right)}; row scrolls ${rowScrolls}`);
  });
}

// ---------- drop: on the panel too, and folders and zips ----------

// Drops `items` on the element with id `on`, and returns what the overlay said while they were over it and whether
// it covered that element. An item is [name, text or { bytes } or { fails: true }] for a file, or [name, items] for
// a folder. A script's DataTransfer has no file system behind it, so webkitGetAsEntry is stubbed with entries that
// read as a dropped folder's do, its reader handing them out in batches until one is empty, as Chromium's does.
function dropItems({ on, items }) {
  const BATCH = 2;
  const file = (name, content) => new File([typeof content === 'string' ? content : new Uint8Array(content.bytes)], name);
  const entry = (path, [name, content]) => {
    const fullPath = `${path}/${name}`;
    if (!Array.isArray(content)) {
      const read = (done, fail) => (content.fails ? fail(new DOMException('gone', 'NotFoundError')) : done(file(name, content)));
      return { isFile: true, isDirectory: false, name, fullPath, file: read };
    }
    const children = content.map((child) => entry(fullPath, child));
    const createReader = () => {
      let next = 0;
      const readEntries = (done) => {
        const batch = children.slice(next, next + BATCH);
        next += BATCH;
        done(batch);
      };
      return { readEntries };
    };
    return { isFile: false, isDirectory: true, name, fullPath, createReader };
  };
  const transfer = new DataTransfer();
  for (const [name, content] of items) transfer.items.add(Array.isArray(content) ? new File([], name) : file(name, content));
  const entries = items.map((item) => entry('', item));
  Object.defineProperty(transfer, 'items', { value: entries.map((e) => ({ kind: 'file', webkitGetAsEntry: () => e })) });
  const target = document.getElementById(on);
  const drag = (type) => target.dispatchEvent(new DragEvent(type, { dataTransfer: transfer, bubbles: true, cancelable: true }));
  drag('dragenter');
  drag('dragover');
  const overlay = document.getElementById('drop-overlay');
  const shown = getComputedStyle(overlay).display !== 'none';
  const box = target.getBoundingClientRect();
  const over = overlay.getBoundingClientRect();
  const covers = shown && over.left <= box.left && over.right >= box.right && over.top <= box.top && over.bottom >= box.bottom;
  drag('drop');
  return { said: shown ? overlay.textContent : null, covers };
}

// Drags files in over the panel, on over the side column, off the panel, then off the side column and so out of the
// frame, with no relatedTarget, as WebKit sends them; whether the overlay was on after each.
function dragAcross() {
  const transfer = new DataTransfer();
  transfer.items.add(new File(['A memo.\n'], 'memo.txt'));
  const fire = (id, type) => document.getElementById(id).dispatchEvent(new DragEvent(type, { dataTransfer: transfer, bubbles: true, cancelable: true }));
  const on = () => document.getElementById('frame').classList.contains('dropping');
  return [['panel', 'dragenter'], ['side', 'dragenter'], ['panel', 'dragleave'], ['side', 'dragleave']].map(([id, type]) => {
    fire(id, type);
    return on();
  });
}

// Drags files in over the panel, then ends the drag outside the frame, dropped on the top bar, then called off;
// whether the overlay was on after each step.
function dragEndsOutside() {
  const transfer = new DataTransfer();
  transfer.items.add(new File(['A memo.\n'], 'memo.txt'));
  const fire = (target, type) => target.dispatchEvent(new DragEvent(type, { dataTransfer: transfer, bubbles: true, cancelable: true }));
  const on = () => document.getElementById('frame').classList.contains('dropping');
  const ends = [() => fire(document.querySelector('.topbar'), 'drop'), () => fire(document, 'dragend')];
  return ends.flatMap((end) => {
    fire(document.getElementById('panel'), 'dragenter');
    const entered = on();
    end();
    return [entered, on()];
  });
}

// A folder of documents, two of one name in two folders; a .py, which r3con's folder walk leaves out by its suffix;
// a hidden file, Word's lock file, a memo in a hidden folder and one under __MACOSX, which the page leaves out unsaid;
// and a file the browser can't read. It has more entries than one batch.
const DROPPED_FOLDER = [
  'reports',
  [
    ['01_summary.txt', 'A summary in a dropped folder.\n'],
    ['.DS_Store', 'x'],
    ['notes.py', "print('not a document')\n"],
    ['~$draft.docx', 'x'],
    ['q3', [['02_incidents.md', 'Incidents in Q3.\n'], ['.hidden', [['03_secret.txt', 'A memo in a hidden folder.\n']]]]],
    ['__MACOSX', [['01_summary.txt', "macOS's metadata, named like a memo.\n"]]],
    ['q4', [['02_incidents.md', 'Incidents in Q4.\n']]],
    ['moved.txt', { fails: true }],
  ],
];
const LOOSE_FILE = ['loose.py', "print('dropped on its own')\n"];
const filesSaid = (page) => page.textContent('#files-said');
const memoNames = (from, to) => Array.from({ length: to - from }, (_, i) => `memo_${String(from + i).padStart(2, '0')}.txt`);

async function dropArea() {
  await withPage('drop', { sample: false }, async (page) => {
    await page.click('#new-tab-btn');
    await page.click('#tab-chat');
    const over = await page.evaluate(dropItems, { on: 'panel', items: [['panel_memo.txt', 'A memo dropped on the panel.\n']] });
    await removeButton(page, 'panel_memo.txt').waitFor();
    const added = { names: (await page.evaluate(filesPane)).names, docs: await docNames(page), filesShown: await page.$eval('#files-pane', (el) => !el.hidden) };
    const panelName = 'files dropped on the panel are added to the shown tab, as on the files pane, which then shows; the overlay covers the panel while they are over it';
    const onPanel = over.said === DROP_ADDS && over.covers && isDeepStrictEqual(added, { names: ['panel_memo.txt'], docs: ['panel_memo.txt'], filesShown: true });
    check('drop', panelName, onPanel, `${JSON.stringify(over)}; ${JSON.stringify(added)}`);

    const across = await page.evaluate(dragAcross);
    const acrossName = 'files dragged over the frame keep the overlay on from the panel onto the side column, and leaving the frame turns it off, with no relatedTarget, as WebKit sends';
    check('drop', acrossName, isDeepStrictEqual(across, [true, true, true, false]), JSON.stringify(across));
    const ended = await page.evaluate(dragEndsOutside);
    const endedName = 'a drag of files that ends outside the frame, dropped there or called off, turns the overlay off, though the frame had no dragleave';
    check('drop', endedName, isDeepStrictEqual(ended, [true, false, true, false]), JSON.stringify(ended));

    await page.evaluate(dropItems, { on: 'panel', items: [DROPPED_FOLDER, LOOSE_FILE] });
    await removeButton(page, LOOSE_FILE[0]).waitFor();
    const folder = { names: (await page.evaluate(filesPane)).names, said: await filesSaid(page), chat: await lastChatLine(page), error: await filesError(page) };
    const expected = {
      names: ['panel_memo.txt', '01_summary.txt', 'loose.py', 'q3/02_incidents.md', 'q4/02_incidents.md'],
      said: '4 files added; 1 skipped.',
      chat: '4 files added; 1 skipped: the next question runs over 5 documents.',
      error: `Not added: moved.txt. ${BROWSER_COULD_NOT_READ}`,
    };
    const folderName =
      "a folder dropped on the panel is read through every batch its reader gives: only the text, PDF and Word files r3con run reads in a folder, named by their paths below it, and, as that walk does, nothing hidden or under __MACOSX; a file dropped beside it is taken as it is; the files pane counts as skipped only what was left out by its suffix, and names a file the browser couldn't read";
    check('drop', folderName, isDeepStrictEqual(folder, expected), JSON.stringify(folder));

    await page.evaluate(dropItems, { on: 'panel', items: [['scripts', [['a.py', 'x'], ['b.py', 'x'], ['.DS_Store', 'x']]]] });
    const none = { error: await becomes(page, (text) => document.getElementById('files-error').textContent === text, `${NO_DOCUMENTS}; 2 skipped.`), names: (await page.evaluate(filesPane)).names };
    check('drop', 'a folder with no file r3con run reads adds nothing, and says how many it skipped', none.error && isDeepStrictEqual(none.names, expected.names), `${await filesError(page)}; ${JSON.stringify(none)}`);

    await page.click('#new-tab-btn');
    const many = memoNames(0, MAX_FILES + 2).map((name) => [name, `${name}\n`]);
    await page.evaluate(dropItems, { on: 'panel', items: [['many', many.reverse()]] });
    await removeButton(page, 'memo_39.txt').waitFor();
    const capped = { names: (await page.evaluate(filesPane)).names, error: await filesError(page), said: await filesSaid(page) };
    const cappedName = `a folder of more files than a tab takes, walked in any order, adds the first ${MAX_FILES} by name, and says how many it left out`;
    const cappedExpected = { names: memoNames(0, MAX_FILES), error: `Not added: 2 files. ${TOO_MANY_FILES}`, said: `${MAX_FILES} files added.` };
    check('drop', cappedName, isDeepStrictEqual(capped, cappedExpected), JSON.stringify({ ...capped, names: `${capped.names.length}: ${capped.names[0]}…${capped.names.at(-1)}` }));

    await page.click('#new-tab-btn');
    const pdfs = ['a_0.pdf', 'a_1.pdf', 'a_2.pdf'].map((name) => [name, '%PDF-1.7']);
    await page.evaluate(dropItems, { on: 'panel', items: [['mixed', [...pdfs, ...memoNames(0, MAX_FILES).map((name) => [name, `${name}\n`])]]] });
    await removeButton(page, 'memo_39.txt').waitFor();
    const filled = { names: (await page.evaluate(filesPane)).names, error: await filesError(page), said: await filesSaid(page) };
    const filledName = `files refused among the first ${MAX_FILES} by name leave their room to the next, so the tab still fills to ${MAX_FILES}, and none is called one too many`;
    const filledExpected = { names: memoNames(0, MAX_FILES), error: `Not added: a_0.pdf, a_1.pdf, a_2.pdf. ${NEEDS_APP}`, said: `${MAX_FILES} files added.` };
    check('drop', filledName, isDeepStrictEqual(filled, filledExpected), JSON.stringify({ ...filled, names: filled.names.length }));

    await showRunTab(page, 1);
    await page.click('#tab-chat');
    await send(page, NEW_QUESTION);
    const late = Array.from({ length: 7 }, (_, i) => [`late_${i + 1}.txt`, 'A late memo.\n']);
    const overLocked = await page.evaluate(dropItems, { on: 'panel', items: [...late, DROPPED_FOLDER] });
    const refused = { error: await filesError(page), names: (await page.evaluate(filesPane)).names, tabs: (await page.evaluate(tabRow)).length };
    const lockedName =
      'files and a folder dropped on the panel of a tab that has run are refused at once, unread, as the overlay over the panel says, naming the first five and counting the rest';
    const lockedExpected = { error: `Not added: late_1.txt, late_2.txt, late_3.txt, late_4.txt, late_5.txt and 3 more. ${LOCKED_DROP}`, names: expected.names, tabs: 4 };
    check('drop', lockedName, overLocked.said === DROP_LOCKED && isDeepStrictEqual(refused, lockedExpected), `${JSON.stringify(overLocked)}; ${JSON.stringify(refused)}`);
  });

  const dir = mkdtempSync(join(tmpdir(), 'r3con-folder-'));
  try {
    const picked = join(dir, 'picked');
    for (const [path, text] of [['01_memo.txt', 'A picked memo.\n'], ['sub/02_memo.md', 'A memo in a subfolder.\n'], ['.DS_Store', 'x'], ['notes.py', 'x'], ['.git/03_memo.txt', 'x']]) {
      mkdirSync(dirname(join(picked, path)), { recursive: true });
      writeFileSync(join(picked, path), text);
    }
    await withPage('drop', { sample: false }, async (page) => {
      await page.click('#tab-files');
      await page.setInputFiles('#folder-input', picked);
      await removeButton(page, '01_memo.txt').waitFor();
      const pane = { names: (await page.evaluate(filesPane)).names, said: await filesSaid(page) };
      const pickedName = 'a folder picked is read as a dropped one is, its files named by their paths below it, and the files pane counts what was left out by its suffix';
      const pickedExpected = { names: ['01_memo.txt', 'sub/02_memo.md'], said: '2 files added, in place of the 5 sample memos; 1 skipped.' };
      check('drop', pickedName, isDeepStrictEqual(pane, pickedExpected), JSON.stringify(pane));
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }

  await withPage('drop', {}, async (page) => {
    await page.evaluate(() => {
      window.everDropping = false;
      const frame = document.getElementById('frame');
      new MutationObserver(() => (window.everDropping ||= frame.classList.contains('dropping'))).observe(frame, { attributes: true });
    });
    const before = await positionOf(page, 'r2-1');
    await dragNode(page, 'r2-1', 60, 40);
    const moved = !isDeepStrictEqual(before, await positionOf(page, 'r2-1'));
    const textTaken = await page.evaluate(() => {
      const transfer = new DataTransfer();
      transfer.setData('text/plain', 'a dragged line');
      const panel = document.getElementById('panel');
      const drag = (type) => {
        const event = new DragEvent(type, { dataTransfer: transfer, bubbles: true, cancelable: true });
        panel.dispatchEvent(event);
        return event.defaultPrevented;
      };
      return ['dragenter', 'dragover', 'drop'].map(drag).some(Boolean);
    });
    const after = { everDropping: await page.evaluate(() => window.everDropping), files: (await page.evaluate(filesPane)).count, chatShown: await page.$eval('#chat-pane', (el) => !el.hidden) };
    const name = 'a node dragged in the panel, and text dragged over it, show no drop overlay and add nothing: only a drag of files is taken';
    check('drop', name, moved && !textTaken && isDeepStrictEqual(after, { everDropping: false, files: String(SAMPLE_DOCS), chatShown: true }), `moved ${moved}; text taken ${textTaken}; ${JSON.stringify(after)}`);
  });
}

// ---------- stop: the shown tab's run, in the prototype ----------

// Some node reads as stopped, and every failed node does.
function nodesStopped() {
  const names = [...document.querySelectorAll('#world .node.failed')].map((el) => el.getAttribute('aria-label'));
  return names.length > 0 && names.every((name) => name.endsWith(', stopped'));
}

// A select ellipsizes its text without scrolling, so the picker's text is measured beside it.
function pickerFits() {
  const pick = document.getElementById('model-pick');
  const probe = document.createElement('span');
  probe.style.cssText = 'position: absolute; visibility: hidden; white-space: pre';
  probe.textContent = pick.selectedOptions[0].textContent;
  pick.parentElement.append(probe);
  const room = pick.clientWidth - parseFloat(getComputedStyle(pick).paddingRight);
  const fits = probe.getBoundingClientRect().width <= room + 0.5;
  probe.remove();
  return fits;
}

function statusStarts(start) {
  return document.getElementById('run-status').textContent.startsWith(start);
}

const drawnAs = ({ nodes, edges, isLit, status }) => ({ nodes, edges, isLit, status });

async function pressStop(page, { keyboard = false } = {}) {
  if (!keyboard) return page.$eval('#stop-btn', (button) => button.click());
  await page.focus('#stop-btn');
  await page.keyboard.press('Enter');
}

// With motion on, a replay takes seconds, so it can be stopped partway.
async function stopArea() {
  await withPage('stop', { sample: false }, async (page) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    const question = await page.inputValue('#question');
    await page.click('#run-btn');
    await page.waitForSelector('#world .node.r1', { state: 'attached' });
    const offered = (await page.evaluate(composer)).stop;
    check('stop', "1440 light: with stop offered, the model picker shows the default model's whole name", offered && (await page.evaluate(pickerFits)));
    await pressStop(page, { keyboard: true });
    const stopped = await page.evaluate(nodesStopped);
    const after = { composer: await page.evaluate(composer), focus: await focusedId(page), working: await count(page, '#world .node.working') };
    // What was animating in as it stopped plays out first, and its class comes off.
    await becomes(page, panelStill);
    const frozen = await page.evaluate(panelSnapshot);
    const offersRun = isDeepStrictEqual(after, { composer: { primary: 'run ⏎', stop: false, newRun: false }, focus: 'question', working: 0 });
    const stopName = "stop, pressed from the keyboard during a replay, ends it where it is: its working nodes read as stopped, the status says where, and the composer offers run, its focus in the question box";
    check('stop', stopName, offered && stopped && frozen.status === STOPPED_IN_RELEVANCE && offersRun, `${frozen.status}; ${JSON.stringify(after)}`);

    await send(page, question);
    const ended = await becomes(page, statusIs, SAMPLE_STATUS, REPLAY_MS);
    await page.locator('.run-tab').first().click();
    const still = isDeepStrictEqual(drawnAs(await page.evaluate(panelSnapshot)), drawnAs(frozen));
    check('stop', 'a stopped replay draws nothing more: once the same replay has ended in another tab, the stopped one is as it was', ended && still, `ended ${ended}`);
  });

  await withPage('stop', { sample: false }, async (page) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    const question = await page.inputValue('#question');
    await page.click('#run-btn');
    await page.waitForSelector('#world .node.doc.working', { state: 'attached' });
    await pressStop(page);
    const docs = await count(page, '#world .node.doc.working, #world .node.doc.failed');
    const reading = await page.textContent('#run-status');
    const readName = 'stop while a replay reads its documents leaves them read, not stopped, and the status says it stopped while reading';
    check('stop', readName, docs === 0 && reading === STOPPED_WHILE_READING, `${docs} documents working or stopped; ${reading}`);

    await send(page, question);
    await becomes(page, lastTurnSays, [`read ${SAMPLE_DOCS} files`], REPLAY_MS);
    await pressStop(page);
    const between = await page.textContent('#run-status');
    check('stop', 'stop between reading and round 1 says only that it stopped', between === STOPPED, between);

    await send(page, question);
    await becomes(page, statusStarts, 'running · parsing', REPLAY_MS);
    await pressStop(page);
    const parsing = await page.textContent('#run-status');
    check('stop', 'stop right after parsing begins says it stopped in parsing', parsing === STOPPED_IN_PARSING, parsing);
  });

  await withPage('stop', { layout: LAYOUTS.phone, sample: false }, async (page) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.click('#run-btn');
    await page.waitForSelector('#world .node.r1', { state: 'attached' });
    const fits = (await page.evaluate(composer)).stop && (await page.evaluate(noHorizontalScroll)) && (await page.evaluate(pickerFits));
    check('stop', "400 dark: with stop offered, the page has no horizontal scroll, and the picker shows the default model's whole name", fits);
  });
}

// ---------- live: served by r3con app, with a scripted model ----------

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
  const release = () => writeFileSync(join(logsDir, RELEASE_FILE), '');
  return { url, stop, release };
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
    const note = await page.$eval('#keys-note', (el) => (el.hidden ? null : el.textContent));
    check('live', 'the settings show no note under the keys', note === null, note);
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
    const answered = await becomes(page, lastTurnSays, [SCRIPTED_ANSWER, `ready · ${DEFAULT_MODEL} · $`], LIVE_RUN_MS);
    const askLog = await page.evaluate(logOf);
    const told = isDeepStrictEqual(askLog.after, [SCRIPTED_ANSWER]) && !askLog.open && askLog.mark === 'ok';
    const askName = 'a follow-up runs an ask and shows its answer in the chat with nothing under it, and its model and cost on its folded log';
    check('live', askName, answered && told, `${askLog.folded}; ${askLog.after.join(' | ')}`);

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
    const ready = lines.filter((text) => text === `${SPLIT_MEMO} — ${SNIPPET_READY}`).length;
    const parsed = lines.includes(`${SPLIT_MEMO} — ${STRUCTURE_READY}`);
    const cited = (await count(page, `.edge.cite[data-key="r1-0>r2-${SAMPLE_DOCS}"]`)) === 1;
    const partsName = 'a document read in two parts, the first with nothing relevant, is drawn and logged from both: ready in both rounds, its round-2 citation kept, both parts parsed';
    check('live', partsName, done && ready === ROUNDS && cited && parsed, `ready in ${ready} rounds; cited ${cited}; parsed ${parsed}`);
    await inspect(page, `parsed-${SAMPLE_DOCS}`);
    const { memos } = await shownParse(page);
    const read = memos.flatMap((memo) => [memo.title, ...memo.lines.map(({ text }) => text)]);
    const both = memos.length === 2 && isDeepStrictEqual(read, splitMemo().buffer.toString().split('\n\n'));
    check('live', "its popup shows both parts' parses, one memo each, which hold its lines in order", both, `${memos.length} memos, ${read.length} lines`);
    const scrolls = await page.$eval('#inspector pre.code', (el) => el.scrollHeight > el.clientHeight);
    check('live', 'a parse taller than the panel scrolls in its popup, which stays inside the panel', scrolls && (await page.evaluate(popupInsidePanel)), JSON.stringify(await popup(page)));
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
    const landed = await becomes(page, landedNode, 'answer-1', LIVE_RUN_MS);
    const docIds = await docIdsOf(page);
    const expected = lineage(SCRIPTED_CITES, docIds, 'answer-1', SCRIPTED_CITES.answer.provenance);
    const lit = await litEdges(page);
    const ownPath = landed && isDeepStrictEqual(lit, expected) && isDeepStrictEqual(await litAnswers(page), ['answer-1']);
    const beforeEnd = await inspect(page, 'answer-1');
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
    const opened = await becomes(page, newRunDone, LIVE_DONE.source, LIVE_RUN_MS);
    const runs = posts.filter((path) => path === '/api/runs').length;
    const nodes = await count(page, '.node');
    const inTab = (await shownTab(page)) === 1;
    const newRunName = "after an ask, new run is offered and opens its run's own tab";
    check('live', newRunName, offered && opened && inTab && runs === 2 && nodes === SAMPLE_COUNTS.nodes, `${runs} runs, ${nodes} nodes`);
    const focused = await focusedId(page);
    check('live', 'new run from the keyboard keeps the focus on new run, still offered while that run goes', focused === 'new-run-btn', `focus on #${focused}`);
    await page.fill('#question', `Which memo is the registry? ${FAILS}`);
    await page.focus('#new-run-btn');
    await page.keyboard.press('Enter');
    const failed = await becomes(page, statusIs, LIVE_FAILED, LIVE_RUN_MS);
    const refocused = await focusedId(page);
    const hidden = !(await page.evaluate(composer)).newRun;
    const failName = 'a new run from the keyboard that fails hides new run, and the focus moves to the question box';
    check('live', failName, failed && hidden && refocused === 'question', `${await page.textContent('#run-status')}; focus on #${refocused}`);
  });
}

// Holds the run's answer back, so its log is seen going.
async function liveLog(url) {
  const before = (page) => page.addInitScript(holdFrom, 'reasoning_finished');
  await withPage('live', { url, sample: false, before }, async (page) => {
    const place = await page.evaluate(composerPlace);
    check('live', "no note under the composer: it takes the note's place, a line at most from the foot of the chat column", place.note === null && place.gap < LINE_PX, JSON.stringify(place));
    await page.click('#run-btn');
    await becomes(page, answerWorking, null, LIVE_RUN_MS);
    const going = await page.evaluate(logOf);
    const goingName = 'while a run goes its log is folded under the step it is on, with a spinner, its button named for the question and that step';
    check('live', goingName, !going.open && going.mark === 'log-spin' && going.folded === 'reasoning' && going.name === `${going.question} reasoning`, going.name);
    const status = await page.textContent('#run-status');
    check('live', 'the status names the last stage as the log does: reasoning', status === 'running · reasoning', status);
    await page.evaluate(() => (window.release = true));
    await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const ended = await page.evaluate(logOf);
    const endedName = 'once it ends its folded line says ready, the model and the cost, and nothing is under its answer';
    check('live', endedName, !ended.open && ended.mark === 'ok' && LIVE_READY.test(ended.folded) && isDeepStrictEqual(ended.after, [SCRIPTED_ANSWER]), `${ended.folded}; ${ended.after.join(' | ')}`);
    await page.focus('.log-toggle');
    await page.keyboard.press('Enter');
    const opened = await page.evaluate(logOf);
    const [sent, ...steps] = stepsOf(opened);
    const sentFirst = sent?.[0].startsWith(`sent ${SAMPLE_DOCS} documents to r3con · `) && sent[1] === 0;
    const listed = opened.open && sentFirst && isDeepStrictEqual(steps, RUN_STEPS);
    check('live', "Enter on its chevron opens the log: what was sent, then each step in turn with its documents' lines under it", listed, JSON.stringify(stepsOf(opened)));
    const docs = recordedSample().docs;
    const ready = docs.map((_, i) => `${stem(i)} — ${STRUCTURE_READY}`).sort();
    check('live', "each parsed document's line says its structure is ready", isDeepStrictEqual(parsedLines(opened), ready), parsedLines(opened).join(' | '));
    await page.click('.log-lines .line[data-node="parsed-0"]');
    const title = (await popup(page))?.title;
    const shown = await shownParse(page);
    const filledName = "a parsed document's line opens its node's popup, which shows the parse the model filled, nested lines and all";
    check('live', filledName, title === '● 01_northgate · parsed' && isDeepStrictEqual(shown, scriptedParse(docs[0].text)), `${title}; ${JSON.stringify(shown)}`);
  });
}

// Holds the second run's answer back until the first tab is shown.
async function liveTabs(url) {
  const before = (page) => page.addInitScript(holdFrom, 'reasoning_finished');
  await withPage('live', { url, sample: false, before }, async (page) => {
    const posts = postsOf(page);
    await page.evaluate(() => (window.release = true));
    await page.click('#run-btn');
    await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const first = await page.evaluate(panelSnapshot);
    const firstStatus = await page.textContent('#run-status');
    const firstChat = await chatLines(page, 0);
    await page.evaluate(() => (window.release = false));
    const question = 'Which memo is the registry?';
    await send(page, question, 'Control+Enter');
    const held = await becomes(page, answerWorking, null, LIVE_RUN_MS);
    await page.locator('.run-tab').first().click();
    await page.evaluate(() => (window.release = true));
    const ended = await becomes(page, tabEnded, 1, LIVE_RUN_MS);
    const untouched = isDeepStrictEqual(await page.evaluate(panelSnapshot), first) && (await page.textContent('#run-status')) === firstStatus;
    const awayName = "a run's events land in its own tab: the tab shown meanwhile is left as it was";
    check('live', awayName, held && ended && untouched, `held ${held}; ended ${ended}`);
    const [firstAfter, second] = [await chatLines(page, 0), await chatLines(page, 1)];
    const ownChat =
      isDeepStrictEqual(firstAfter, firstChat) && isDeepStrictEqual(second.slice(0, 2), [loadedLine(SAMPLE_DOCS), question]) && LIVE_READY.test(second[2]) && second.at(-1) === SCRIPTED_ANSWER;
    const ownName = "a run in tab 2 writes into tab 2's chat alone, its answer and its end too, which land while tab 1 is shown";
    check('live', ownName, ownChat, `tab 1 ${firstAfter.length} lines, was ${firstChat.length}; tab 2: ${second.slice(0, 3).join(' | ')} … ${second.at(-1)}`);
    await page.locator('.run-tab').nth(1).click();
    const nodes = await count(page, '.node');
    const working = await count(page, '.node.working');
    const lit = await litEdges(page);
    const drawn = nodes === SAMPLE_COUNTS.nodes && working === 0 && isDeepStrictEqual(lit, lineage(SCRIPTED_CITES, await docIdsOf(page)));
    const status = await page.textContent('#run-status');
    const backName = "switching to that tab shows all it drew in the background: every node, its lit answer and its cost";
    check('live', backName, drawn && LIVE_DONE.test(status), `${nodes} nodes, ${working} working; ${status}`);

    await page.evaluate(() => (window.release = false));
    await send(page, 'Which memo is the registry, a third time?', 'Control+Enter');
    await becomes(page, answerWorking, null, LIVE_RUN_MS);
    await page.locator('.run-tab').first().click();
    const asked = page.waitForRequest((request) => request.method() === 'POST' && request.url().endsWith('/asks'), { timeout: SETTLE_TIMEOUT_MS }).catch(() => null);
    await send(page, 'Which memo mentions the storm?');
    await asked;
    const sent = posts.filter((path) => path.endsWith('/asks')).length;
    await page.evaluate(() => (window.release = true));
    const answered = await becomes(page, landedNode, 'answer-1', LIVE_RUN_MS);
    const askName = "while another tab's run is going, an ask over a finished tab goes ahead, and lands there";
    check('live', askName, sent === 1 && answered && (await shownTab(page)) === 0, `${sent} asks sent`);
  });
}

async function liveAskAfterFailed(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    await page.click('#run-btn');
    await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    await send(page, `Which memo is the registry? ${FAILS}`);
    const failed = await becomes(page, failedNode, 'answer-1', LIVE_RUN_MS);
    await clickNode(page, 'answer-1');
    await send(page, 'Which memo is the registry?');
    const landed = await becomes(page, landedNode, 'answer-2', LIVE_RUN_MS);
    const lit = await litAnswers(page);
    const openOn = (await popup(page))?.title;
    const litName = 'a landing ask is lit while the open popup is an answer that never landed';
    check('live', litName, failed && landed && isDeepStrictEqual(lit, ['answer-2']), `lit ${lit}; popup ${openOn}`);
  });
}

// Holds the run's end back, so its answer has landed while it is still going.
async function liveWhileGoing(url) {
  const before = (page) => page.addInitScript(holdFrom, 'run_finished');
  await withPage('live', { url, sample: false, before }, async (page) => {
    await page.click('#run-btn');
    const answered = await becomes(page, lastTurnSays, [SCRIPTED_ANSWER], LIVE_RUN_MS);
    const offered = await page.evaluate(composer);
    await page.fill('#question', 'Which memo is the registry?');
    await page.click('#run-btn');
    const refused = await page.textContent('#run-status');
    const started = page.waitForRequest((request) => request.method() === 'POST' && request.url().endsWith('/api/runs'), { timeout: SETTLE_TIMEOUT_MS }).catch(() => null);
    await page.$eval('#new-run-btn', (button) => button.click());
    const second = (await started) !== null && (await shownTab(page)) === 1;
    const goingName = "while a run is going, even with its answer landed, the composer offers ask ⏎, which waits for its end as Enter does, and new run, which starts a second run";
    check('live', goingName, answered && offered.primary === 'ask ⏎' && offered.newRun && refused === ASK_WHEN_DONE && second, `${JSON.stringify(offered)}; ${refused}`);
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
    await page.fill('#question', 'Which memo is the registry, once more?');
    await page.click('#new-run-btn');
    const ended = await becomes(page, newRunDone, LIVE_DONE.source, LIVE_RUN_MS);
    const runs = posts.filter((path) => path === '/api/runs').length;
    const row = (await page.evaluate(tabRow)).map((tab) => [tab.selected, tab.closable]);
    const askTurn = await page.locator('.transcript').first().locator('.turn').last().locator('.line .t').allTextContents();
    const stillAsking = askTurn.includes(`asking · ${DEFAULT_MODEL}`);
    const askName = "while an ask is going, a new run goes ahead in a tab of its own, and the asking tab can't be closed";
    check('live', askName, asking && ended && runs === 2 && isDeepStrictEqual(row, [[false, false], [true, true]]) && stillAsking, `${runs} runs; ${JSON.stringify(row)}`);
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
    const failedLog = await page.evaluate(logOf);
    const inView = !failedLog.open && failedLog.mark === 'err' && failedLog.folded === 'failed in relevance' && isDeepStrictEqual(failedLog.after, [LIVE_ERROR]);
    check('live', "a failed run's log ends on where it failed, its spinner gone, and its error is in view below it", inView, `${failedLog.folded}; ${failedLog.after.join(' | ')}`);
    const label = await page.textContent('#run-btn');
    check('live', 'after a failed run the next question starts a new run', label === 'run ⏎', label);
  });
}

const openFileView = async (page, name) => {
  await page.click(`#file-list .file:has-text("${name}")`);
  return page.$eval('#file-view', (view) => ({ meta: view.querySelector(':scope > .k').textContent, text: view.querySelector('pre, p.k').textContent }));
};

const fmtKb = (n) => `${(n / 1024).toFixed(1)} KB`;
// As the page writes a file's size.
const fmtSize = (n) => (n < 1024 ? `${n} B` : fmtKb(n));
const runPosted = (page) =>
  page.waitForRequest((request) => request.method() === 'POST' && new URL(request.url()).pathname === '/api/runs', { timeout: SETTLE_TIMEOUT_MS }).catch(() => null);

async function livePdf(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    const posts = postsOf(page);
    const memoPdf = documents['01_memo.pdf'];
    const padded = documents['03_padded.pdf'];
    await page.setInputFiles('#file-input', [memoPdf, documents['02_scan.pdf'], padded]);
    const two = await becomes(page, () => document.querySelectorAll('.node.doc').length === 2);
    await page.click('#tab-files');
    const listed = await removeNames(page);
    const error = await filesError(page);
    const sentToApp = isDeepStrictEqual(posts, Array(3).fill('/api/documents/text'));
    const scanRefused = error === `Not added: 02_scan.pdf. ${NO_TEXT_LAYER}` && isDeepStrictEqual(listed, ['remove 01_memo.pdf', 'remove 03_padded.pdf']);
    const scanName = 'each PDF added goes to r3con app for its text, and a scan, with none, is refused as it is added, named, saying why';
    check('live', scanName, two && sentToApp && scanRefused, `${listed.join(', ')}; ${error}; ${posts.join(', ')}`);

    const shown = await openFileView(page, '01_memo.pdf');
    const extracted = isDeepStrictEqual(shown, { meta: `${memoPdf.buffer.length} B · your file`, text: PDF_TEXT });
    check('live', "the files tab shows a PDF's extracted text and its file's size", extracted, JSON.stringify(shown));
    await page.click('#file-back');
    const big = await openFileView(page, '03_padded.pdf');
    const whole = isDeepStrictEqual(big, { meta: `${fmtKb(padded.buffer.length)} · your file`, text: PADDED_TEXT });
    const bigName = "a PDF of more bytes than the preview's characters, but fewer characters of text, shows its whole text, with no note that only its start is shown";
    check('live', bigName, padded.buffer.length > PREVIEW_CHARS && whole, JSON.stringify(big));

    await page.click('#tab-chat');
    const started = runPosted(page);
    await page.click('#run-btn');
    const sent = (await started)?.postDataJSON().documents;
    const bytes = Buffer.byteLength(PDF_TEXT + PADDED_TEXT);
    const counted = await becomes(page, lastTurnSays, [`sent 2 documents to r3con · ${bytes} B`]);
    const runName = "a run is sent the PDFs' extracted texts, and says it sent their size, not the PDFs'";
    const said = (await page.evaluate(lastTurnLines)).find((text) => text.startsWith('sent '));
    check('live', runName, isDeepStrictEqual(sent, [PDF_TEXT, PADDED_TEXT]) && counted, `${JSON.stringify(sent)}; ${said}`);
  });
}

// The first PDF's text is held at the server while a second batch is added: that batch waits its turn, so
// the files pane says each batch's lines in order.
async function liveBatches(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    let release;
    const held = new Promise((resolve) => (release = resolve));
    let requests = 0;
    await page.route('**/api/documents/text?*', async (route) => {
      requests += 1;
      if (requests === 1) await held;
      await route.continue();
    });
    await page.evaluate(() => {
      const said = document.getElementById('files-said');
      window.saidLines = [];
      new MutationObserver(() => window.saidLines.push(said.textContent)).observe(said, { childList: true });
    });
    await page.setInputFiles('#file-input', documents['01_memo.pdf']);
    await page.setInputFiles('#file-input', [oldWord(), documents['03_padded.pdf']]);
    release();
    const lines = ['Reading 1 file…', '1 file added, in place of the 5 sample memos.', 'Reading 2 files…', '1 file added.'];
    const inOrder = await becomes(page, (expected) => JSON.stringify(window.saidLines) === JSON.stringify(expected), lines);
    const refused = (await filesError(page)) === `Not added: old.doc. ${OLD_WORD}`;
    const docs = await count(page, '.node.doc');
    const name = "a batch added while a PDF is still being read waits its turn, then each batch's lines come in order, and the second's refusal stands";
    check('live', name, inOrder && refused && docs === 2, `${(await page.evaluate(() => window.saidLines)).join(' | ')}; ${await filesError(page)}; ${docs} documents`);
  });
}

async function liveWord(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    const posts = postsOf(page);
    const report = documents['report.docx'];
    // Word's lock file for report.docx, which is not added or sent.
    const lockFile = { name: '~$report.docx', mimeType: 'application/octet-stream', buffer: Buffer.from('\x07Michael') };
    await page.setInputFiles('#file-input', [documents['empty.docx'], report, lockFile]);
    const one = await becomes(page, () => document.querySelectorAll('.node.doc').length === 1);
    await page.click('#tab-files');
    const listed = await removeNames(page);
    const error = await filesError(page);
    const sentToApp = isDeepStrictEqual(posts, Array(2).fill('/api/documents/text'));
    const emptyRefused = error === `Not added: empty.docx. ${NO_TEXT}` && isDeepStrictEqual(listed, ['remove report.docx']);
    const emptyName =
      "each Word file added goes to r3con app for its text, one with none is refused as it is added, named, saying why, and Word's lock file is left out";
    check('live', emptyName, one && sentToApp && emptyRefused, `${listed.join(', ')}; ${error}; ${posts.join(', ')}`);

    const shown = await openFileView(page, 'report.docx');
    const extracted = isDeepStrictEqual(shown, { meta: `${fmtSize(report.buffer.length)} · your file`, text: WORD_TEXT });
    check('live', "the files tab shows a Word file's text as Markdown, its table and all, and its file's size", extracted, JSON.stringify(shown));

    await page.click('#tab-chat');
    const started = runPosted(page);
    await page.click('#run-btn');
    const sent = (await started)?.postDataJSON().documents;
    check('live', "a run is sent the Word file's Markdown", isDeepStrictEqual(sent, [WORD_TEXT]), JSON.stringify(sent));
  });
}

const utf16 = (name, text, bom = []) => ({ name, mimeType: 'text/plain', buffer: Buffer.concat([Buffer.from(bom), Buffer.from(text, 'utf16le')]) });
// Its ASCII text, each character four bytes, low byte first.
const utf32 = (name, text) => ({ name, mimeType: 'text/plain', buffer: Buffer.from([...UTF32LE_BOM, ...[...text].flatMap((c) => [c.charCodeAt(0), 0, 0, 0])]) });

async function liveNotText(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    const posts = postsOf(page);
    const binary = { name: 'data.bin', mimeType: 'application/octet-stream', buffer: Buffer.from('a header\0with a NUL byte') };
    const files = [
      oldWord(),
      oldPowerPoint(),
      binary,
      memo('notes.log', 'A log line.\n'),
      memo('page.html', '<p>An HTML note.</p>\n'),
      memo('README', 'An extensionless note.\n'),
      utf16('utf16_bom.txt', 'A Notepad note.\n', UTF16LE_BOM),
      utf16('utf16_no_bom.txt', 'No byte-order mark.\n'),
      utf32('utf32.txt', 'Four bytes a character.\n'),
      { name: 'ff_fe.bin', mimeType: 'application/octet-stream', buffer: Buffer.from([...UTF16LE_BOM, 0x13, 0x37, 0x00, 0x00, 0x2a]) },
    ];
    await page.setInputFiles('#file-input', files);
    const four = await becomes(page, () => document.querySelectorAll('.node.doc').length === 4);
    await page.click('#tab-files');
    const listed = await removeNames(page);
    const error = await filesError(page);
    const refused =
      error === `Not added: data.bin, ff_fe.bin, old.ppt, utf16_no_bom.txt, utf32.txt. ${NOT_TEXT} Not added: old.doc. ${OLD_WORD}` &&
      isDeepStrictEqual(listed, ['remove README', 'remove notes.log', 'remove page.html', 'remove utf16_bom.txt']);
    const name =
      'an old .doc and .ppt, a file with a NUL byte, UTF-16 without a byte-order mark, UTF-32, and binary that starts like UTF-16 are refused, named, with what to do, and nothing is sent; a .log, an .html, an extensionless file and UTF-16 with a byte-order mark are added as text';
    check('live', name, four && refused && posts.length === 0, `${listed.join(', ')}; ${error}; ${posts.length} posts`);
    await page.click('#tab-chat');
    const started = runPosted(page);
    await page.click('#run-btn');
    const sent = (await started)?.postDataJSON().documents;
    const texts = ['An extensionless note.\n', 'A log line.\n', '<p>An HTML note.</p>\n', 'A Notepad note.\n'];
    check('live', 'a run is sent only the text files, the UTF-16 one decoded', isDeepStrictEqual(sent, texts), JSON.stringify(sent));
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
    await page.click('#tab-files');
    await page.getByRole('button', { name: 'remove 02_blank.txt', exact: true }).click();
    await page.click('#tab-chat');
    const started = page.waitForRequest((request) => request.method() === 'POST' && new URL(request.url()).pathname === '/api/runs', { timeout: SETTLE_TIMEOUT_MS }).catch(() => null);
    await page.click('#run-btn');
    const sent = (await started)?.postDataJSON().documents;
    check('live', 'once the unreadable file is removed, the next run sends only the readable one', isDeepStrictEqual(sent, ['A memo.\n']), JSON.stringify(sent));
  });
}

// A zip dropped on the panel goes to r3con app, which unpacks it, sent the room left in the tab; its documents are
// added and run as dropped files are.
async function liveZip(url) {
  await withPage('live', { url, sample: false, expectedErrors: [REFUSED_LOG] }, async (page) => {
    const limits = [];
    page.on('request', (request) => request.url().includes('/api/documents/zip') && limits.push(new URL(request.url()).searchParams.get('limit')));
    const zipped = (name) => [name, { bytes: [...documents[name].buffer] }];
    await page.evaluate(dropItems, { on: 'panel', items: [zipped('memos.zip'), zipped('photos.zip'), ['broken.zip', 'not a zip']] });
    await removeButton(page, '01_zipped.txt').waitFor();
    const pane = { names: (await page.evaluate(filesPane)).names, said: await filesSaid(page), error: await filesError(page), limits };
    const expected = {
      names: ['01_zipped.txt'],
      said: '1 file added, in place of the 5 sample memos; 1 skipped.',
      error: `Not added: broken.zip. ${UNREADABLE_ZIP} Not added: 02_blank.txt. ${NO_TEXT} Not added: photos.zip. ${NO_DOCUMENTS}; 1 skipped.`,
      limits: ['40', '40', '39'],
    };
    const name =
      "zips dropped on the panel are unpacked by r3con app one at a time, each sent the room left: their documents are added by their paths below the folder they share, one with no text is refused, the files pane counts what was left out by its suffix, and a zip with no documents, or one r3con app can't open, is refused saying so";
    check('live', name, isDeepStrictEqual(pane, expected), JSON.stringify(pane));
    await page.click('#tab-chat');
    const started = runPosted(page);
    await page.click('#run-btn');
    const sent = (await started)?.postDataJSON().documents;
    check('live', "a run is sent the zipped memo's text", isDeepStrictEqual(sent, [ZIPPED_TEXT]), JSON.stringify(sent));
  });
}

// A + tab's run and ask are sent its own files, and tab 1 keeps the sample memos, not run.
async function liveTabFiles(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    const posts = postsOf(page);
    const text = 'A memo for the second tab.\n';
    await page.click('#new-tab-btn');
    await page.setInputFiles('#file-input', memo('07_second_tab.txt', text));
    await removeButton(page, '07_second_tab.txt').waitFor();
    await page.click('#tab-chat');
    const first = runPosted(page);
    await send(page, 'Which memo is the registry?');
    const sent = (await first)?.postDataJSON().documents;
    const done = await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const inPlace = done && isDeepStrictEqual(sent, [text]) && (await page.evaluate(tabRow)).length === 2 && (await shownTab(page)) === 1;
    check('live', "the first question in a + tab runs in place, sent only that tab's files", inPlace, JSON.stringify(sent));

    await send(page, 'Which memo mentions the storm?');
    const asked = await becomes(page, landedNode, 'answer-1', LIVE_RUN_MS);
    const second = runPosted(page);
    await send(page, 'Which memo is the registry, once more?', 'Control+Enter');
    const copied = (await second)?.postDataJSON().documents;
    const ended = await becomes(page, newRunDone, LIVE_DONE.source, LIVE_RUN_MS);
    const runs = posts.filter((path) => path === '/api/runs').length;
    const asks = posts.filter((path) => path.endsWith('/asks')).length;
    const newRun = asked && ended && isDeepStrictEqual(copied, [text]) && runs === 2 && asks === 1 && (await shownTab(page)) === 2;
    check('live', "Enter then asks over the + tab's run, and new run opens a third tab, sent a copy of its files", newRun, `${runs} runs, ${asks} asks; ${JSON.stringify(copied)}`);

    await showRunTab(page, 0);
    const pane = await paneOf(page);
    const label = (await page.evaluate(tabRow))[0].label;
    const untouched = label === FIRST_TAB_BEFORE_RUN && pane.removes === SAMPLE_DOCS && pane.adds && pane.locked === null;
    check('live', 'tab 1 keeps the sample memos, not run and not locked', untouched, `${label}; ${JSON.stringify(pane)}`);
  });
}

// A sample memo dropped on the can is left out of the next run.
async function liveTrash(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    const docs = recordedSample().docs;
    const id = await docNodeId(page, docs[0].name);
    await dragOver(page, await centre(page, id), await boxCentre(page, '#trash'));
    const started = runPosted(page);
    await page.click('#run-btn');
    const sent = (await started)?.postDataJSON().documents;
    const done = await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const expected = docs.slice(1).map((d) => d.text);
    check('live', 'a memo dropped on the can before a run is not sent with the run', done && isDeepStrictEqual(sent, expected), JSON.stringify(sent?.length));
  });
}

// Holds the first PDF's text at the server until the tab it was added in has started its run; three more
// batches wait behind it, the last with only a hidden file.
async function liveLockedWhileReading(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    let release;
    const held = new Promise((resolve) => (release = resolve));
    let extractions = 0;
    await page.route('**/api/documents/text?*', async (route) => {
      extractions += 1;
      await held;
      await route.continue();
    });
    await page.evaluate(() => {
      const said = document.getElementById('files-said');
      window.saidLines = [];
      new MutationObserver(() => window.saidLines.push(said.textContent)).observe(said, { childList: true });
    });
    const started = runPosted(page);
    const names = ['01_memo.pdf', '03_padded.pdf', 'report.docx'];
    for (const name of names) await page.setInputFiles('#file-input', documents[name]);
    await page.setInputFiles('#file-input', memo('.notes.txt', 'A hidden note.\n'));
    await page.click('#run-btn');
    const sent = (await started)?.postDataJSON().documents.length;
    release();
    await page.click('#tab-files');
    const refusal = [...names.map((name) => `Not added: ${name}. ${RAN_FIRST}`), RAN_FIRST].join(' ');
    const refused = await becomes(page, (text) => document.getElementById('files-error').textContent === text, refusal, LIVE_RUN_MS);
    const pane = await page.evaluate(filesPane);
    const said = await page.evaluate(() => window.saidLines);
    const name =
      "files being read, or queued, when their tab's run starts are not added: a queued batch is never read and says no reading line, every file refused stays named, and a batch with none a pick would take says the tab ran first";
    const kept = sent === SAMPLE_DOCS && refused && pane.names.length === SAMPLE_DOCS && extractions === 1 && isDeepStrictEqual(said, ['Reading 1 file…', '']);
    check('live', name, kept, `${sent} sent; ${extractions} read; said ${said.join(' | ')}; ${await filesError(page)}`);
  });
}

// Records each write to the files pane's two lines, and its live region's setting as it was written, in window.lineWrites.
function recordFilesLines() {
  window.lineWrites = [];
  for (const id of ['files-said', 'files-error']) {
    const el = document.getElementById(id);
    new MutationObserver(() => window.lineWrites.push({ id, text: el.textContent, live: el.getAttribute('aria-live') })).observe(el, { childList: true });
  }
}

const filesLines = async (page) => ({ said: await page.textContent('#files-said'), error: await filesError(page) });

// Holds the PDFs' text at the server until released.
async function holdExtraction(page) {
  let release;
  const held = new Promise((resolve) => (release = resolve));
  await page.route('**/api/documents/text?*', async (route) => {
    await held;
    await route.continue();
  });
  return release;
}

// Holds tab 2's PDF text at the server while tab 1 is shown.
async function liveBatchesPerTab(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    const release = await holdExtraction(page);
    await page.evaluate(recordFilesLines);
    await page.click('#new-tab-btn');
    await page.setInputFiles('#file-input', [documents['01_memo.pdf'], oldWord()]);
    await showRunTab(page, 0);
    await page.evaluate(() => (window.lineWrites = []));
    await page.setInputFiles('#file-input', memo('08_first_tab.txt', 'A memo for the first tab.\n'));
    const ownQueue = await becomes(page, () => document.querySelector('[aria-label="remove 08_first_tab.txt"]') !== null);
    check('live', "a tab's batch waits only for that tab's own: a text file is added to tab 1 while tab 2's PDF is still being read", ownQueue);
    const own = await page.evaluate(() => window.lineWrites);
    await showRunTab(page, 1);
    const reading = await page.textContent('#files-said');
    check('live', "a tab's reading line stays while its batch is read, through a switch away and back", reading === 'Reading 2 files…', reading);

    await showRunTab(page, 0);
    await page.evaluate(() => (window.lineWrites = []));
    release();
    const landed = await becomes(page, () => document.getElementById('run-tab-2').textContent === '2 new · 1 file', null, LIVE_RUN_MS);
    const first = { names: (await page.evaluate(filesPane)).names, ...(await filesLines(page)) };
    await showRunTab(page, 1);
    const second = { names: (await page.evaluate(filesPane)).names, ...(await filesLines(page)) };
    await showRunTab(page, 0);
    await showRunTab(page, 1);
    const again = { names: (await page.evaluate(filesPane)).names, ...(await filesLines(page)) };
    const restored = await page.evaluate(() => window.lineWrites);
    const firstOwn = { names: ['08_first_tab.txt'], said: '1 file added, in place of the 5 sample memos.', error: null };
    const landsName = 'a batch that lands while another tab is shown goes to the tab it was added in, and the shown tab keeps its own files and lines';
    check('live', landsName, landed && isDeepStrictEqual(first, firstOwn) && isDeepStrictEqual(second.names, ['01_memo.pdf']), JSON.stringify({ first, second }));
    const toldName = "that tab's pane says what happened to the batch once the tab is shown, its refusal too, and still says it after a switch away and back";
    const told = second.said === '1 file added.' && second.error === `Not added: old.doc. ${OLD_WORD}` && isDeepStrictEqual(again, second);
    check('live', toldName, told, JSON.stringify({ second, again }));
    const quietName = "a tab's lines are written with their live region on as they come while it is shown, and with it off when a switch restores them";
    const quiet = own.length > 0 && own.every((write) => write.live !== 'off') && restored.length > 0 && restored.every((write) => write.live === 'off');
    check('live', quietName, quiet, JSON.stringify({ own, restored }));
  });
}

// Holds tab 1's PDF text at the server until the chat pane is shown.
async function liveRefusalWhileChatting(url) {
  await withPage('live', { url, sample: false }, async (page) => {
    const release = await holdExtraction(page);
    await page.setInputFiles('#file-input', [documents['01_memo.pdf'], oldWord()]);
    release();
    await becomes(page, () => document.getElementById('run-tab-1').textContent === '1 new · 1 file', null, LIVE_RUN_MS);
    await page.click('#new-tab-btn');
    await showRunTab(page, 0);
    const lines = await filesLines(page);
    const expected = { said: '1 file added, in place of the 5 sample memos.', error: `Not added: old.doc. ${OLD_WORD}` };
    const name = 'a refusal that lands while the chat pane is shown is in the files pane once it is opened, after a switch to another tab and back';
    check('live', name, isDeepStrictEqual(lines, expected), JSON.stringify(lines));
  });
}

// Holds a scan's text at the server while two batches queue behind it; the files list is taken out of the page, so
// the last batch, which reads a file, fails to be added.
async function liveRefusalsThenFailure(url) {
  await withPage('live', { url, sample: false, expectedErrors: ["couldn't add files"] }, async (page) => {
    const release = await holdExtraction(page);
    await page.click('#tab-files');
    await page.setInputFiles('#file-input', documents['02_scan.pdf']);
    await page.setInputFiles('#file-input', oldWord());
    await page.setInputFiles('#file-input', memo('first.txt', 'A first memo.\n'));
    await page.evaluate(() => document.getElementById('file-list').remove());
    release();
    const expected = `Not added: 02_scan.pdf. ${NO_TEXT_LAYER} Not added: old.doc. ${OLD_WORD} Couldn't add those files.`;
    const kept = await becomes(page, (text) => document.getElementById('files-error').textContent === text, expected, LIVE_RUN_MS);
    check('live', "queued batches' refusals and a later failure to add all stay in the error line", kept, await filesError(page));
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
    const problemName = 'a connection lost at load stays in the status, through typing and a 500, until a call to r3con app succeeds';
    check('live', problemName, atLoad && typing && unwell && cleared, `typing ${typing}; 500 ${unwell}; then ${status}`);
  });
}

// Loses the connection at load, so the status holds the error, and then removes a file.
async function liveRemoveKeepsProblem(url) {
  const before = (page) => page.route('**/api/keys', (route) => route.abort());
  await withPage('live', { url, sample: false, before, expectedErrors: [ABORTED_LOG] }, async (page) => {
    const atLoad = await becomes(page, statusIs, LOST_CONNECTION);
    check('live', 'a connection lost at load shows in the status', atLoad, await page.textContent('#run-status'));
    await page.click('#tab-files');
    await page.locator('#file-list .remove').first().click();
    const status = await page.textContent('#run-status');
    check('live', 'removing a file before a run leaves a connection error in the status', status === LOST_CONNECTION, status);
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
    const unstarted = await page.evaluate(logOf);
    const endedName = 'a run that fails before it starts ends its log on failed, not on starting, and says why below it';
    check('live', endedName, unstarted.folded === 'failed' && unstarted.mark === 'err' && isDeepStrictEqual(unstarted.after, [STALE_TOKEN]), `${unstarted.folded}; ${unstarted.after.join(' | ')}`);
  });
}

// Stops the server it is given: its own.
async function liveGoingThenLost(url, server) {
  await withPage('live', { url, sample: false, expectedErrors: [CUT_STREAM_LOG] }, async (page) => {
    const posts = postsOf(page);
    await page.fill('#question', `Which site logged the most incidents? ${HANGS}`);
    await page.click('#run-btn');
    const held = await becomes(page, roundOneWorking, null, LIVE_RUN_MS);
    const question = 'Which site logged the fewest?';
    await page.fill('#question', question);
    await page.click('#run-btn');
    const asked = [await page.textContent('#run-status'), await page.inputValue('#question')];
    const askName = 'while a run is going, a question stays in the box, unsent, and the status says to ask once it has finished';
    check('live', askName, held && isDeepStrictEqual(asked, [ASK_WHEN_DONE, question]) && posts.length === 1, asked[0]);
    await page.press('#question', 'Control+Enter');
    const ended = await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const second = [await shownTab(page), await count(page, '.node'), await count(page, '.node.working')];
    await page.locator('.run-tab').first().click();
    const status = await page.textContent('#run-status');
    const first = [await count(page, '.node.r1.working'), await count(page, '.node.answer')];
    const apart = ended && isDeepStrictEqual(second, [1, SAMPLE_COUNTS.nodes, 0]) && status.startsWith('running · ') && first[0] > 0 && first[1] === 0;
    const runName = "while a run is going, a new run opens a tab of its own, and each run's events land in its own tab";
    check('live', runName, apart && isDeepStrictEqual(posts, ['/api/runs', '/api/runs']), `${posts.join(' ')}; second ${second}; first ${status}, ${first}`);

    await page.locator('.run-tab').nth(1).click();
    await send(page, `Which site is quietest? ${HANGS}`);
    const asking = await becomes(page, lastTurnSays, [`asking · ${DEFAULT_MODEL}`]);
    await page.locator('.run-tab').first().click();
    await server.stop();
    const lost = await becomes(page, statusIs, LOST_CONNECTION, LIVE_RUN_MS);
    const lostName = 'a run whose server goes away mid-stream says the connection was lost';
    check('live', lostName, lost, await page.textContent('#run-status'));
    await page.locator('.run-tab').nth(1).click();
    const cut = await page.evaluate(logOf);
    const cutName = "an ask whose stream ends with no last event ends its log on failed, not spinning on asking, and says why below it";
    check('live', cutName, asking && cut.folded === 'failed' && cut.mark === 'err' && isDeepStrictEqual(cut.after, [LOST_CONNECTION.slice(2)]), `${cut.folded}; ${cut.after.join(' | ')}`);
  });
}

const isCancel = (request) => request.method() === 'POST' && request.url().endsWith('/cancel');
const cancelsIn = (posts) => posts.filter((path) => path.endsWith('/cancel'));
const startedJob = (page, path) => page.waitForResponse((response) => response.request().method() === 'POST' && new URL(response.url()).pathname.endsWith(path));

// Stalls both tabs' runs on its server until it releases them, after the second is stopped.
async function liveStop(url, server) {
  await withPage('live', { url, sample: false }, async (page) => {
    const posts = postsOf(page);
    const absentBefore = !(await page.evaluate(composer)).stop;
    await send(page, `Which site logged the most incidents? ${STALLS}`);
    await becomes(page, roundOneWorking, null, LIVE_RUN_MS);
    const started = startedJob(page, '/api/runs');
    await send(page, `Which memo is the registry? ${STALLS}`, 'Control+Enter');
    const runId = (await (await started).json()).run_id;
    await becomes(page, roundOneWorking, null, LIVE_RUN_MS);
    check('live', 'stop is absent before a run, and offered while one goes', absentBefore && (await page.evaluate(composer)).stop);
    const sent = page.waitForRequest(isCancel);
    await pressStop(page);
    await (await sent).response();
    const stoppingSaid = await page.textContent('#run-status');
    const disabled = await page.$eval('#stop-btn', (button) => button.disabled);
    await pressStop(page);
    check('live', 'while a stopped run finishes its call in flight, the status says it is stopping, and stop is disabled', stoppingSaid === STOPPING && disabled, stoppingSaid);
    await page.click('#tab-files');
    const pane = await page.evaluate(filesPane);
    await page.click('#new-tab-btn');
    await page.locator('.run-tab').nth(1).click();
    const back = await page.textContent('#run-status');
    await page.click('#tab-chat');
    check('live', "a stopping run's files are locked, and its stopping line is back once a new tab has been shown and left", pane.removes === 0 && back === STOPPING, back);

    server.release();
    const ended = await becomes(page, statusIs, STOPPED_IN_RELEVANCE, LIVE_RUN_MS);
    const stopped = await page.evaluate(nodesStopped);
    const closable = (await page.evaluate(tabRow))[1].closable;
    const after = await page.evaluate(composer);
    const cancels = cancelsIn(posts);
    const runName = "stop sends one cancel, for the shown tab's run, even when pressed twice; the run ends stopped where it was, its × enabled, and stop gives way to run";
    check('live', runName, isDeepStrictEqual(cancels, [`/api/runs/${runId}/cancel`]) && ended && stopped && closable && !after.stop && after.primary === 'run ⏎', `${cancels.join(' ')}; ${JSON.stringify(after)}`);

    const firstEnded = await becomes(page, tabEnded, 0, LIVE_RUN_MS);
    await page.locator('.run-tab').first().click();
    const first = { nodes: await count(page, '.node'), working: await count(page, '.node.working'), stop: (await page.evaluate(composer)).stop };
    const status = await page.textContent('#run-status');
    const goesOn = firstEnded && isDeepStrictEqual(first, { nodes: SAMPLE_COUNTS.nodes, working: 0, stop: false }) && LIVE_DONE.test(status);
    check('live', "stop in one tab leaves another tab's run going: its events still land there, and once it is done stop is gone", goesOn, `${JSON.stringify(first)}; ${status}`);
  });
}

// Holds the run's POST back until stop has been pressed, so the cancel has to wait for the run's id.
async function liveStopEarly(url, server) {
  let letThrough;
  const posted = new Promise((resolve) => (letThrough = resolve));
  const before = (page) => page.route('**/api/runs', async (route) => {
    await posted;
    await route.continue();
  });
  await withPage('live', { url, sample: false, before }, async (page) => {
    const posts = postsOf(page);
    await send(page, `Which site logged the most incidents? ${STALLS}`);
    await becomes(page, () => !document.getElementById('stop-btn').hidden);
    await pressStop(page);
    const beforeId = cancelsIn(posts).length;
    const sent = page.waitForRequest(isCancel, { timeout: SETTLE_TIMEOUT_MS }).catch(() => null);
    letThrough();
    await sent;
    server.release();
    const ended = await becomes(page, statusIs, STOPPED_IN_RELEVANCE, LIVE_RUN_MS);
    const name = "stop pressed before the run's id has come back sends its cancel once it does, and the run ends stopped";
    check('live', name, beforeId === 0 && cancelsIn(posts).length === 1 && ended, await page.textContent('#run-status'));
  });
}

// Stalls two asks' calls until it releases them; their replies then miss their schema, so r3con would send again.
async function liveStopAsk(url, server) {
  await withPage('live', { url, sample: false }, async (page) => {
    const posts = postsOf(page);
    await page.click('#run-btn');
    await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const ask = async (question) => {
      const started = startedJob(page, '/asks');
      await send(page, `${question} ${STALLS} ${OFF_SCHEMA}`);
      return (await (await started).json()).ask_id;
    };
    const first = await ask('Which site is quietest?');
    const sentFirst = page.waitForRequest(isCancel);
    await pressStop(page);
    await sentFirst;
    const second = await ask('Which site is loudest?');
    const offered = await page.$eval('#stop-btn', (button) => !button.hidden && !button.disabled);
    const sentSecond = page.waitForRequest(isCancel, { timeout: SETTLE_TIMEOUT_MS }).catch(() => null);
    await pressStop(page);
    await sentSecond;
    const spent = await page.$eval('#stop-btn', (button) => button.disabled);
    const cancels = cancelsIn(posts);
    const laterName = 'a later ask can be stopped while an earlier one is still stopping: stop is offered again, and cancels only the later one';
    check('live', laterName, offered && spent && isDeepStrictEqual(cancels, [`/api/asks/${first}/cancel`, `/api/asks/${second}/cancel`]), cancels.join(' '));

    server.release();
    const ended = (await becomes(page, failedNode, 'answer-1', LIVE_RUN_MS)) && (await becomes(page, failedNode, 'answer-2', LIVE_RUN_MS));
    const names = await page.$$eval('.node.answer.failed', (els) => els.map((el) => el.getAttribute('aria-label')));
    const askLog = await page.evaluate(logOf);
    const status = await page.textContent('#run-status');
    const chatSays = askLog.folded === STOPPED_ASK_LINE && askLog.mark === 'err' && askLog.after.length === 0;
    const stoppedAsks = ended && names.length === 2 && names.every((name) => name.endsWith(', stopped')) && chatSays && status === STOPPED_ASK && !(await page.evaluate(composer)).stop;
    const askName = "stop during an ask over a finished tab cancels that ask; its answer node, its log's folded line and the status say it stopped";
    check('live', askName, stoppedAsks, `${names.length} stopped; ${status}`);

    await send(page, 'Which memo is the registry?', 'Control+Enter');
    await becomes(page, newRunDone, LIVE_DONE.source, LIVE_RUN_MS);
    await page.locator('.run-tab').first().click();
    const back = await page.textContent('#run-status');
    check('live', "after a stopped ask, the tab's own run line is back once the tab is shown again", LIVE_DONE.test(back), back);
  });
}

// Its cancel is answered 404, so the run, held at its answer, goes on to the end.
async function liveStopRefused(url) {
  const before = async (page) => {
    await page.addInitScript(holdFrom, 'reasoning_finished');
    await page.route('**/cancel', (route) => route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ error: CANCEL_REFUSED }) }));
  };
  await withPage('live', { url, sample: false, before, expectedErrors: [NOT_FOUND_LOG] }, async (page) => {
    await page.click('#run-btn');
    await becomes(page, answerWorking, null, LIVE_RUN_MS);
    await page.evaluate(recordStatuses);
    await pressStop(page);
    const refusal = `● couldn't stop · ${CANCEL_REFUSED}`;
    const refused = await becomes(page, statusIs, refusal);
    const again = await page.$eval('#stop-btn', (button) => !button.hidden && !button.disabled);
    await page.evaluate(() => (window.release = true));
    const done = await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const said = await page.evaluate(() => window.statuses);
    const goesOn = refused && again && done && !said.slice(said.indexOf(refusal)).includes(STOPPING);
    check('live', 'a refused cancel says so in the status, offers stop again, and the run goes on to the end, no longer stopping', goesOn, said.join(' | '));
  });
}

// Holds the cancel until its run has ended on its own, then refuses it.
async function liveStopLate(url) {
  let refuse;
  const ran = new Promise((resolve) => (refuse = resolve));
  const before = async (page) => {
    await page.addInitScript(holdFrom, 'reasoning_finished');
    await page.route('**/cancel', async (route) => {
      await ran;
      await route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ error: CANCEL_REFUSED }) });
    });
  };
  await withPage('live', { url, sample: false, before, expectedErrors: [NOT_FOUND_LOG] }, async (page) => {
    await page.click('#run-btn');
    await becomes(page, answerWorking, null, LIVE_RUN_MS);
    await pressStop(page);
    await page.evaluate(() => (window.release = true));
    const done = await becomes(page, statusMatches, LIVE_DONE.source, LIVE_RUN_MS);
    const finished = page.waitForEvent('requestfinished', isCancel);
    refuse();
    await finished;
    const overwritten = await becomes(page, (source) => !new RegExp(source).test(document.getElementById('run-status').textContent), LIVE_DONE.source);
    check('live', 'a cancel refused once its run has ended leaves the done status alone', done && !overwritten, await page.textContent('#run-status'));
  });
}

// A copy of the page served from another host must not take a token or call any API.
async function liveElsewhere() {
  const elsewhere = 'http://r3con.example/';
  const calls = [];
  const before = (page) => {
    page.on('request', (request) => request.url().includes('/api/') && calls.push(request.url()));
    return page.route(`${elsewhere}**`, (route) => {
      const name = PAGE_FILES[new URL(route.request().url()).pathname];
      return route.fulfill(name ? { path: fileURLToPath(new URL(name, APP_DIR)) } : { status: 404 });
    });
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
    await introCounts('live', { url, sample: false });
    await liveCutStream(url);
    await liveAnswers(url);
    await liveLog(url);
    await liveAskAfterFailed(url);
    await liveTabs(url);
    await liveWhileGoing(url);
    await liveParts(url);
    await liveFailure(url);
    await liveUnreadable(url);
    await livePdf(url);
    await liveWord(url);
    await liveBatches(url);
    await liveNotText(url);
    await liveStaleToken(url);
    await liveStopRefused(url);
    await liveStopLate(url);
    await liveRemoveKeepsProblem(url);
    await liveTabFiles(url);
    await liveTrash(url);
    await liveZip(url);
    await liveLockedWhileReading(url);
    await liveBatchesPerTab(url);
    await liveRefusalWhileChatting(url);
    await liveRefusalsThenFailure(url);
  });
  const stateful = [liveSession, liveKeySaves, liveLoadProblem, liveAskGoing, liveGoingThenLost, liveStop, liveStopEarly, liveStopAsk];
  for (const body of stateful) await withFakeServer(body);
  await liveElsewhere();
}

try {
  documents = writeDocuments(DOCUMENT_SPECS);
  browser = await require('playwright').chromium.launch();
  const areas = [load, beforeRun, sampleReplay, chatLog, arrange, runView, filesNotText, filesAddFails, removeArea, trashArea, answers, tabs, tabFiles, dropArea, stopArea, settings, popupArea, gestures, scale, live];
  for (const area of areas) await area();
} catch (error) {
  check('setup', firstLine(error), false);
} finally {
  await browser?.close();
}
console.log(failures ? `${failures} failed` : 'all checks passed');
process.exit(failures ? 1 : 0);
