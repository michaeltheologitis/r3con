// Checks the run panel in app/web/index.html in headless Chromium: node app/checks/check.mjs
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { isDeepStrictEqual } from 'node:util';

// An ESM import ignores NODE_PATH, where a global Playwright is found; require honours it.
const require = createRequire(import.meta.url);

const PAGE_FILE = fileURLToPath(new URL('../web/index.html', import.meta.url));
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
const TEST_KEY = 'sk-check-not-a-real-key-7f3a';
const TEST_ENDPOINT = 'http://10.0.0.7:8000/v1';
const LISTED_MODEL = 'gemini/gemini-3.8-flash';
const OTHER_MODEL = 'hosted_vllm/qwen3-32b';
const LARGE_RUN_DOCS = 40;
const LARGE_RUN_COUNTS = { nodes: 162, edges: 3440 };

let browser;
let failures = 0;

function check(area, name, pass, detail = '') {
  if (!pass) failures += 1;
  console.log(`${pass ? 'ok  ' : 'FAIL'}  ${area} · ${name}${detail ? `  (${detail})` : ''}`);
}

const firstLine = (error) => String(error?.message ?? error).split('\n')[0];

async function withPage(area, { layout = LAYOUTS.desktop, sample = true }, body) {
  const errors = [];
  let context;
  try {
    context = await browser.newContext({ reducedMotion: 'reduce', ...layout.options });
    context.setDefaultTimeout(ACTION_TIMEOUT_MS);
    const page = await context.newPage();
    page.on('console', (m) => ['error', 'warning'].includes(m.type()) && errors.push(m.text()));
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(PAGE_URL);
    if (sample) await runSample(page);
    await body(page);
  } catch (error) {
    check(area, firstLine(error), false);
  } finally {
    await context?.close();
  }
  check(area, `${layout.name}: no console errors or warnings`, errors.length === 0, errors.slice(0, 3).join(' | '));
}

async function runSample(page) {
  await page.click('#run-btn');
  await page.waitForSelector('#panel.is-lit');
}

// Waits for a condition in the page; false if it never holds.
async function becomes(page, predicate, arg) {
  try {
    await page.waitForFunction(predicate, arg, { timeout: SETTLE_TIMEOUT_MS });
    return true;
  } catch (error) {
    if (error.name !== 'TimeoutError') throw error;
    return false;
  }
}

const nodeSel = (id) => `.node[data-id="${id}"]`;
const count = (page, selector) => page.$$eval(selector, (els) => els.length);
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

async function rowsOf(page, id) {
  await page.keyboard.press('Escape');
  await page.click(nodeSel(id));
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

function lineage(sample, docIds) {
  const { summary_ids, parse_ids } = sample.answer.provenance;
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
  const nodes = new Set(['answer']);
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

function placesHolding(key) {
  const fields = [...document.querySelectorAll('input, textarea')].filter((el) => el.value.includes(key));
  const holding = (store) => Object.keys(store).filter((k) => store.getItem(k).includes(key));
  const stores = [localStorage, sessionStorage].flatMap(holding);
  const markup = document.documentElement.outerHTML.includes(key) ? ['markup'] : [];
  return [...fields.map((el) => `#${el.id}`), ...stores, ...markup];
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
    const ready = `${SAMPLE_DOCS} documents · ready to run`;
    check('before a run', 'the hint says the documents are ready to run', hint === ready, hint);
    const status = await page.textContent('#run-status');
    const named = `${SAMPLE_DOCS} documents ready · ${DEFAULT_MODEL}`;
    check('before a run', 'the status names the default model', status === named, status);
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
    const docIds = await page.$$eval('.node.doc', (els) => els.map((el) => el.dataset.id));
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
    await withPage('settings', { layout, sample: false }, async (page) => {
      await keys(page, layout.name);
      await picker(page, layout.name);
    });
  }
}

async function keys(page, layoutName) {
  const at = (name) => `${layoutName}: ${name}`;
  const states = () => page.evaluate(keyStates);
  const blocked = async () => (await page.textContent('#settings-saved')) === '' && (await states()).length === 0;
  const fresh = await page.evaluate(pickerState);
  const defaultOnly = { value: DEFAULT_MODEL, groups: [], models: [DEFAULT_MODEL, ''], hint: true };
  const freshName = 'with no keys the picker shows the default model, other model… and the hint';
  check('settings', at(freshName), isDeepStrictEqual(fresh, defaultOnly), JSON.stringify(fresh));

  await page.click(nodeSel('doc-0'));
  await page.click('#settings-btn');
  await page.fill('#endpoint-other', 'localhost:8000/v1');
  await page.click('#settings-save');
  check('settings', at('an endpoint without http(s) blocks the save'), await blocked());
  await page.fill('#endpoint-other', '');
  await page.fill('#key-other', TEST_KEY);
  await page.click('#settings-save');
  check('settings', at('an Other key without an endpoint blocks the save'), await blocked());

  await page.fill('#endpoint-other', TEST_ENDPOINT);
  await page.fill('#key-openai', TEST_KEY);
  await page.fill('#key-gemini', TEST_KEY);
  await page.click('#settings-save');
  await becomes(page, () => document.getElementById('settings-saved').textContent === 'saved');
  const set = await states();
  const marked = isDeepStrictEqual(set, ['state-openai', 'state-gemini', 'state-other']);
  check('settings', at('a save marks OpenAI, Google and Other as set'), marked, set.join(', '));
  const keyed = await page.evaluate(pickerState);
  const noAnthropic = !keyed.models.some((model) => model.startsWith('anthropic/'));
  const onlyKeyed = isDeepStrictEqual(keyed.groups, ['OpenAI', 'Google']) && noAnthropic && !keyed.hint;
  check('settings', at('the picker offers only the providers with a key'), onlyKeyed, keyed.groups.join(', '));

  await page.keyboard.press('Escape');
  const settingsOpen = await page.$eval('#settings', (el) => el.matches(':popover-open'));
  check('settings', at('Escape closes the settings but not the popup'), !settingsOpen && Boolean(await popup(page)));
  const holding = (value) => page.evaluate(placesHolding, value);
  const holders = [...(await holding(TEST_KEY)), ...(await holding(TEST_ENDPOINT))];
  const notKept = 'once closed, no field, storage or markup holds a key or the endpoint';
  check('settings', at(notKept), holders.length === 0, holders.join(', '));
  await page.click('#settings-btn');
  const filled = await page.$$eval('#settings input', (els) => els.filter((el) => el.value).map((el) => el.id));
  const reopened = filled.length === 0 && isDeepStrictEqual(await states(), set);
  check('settings', at('reopened, every field is empty and the providers are still set'), reopened, filled.join(', '));
  await page.keyboard.press('Escape');
}

async function picker(page, layoutName) {
  const at = (name) => `${layoutName}: ${name}`;
  const status = () => page.textContent('#run-status');
  const stored = () => page.evaluate(storedEntries);
  await page.selectOption('#model-pick', LISTED_MODEL);
  const named = await status();
  const namesIt = named === `${SAMPLE_DOCS} documents ready · ${LISTED_MODEL}`;
  check('settings', at('the status names the picked model'), namesIt, named);

  await page.selectOption('#model-pick', '');
  await page.click('#run-btn');
  const invalid = await page.$eval('#model-other', (el) => !el.hidden && el.matches(':invalid'));
  const noRun = (await count(page, '.turn')) === 0;
  const kept = await stored();
  const rejected = invalid && noRun && isDeepStrictEqual(kept, { [MODEL_KEY]: LISTED_MODEL });
  const rejectName = 'other model… rejects an empty model and keeps the last choice';
  check('settings', at(rejectName), rejected, JSON.stringify(kept));
  await page.fill('#model-other', OTHER_MODEL);
  const typed = await stored();
  const onlyModel = isDeepStrictEqual(typed, { [MODEL_KEY]: OTHER_MODEL });
  const typedName = 'other model… takes a typed model, and storage holds only that string';
  check('settings', at(typedName), onlyModel, JSON.stringify(typed));
  await page.click('#run-btn');
  const ran = await becomes(page, () => document.querySelector('#panel.is-lit'));
  check('settings', at('a run starts with the typed model'), ran);

  await page.reload();
  const after = await page.evaluate(pickerState);
  const keysGone = after.groups.length === 0 && after.hint && (await page.evaluate(keyStates)).length === 0;
  const survives = after.value === OTHER_MODEL && keysGone && (await status()).endsWith(OTHER_MODEL);
  check('settings', at('after a reload the choice is kept and the keys are not'), survives, JSON.stringify(after));
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

try {
  browser = await require('playwright').chromium.launch();
  const areas = [load, beforeRun, sampleReplay, arrange, files, settings, popupArea, gestures, scale];
  for (const area of areas) await area();
} catch (error) {
  check('setup', firstLine(error), false);
} finally {
  await browser?.close();
}
console.log(failures ? `${failures} failed` : 'all checks passed');
process.exit(failures ? 1 : 0);
