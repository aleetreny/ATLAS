/* Entry point.
 *
 * The guards here recompute the standard the whole page is scored against: on
 * the deep pass the browser solves the drawn position exactly, by the same
 * backwards induction the generator used and in a different language, and
 * checks every column's value against the file. The cheap pass checks the rules
 * and that a search finds a win that is one move away, which any correct search
 * must and a search with its signs crossed does not.
 */
import { initBoardWidget } from './board-widget.js';
import { initTreeScrolly } from './scrolly-tree.js';
import { initDialsWidget } from './dials-widget.js';
import { initModelWidget } from './model-widget.js';
import { initProse } from './prose.js';
import { makeGame, lcg, search, solveExact, stateFromGrid } from './gamekit.js';

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
      { left: '$', right: '$', display: false },
    ],
    throwOnError: false,
  });
}

function safely(name, fn) {
  try {
    return fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
    return null;
  }
}

function check(label, ok, detail) {
  if (!ok) console.warn(`[mcts] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const M = data.meta;
  const game = makeGame(M.rows, M.cols, M.connect);
  const state = stateFromGrid(data.tree.grid);

  /* 1. the rules: the position the file drew has the legal moves it claims */
  const legal = game.moves(state);
  check('the legal moves', legal.join(',') === data.tree.moves.join(','),
    `${legal.join(',')} here against ${data.tree.moves.join(',')} in the file`);
  check('the position is alive', game.terminal(state) === null,
    'the drawn position is already over, which makes the whole widget meaningless');

  /* 2. the file is consistent with itself: the best moves are the ones whose
   *    exact value is the value of the position */
  const exact = data.tree.frames[0].exact;
  const top = Math.max(...Object.values(exact));
  const best = Object.keys(exact).filter((k) => exact[k] === top).map(Number);
  check('the best moves', best.join(',') === data.tree.best.join(',')
    && Math.abs(top - data.tree.value) < 1e-9,
  `the file lists ${data.tree.best.join(',')} but its own values point at ${best.join(',')}`);

  /* 3. a search with its signs crossed still counts visits, so the cheap test
   *    is a position with a win one move away: any correct search finds it */
  let found = 0;
  for (let s = 0; s < 3; s++) {
    let win = [[], [], [], [], []].slice(0, M.cols).map(() => []);
    [1, 2, 3].forEach((c) => { win = game.play(win, c, 1); });
    [1, 2].forEach((c) => { win = game.play(win, c, 2); });
    const res = search(game, win, 1, 200, { rand: lcg(11 + s * 7717) });
    if (game.winner(game.play(win, res.pick, 1)) === 1) found += 1;
  }
  check('a win one move away', found === 3, `found in ${found} of 3 runs`);

  /* 4. the same, from the other side: a threat that has to be blocked */
  let blocked = 0;
  for (let s = 0; s < 3; s++) {
    let thr = [[], [], [], [], []].slice(0, M.cols).map(() => []);
    [1, 2, 3].forEach((c) => { thr = game.play(thr, c, 1); });
    [1, 2].forEach((c) => { thr = game.play(thr, c, 2); });
    const res = search(game, thr, 2, 400, { rand: lcg(23 + s * 6151) });
    if (res.pick === 0 || res.pick === 4) blocked += 1;
  }
  check('a threat that must be blocked', blocked === 3, `blocked in ${blocked} of 3 runs`);

  /* 5. the model panel: the error has to grow with the horizon in every row,
   *    or the claim the panel is making is not in its own numbers */
  let broken = 0;
  data.model.rows.forEach((row) => {
    row.horizon.forEach((d, i) => {
      if (i && d.error < row.horizon[i - 1].error) broken += 1;
    });
  });
  check('the horizon', broken === 0, `${broken} points where imagining further hurt less`);

  /* 6. nothing may read undefined or NaN */
  const empty = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        empty.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', empty.length === 0, `undefined or NaN appears in: ${empty.join(', ')}`);

  /* 7. the expensive one: solve the drawn position here, from scratch */
  if (deep) {
    const memo = new Map();
    const value = solveExact(game, state, data.tree.who, memo);
    check('the value of the position', Math.abs(value - data.tree.value) < 1e-9,
      `${value} here against ${data.tree.value} in the file`);
    let worst = null;
    legal.forEach((c) => {
      const v = -solveExact(game, game.play(state, c, data.tree.who), 3 - data.tree.who, memo);
      if (Math.abs(v - exact[String(c)]) > 1e-9) worst = `column ${c}: ${v} here, ${exact[String(c)]} in the file`;
    });
    check('every column solved', worst === null, worst || '');
    console.info(`[mcts] deep check: the position solved here over ${memo.size} nodes, `
      + `value ${value}, and every column agrees with the file`);
  }

  console.info('[mcts] load-time checks complete: rules, the file against itself, '
    + 'a win in one and a block in one, and the horizon');
  return { legal, best };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/mcts.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the board', () => initBoardWidget(data));
  safely('the four steps', () => initTreeScrolly(data));
  safely('the dials', () => initDialsWidget(data));
  safely('the model', () => initModelWidget(data));

  renderMath();

  window.__atlasCheck = (deep = false) => safely('guards', () => runGuards(data, deep));
  const guards = () => window.__atlasCheck(false);
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
