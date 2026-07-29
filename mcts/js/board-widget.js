/* A real position, a real search, run here.
 *
 * The exact value of every legal move comes from the file, where the whole game
 * was solved; the visit counts come from a search that runs when the button is
 * pressed. So the reader is not watching a recording: they are watching a search
 * that can, and at small budgets does, get the answer wrong.
 */
import { makeChart } from '../../assets/js/chart.js';
import { makeGame, lcg, search, stateFromGrid } from './gamekit.js';

const BUDGETS = [16, 64, 256, 1024];
const VERDICT = { 1: 'wins', 0: 'draws', '-1': 'loses' };

export function initBoardWidget(data) {
  const M = data.meta;
  const T = data.tree;
  const game = makeGame(M.rows, M.cols, M.connect);
  const state = stateFromGrid(T.grid);
  const readout = document.querySelector('#board-readout');
  const controls = d3.select('#board-controls');
  let sims = 64;
  let seed = 4242;

  const chart = makeChart('#board-chart', {
    width: 640, height: 400, margin: { top: 46, right: 24, bottom: 34, left: 24 },
  });
  const { g, w, h } = chart;
  /* the board is drawn at a fixed size and centred, and everything under it is
     placed from its bottom edge rather than from the height of the drawing */
  const cell = 46;
  const bw = cell * M.cols;
  const left = (w - bw) / 2;
  const boardH = cell * M.rows;
  const barTop = boardH + 38;
  const barH = h - barTop - 30;
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -26)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -8)
    .style('font-size', '11.5px');
  const board = g.append('g');
  const bars = g.append('g');

  /* who is who, next to the board: the colours are the only thing saying it */
  [[T.who, 'to move'], [3 - T.who, 'the other one']].forEach(([p, label], i) => {
    const y = 26 + i * 24;
    g.append('circle').attr('cx', left + bw + 26).attr('cy', y).attr('r', 9)
      .attr('fill', p === T.who ? 'var(--primary)' : 'var(--anchor)');
    g.append('text').attr('class', 'chart-note').attr('x', left + bw + 40).attr('y', y + 4)
      .style('font-size', '10.5px').text(label);
  });

  /* the board itself, drawn once: the position never changes */
  T.grid.forEach((row, r) => row.forEach((v, c) => {
    board.append('rect').attr('x', left + c * cell).attr('y', r * cell)
      .attr('width', cell - 3).attr('height', cell - 3)
      .attr('fill', '#e7eaec').attr('rx', 3);
    if (!v) return;
    board.append('circle').attr('cx', left + c * cell + (cell - 3) / 2)
      .attr('cy', r * cell + (cell - 3) / 2).attr('r', cell * 0.32)
      .attr('fill', v === T.who ? 'var(--primary)' : 'var(--anchor)');
  }));

  function render() {
    const rand = lcg(seed);
    const res = search(game, state, T.who, sims, { rand });
    const total = Object.values(res.counts).reduce((a, b) => a + b, 0) || 1;
    const exact = T.frames[0].exact;
    bars.selectAll('*').remove();
    T.moves.forEach((c) => {
      const share = (res.counts[c] || 0) / total;
      const good = T.best.includes(c);
      bars.append('rect').attr('x', left + c * cell + cell * 0.12)
        .attr('y', barTop + barH * (1 - share)).attr('width', (cell - 3) * 0.76)
        .attr('height', Math.max(barH * share, 1))
        .attr('fill', good ? 'var(--primary)' : '#c9ced0');
      bars.append('text').attr('class', 'chart-note')
        .attr('x', left + c * cell + (cell - 3) / 2).attr('y', barTop + barH + 14)
        .attr('text-anchor', 'middle').style('font-size', '10.5px')
        .text(`${(share * 100).toFixed(0)}%`);
      bars.append('text').attr('class', 'chart-note')
        .attr('x', left + c * cell + (cell - 3) / 2).attr('y', barTop + barH + 26)
        .attr('text-anchor', 'middle').style('font-size', '10px')
        .attr('fill', good ? 'var(--primary)' : '#8e9aa6')
        .text(VERDICT[String(exact[String(c)])] || '');
    });
    bars.append('text').attr('class', 'chart-note').attr('x', left)
      .attr('y', boardH + 24).style('font-size', '10.5px')
      .text('share of the simulations each column got');

    const right = T.best.includes(res.pick);
    title.text(`${sims} simulations, run here, on a position the file solved`);
    sub.text(`it played column ${res.pick}, which is ${right ? 'one of the best moves' : 'not a best move'}`);

    const gen = T.frames.find((f) => f.sims === sims);
    const genTotal = gen ? Object.values(gen.counts).reduce((a, b) => a + b, 0) : 0;
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>column</th>${T.moves.map((c) => `<th>${c}</th>`).join('')}</tr>
        <tr><td>what it is worth</td>
            ${T.moves.map((c) => `<td>${VERDICT[String(exact[String(c)])]}</td>`).join('')}</tr>
        <tr><td>this run, share</td>
            ${T.moves.map((c) => `<td>${(((res.counts[c] || 0) / total) * 100).toFixed(0)}%</td>`).join('')}</tr>
        ${gen ? `<tr><td>the generator's run</td>
            ${T.moves.map((c) => `<td>${((Number(gen.counts[String(c)] || 0) / genTotal) * 100).toFixed(0)}%</td>`).join('')}</tr>` : ''}
      </table>
      <div class="gen-note">
        The bottom two rows are two different searches on the same position, one
        written in Python and one in JavaScript, drawing different random
        numbers. They do not agree pull for pull and they are not supposed to:
        what a search promises is the column, not the count. At ${sims}
        simulations this one plays <span class="value">${res.pick}</span>
        ${right ? 'and is right' : 'and is wrong'}${gen ? `, the generator's plays ${gen.pick} and is ${gen.correct ? 'right' : 'wrong'}` : ''}.
        Press the same button twice: the counts move, and at the smaller budgets
        so does the answer.
      </div>`;
  }

  controls.selectAll('button.budget').data(BUDGETS).join('button')
    .attr('class', (d) => `budget atlas-btn${d === sims ? '' : ' ghost'}`)
    .text((d) => `${d} simulations`)
    .on('click', (_, d) => {
      sims = d;
      controls.selectAll('button.budget')
        .attr('class', (q) => `budget atlas-btn${q === sims ? '' : ' ghost'}`);
      render();
    });
  controls.append('button').attr('class', 'atlas-btn ghost').text('run it again')
    .on('click', () => { seed = (seed * 7 + 13) % 2147483647; render(); });
  render();
  return { render };
}
