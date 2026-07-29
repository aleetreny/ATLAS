/* The three things you can turn, each scored against the solved game.
 *
 * Every panel is drawn against the same floor: the same configuration rerun
 * with different search seeds moves on its own, and a difference smaller than
 * that is not a difference. That band is why one of these three panels ends in
 * "nothing happened" instead of a recommendation.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const n0 = (v) => Number(v).toLocaleString('en-US');

const VIEWS = [
  { key: 'sims', label: 'how long it thinks' },
  { key: 'c', label: 'the exploration constant' },
  { key: 'vs', label: 'against minimax' },
];

export function initDialsWidget(data) {
  const A = data.accuracy;
  const C = data.constant;
  const NZ = data.noise;
  const V = data.versus;
  const readout = document.querySelector('#dials-readout');
  const controls = d3.select('#dials-controls');
  let view = 'sims';

  const chart = makeChart('#dials-chart', {
    width: 640, height: 360, margin: { top: 50, right: 128, bottom: 56, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const plot = g.append('g');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -32)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -13)
    .style('font-size', '11.5px');

  const floor = NZ.rows.find((d) => d.sims === 256 && !d.smart);
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  /* a gap between two shares is points, not percent: "24.7% more" reads as a
     ratio and it is not one */
  const pts = (v) => `${(v * 100).toFixed(1)} points`;
  /* a legend glyph has to look like the mark it stands for: the band is a
     shaded rectangle, so its swatch is one too, not a line */
  const legend = (items) => {
    key.selectAll('*').remove();
    items.forEach(([col, lines, dash], i) => {
      if (dash === 'band') {
        key.append('rect').attr('x', 0).attr('y', i * 42).attr('width', 16).attr('height', 12)
          .attr('fill', col).attr('fill-opacity', 0.16);
      } else {
        key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 42 + 6)
          .attr('y2', i * 42 + 6).attr('stroke', col).attr('stroke-width', 2.6)
          .attr('stroke-dasharray', dash || null);
      }
      lines.forEach((t, j) => {
        key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 42 + 20 + j * 11)
          .style('font-size', '10px').text(t);
      });
    });
  };

  function line(x, y, rows, fx, fy, col, dash) {
    plot.append('path').attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2.6)
      .attr('stroke-dasharray', dash || null)
      .attr('d', d3.line().x((d) => x(fx(d))).y((d) => y(fy(d)))(rows));
    plot.selectAll(null).data(rows).join('circle')
      .attr('cx', (d) => x(fx(d))).attr('cy', (d) => y(fy(d))).attr('r', 3.6).attr('fill', col);
  }

  /* the band a rerun of the same thing covers, so every panel is read against it */
  function band(x, y, centre, spread) {
    plot.append('rect').attr('x', 0).attr('y', y(centre + spread)).attr('width', w)
      .attr('height', Math.abs(y(centre - spread) - y(centre + spread)))
      .attr('fill', '#8e9aa6').attr('fill-opacity', 0.14);
  }

  function renderSims() {
    const sims = [...new Set(A.rows.map((d) => d.sims))];
    const x = d3.scaleLog().domain([sims[0] * 0.8, sims[sims.length - 1] * 1.25]).range([0, w]);
    const y = d3.scaleLinear().domain([0.35, 0.85]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: sims, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: sims, yTicks: 5, xFmt: d3.format(',d'),
      yFmt: d3.format('.0%') });
    axisLabels(g, w, h, { x: 'simulations before each move',
      y: 'best moves found', leftBudget: margin.left });
    NZ.rows.filter((d) => !d.smart).forEach((d) => {
      plot.append('rect').attr('x', x(d.sims * 0.86)).attr('y', y(d.mean + d.spread))
        .attr('width', x(d.sims * 1.16) - x(d.sims * 0.86))
        .attr('height', Math.abs(y(d.mean - d.spread) - y(d.mean + d.spread)))
        .attr('fill', '#8e9aa6').attr('fill-opacity', 0.16);
    });
    line(x, y, A.rows.filter((d) => !d.smart), (d) => d.sims, (d) => d.share, 'var(--primary)');
    line(x, y, A.rows.filter((d) => d.smart), (d) => d.sims, (d) => d.share, 'var(--anchor)', '5 4');
    legend([['var(--primary)', ['moves at random', 'in the play out']],
      ['var(--anchor)', ['take a win,', 'block a loss'], '5 4'],
      ['#8e9aa6', ['the same thing', 'rerun on other', 'seeds'], 'band']]);
    title.text(`${A.positions} positions, and the file knows the answer to every one`);
    sub.text(`the grey bands are one configuration rerun on ${NZ.rows[0].seeds} search seeds`);

    const plain = A.rows.filter((d) => !d.smart);
    const lo = plain[0];
    const hi = plain[plain.length - 1];
    /* the two ends of the single seed sweep, read rather than remembered: where
       the hint looks worst and where it looks best */
    const gaps = plain.map((d) => ({ sims: d.sims,
      gap: A.rows.find((q) => q.smart && q.sims === d.sims).share - d.share }));
    const worstGap = gaps.reduce((a, b) => (b.gap < a.gap ? b : a), gaps[0]);
    const bestGap = gaps.reduce((a, b) => (b.gap > a.gap ? b : a), gaps[0]);
    const rows = NZ.rows;
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>simulations</th>${[...new Set(rows.map((d) => d.sims))].map((s) => `<th>${s}</th>`).join('')}</tr>
        <tr><td>at random</td>${rows.filter((d) => !d.smart).map((d) => `<td>${pct(d.mean)}</td>`).join('')}</tr>
        <tr><td>with the hint</td>${rows.filter((d) => d.smart).map((d) => `<td>${pct(d.mean)}</td>`).join('')}</tr>
        <tr><td>spread, at random</td>${rows.filter((d) => !d.smart).map((d) => `<td>${pct(d.spread)}</td>`).join('')}</tr>
        <tr><td>spread, with the hint</td>${rows.filter((d) => d.smart).map((d) => `<td>${pct(d.spread)}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        Thinking longer works, and stops working: from
        <span class="value">${pct(lo.share)}</span> at ${lo.sims} simulations to
        ${pct(hi.share)} at ${n0(hi.sims)}, which is ${hi.sims / lo.sims} times the
        thinking for ${hi.correct - lo.correct} more positions out of ${hi.of}. The hint in the play out is the panel
        that returns nothing: over ${NZ.rows[0].seeds} seeds it is
        ${pts(Math.abs(rows.find((d) => d.sims === floor.sims && d.smart).mean
          - floor.mean))} away from playing at random at ${floor.sims}
        simulations, against a spread of ${pts(floor.spread)}
        between seeds of the same thing. The single sweep above shows it losing
        by ${pts(-worstGap.gap)} at ${worstGap.sims} simulations and winning by
        ${pts(bestGap.gap)} at ${bestGap.sims}, which is what a difference that
        is not there looks like when you measure it once.
      </div>`;
  }

  function renderC() {
    const cs = C.rows.map((d) => d.c);
    const x = d3.scalePoint().domain(cs.map(String)).range([0, w]).padding(0.5);
    const y = d3.scaleLinear().domain([0.6, 0.8]).range([h, 0]);
    drawGrid(g, x, y, w, h, { yTicks: 5 });
    drawAxes(g, x, y, w, h, { yTicks: 5, yFmt: d3.format('.0%') });
    axisLabels(g, w, h, { x: 'how much the formula pays for being untried',
      y: 'best moves found', leftBudget: margin.left });
    band(x, y, floor.mean, floor.spread);
    /* inside the band and on the left, which is the only part of it the curve
       does not cross */
    plot.append('text').attr('class', 'chart-note').attr('x', 4)
      .attr('y', y(floor.mean) + 4).style('font-size', '10.5px')
      .text('one seed either way');
    line(x, y, C.rows, (d) => String(d.c), (d) => d.share, 'var(--primary)');
    const book = C.rows.find((d) => d.c === C.textbook_c);
    plot.append('circle').attr('cx', x(String(book.c))).attr('cy', y(book.share)).attr('r', 7)
      .attr('fill', 'none').attr('stroke', 'var(--anchor)').attr('stroke-width', 2);
    plot.append('text').attr('class', 'chart-note').attr('x', x(String(book.c)))
      .attr('y', y(book.share) + 24).attr('text-anchor', 'middle').style('font-size', '10.5px')
      .attr('fill', 'var(--anchor)').text('what the textbook says');
    legend([['var(--primary)', ['best moves found']],
      ['#8e9aa6', ['what a rerun', 'covers on its own'], 'band']]);
    title.text(`the constant swept at ${C.sims} simulations, ${C.positions} positions`);
    sub.text('the shaded band is the same search rerun with other seeds, not an error bar');

    const best = C.rows.reduce((a, b) => (b.share > a.share ? b : a), C.rows[0]);
    const worst = C.rows.reduce((a, b) => (b.share < a.share ? b : a), C.rows[0]);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>constant</th>${C.rows.map((d) => `<th>${d.c}</th>`).join('')}</tr>
        <tr><td>best moves found</td>
            ${C.rows.map((d) => `<td>${d === best ? '<span class="value">' : ''}${pct(d.share)}${d === best ? '</span>' : ''}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        The sweep moves ${pts(best.share - worst.share)} between its best
        (<span class="value">${best.c}</span>) and its worst (${worst.c}),
        against ${pts(floor.spread)} of movement from changing nothing but the
        seed. So the shape is real and the ranking of neighbours is not: the
        textbook ${C.textbook_c} scores ${pct(book.share)} here, the same as
        turning exploration off entirely, and ${best.c} scores
        ${pct(best.share)}. Note what zero does not mean: with the constant at
        zero the search still tries every child once, because a child with no
        visits has no average to exploit, so pure greed still looks at all five
        columns before it stops looking.
      </div>`;
  }

  function renderVs() {
    const rows = V.minimax.map((d, i) => ({ ...d, tree: V.tree[i] }));
    const x = d3.scaleLog().domain([4, 4000]).range([0, w]);
    const y = d3.scaleLinear().domain([0.35, 0.85]).range([h, 0]);
    const ticks = rows.map((d) => d.nodes);
    drawGrid(g, x, y, w, h, { xValues: ticks, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: ticks, yTicks: 5, xFmt: d3.format(',d'),
      yFmt: d3.format('.0%') });
    axisLabels(g, w, h, { x: 'positions looked at, per move',
      y: 'best moves found', leftBudget: margin.left });
    line(x, y, rows, (d) => d.nodes, (d) => d.tree.share, 'var(--primary)');
    line(x, y, rows, (d) => d.nodes, (d) => d.share, 'var(--anchor)', '5 4');
    legend([['var(--primary)', ['the tree search']],
      ['var(--anchor)', ['minimax to a', 'fixed depth'], '5 4']]);
    title.text(`the same budget spent two ways, on ${V.positions} positions`);
    sub.text('neither of the two knows anything about the game beyond its rules');

    const last = rows[rows.length - 1];
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>positions seen</th>${rows.map((d) => `<th>${d.nodes}</th>`).join('')}</tr>
        <tr><td>minimax, depth</td>${rows.map((d) => `<td>${d.depth}</td>`).join('')}</tr>
        <tr><td>minimax finds</td>${rows.map((d) => `<td>${pct(d.share)}</td>`).join('')}</tr>
        <tr><td>the tree finds</td>${rows.map((d) => `<td>${pct(d.tree.share)}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        Minimax stops improving after depth ${rows[1].depth}: ${pct(rows[1].share)}
        at ${n0(rows[1].nodes)} positions and ${pct(last.share)} at ${n0(last.nodes)},
        which is ${(last.nodes / rows[1].nodes).toFixed(0)} times the work for
        ${last.correct - rows[1].correct} more position out of ${last.of}. That is not a fault in minimax, it is
        the honest version of it: cut off before the end of the game it has
        nothing to say about the leaf it stopped at, and here it says zero,
        because giving it a hand written evaluation would be giving it knowledge
        the tree search does not have. The tree search keeps climbing to
        <span class="value">${pct(last.tree.share)}</span> on the same budget,
        and the reason is the one thing it does that minimax does not: it decides
        where to spend the next position rather than spreading them evenly.
      </div>`;
  }

  function render() {
    g.selectAll('g.axis').remove();
    g.selectAll('g.grid').remove();
    g.selectAll('text.axis-label').remove();
    plot.selectAll('*').remove();
    if (view === 'sims') renderSims();
    else if (view === 'c') renderC();
    else renderVs();
  }

  controls.selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (_, d) => {
      view = d.key;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render };
}
