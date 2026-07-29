/* Neural additive models: nine curves, and a prediction that is their sum.
 *
 * The shapes come from a real torch model trained by
 * src/utils/generate_deeptabular_data.py. Each one was centred on its mean over
 * the training set, which is what makes the waterfall below exact rather than
 * merely illustrative: because the model is additive by construction,
 *
 *     F(x) = base + sum_j ( f_j(x_j) - E[f_j] )
 *
 * holds with no error term. The generator asserts it to the cent before
 * exporting. So the bar chart is not a post-hoc attribution method with its own
 * assumptions, the way SHAP or a partial dependence plot would be. It is the
 * arithmetic the model actually performed.
 */
import { makeChart, tooltip } from '../../assets/js/chart.js';

const money = (v) => (v < 0 ? '−' : '') + '$' + Math.abs(Math.round(v)).toLocaleString('en-US');
const signed = (v) => (v >= 0 ? '+' : '−') + '$' + Math.abs(Math.round(v)).toLocaleString('en-US');

export function initNamWidget(data) {
  const shapes = data.shapes;
  const base = data.nam_base;
  const stones = data.examples;
  let stone = 3;
  let sameScale = true;

  const tip = tooltip();

  /* --------------------------------------------------- nine small multiples */
  const PW = 212;
  const PH = 138;
  const PAD = { top: 26, right: 14, bottom: 26, left: 54 };

  /* An svg with a fixed viewBox scales its text along with everything else, so
   * three columns of these on a phone render at about 5px and are simply not
   * readable. Fewer columns gives a bigger scale factor and the font multiplier
   * takes it the rest of the way. The padding has to grow by the same factor or
   * the enlarged axis labels walk straight off the left edge, so every panel
   * dimension is a function of it, and the secondary "spans $X" label steps
   * aside because at two columns there is no room for both it and the feature
   * name on one line. */
  const colsFor = (px) => (px < 760 ? 2 : 3);
  let COLS = colsFor(window.innerWidth);
  const F = () => (COLS === 3 ? 1 : 1.45);
  const padL = () => PAD.left * F();
  const padT = () => PAD.top * F();
  const iw = () => PW - (PAD.left + PAD.right) * F();
  const ih = () => PH - (PAD.top + PAD.bottom) * F();

  const svg = d3
    .select('#nam-shapes')
    .append('svg')
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const globalMax = d3.max(shapes, (s) => d3.max(s.y, Math.abs));

  const panels = svg.selectAll('g.panel').data(shapes).join('g').attr('class', 'panel');

  const title = panels.append('text').attr('class', 'chart-annotation').attr('x', 0).text((d) => d.feature);

  const spanLabel = panels
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('text-anchor', 'end')
    .style('text-transform', 'none')
    .attr('fill', 'var(--primary)');

  const box = panels.append('rect').attr('fill', 'var(--white)').attr('stroke', 'var(--stone)');

  const zeroLine = panels
    .append('line')
    .attr('x1', 0)
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 0.8)
    .attr('stroke-dasharray', '2 3')
    .attr('stroke-opacity', 0.55);

  const curve = panels.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
  const marker = panels.append('circle').attr('r', 4.5).attr('fill', 'var(--cosmos)').attr('stroke', 'var(--white)').attr('stroke-width', 1.4);
  const axTop = panels.append('text').attr('class', 'chart-annotation ax').attr('text-anchor', 'end').style('text-transform', 'none');
  const axBot = panels.append('text').attr('class', 'chart-annotation ax').attr('text-anchor', 'end').style('text-transform', 'none');
  const xLo = panels.append('text').attr('class', 'chart-annotation ax').attr('x', 0).style('text-transform', 'none');
  const xHi = panels.append('text').attr('class', 'chart-annotation ax').attr('text-anchor', 'end').style('text-transform', 'none');

  /* Everything that depends on the column count. drawShapes() then supplies the
   * vertical positions, so the two together fully specify the panel. */
  function layout() {
    const f = F();
    const W = iw();
    const H = ih();
    const rows = Math.ceil(shapes.length / COLS);
    svg.attr('viewBox', `0 0 ${COLS * PW} ${rows * PH + 8}`);
    panels.attr('transform', (d, i) => `translate(${(i % COLS) * PW + padL()},${Math.floor(i / COLS) * PH + padT()})`);

    title.style('font-size', `${0.6 * f}rem`).attr('y', -10 * f);
    spanLabel
      .style('font-size', `${0.58 * f}rem`)
      .attr('x', W)
      .attr('y', -10 * f)
      .attr('display', COLS === 3 ? null : 'none');
    hitBox.attr('width', W).attr('height', H);
    box.attr('width', W).attr('height', H);
    zeroLine.attr('x2', W);
    panels.selectAll('text.ax').style('font-size', `${0.55 * f}rem`).attr('x', -6 * f);
    xLo.attr('x', 0).attr('y', H + 14 * f);
    xHi.attr('x', W).attr('y', H + 14 * f);
  }

  let lastCols = COLS;
  window.addEventListener('resize', () => {
    const next = colsFor(window.innerWidth);
    if (next === lastCols) return;
    lastCols = next;
    COLS = next;
    layout();
    drawShapes();
  });

  const hitBox = panels
    .append('rect')
    .attr('fill', 'transparent')
    .on('mousemove', function (event, d) {
      const j = shapes.indexOf(d);
      tip.show(
        `<span style="color:var(--smile)">${d.feature}</span><br/>` +
          `this stone: ${fmtX(stones[stone].values[j])}<br/>` +
          `contribution ${signed(stones[stone].contrib[j])}<br/>` +
          `span, 1st to 99th pct ${money(d.range)}`,
        event
      );
    })
    .on('mouseleave', () => tip.hide());

  const fmtK = (v) => (v >= 1000 ? '$' + (Math.round(v / 100) / 10).toFixed(1) + 'k' : '$' + Math.round(v));
  const fmtX = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2));
  /* Shared scale is the default because it carries the finding: on the same
   * axis, carat towers over everything and table% is a flat line. Own-scale
   * shows each curve's shape at the cost of hiding how little most of them
   * matter, which is exactly the mistake the toggle exists to make visible. */
  const capOf = (d) => (sameScale ? globalMax : Math.max(1, d3.max(d.y, Math.abs)));
  const scaleY = (d) => d3.scaleLinear().domain([-capOf(d), capOf(d)]).range([ih(), 0]);
  const scaleX = (d) => d3.scaleLinear().domain([d.lo, d.hi]).range([0, iw()]);

  function drawShapes() {
    const line = d3.line();
    curve.attr('d', (d) => {
      const sx = scaleX(d);
      const sy = scaleY(d);
      return line.x((v, i) => sx(d.x[i])).y((v) => sy(v))(d.y);
    });
    zeroLine.attr('y1', (d) => scaleY(d)(0)).attr('y2', (d) => scaleY(d)(0));
    marker
      .attr('cx', (d, j) => scaleX(d)(clampValue(d, j)))
      .attr('cy', (d, j) => scaleY(d)(shapeAt(d, j)));
    axTop.attr('y', 4 * F()).text((d) => '+' + fmtK(capOf(d)));
    axBot.attr('y', ih() + 3 * F()).text((d) => '−' + fmtK(capOf(d)));
    xLo.text((d) => fmtX(d.lo));
    xHi.text((d) => fmtX(d.hi));
    spanLabel.text((d) => 'spans ' + fmtK(d.range));
  }

  const clampValue = (d, j) => Math.max(d.lo, Math.min(d.hi, stones[stone].values[j]));
  const shapeAt = (d, j) => {
    const i = Math.round(((clampValue(d, j) - d.lo) / (d.hi - d.lo)) * (d.y.length - 1));
    return d.y[i];
  };

  /* ------------------------------------------------------------- waterfall */
  const { g: wg, w: ww, h: wh } = makeChart('#nam-waterfall', {
    width: 640,
    height: 300,
    margin: { top: 26, right: 118, bottom: 40, left: 96 },
  });

  function drawWaterfall() {
    const s = stones[stone];
    const order = d3
      .range(shapes.length)
      .sort((a, b) => Math.abs(s.contrib[b]) - Math.abs(s.contrib[a]));

    let running = base;
    const bars = [{ label: 'average stone', from: 0, to: base, kind: 'base' }];
    for (const j of order) {
      const next = running + s.contrib[j];
      bars.push({ label: shapes[j].feature, from: running, to: next, kind: s.contrib[j] >= 0 ? 'up' : 'down', delta: s.contrib[j] });
      running = next;
    }
    /* The running total and the exported prediction agree to well under a
     * dollar, but they round differently, and a page arguing that this sum is
     * exact cannot afford to print $5,642 in the bar and $5,641 in the text.
     * One source of truth: the model's own output. */
    bars.push({ label: 'prediction', from: 0, to: s.pred, kind: 'total' });

    const lo = Math.min(0, d3.min(bars, (b) => Math.min(b.from, b.to)));
    const hi = Math.max(d3.max(bars, (b) => Math.max(b.from, b.to)), s.price) * 1.06;
    const x = d3.scaleLinear().domain([lo, hi]).range([0, ww]);
    const y = d3.scaleBand().domain(bars.map((b) => b.label)).range([0, wh]).padding(0.22);

    wg.selectAll('g.axis.x')
      .data([0])
      .join((e) => e.append('g').attr('class', 'axis x'))
      .attr('transform', `translate(0,${wh})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat((v) => '$' + d3.format(',')(v)).tickSizeOuter(0));
    wg.selectAll('g.axis.y')
      .data([0])
      .join((e) => e.append('g').attr('class', 'axis y'))
      .call(d3.axisLeft(y).tickSizeOuter(0));

    const colour = (b) =>
      b.kind === 'base' ? 'var(--stone)' : b.kind === 'total' ? 'var(--primary)' : b.kind === 'up' ? 'var(--seablue)' : 'var(--cosmos)';

    wg.selectAll('rect.bar')
      .data(bars, (b) => b.label)
      .join((e) => e.append('rect').attr('class', 'bar').attr('stroke', 'var(--squidink)').attr('stroke-width', 0.8))
      .attr('y', (b) => y(b.label))
      .attr('height', y.bandwidth())
      .attr('fill', colour)
      .transition()
      .duration(320)
      .attr('x', (b) => x(Math.min(b.from, b.to)))
      .attr('width', (b) => Math.max(1, Math.abs(x(b.to) - x(b.from))));

    wg.selectAll('text.blabel')
      .data(bars, (b) => b.label)
      .join((e) => e.append('text').attr('class', 'blabel chart-annotation').style('font-size', '0.72rem').style('text-transform', 'none'))
      .attr('y', (b) => y(b.label) + y.bandwidth() / 2 + 3.5)
      .transition()
      .duration(320)
      .attr('x', (b) => Math.max(x(b.from), x(b.to)) + 6)
      .text((b) => (b.delta === undefined ? money(b.to) : signed(b.delta)));

    /* the actual price, which is the only number on the chart the model did not
     * produce */
    wg.selectAll('line.truth')
      .data([s.price])
      .join((e) => e.append('line').attr('class', 'truth').attr('stroke', 'var(--squidink)').attr('stroke-width', 2).attr('stroke-dasharray', '5 4'))
      .transition()
      .duration(320)
      .attr('x1', (d) => x(d))
      .attr('x2', (d) => x(d))
      .attr('y1', -8)
      .attr('y2', wh);
    wg.selectAll('text.truthlab')
      .data([s.price])
      .join((e) => e.append('text').attr('class', 'truthlab chart-annotation').attr('text-anchor', 'middle').style('font-size', '0.72rem'))
      .attr('y', -14)
      .transition()
      .duration(320)
      .attr('x', (d) => x(d))
      .text((d) => 'actual ' + money(d));
  }

  /* ---------------------------------------------------------------- readout */
  function readout() {
    const s = stones[stone];
    const sum = s.pred - base;
    const order = d3.range(shapes.length).sort((a, b) => Math.abs(s.contrib[b]) - Math.abs(s.contrib[a]));
    const top = order.slice(0, 2).map((j) => `${shapes[j].feature} ${signed(s.contrib[j])}`).join(', ');
    const err = s.pred - s.price;
    document.querySelector('#nam-readout').innerHTML =
      `<div class="slider-label">A <span class="value">${s.values[0]} carat</span> ${s.cut}, colour ${s.colour}, clarity ${s.clarity}. ` +
      `The model starts at the average stone, <span class="value">${money(base)}</span>, and adds nine numbers: ` +
      `${top} lead. They come to <span class="value">${signed(sum)}</span>, so it predicts ` +
      `<span class="value" style="color:var(--primary)">${money(s.pred)}</span> against an actual ` +
      `<span class="value">${money(s.price)}</span>, ${err >= 0 ? 'over' : 'under'} by ${money(Math.abs(err))}.</div>` +
      `<div class="slider-label" style="margin-top:.5rem">Nothing was estimated after the fact here. Because the network is a sum of nine one-feature networks, this <i>is</i> the computation it ran.</div>`;
  }

  function render() {
    layout();
    drawShapes();
    drawWaterfall();
    readout();
    document.querySelectorAll('#nam-stones button').forEach((b, i) => {
      b.classList.toggle('ghost', i !== stone);
    });
  }

  document.querySelectorAll('#nam-stones button').forEach((b, i) => {
    b.addEventListener('click', () => {
      stone = i;
      render();
    });
  });
  const scaleBtn = document.querySelector('#nam-scale');
  scaleBtn.addEventListener('click', () => {
    sameScale = !sameScale;
    scaleBtn.textContent = sameScale ? 'shared scale' : 'own scale each';
    drawShapes();
  });

  render();
}
