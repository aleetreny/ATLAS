/* The fifteen percent, swept, with the two forces drawn on the same axis.
 *
 * The table underneath has all five rows and is the record. What the table
 * cannot do is show the shape, and the shape is the argument: the bars (how
 * many predictions the objective was graded on) climb without ever turning
 * around, because hiding more text always buys more predictions, and the line
 * (what the frozen representation turned out to know) is under no obligation to
 * follow them.
 *
 * Whether it does follow them is the measurement, and the readout is written
 * with a branch for each answer rather than a sentence for the expected one.
 * The planned paragraph described a line that peaks in the middle; if the run
 * says otherwise, the paragraph says otherwise.
 *
 * The strip above is the same twenty-four words masked at the selected rate,
 * by the generator, at every rate. The five draws share a seed on purpose, so
 * the picks are nested: moving the slider up ADDS placeholders instead of
 * reshuffling them, and a reader can watch the same sentence being eaten
 * rather than five unrelated sentences flashing past.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { seriesLabel, tokenStrip, legend } from '../../assets/js/textdraw.js';

export function initRateWidget(data) {
  const S = data.rate_sweep;
  const X = data.rate_examples;
  if (!S || !S.length) return null;
  let i = S.findIndex((r) => Math.abs(r.rate - 0.15) < 1e-9);
  if (i < 0) i = 0;

  const slider = document.querySelector('#rate-slider');
  const label = document.querySelector('#rate-slider-label');
  if (slider) {
    slider.min = 0;
    slider.max = S.length - 1;
    slider.step = 1;
    slider.value = i;
    slider.addEventListener('input', () => {
      i = +slider.value;
      render();
    });
  }

  const stripChart = makeChart('#rate-strip', {
    width: 640, height: 120, margin: { top: 26, right: 16, bottom: 12, left: 16 },
  });
  const chart = makeChart('#rate-chart', {
    width: 640, height: 320, margin: { top: 44, right: 24, bottom: 52, left: 62 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#rate-readout');

  const x = d3.scalePoint().domain(S.map((r) => r.rate)).range([0, w]).padding(0.6);
  const yProbe = d3.scaleLinear()
    .domain([Math.min(...S.map((r) => r.probe)) - 0.02,
      Math.max(...S.map((r) => r.probe)) + 0.02])
    .range([h, 0]);
  const yPred = d3.scaleLinear()
    .domain([0, Math.max(...S.map((r) => r.supervised)) * 1.18])
    .range([h, 0]);

  function drawStrip() {
    const ex = X && X[i];
    stripChart.g.selectAll('*').remove();
    if (!ex) return;
    const gradedSet = new Set(ex.graded);
    seriesLabel(stripChart.g, 0, -8,
      `the same passage with ${(100 * ex.rate).toFixed(0)}% of it hidden`,
      { size: 11, opacity: 0.75 });
    /* Its own child group, because `tokenStrip` clears whatever group it is
     * handed: drawing it straight into `stripChart.g` silently deleted the
     * caption written on the line above, which no numeric check can see. */
    const body = stripChart.g.append('g');
    const laid = tokenStrip(body, ex.input, stripChart.w, {
      size: 12,
      fill: (t, k) => (gradedSet.has(k) ? 'var(--primary)' : 'transparent'),
      textFill: (t, k) => (gradedSet.has(k) ? 'var(--paper)' : 'var(--squidink)'),
      opacity: (t, k) => (gradedSet.has(k) ? 0.92 : 1),
      title: (t, k) => (gradedSet.has(k) ? 'hidden, and graded on' : null),
    });
    stripChart.svg.attr('viewBox',
      `0 0 ${stripChart.width} ${laid.height + stripChart.margin.top + stripChart.margin.bottom}`);
  }

  function render() {
    drawStrip();
    g.selectAll('*').remove();
    drawGrid(g, x, yProbe, w, h, { xValues: S.map((r) => r.rate), yTicks: 5 });

    const bw = Math.min(46, (w / S.length) * 0.45);
    S.forEach((r, k) => {
      g.append('rect')
        .attr('x', x(r.rate) - bw / 2).attr('y', yPred(r.supervised))
        .attr('width', bw).attr('height', h - yPred(r.supervised))
        .attr('fill', 'var(--secondary)')
        .attr('opacity', k === i ? 0.34 : 0.16);
      g.append('text')
        .attr('x', x(r.rate)).attr('y', yPred(r.supervised) - 5)
        .attr('text-anchor', 'middle')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '10px')
        .style('fill', 'var(--squidink)')
        .style('opacity', k === i ? 0.9 : 0.45)
        .text(`${(r.supervised / 1e6).toFixed(2)}M`);
    });

    const line = d3.line().x((r) => x(r.rate)).y((r) => yProbe(r.probe));
    g.append('path').datum(S).attr('d', line)
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
    S.forEach((r, k) => {
      g.append('circle')
        .attr('cx', x(r.rate)).attr('cy', yProbe(r.probe))
        .attr('r', k === i ? 6.5 : 4)
        .attr('fill', k === i ? 'var(--primary)' : 'var(--paper)')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2);
    });

    drawAxes(g, x, yProbe, w, h, {
      xValues: S.map((r) => r.rate),
      xFmt: (v) => `${(100 * v).toFixed(0)}%`,
      yTicks: 5,
      yFmt: (v) => `${(100 * v).toFixed(0)}%`,
    });
    axisLabels(g, w, h, { x: 'fraction of the tokens hidden', y: 'frozen probe' });
    legend(g, [
      { label: 'what the representation knows', colour: 'var(--primary)', shape: 'line' },
      { label: 'predictions it was graded on', colour: 'var(--secondary)', shape: 'square' },
    ], { x0: 0, y0: -26, gap: 250, cols: 2 });

    const r = S[i];
    const best = S.reduce((p, c) => (c.probe > p.probe ? c : p), S[0]);
    const most = S[S.length - 1];
    readout.html(
      `<div>At <span class="bold">${(100 * r.rate).toFixed(0)}%</span> the objective was graded `
      + `on <span class="value">${r.supervised.toLocaleString('en-US')}</span> predictions, `
      + `finished at a loss of <span class="value">${r.final_loss.toFixed(4)}</span>, and its `
      + `frozen representation scores <span class="value">${(100 * r.probe).toFixed(2)}%</span>`
      + (r.rate === best.rate
        ? ', which is the best of the five.</div>'
        : `, against ${(100 * best.probe).toFixed(2)}% for the best of the five at `
          + `${(100 * best.rate).toFixed(0)}%.</div>`)
      + `<div>The two marks are here to be compared, not to agree. Hiding more always buys more `
      + `supervision: ${(most.supervised / S[0].supervised).toFixed(1)} times as many graded `
      + `predictions at ${(100 * most.rate).toFixed(0)}% as at ${(100 * S[0].rate).toFixed(0)}%, `
      + `with no turning point anywhere in the bars. What each of those predictions is worth is `
      + `the other force, because the context needed to make them is exactly what is being `
      + `destroyed to create them. `
      + (best.rate === most.rate
        ? `Over the range swept here the line follows the bars all the way up, so within this `
          + `budget the supervision is still worth more than the context it costs.`
          + `</div>`
        : `The line parts company with the bars at ${(100 * best.rate).toFixed(0)}%, which is `
          + `where the second force starts winning.</div>`)
    );
    if (label) label.textContent = `${(100 * r.rate).toFixed(0)}%`;
    if (slider) slider.value = i;
  }

  render();
  return { render };
}
