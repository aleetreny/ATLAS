/* Splitting a series, three ways, plus the control that isolates the cause.
 *
 * The second bar is the one that turns an anecdote into a measurement. Shuffle
 * the rows of the same table and split at random again: same rows, same
 * features, same model, same number of folds, and the only thing removed is the
 * order. If the optimism survives the shuffle it was never about time.
 *
 * The second view is the failure that survives forward chaining, which almost
 * nobody guards against: with a rolling mean of width w in the features, the
 * rows on either side of a cut share observations, so the split leaks even
 * though no future row is ever used to predict a past one. The cure is a gap,
 * and the width of the gap is measurable rather than folkloric.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initTimeWidget(data) {
  const T = data.time;
  const buttons = document.querySelector('#time-buttons');
  const readout = document.querySelector('#time-readout');
  const st = { view: 'protocols' };

  const chart = makeChart('#time-chart', {
    width: 640, height: 380, margin: { top: 52, right: 34, bottom: 106, left: 86 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['protocols', 'four ways to split'],
    ['embargo', 'the gap that fixes the overlap'],
    ['horizon', 'how far ahead you are asking']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    if (st.view === 'protocols') {
      const rows = T.rows;
      const x = d3.scaleBand().domain(rows.map((r) => r.arm)).range([0, w]).padding(0.32);
      /* the reference line is more than twice the tallest bar, so a domain built
         from the bars alone drew it 272 units above the top of the plot. It goes
         in the domain: the bars stay legible, and the thing the line is for,
         that all four protocols sit well under the series' own month to month
         step, is the comparison the reader cannot make from the numbers alone */
      const y = d3.scaleLinear()
        .domain([0, Math.max(d3.max(rows.map((r) => r.mae)) * 1.2, T.scale * 1.12)])
        .range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
      drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.3f') });
      axisLabels(g, w, h, { x: '', y: 'error, parts per million' });
      rows.forEach((r, i) => {
        const honest = r.arm.startsWith('forward');
        layer.append('rect').attr('x', x(r.arm)).attr('width', x.bandwidth())
          .attr('y', y(r.mae)).attr('height', h - y(r.mae))
          .attr('fill', honest ? 'var(--primary)' : 'var(--cosmos)')
          .attr('opacity', i === 1 ? 0.55 : 1);
        layer.append('text').attr('class', 'chart-note')
          .attr('x', x(r.arm) + x.bandwidth() / 2).attr('y', y(r.mae) - 8)
          .attr('text-anchor', 'middle').style('font-size', '11.5px')
          .text(r.mae.toFixed(3));
        layer.append('text').attr('class', 'chart-note')
          .attr('transform', `translate(${x(r.arm) + x.bandwidth() / 2},${h + 8}) rotate(-28)`)
          .attr('text-anchor', 'end').style('font-size', '10.5px').text(r.arm);
      });
      layer.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(T.scale)).attr('y2', y(T.scale))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.3).attr('stroke-dasharray', '5 3');
      layer.append('text').attr('class', 'chart-note').attr('x', w - 2).attr('y', y(T.scale) - 6)
        .attr('text-anchor', 'end').style('font-size', '11px')
        .text(`for scale: the average month to month step, ${T.scale.toFixed(3)}`);
    } else if (st.view === 'embargo') {
      const E = T.embargo;
      const x = d3.scaleLinear().domain([0, E[E.length - 1].embargo]).range([0, w]);
      const vals = E.map((r) => r.mae).concat([T.embargo_random]);
      const y = d3.scaleLinear().domain([0, d3.max(vals) * 1.15]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: E.map((r) => r.embargo), yTicks: 5 });
      drawAxes(g, x, y, w, h, {
        xValues: E.map((r) => r.embargo), yTicks: 5, yFmt: d3.format('.3f'),
      });
      axisLabels(g, w, h, { x: 'months dropped either side of the cut', y: 'error, parts per million' });
      layer.append('path').attr('d', d3.line().x((r) => x(r.embargo)).y((r) => y(r.mae))(E))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
      E.forEach((r) => layer.append('circle').attr('cx', x(r.embargo)).attr('cy', y(r.mae))
        .attr('r', 4).attr('fill', 'var(--primary)'));
      layer.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(T.embargo_random)).attr('y2', y(T.embargo_random))
        .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
      layer.append('text').attr('class', 'chart-note').attr('x', 2)
        .attr('y', y(T.embargo_random) - 6).style('font-size', '11px').attr('fill', 'var(--cosmos)')
        .text(`splitting at random with the same features: ${T.embargo_random.toFixed(3)}`);
      layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -34)
        .style('font-size', '11px')
        .text(`a ${T.ma} month rolling mean is in the features, so rows near a cut overlap`);
    } else {
      const H = T.horizon;
      const x = d3.scaleLog().domain([H[0].horizon * 0.8, H[H.length - 1].horizon * 1.3])
        .range([0, w]);
      const vals = H.flatMap((r) => [r.random, r.forward]);
      const y = d3.scaleLog().domain([d3.min(vals) * 0.7, d3.max(vals) * 1.4]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: H.map((r) => r.horizon), yTicks: 5 });
      drawAxes(g, x, y, w, h, {
        xValues: H.map((r) => r.horizon), yTicks: 5,
        xFmt: (v) => String(v), yFmt: d3.format('.2f'),
      });
      axisLabels(g, w, h, { x: 'months ahead the model is asked to predict', y: 'error, parts per million' });
      [['random', 'var(--cosmos)', 'split at random'],
        ['forward', 'var(--primary)', 'forward chaining']].forEach(([key, colour, name], i) => {
        layer.append('path').attr('d', d3.line().x((r) => x(r.horizon)).y((r) => y(r[key]))(H))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2);
        H.forEach((r) => layer.append('circle').attr('cx', x(r.horizon)).attr('cy', y(r[key]))
          .attr('r', 4).attr('fill', colour));
        layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -34 + i * 20)
          .style('font-size', '11px').attr('fill', colour).text(name);
      });
    }

    const rows = T.rows;
    const rand = rows[0];
    const shuf = rows[1];
    const fwd = rows[2];
    const E = T.embargo;
    const noGap = E[0];
    const bigGap = E[E.length - 1];
    readout.innerHTML = `<div class="gen-note">The series is the ${T.n} monthly carbon `
      + `dioxide readings the <a href="../arima/">forecasting article</a> published, asserted `
      + `by digest before anything ran, with ${T.lags} lags as features. Splitting at random `
      + `reports ${rand.mae.toFixed(3)} parts per million; forward chaining reports `
      + `${fwd.mae.toFixed(3)}, which is `
      + `${(fwd.mae / rand.mae).toFixed(1)} times larger. The control says the difference is `
      + `the order and nothing else: shuffle the same rows and split at random again and it `
      + `is ${shuf.mae.toFixed(3)}, back within `
      + `${Math.abs(shuf.mae - rand.mae).toFixed(3)} of the optimistic figure. `
      + `And the failure forward chaining does not fix: add a ${T.ma} month rolling mean to `
      + `the features and rows on either side of a cut share observations, so forward `
      + `chaining reports ${noGap.mae.toFixed(3)} with no gap and `
      + `${bigGap.mae.toFixed(3)} once ${bigGap.embargo} months are dropped either side, `
      + `a factor of ${(bigGap.mae / noGap.mae).toFixed(1)}. `
      + (() => {
        /* the sweep is not monotone and the reason is worth a sentence: past a
           point the gap is deleting training rows faster than it is deleting
           leakage, so the curve comes back down and a reader who only saw the
           two ends would take the wrong lesson from it */
        const peak = E.reduce((a, b) => (b.mae > a.mae ? b : a));
        return peak.embargo !== bigGap.embargo
          ? `The sweep is not a ramp: the pessimism peaks at ${peak.embargo} months with `
            + `${peak.mae.toFixed(3)} and then falls back, because past that point the gap `
            + `is throwing away more training rows than leakage. The honest reading is that `
            + `the gap has to be at least as wide as the feature, ${T.ma} months here, and `
            + `that wider is not automatically safer.`
          : `The sweep rises all the way, so on this series the widest gap tested is also `
            + `the most pessimistic one.`;
      })()
      + `</div>`;
  }

  render();
  return { render, state: st };
}
