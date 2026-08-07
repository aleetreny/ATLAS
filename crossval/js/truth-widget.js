/* k against a risk that was computed rather than estimated.
 *
 * Two views because k trades two things against each other and the trade is
 * invisible if only one is drawn. Small k trains each fold on fewer rows, so it
 * measures a worse model than the one you are going to ship: that is the bias,
 * and it is the distance from the dashed line. Large k has every fold sharing
 * almost all its training data with every other, so the fold scores stop being
 * independent: that is the variance, and it is the second view.
 *
 * The second view carries the number this section exists for. The error bar
 * everyone prints is the spread of the fold scores over the root of k, as if
 * the folds were independent samples. They are not, and the ratio between what
 * that claims and what the estimator really does across datasets is measurable
 * here because the datasets can be redrawn.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initTruthWidget(data) {
  const T = data.truth;
  const buttons = document.querySelector('#truth-buttons');
  const readout = document.querySelector('#truth-readout');
  const st = { view: 'bias' };

  const chart = makeChart('#truth-chart', {
    width: 640, height: 380, margin: { top: 58, right: 34, bottom: 58, left: 82 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['bias', 'what it estimates'],
    ['spread', 'how much it moves'], ['repeat', 'repeating the split']]) {
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
    const rows = T.rows;

    if (st.view === 'repeat') {
      const R = T.repeats;
      const x = d3.scaleLog().domain([R[0].repeats * 0.8, R[R.length - 1].repeats * 1.25])
        .range([0, w]);
      const y = d3.scaleLinear().domain([0, d3.max(R.map((r) => r.sd)) * 1.15]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: R.map((r) => r.repeats), yTicks: 5 });
      drawAxes(g, x, y, w, h, {
        xValues: R.map((r) => r.repeats), yTicks: 5,
        xFmt: (v) => String(v), yFmt: d3.format('.4f'),
      });
      axisLabels(g, w, h, { x: 'times the whole split is repeated', y: 'spread of the estimate' });
      layer.append('path').attr('d', d3.line().x((r) => x(r.repeats)).y((r) => y(r.sd))(R))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
      R.forEach((r) => layer.append('circle').attr('cx', x(r.repeats)).attr('cy', y(r.sd))
        .attr('r', 4).attr('fill', 'var(--primary)'));
      /* The floor is what the sweep converges to, solved from its two ends, and
         it is NOT T.true_sd: that is the spread of the true risk, four times
         smaller, and drawing it here made a curve that is already flat look as
         though it had most of its fall still to come. */
      const pv = (R[0].sd ** 2 - R[R.length - 1].sd ** 2) / (1 - 1 / R[R.length - 1].repeats);
      const fl = Math.sqrt(Math.max(R[0].sd ** 2 - pv, 0));
      layer.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(fl)).attr('y2', y(fl))
        .attr('stroke', 'var(--smile)').attr('stroke-width', 1.4).attr('stroke-dasharray', '6 4');
      layer.append('text').attr('class', 'chart-note').attr('x', w - 2)
        .attr('y', y(fl) - 6).attr('text-anchor', 'end')
        .style('font-size', '11px').attr('fill', 'var(--smile)')
        .text(`the floor no amount of resplitting reaches below, ${fl.toFixed(4)}`);
    } else {
      const ks = rows.map((r) => r.k);
      const x = d3.scaleLog().domain([ks[0] * 0.8, ks[ks.length - 1] * 1.3]).range([0, w]);
      const bias = st.view === 'bias';
      const vals = bias ? rows.map((r) => r.mean).concat([T.expected_risk])
        : rows.flatMap((r) => [r.sd_true, r.sd_claimed]);
      const y = d3.scaleLinear()
        .domain([d3.min(vals) - (bias ? 0.006 : 0.001), d3.max(vals) + (bias ? 0.006 : 0.002)])
        .range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: ks, yTicks: 5 });
      drawAxes(g, x, y, w, h, {
        xValues: ks, yTicks: 5, xFmt: (v) => (v === ks[ks.length - 1] ? `${v} (leave one out)` : String(v)),
        yFmt: bias ? d3.format('.3f') : d3.format('.4f'),
      });
      axisLabels(g, w, h, {
        x: 'folds', y: bias ? 'accuracy' : 'how much the estimate moves',
      });

      if (bias) {
        layer.append('line').attr('x1', 0).attr('x2', w)
          .attr('y1', y(T.expected_risk)).attr('y2', y(T.expected_risk))
          .attr('stroke', 'var(--smile)').attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
        layer.append('text').attr('class', 'chart-note').attr('x', w - 2)
          .attr('y', y(T.expected_risk) + 15).attr('text-anchor', 'end')
          .style('font-size', '11px').attr('fill', 'var(--smile)')
          .text(`the risk of fitting on all ${T.n} rows, ${T.expected_risk.toFixed(4)}`);
        layer.append('path').attr('d', d3.line().x((r) => x(r.k)).y((r) => y(r.mean))(rows))
          .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
        rows.forEach((r) => layer.append('circle').attr('cx', x(r.k)).attr('cy', y(r.mean))
          .attr('r', 4.4).attr('fill', 'var(--primary)'));
      } else {
        [['sd_true', 'var(--primary)', 'how much it really moves between datasets', null],
          ['sd_claimed', 'var(--anchor)', 'what the fold spread claims', '5 3']]
          .forEach(([key, colour, name, dash], i) => {
            layer.append('path').attr('d', d3.line().x((r) => x(r.k)).y((r) => y(r[key]))(rows))
              .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
              .attr('stroke-dasharray', dash);
            rows.forEach((r) => layer.append('circle').attr('cx', x(r.k)).attr('cy', y(r[key]))
              .attr('r', 4).attr('fill', colour));
            layer.append('line').attr('x1', 0).attr('x2', 22).attr('y1', -44 + i * 20 - 4)
              .attr('y2', -44 + i * 20 - 4).attr('stroke', colour).attr('stroke-width', 2.4)
              .attr('stroke-dasharray', dash);
            layer.append('text').attr('class', 'chart-note').attr('x', 28).attr('y', -44 + i * 20)
              .style('font-size', '11px').attr('fill', colour).text(name);
          });
      }
    }

    const tbl = rows.map((r) => `<tr><td>${r.k === T.n ? `${r.k} (leave one out)` : r.k}</td>`
      + `<td>${r.n_train}</td><td>${r.mean.toFixed(4)}</td>`
      + `<td>${r.bias >= 0 ? '+' : ''}${r.bias.toFixed(4)}</td>`
      + `<td>${r.sd_true.toFixed(4)}</td><td>${r.sd_claimed.toFixed(4)}</td>`
      + `<td>${r.ratio.toFixed(2)} times</td></tr>`).join('');
    const ten = rows.find((r) => r.k === 10) || rows[rows.length - 2];
    readout.innerHTML = `<table class="gen-table"><thead><tr><th>folds</th>`
      + `<th>rows per fit</th><th>estimate</th><th>bias</th><th>really moves</th>`
      + `<th>claims to move</th><th>understated by</th></tr></thead>`
      + `<tbody>${tbl}</tbody></table>`
      + `<div class="gen-note">The truth here is not a number anyone measured on data: with `
      + `${T.sets} independent datasets of ${T.n} rows and a population of `
      + `${T.pop.toLocaleString('en-US')} fresh rows, the risk of fitting on `
      + `${T.n} rows is ${T.expected_risk.toFixed(4)} and the best any model could do is `
      + `${T.bayes.toFixed(4)}. With ${ten.k} folds the estimate sits `
      + `${ten.bias >= 0 ? '+' : ''}${ten.bias.toFixed(4)} from it, moves `
      + `${ten.sd_true.toFixed(4)} between datasets, and the error bar people print claims `
      + `${ten.sd_claimed.toFixed(4)}: <span class="bold">understated by a factor of `
      + `${ten.ratio.toFixed(2)}</span>.</div>`;
  }

  render();
  return { render, state: st };
}
