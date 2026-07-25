/* Three models, four metrics, one held-out split, with the cross-validated
 * spread drawn alongside so a single lucky test set cannot carry the argument.
 *
 * The metric buttons name the metric currently shown, which is the site's
 * convention for a toggle, and the readout always states which direction is
 * better, because two of these four are scored upside down relative to the
 * other two and that is a reliable way to misread a chart.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

/* Axis labels are kept short on purpose: the long-form description of what QDA
 * is doing here belongs in the caption, not stretching a left margin that would
 * then swallow the plot. */
const ROWS = [
  { key: 'logistic', label: 'logistic regression', colour: 'var(--primary)' },
  { key: 'naive_bayes', label: 'naive Bayes', colour: 'var(--cosmos)' },
  { key: 'qda', label: 'QDA', colour: 'var(--aqua)' },
];

const METRICS = {
  brier: { label: 'brier score', lower: true, fmt: d3.format('.4f'), btn: '#bench-brier' },
  logloss: { label: 'log loss', lower: true, fmt: d3.format('.3f'), btn: '#bench-logloss' },
  acc: { label: 'accuracy', lower: false, fmt: d3.format('.4f'), btn: '#bench-acc' },
  auc: { label: 'roc auc', lower: false, fmt: d3.format('.4f'), btn: '#bench-auc' },
};

export function initBenchWidget(data) {
  const { g, w, h } = makeChart('#bench-chart', {
    width: 700,
    height: 300,
    margin: { top: 34, right: 64, bottom: 60, left: 132 },
  });

  const y = d3.scaleBand().domain(ROWS.map((r) => r.key)).range([0, h]).padding(0.34);
  const x = d3.scaleLinear().range([0, w]);

  const barsG = g.append('g');
  const cvG = g.append('g');
  const labelG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  g.append('g').attr('class', 'axis y')
    .call(d3.axisLeft(y).tickSize(0))
    .selectAll('text')
    .style('font-family', 'var(--font-mono)')
    .style('font-size', '11px')
    .text((k) => ROWS.find((r) => r.key === k).label);

  const state = { metric: 'brier' };
  const readout = document.querySelector('#bench-readout');

  function render() {
    const m = METRICS[state.metric];
    const vals = ROWS.map((r) => data.benchmark[r.key][state.metric]);
    const cv = ROWS.map((r) => data.cv[r.key][state.metric]);
    const hi = d3.max(vals.concat(cv.map((c) => c.mean + c.sd)));
    const lo = m.lower ? 0 : Math.min(0.85, d3.min(vals.concat(cv.map((c) => c.mean - c.sd))) * 0.995);
    x.domain([lo, hi * 1.08]);

    drawGrid(g, x, y, w, h, { xTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, xFmt: m.fmt });
    axisLabels(g, w, h, { x: `${m.label} on the 171 held-out biopsies (${m.lower ? 'lower' : 'higher'} is better)` });
    g.select('g.axis.y').call(d3.axisLeft(y).tickSize(0))
      .selectAll('text').style('font-family', 'var(--font-mono)').style('font-size', '11px')
      .text((k) => ROWS.find((r) => r.key === k).label);

    barsG.selectAll('rect').data(ROWS, (r) => r.key).join('rect')
      .attr('y', (r) => y(r.key))
      .attr('height', y.bandwidth())
      .attr('fill', (r) => r.colour)
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 0.9)
      .transition('bar').duration(420)
      .attr('x', x(lo))
      .attr('width', (r) => Math.max(1.5, x(data.benchmark[r.key][state.metric]) - x(lo)));

    /* The five-fold interval, drawn on top of the single-split bar. If the bar
     * ends outside its own whiskers the split was lucky and the reader should
     * be able to see that without being told. */
    const wh = cvG.selectAll('g.whisker').data(ROWS, (r) => r.key).join((e) => {
      const grp = e.append('g').attr('class', 'whisker');
      grp.append('line').attr('class', 'bar');
      grp.append('line').attr('class', 'capL');
      grp.append('line').attr('class', 'capR');
      return grp;
    });
    wh.each(function (r, i) {
      const c = cv[i];
      const cy = y(r.key) + y.bandwidth() / 2;
      const a = x(Math.max(lo, c.mean - c.sd));
      const b = x(c.mean + c.sd);
      const s = d3.select(this);
      s.select('line.bar').transition('bar').duration(420)
        .attr('x1', a).attr('x2', b).attr('y1', cy).attr('y2', cy)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6);
      s.select('line.capL').transition('bar').duration(420)
        .attr('x1', a).attr('x2', a).attr('y1', cy - 5).attr('y2', cy + 5)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6);
      s.select('line.capR').transition('bar').duration(420)
        .attr('x1', b).attr('x2', b).attr('y1', cy - 5).attr('y2', cy + 5)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6);
    });

    labelG.selectAll('text').data(ROWS, (r) => r.key).join('text')
      .attr('class', 'chart-annotation')
      .attr('y', (r) => y(r.key) + y.bandwidth() / 2 + 4)
      .style('font-size', '0.66rem')
      .style('text-transform', 'none')
      .attr('fill', 'var(--squidink)')
      .transition('bar').duration(420)
      .attr('x', (r) => x(data.benchmark[r.key][state.metric]) + 8)
      .text((r) => m.fmt(data.benchmark[r.key][state.metric]));

    title.text(`${m.label}: bar is one split, whisker is the five-fold spread`);
    for (const [k, cfg] of Object.entries(METRICS)) {
      document.querySelector(cfg.btn).classList.toggle('ghost', k !== state.metric);
    }

    const lr = data.benchmark.logistic[state.metric];
    const nb = data.benchmark.naive_bayes[state.metric];
    const qda = data.benchmark.qda[state.metric];
    const ratio = m.lower ? nb / lr : lr / nb;
    const recovered = m.lower ? (nb - qda) / (nb - lr) : (qda - nb) / (lr - nb);
    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5">` +
      /* Brier and log loss are ratios of a cost, so a multiple reads naturally.
       * Accuracy and AUC live on 0 to 1 and are quoted in points everywhere
       * else on this page, so a percentage of a percentage would be a third
       * unit for the same quantity. */
      `On <span class="bold">${m.label}</span>, naive Bayes is ` +
      `<span class="value">${m.lower ? ratio.toFixed(1) + '×' : (Math.abs(nb - lr) * 100).toFixed(1) + ' points'}</span> ` +
      `${m.lower ? 'worse than' : 'behind'} logistic regression. ` +
      `Putting the covariances back recovers ` +
      `<span class="value">${(Math.max(0, Math.min(1, recovered)) * 100).toFixed(0)}%</span> of that gap. ` +
      `Same Gaussians, same Bayes' theorem, one assumption dropped.` +
      `</div>`;
  }

  for (const [k, cfg] of Object.entries(METRICS)) {
    document.querySelector(cfg.btn).addEventListener('click', () => {
      state.metric = k;
      render();
    });
  }
  render();
}
