/* Confidence against lift, on rules mined here.
 *
 * The line at lift one is the whole section: everything below it is a rule that
 * makes its own conclusion LESS likely than it already was, and confidence
 * cannot see that line at all, because confidence does not know how common the
 * conclusion is.
 */
import { makeChart, drawAxes, axisLabels, drawGrid } from '../../assets/js/chart.js';
import { apriori, rulesFrom } from './basketkit.js';
import { legend, seriesLabel } from './draw.js';

const CONFS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5];

export function initRulesWidget(data, db) {
  const SUPPORTS = data.groceries.ship_supports;
  let si = SUPPORTS.indexOf(0.01) >= 0 ? SUPPORTS.indexOf(0.01) : 4;
  let ci = CONFS.indexOf(0.1) >= 0 ? CONFS.indexOf(0.1) : 1;
  let sortBy = 'lift';
  let focus = null;

  const chart = makeChart('#rules-chart', {
    width: 640, height: 400, margin: { top: 54, right: 30, bottom: 56, left: 62 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#rules-readout');
  const sSlider = d3.select('#rules-support')
    .attr('min', 0).attr('max', SUPPORTS.length - 1).attr('step', 1).property('value', si);
  const cSlider = d3.select('#rules-conf')
    .attr('min', 0).attr('max', CONFS.length - 1).attr('step', 1).property('value', ci);

  const SORTS = [{ key: 'lift', label: 'sorted by lift' },
    { key: 'confidence', label: 'sorted by confidence' }];
  d3.select('#rules-sorts').selectAll('button').data(SORTS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === sortBy ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      sortBy = d.key;
      d3.select('#rules-sorts').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === sortBy ? '' : ' ghost'}`);
      render();
    });
  d3.select('#rules-trap').on('click', () => {
    focus = 'trap';
    render();
  });

  /* the frequent sets at each threshold, once each */
  const CACHE = new Map();
  function mine(sup) {
    if (!CACHE.has(sup)) {
      const minCount = Math.ceil(sup * db.n);
      const ap = apriori(db, db.names.length, minCount);
      const counts = new Map();
      ap.levels.forEach((lv) => lv.forEach((e) => counts.set(e.key.join(','), e.count)));
      CACHE.set(sup, { ap, counts });
    }
    return CACHE.get(sup);
  }

  const fmt = (r) => `{${r.a.join(', ')}} &rarr; {${r.c.join(', ')}}`;

  function render() {
    si = +sSlider.property('value');
    ci = +cSlider.property('value');
    const sup = SUPPORTS[si];
    const conf = CONFS[ci];
    const { ap, counts } = mine(sup);
    const rules = rulesFrom(counts, db.n, db.names, { minConf: conf });

    const below = rules.filter((r) => r.lift < 1);
    const trap = below.length
      ? below.reduce((a, b) => (b.confidence > a.confidence ? b : a)) : null;
    if (focus === 'trap' && !trap) focus = null;

    const x = d3.scaleLinear().domain([Math.min(conf, ...rules.map((r) => r.confidence)) - 0.02,
      Math.max(0.3, ...rules.map((r) => r.confidence)) + 0.02]).range([0, w]);
    const yMax = Math.max(1.4, ...rules.map((r) => r.lift)) * 1.05;
    const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

    g.selectAll('*').remove();
    drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 5, xFmt: d3.format('.0%') });
    axisLabels(g, w, h, { x: 'confidence', y: 'lift' });

    /* the line that confidence cannot see */
    g.append('rect').attr('x', 0).attr('y', y(1)).attr('width', w).attr('height', h - y(1))
      .attr('fill', 'var(--cosmos)').attr('opacity', 0.07);
    g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(1)).attr('y2', y(1))
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.4).attr('stroke-dasharray', '4 3');
    /* The caption sits at the FLOOR of the shaded band, not just under the
       line: right under the line is where the rules with lift near one live,
       and where the button below points its label. */
    seriesLabel(g, w - 4, h - 8, 'below here the rule makes its conclusion less likely',
      { anchor: 'end', size: 10.5, opacity: 0.8 });

    const rr = d3.scaleSqrt().domain([0, Math.max(...rules.map((r) => r.support), 0.01)])
      .range([1.6, 9]);
    g.selectAll('circle.rule').data(rules).join('circle').attr('class', 'rule')
      .attr('cx', (d) => x(d.confidence)).attr('cy', (d) => y(d.lift))
      .attr('r', (d) => rr(d.support))
      .attr('fill', (d) => (d.lift < 1 ? 'var(--cosmos)' : 'var(--primary)'))
      .attr('opacity', 0.6)
      .attr('stroke', 'var(--paper)').attr('stroke-width', 0.6);

    if (focus === 'trap' && trap) {
      g.append('circle')
        .attr('cx', x(trap.confidence)).attr('cy', y(trap.lift)).attr('r', 13)
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4);
      const left = x(trap.confidence) > 0.6 * w;
      seriesLabel(g, x(trap.confidence) + (left ? -16 : 16), y(trap.lift) - 16,
        `${trap.a.join(', ')} then ${trap.c.join(', ')}`,
        { anchor: left ? 'end' : 'start', size: 10.5, opacity: 0.95 });
    }

    legend(g, [
      { label: `lift above one (${rules.length - below.length})`, shape: 'dot', colour: 'var(--primary)' },
      { label: `lift below one (${below.length})`, shape: 'dot', colour: 'var(--cosmos)' },
      { label: 'size is support', shape: 'dot', colour: 'var(--stone)' },
    ], { x0: 2, y0: -30, gap: 200 });

    d3.select('#rules-support-label').text(`${(100 * sup).toFixed(2)}%`);
    d3.select('#rules-conf-label').text(`${(100 * conf).toFixed(0)}%`);

    const top = rules.slice().sort((a, b) => b[sortBy] - a[sortBy]).slice(0, 6);
    const table = `<table class="rule-table"><thead><tr><th>Rule</th><th>Support</th>`
      + `<th>Confidence</th><th>Lift</th></tr></thead><tbody>`
      + top.map((r) => `<tr><td>${fmt(r)}</td><td>${(100 * r.support).toFixed(2)}%</td>`
        + `<td>${(100 * r.confidence).toFixed(1)}%</td>`
        + `<td${r.lift < 1 ? ' class="bold"' : ''}>${r.lift.toFixed(3)}</td></tr>`).join('')
      + `</tbody></table>`;

    readout.html(
      `<span class="slider-label">minimum support <span class="value">`
      + `${(100 * sup).toFixed(2)}%</span></span> `
      + `<span class="slider-label">minimum confidence <span class="value">`
      + `${(100 * conf).toFixed(0)}%</span></span>. `
      + `${ap.work.frequent.toLocaleString('en-US')} frequent sets over the `
      + `${db.names.length} commonest items, and `
      + `<span class="bold">${rules.length.toLocaleString('en-US')}</span> rules out of them. `
      + (trap
        ? `<span class="bold">${below.length}</span> of those have lift below one. The worst is `
          + `${fmt(trap)} at ${(100 * trap.confidence).toFixed(1)}% confidence and lift `
          + `${trap.lift.toFixed(3)}: it looks like a rule and it is an anti-rule, because `
          + `${trap.c.join(', ')} is in ${(100 * db.counts[trap.cIdx[0]] / db.n).toFixed(2)}% of `
          + `all baskets anyway and the rule gets to `
          + `${(100 * trap.confidence).toFixed(2)}%. Knowing the antecedent made you `
          + `<span class="bold">worse</span> off.`
        : `None of them has lift below one at these thresholds; lower the confidence and they `
          + `appear.`)
      + table
    );
  }

  sSlider.on('input', () => { focus = null; render(); });
  cSlider.on('input', () => { focus = null; render(); });
  render();
  return { render };
}
