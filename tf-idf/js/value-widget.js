/* What idf rewards, against what actually separates the categories.
 *
 * idf is a function of one number: how many documents hold the term. It never
 * sees a label, because it is computed before there is a task. So the question
 * with an answer is whether that proxy ranks terms the way something that does
 * see the labels would, and the something is the mutual information between
 * "this term occurs" and "the category is this".
 *
 * The line across the cloud is not a fit. It is the exact ceiling: the
 * information the presence of a term can carry is bounded by the entropy of
 * its own presence, H(df/N), which is zero at df = 1 and largest at df = N/2.
 * Drawing it is the difference between "these two disagree here" and "these
 * two cannot agree anywhere", and it is the reason the article does not
 * present the correlation as a surprise.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { seriesLabel, legend, wordBars } from '../../assets/js/textdraw.js';

const VIEWS = [
  { key: 'cloud', label: 'every term' },
  { key: 'idf', label: 'top fifteen by rarity' },
  { key: 'mi', label: 'top fifteen by information' },
];

export function initValueWidget(data) {
  const V = data.term_value;
  let view = 'cloud';

  const chart = makeChart('#value-chart', {
    width: 640, height: 400, margin: { top: 40, right: 24, bottom: 56, left: 66 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#value-readout');

  d3.select('#value-views').selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      view = d.key;
      d3.select('#value-views').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });

  const maxMi = Math.max(V.best_mi, ...V.ceiling.map((c) => c.cap));

  function drawCloud() {
    const x = d3.scaleLinear()
      .domain([1, Math.max(...V.cloud.map((p) => p[0])) * 1.02]).range([0, w]);
    const y = d3.scaleLinear().domain([0, maxMi * 1.05]).range([h, 0]);
    g.append('g').attr('class', 'grid').call((s) => drawGrid(s, x, y, w, h, { xTicks: 6, yTicks: 5 }));
    const axes = g.append('g');
    drawAxes(axes, x, y, w, h, { xTicks: 6, yTicks: 5, yFmt: (v) => v.toFixed(2) });
    axisLabels(axes, w, h, {
      x: 'inverse document frequency, what tf-idf multiplies by',
      y: 'information about the category, in nats',
    });

    /* the ceiling first, so the cloud sits on top of it */
    const line = d3.line().x((d) => x(d.idf)).y((d) => y(d.cap))
      .curve(d3.curveMonotoneX);
    g.append('path')
      .datum(V.ceiling.slice().sort((a, b) => a.idf - b.idf))
      .attr('fill', 'none')
      .attr('stroke', 'var(--cosmos)')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '7 4')
      .attr('d', line);

    const pts = g.append('g');
    V.cloud.forEach(([xi, mi]) => {
      pts.append('circle')
        .attr('cx', x(xi)).attr('cy', y(mi)).attr('r', 1.9)
        .attr('fill', 'var(--primary)').attr('opacity', 0.4);
    });

    legend(g, [
      { label: 'one word', colour: 'var(--primary)', shape: 'dot' },
      { label: 'the most it could carry', colour: 'var(--cosmos)', shape: 'dash' },
    ], { x0: 0, y0: -16, gap: 190, size: 11 });

    const med = V.median_idf;
    g.append('line')
      .attr('x1', x(med)).attr('x2', x(med)).attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 3').attr('opacity', 0.5);
    /* anchor the label on whichever side has room, measured against the width */
    const right = x(med) < 0.6 * w;
    seriesLabel(g, x(med) + (right ? 7 : -7), 16, 'median idf',
      { anchor: right ? 'start' : 'end', size: 11 });
  }

  function drawTop(rows, colour, valueKey) {
    const data = rows.map((r) => ({
      label: r.w, value: r[valueKey], df: r.df, other: r[valueKey === 'idf' ? 'mi' : 'idf'],
    }));
    wordBars(g.append('g'), data, w, h, {
      colour,
      labelWidth: 116,
      size: 11.5,
      valueFmt: (v) => (valueKey === 'idf' ? v.toFixed(2) : v.toFixed(4)),
    });
    seriesLabel(g, 0, -16,
      valueKey === 'idf'
        ? `highest idf: ${V.tied_at_max_idf.toLocaleString('en-US')} terms tie here, fifteen of them`
        : 'most information about the category: what actually separates them',
      { anchor: 'start', size: 11 });
  }

  function render() {
    g.selectAll('*').remove();
    if (view === 'cloud') drawCloud();
    else if (view === 'idf') drawTop(V.top_idf, 'var(--primary)', 'idf');
    else drawTop(V.top_mi, 'var(--anchor)', 'mi');

    const best = V.top_mi[0];
    const rarest = V.top_idf[0];
    readout.html(
      `<div>Across all <span class="value">${V.terms.toLocaleString('en-US')}</span> terms the `
      + `rank correlation between rarity and information is `
      + `<span class="bold">${V.spearman.toFixed(4)}</span>, which is not weak, it is `
      + `<span class="bold">negative</span>: the rarer half of the vocabulary is the less `
      + `informative half. Of the 200 most informative terms, `
      + `<span class="value">${V.informative_above_median_idf}</span> sit above the median idf.</div>`
      + `<div>The single most informative term is `
      + `<span class="bold">${escapeHtml(V.best_mi_word)}</span>, which occurs in `
      + `<span class="value">${V.best_mi_df.toLocaleString('en-US')}</span> of the `
      + `<span class="value">${V.n_train.toLocaleString('en-US')}</span> training documents. `
      + `The highest-idf term is <span class="bold">${escapeHtml(rarest.w)}</span>, which occurs in `
      + `<span class="value">${rarest.df}</span>, and carries `
      + `<span class="value">${rarest.mi.toFixed(5)}</span> nats against `
      + `<span class="value">${best.mi.toFixed(4)}</span>.</div>`
      + `<div><span class="value">${V.top_mi_stopwords}</span> of those 200 most informative terms `
      + `are on the standard English stopword list, the one every tutorial deletes first.</div>`
    );
  }

  render();
  return { render };
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}
