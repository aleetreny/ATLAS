/* Rank and regularisation, with the training error on the same axes.
 *
 * Two curves per view, because a test curve alone cannot tell the difference
 * between a model that is too small and a model that is memorising: the gap
 * between them is the diagnosis and the test curve is only the verdict.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';

export function initSweepWidget(data) {
  const S = data.sweeps;
  const buttons = document.querySelector('#sweep-buttons');
  const readout = document.querySelector('#sweep-readout');
  const st = { view: 'rank' };

  const chart = makeChart('#sweep-chart', {
    width: 640, height: 380, margin: { top: 46, right: 128, bottom: 58, left: 74 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const btns = {};
  for (const [key, text] of [['rank', 'how many factors'], ['lam', 'how much regularisation']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    clear();
    g.selectAll('text.chart-note').remove();

    const rows = st.view === 'rank' ? S.rank : S.lam;
    const key = st.view === 'rank' ? 'k' : 'lam';
    const xs = rows.map((d) => d[key]);
    const x = st.view === 'rank'
      ? d3.scaleLog().domain([xs[0] * 0.85, xs[xs.length - 1] * 1.15]).range([0, w])
      : d3.scalePoint().domain(xs).range([0, w]).padding(0.5);
    const vals = rows.flatMap((d) => [d.train, d.test]);
    const y = d3.scaleLinear().domain([d3.min(vals) * 0.96, d3.max(vals) * 1.04]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: st.view === 'rank' ? xs : [], yTicks: 5 });

    const ends = [];
    [['test', 'held out', 'var(--primary)'], ['train', 'on what it saw', 'var(--anchor)']]
      .forEach(([field, label, colour]) => {
        plot.append('path').attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
          .attr('d', d3.line().x((d) => x(d[key])).y((d) => y(d[field]))(rows));
        plot.selectAll(null).data(rows).enter().append('circle')
          .attr('cx', (d) => x(d[key])).attr('cy', (d) => y(d[field])).attr('r', 3)
          .attr('fill', colour);
        ends.push({ label, colour, y: y(rows[rows.length - 1][field]) + 3.5, yMax: h + 8 });
      });
    /* on the regularisation view the two curves finish on top of each other,
       so the labels are separated by value instead of by their own y */
    spread(ends, 13).forEach((d) => {
      g.append('text').attr('class', 'chart-note').attr('x', w + 6).attr('y', d.y)
        .style('font-size', '11px').style('fill', d.colour).text(d.label);
    });

    const best = rows.reduce((a, d) => (d.test < a.test ? d : a));
    plot.append('line').attr('x1', x(best[key])).attr('x2', x(best[key]))
      .attr('y1', 0).attr('y2', h).attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);

    if (st.view === 'rank') {
      drawAxes(g, x, y, w, h, { xValues: xs, yTicks: 5, xFmt: (v) => String(v), yFmt: d3.format('.3f') });
      axisLabels(g, w, h, { x: 'factors per reader and per book', y: 'root mean squared error' });
    } else {
      drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.3f') });
      axisLabels(g, w, h, { y: 'root mean squared error' });
      rows.forEach((d) => {
        g.append('text').attr('class', 'chart-note')
          .attr('x', x(d.lam)).attr('y', h + 18).attr('text-anchor', 'middle')
          .style('font-size', '10.5px').text(String(d.lam));
      });
      g.append('text').attr('class', 'chart-note')
        .attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle').style('font-size', '11.5px')
        .text('regularisation');
    }
    g.append('text').attr('class', 'chart-note')
      .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px')
      .text(st.view === 'rank'
        ? `${data.meta.epochs} alternating passes at each rank, regularisation ${data.meta.lam}`
        : `${data.meta.epochs} alternating passes at each setting, ${data.meta.k} factors`);

    const first = rows[0];
    const last = rows[rows.length - 1];
    readout.innerHTML = st.view === 'rank'
      ? `The held out error falls all the way across the sweep, from `
        + `<span class="value">${first.test.toFixed(4)}</span> with ${first.k} factor`
        + `${first.k === 1 ? '' : 's'} to <span class="value">${best.test.toFixed(4)}</span> with `
        + `${best.k}. `
        + `The training error only ever goes down, from ${first.train.toFixed(4)} to `
        + `${last.train.toFixed(4)}, which is what a rank is for: `
        + `<span class="value">${last.params.toLocaleString('en-US')}</span> free numbers at rank `
        + `${last.k} against ${first.params.toLocaleString('en-US')} at rank ${first.k}, on `
        + `${data.meta.train_ratings.toLocaleString('en-US')} ratings. `
        + `${best.k === last.k
          ? 'The held out curve has not turned inside this range: with the penalty fixed, the rank '
            + 'is not what is limiting this fit, which is what the other view is for.'
          : 'Past that rank the two curves separate and the held out one goes back up, which is the '
            + 'shape everything in this atlas that has a capacity knob has.'}`
      : `Regularisation is not a tidiness preference here, it is what makes the fit possible at all: `
        + `at <span class="value">${first.lam}</span> the held out error is `
        + `<span class="value">${first.test.toFixed(4)}</span> with a training error of `
        + `${first.train.toFixed(4)}, and at the best setting `
        + `(<span class="value">${best.lam}</span>) it is `
        + `<span class="value">${best.test.toFixed(4)}</span> with a training error of `
        + `${best.train.toFixed(4)}. The average length of a reader's vector goes from `
        + `<span class="value">${first.norm.toFixed(3)}</span> to `
        + `<span class="value">${last.norm.toFixed(3)}</span> across the sweep. A reader with `
        + `fourteen ratings is being given ${data.meta.k} numbers, and nothing but the penalty stops `
        + `those numbers from fitting the fourteen exactly.`;
  }

  render();
  return { render, state: st };
}
