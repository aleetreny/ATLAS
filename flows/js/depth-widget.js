/* What depth buys, scored by the exact KL rather than by a held out average.
 *
 * Three arms on the same axes: affine coupling with the mask alternating, the
 * same depths with additive coupling (volume preserving), and the same depth
 * with a mask that never alternates, which is a single point because it is a
 * failure rather than a curve. Every point carries its seed range, and the
 * "flat from here" claim is the generator's, checked against every later row
 * rather than against the first one that qualifies.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initDepthWidget(data) {
  const D = data.depth;
  const A = data.additive;
  const M = data.mask;
  const chart = makeChart('#depth-chart', {
    width: 640, height: 360, margin: { top: 52, right: 28, bottom: 56, left: 74 },
  });
  const { g, w, h } = chart;
  const buttons = document.querySelector('#depth-buttons');
  const readout = document.querySelector('#depth-readout');
  const st = { view: 'kl' };

  const btns = {};
  for (const [key, text] of [['kl', 'distance from the truth'],
    ['held', 'log density on held out rows'],
    ['params', 'against the weights it costs']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    const depths = D.rows.map((r) => r.depth);
    const xVals = st.view === 'params' ? D.rows.map((r) => r.params) : depths;
    const x = st.view === 'params'
      ? d3.scaleLog().domain([Math.min(...xVals) * 0.8, Math.max(...xVals) * 1.25]).range([0, w])
      : d3.scaleLinear().domain([0, Math.max(...depths) + 1]).range([0, w]);

    const pick = st.view === 'held' ? (r) => r.held : (r) => r.kl;
    const vals = D.rows.map(pick).concat(A.rows.map((r) => r.kl));
    let y;
    let yValues = null;
    if (st.view === 'held') {
      const lo = Math.min(...D.rows.map((r) => r.held), data.meta.ceiling);
      const hi = Math.max(...D.rows.map((r) => r.held), data.meta.ceiling);
      const pad = (hi - lo) * 0.12 + 1e-6;
      y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([h, 0]);
    } else {
      y = d3.scaleLog()
        .domain([Math.min(...vals) * 0.65, Math.max(...vals, M.kl) * 1.5]).range([h, 0]);
      yValues = [1e-3, 1e-2, 1e-1, 1, 10]
        .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
    }

    drawGrid(g, x, y, w, h,
      st.view === 'params' ? { xTicks: 4, yValues } : { xValues: depths, yValues, yTicks: 5 });
    drawAxes(g, x, y, w, h, st.view === 'params'
      ? { xTicks: 4, yValues, yTicks: 5, yFmt: st.view === 'held' ? null : d3.format('.0e') }
      : { xValues: depths, yValues, yTicks: 5, yFmt: st.view === 'held' ? null : d3.format('.0e') });
    axisLabels(g, w, h, {
      x: st.view === 'params' ? 'weights in the flow' : 'coupling layers',
      y: st.view === 'held' ? 'nats per held out row' : 'exact distance from the truth, in nats',
    });

    const xa = (r, i) => (st.view === 'params' ? x(r.params) : x(r.depth));
    const series = [['affine coupling', D.rows, pick, 'var(--primary)']];
    if (st.view !== 'held') {
      series.push(['additive, volume preserving', A.rows, (r) => r.kl, 'var(--anchor)']);
    }
    series.forEach(([label, rows, f, colour], si) => {
      layer.append('path')
        .attr('d', d3.line().x((r, i) => xa(r, i)).y((r) => y(f(r)))(rows))
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2);
      layer.selectAll(`circle.s${si}`).data(rows).join('circle')
        .attr('class', `s${si}`)
        .attr('cx', (r, i) => xa(r, i)).attr('cy', (r) => y(f(r))).attr('r', 4)
        .attr('fill', colour);
      layer.append('text').attr('class', 'chart-note')
        .attr('x', w - 4).attr('y', 12 + si * 16).attr('text-anchor', 'end')
        .style('font-size', '11.5px').attr('fill', colour).text(label);
    });
    /* the seed range on the arm the article's claims are about */
    D.rows.forEach((r, i) => {
      if (st.view === 'held') return;
      layer.append('line')
        .attr('x1', xa(r, i)).attr('x2', xa(r, i))
        .attr('y1', y(r.kl_min)).attr('y2', y(r.kl_max))
        .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4).attr('opacity', 0.8);
    });

    if (st.view === 'held') {
      layer.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(data.meta.ceiling)).attr('y2', y(data.meta.ceiling))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
      layer.append('text').attr('class', 'chart-note')
        .attr('x', 2).attr('y', y(data.meta.ceiling) + 14).style('font-size', '11.5px')
        .attr('fill', 'var(--anchor)').text('the truth itself');
    }

    wrapLabel(title, st.view === 'params'
      ? 'the same rows, against what they cost'
      : 'every point is the exact integral, not an average over samples', w - 8);

    const best = D.rows.find((r) => r.depth === D.best);
    const shown = D.rows.find((r) => r.depth === data.meta.depth_show);
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>layers</th><th>weights</th><th>distance from the truth</th>
          <th>held out log density</th><th>additive, same depth</th>
        </tr>
        ${D.rows.map((r) => {
    const add = A.rows.find((a) => a.depth === r.depth);
    return `<tr>
          <td>${r.depth === D.best ? '<span class="bold">' + r.depth + '</span>' : r.depth}</td>
          <td>${r.params.toLocaleString('en-US')}</td>
          <td><span class="value">${r.kl}</span> (${r.kl_min} to ${r.kl_max})</td>
          <td>${r.held}</td>
          <td>${add ? add.kl : ''}</td>
        </tr>`;
  }).join('')}
      </table>
      <div class="gen-note">
        The seed floor of this sweep is ${D.floor} nats, measured as the median spread of three
        seeds at each depth, and every claim on this page is read against it. The best row is
        ${D.best} layers at ${best.kl};
        ${D.flat_from
    ? `the staircase is flat from ${D.flat_from} layers onwards, which is a claim about every
       later row and is checked against every later row`
    : 'no depth in this sweep is flat under that test, so the page does not say the word'}
        ${D.backwards.length
    ? `, and it does go backwards between ${D.backwards.map((p) => `${p[0]} and ${p[1]}`).join(', ')} layers`
    : ', and it never goes backwards by more than the floor'}.
        The additive arm is the same flow with the scale removed, so every layer preserves
        volume: its log determinant is ${A.logdet_max} at worst over the probe points, which is
        the guard that it really is what it claims. At ${shown.depth} layers it costs
        ${A.rows.find((a) => a.depth === shown.depth).gap} nats.
      </div>`;
  }

  render();
  return st;
}
