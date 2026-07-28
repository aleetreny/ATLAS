/* Three ways to split the same table, drawn as strips.
 *
 * The strip is the argument: a shuffled fold takes its test rows from all over
 * the series and trains on rows from either side of every one of them, which
 * is not a forecast, it is an interpolation. The bar underneath is what each
 * protocol then reports for the same model on the same data.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initLeakWidget(data) {
  const L = data.leakage;
  const PROTOCOLS = [
    { key: 'shuffled', label: `shuffled ${L.folds} fold`, mase: L.shuffled_mase },
    { key: 'blocked', label: `${L.folds} contiguous blocks`, mase: L.blocked_mase },
    { key: 'rolling', label: 'rolling origin', mase: L.rolling_mase },
  ];
  let current = 0;
  const readout = document.querySelector('#leak-readout');
  const controls = d3.select('#leak-controls');
  const chart = makeChart('#leak-chart', {
    width: 640, height: 400, margin: { top: 44, right: 24, bottom: 50, left: 66 },
  });
  const { g, w, h, margin } = chart;
  const stripH = 150;
  const botY = stripH + 66;
  const botH = h - botY;
  const gStrip = g.append('g');
  const gBar = g.append('g').attr('transform', `translate(0,${botY})`);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px');

  const n = L.n_rows;
  const lanes = 5;
  const laneH = stripH / (lanes + 1.4);

  /* the same deterministic shuffle every time this page loads: a picture of a
     protocol should not change between two readers */
  const order = d3.range(n);
  let seed = 12345;
  for (let i = n - 1; i > 0; i--) {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    const j = seed % (i + 1);
    const t = order[i]; order[i] = order[j]; order[j] = t;
  }

  function assignment(kind) {
    /* returns, per lane, an array of 'train' | 'test' | 'unused' per row */
    const out = [];
    if (kind === 'shuffled') {
      for (let f = 0; f < lanes; f++) {
        const row = new Array(n).fill('train');
        for (let k = 0; k < n; k++) {
          if (order[k] % lanes === f) row[order[k]] = 'test';
        }
        out.push(row);
      }
    } else if (kind === 'blocked') {
      const size = Math.floor(n / lanes);
      for (let f = 0; f < lanes; f++) {
        const row = new Array(n).fill('train');
        for (let k = f * size; k < (f === lanes - 1 ? n : (f + 1) * size); k++) row[k] = 'test';
        out.push(row);
      }
    } else {
      const firstOrigin = n - L.n_origins;
      for (let f = 0; f < lanes; f++) {
        const o = Math.round(firstOrigin + (f * (L.n_origins - 1)) / (lanes - 1));
        const row = new Array(n).fill('unused');
        for (let k = 0; k < o; k++) row[k] = 'train';
        row[o] = 'test';
        out.push(row);
      }
    }
    return out;
  }

  const colour = { train: 'var(--anchor)', test: 'var(--cosmos)', unused: '#dfe3e3' };

  function render() {
    const P = PROTOCOLS[current];
    const lanesData = assignment(P.key);
    const x = d3.scaleLinear().domain([0, n]).range([0, w]);
    gStrip.selectAll('*').remove();
    lanesData.forEach((row, li) => {
      const runs = [];
      let k = 0;
      while (k < n) {
        let j = k;
        while (j < n && row[j] === row[k]) j += 1;
        runs.push([k, j, row[k]]);
        k = j;
      }
      const gl = gStrip.append('g').attr('transform', `translate(0,${li * laneH * 1.16})`);
      gl.selectAll('rect').data(runs).join('rect')
        .attr('x', (d) => x(d[0])).attr('width', (d) => Math.max(1.2, x(d[1]) - x(d[0])))
        .attr('y', 0).attr('height', laneH * 0.82)
        .attr('fill', (d) => colour[d[2]]);
      gl.append('text').attr('class', 'chart-note').attr('x', -6).attr('y', laneH * 0.6)
        .attr('text-anchor', 'end').style('font-size', '10px')
        .text(P.key === 'rolling' ? `origin ${li + 1}` : `fold ${li + 1}`);
    });
    gStrip.append('text').attr('class', 'chart-note').attr('x', 0)
      .attr('y', lanes * laneH * 1.16 + 14).style('font-size', '11px')
      .attr('fill', 'var(--anchor)').text('blue: trained on');
    gStrip.append('text').attr('class', 'chart-note').attr('x', 122)
      .attr('y', lanes * laneH * 1.16 + 14).style('font-size', '11px')
      .attr('fill', 'var(--cosmos)').text('red: scored on');

    const xb = d3.scalePoint().domain(PROTOCOLS.map((p) => p.label)).range([0, w]).padding(0.5);
    const yb = d3.scaleLinear().domain([0, d3.max(PROTOCOLS, (p) => p.mase) * 1.15])
      .range([botH, 0]);
    gBar.selectAll('*').remove();
    drawGrid(gBar, xb, yb, w, botH, { yTicks: 4 });
    drawAxes(gBar, xb, yb, w, botH, { yTicks: 4 });
    axisLabels(gBar, w, botH, {
      y: 'error it reports', leftBudget: margin.left, xOffset: 38,
    });
    gBar.selectAll('rect').data(PROTOCOLS).join('rect')
      .attr('x', (d) => xb(d.label) - 34).attr('width', 68)
      .attr('y', (d) => yb(d.mase)).attr('height', (d) => botH - yb(d.mase))
      .attr('fill', (d) => (d.key === P.key ? 'var(--primary)' : '#c9ced0'));
    gBar.selectAll('text.v').data(PROTOCOLS).join('text').attr('class', 'chart-note v')
      .attr('x', (d) => xb(d.label)).attr('y', (d) => yb(d.mase) - 6)
      .attr('text-anchor', 'middle').style('font-size', '11px')
      .text((d) => d.mase.toFixed(3));

    wrapLabel(title, `${P.label}: which rows are trained on, and which are scored, `
      + `over the ${n.toLocaleString('en-US')} rows of the table`, w - 8);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>protocol</th><th>what it reports</th><th>against the honest number</th></tr>
        <tr><td>shuffled ${L.folds} fold</td>
            <td><span class="value">${L.shuffled_mase}</span></td>
            <td>${(L.rolling_mase / L.shuffled_mase).toFixed(3)} times too good</td></tr>
        <tr><td>${L.folds} contiguous blocks, still training on both sides</td>
            <td><span class="value">${L.blocked_mase}</span></td>
            <td>${(L.rolling_mase / L.blocked_mase).toFixed(3)} times too good</td></tr>
        <tr><td>rolling origin, ${L.n_origins} of them</td>
            <td><span class="value">${L.rolling_mase}</span></td>
            <td>this is the honest number</td></tr>
      </table>
      <div class="gen-note">
        ${P.key === 'shuffled'
    ? `Look at any red cell and notice what is blue on either side of it. The
           model is asked for a value it is surrounded by, which on a smooth
           series is barely a question at all: it reports ${L.shuffled_mase}
           where the honest experiment says ${L.rolling_mase}, so the number a
           tutorial would print is ${((L.rolling_mase / L.shuffled_mase - 1) * 100).toFixed(0)}%
           optimistic.`
    : P.key === 'blocked'
      ? `Contiguous blocks fix the interpolation and introduce something worse.
             The first block is scored by a model trained entirely on <i>later</i>
             data, and this series trends, so that model is being asked for values
             below everything it ever saw: ${L.blocked_mase}, which is
             ${(L.blocked_mase / L.rolling_mase).toFixed(1)} times the honest
             number in the other direction. Two wrong protocols, failing
             opposite ways.`
      : `The only arrangement with no blue to the right of any red. Each origin
             trains on everything above it and is scored on what comes next,
             which is the experiment the deployed model will actually face:
             ${L.rolling_mase}.`}
      </div>`;
  }

  controls.selectAll('button').data(PROTOCOLS).join('button')
    .attr('class', (d, i) => `atlas-btn${i === current ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (evt, d) => {
      current = PROTOCOLS.indexOf(d);
      controls.selectAll('button')
        .attr('class', (dd) => `atlas-btn${dd === d ? '' : ' ghost'}`);
      render();
    });

  render();
  return { render };
}
