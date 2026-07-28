/* The same 500 points the manifold article unrolled, through a bottleneck.
 *
 * The generator asserts the roll is coordinate for coordinate the one
 * manifolds/data/manifolds.json ships, so the comparison against Isomap's score
 * is a comparison and not a coincidence of two similar datasets.
 */
import { makeChart, wrapLabel, equalScales } from '../../assets/js/chart.js';
import { procrustes, toRows } from '../../assets/js/embed.js';

export function initRollWidget(data) {
  const R = data.roll;
  const chart = makeChart('#roll-chart', {
    width: 780, height: 300, margin: { top: 66, right: 22, bottom: 30, left: 40 },
  });
  const { g, w, h } = chart;
  const buttons = document.querySelector('#roll-buttons');
  const readout = document.querySelector('#roll-readout');
  const st = { key: 'ae' };

  const chartTrue = toRows(R.chart);
  const coords = toRows(R.coords);
  const sVals = chartTrue.map((c) => c[0]);
  const ramp = d3.scaleSequential(d3.interpolateTurbo)
    .domain([Math.min(...sVals), Math.max(...sVals)]);

  const btns = {};
  for (const [key, text] of [['ae', 'the bottleneck'], ['tries', 'every attempt at it']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === st.key ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      st.key = key;
      render();
    });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -48).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');

  function render() {
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== st.key);
    body.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').remove();

    if (st.key === 'ae') {
      const gap = 20;
      const half = (w - gap) / 2;
      const left = body.append('g');
      const { x: cx, y: cy } = equalScales(chartTrue, half, h - 16, { pad: 1.06 });
      left.selectAll('circle').data(chartTrue).join('circle')
        .attr('cx', (d) => cx(d[0])).attr('cy', (d) => cy(d[1]) + 14).attr('r', 2.6)
        .attr('fill', (d) => ramp(d[0])).attr('opacity', 0.9);
      left.append('text').attr('x', half / 2).attr('y', 0).attr('text-anchor', 'middle')
        .attr('class', 'chart-annotation').style('font-size', '11.5px')
        .style('text-transform', 'none').text('the sheet, as it really is');
      const right = body.append('g');
      const A = procrustes(chartTrue, coords);
      const { x, y } = equalScales(A.reference.concat(A.coords), half, h - 16, { pad: 1.1 });
      right.selectAll('circle.t').data(A.reference).join('circle').attr('class', 't')
        .attr('cx', (d) => half + gap + x(d[0])).attr('cy', (d) => y(d[1]) + 14).attr('r', 4.2)
        .attr('fill', 'none').attr('stroke', 'var(--stone)').attr('stroke-width', 1);
      right.selectAll('circle.z').data(A.coords).join('circle').attr('class', 'z')
        .attr('cx', (d) => half + gap + x(d[0])).attr('cy', (d) => y(d[1]) + 14).attr('r', 2.6)
        .attr('fill', (d, i) => ramp(chartTrue[i][0])).attr('opacity', 0.92);
      right.append('text').attr('x', half + gap + half / 2).attr('y', 0)
        .attr('text-anchor', 'middle').attr('class', 'chart-annotation')
        .style('font-size', '11.5px').style('text-transform', 'none')
        .text(`what a two unit bottleneck makes of it`);
      wrapLabel(title, 'the same 500 points, coloured by where they really sit along the sheet',
        w - 8);
    } else {
      const rows = R.attempts;
      const x = d3.scalePoint().domain(rows.map((_, i) => i)).range([24, w - 24]);
      const y = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);
      const bw = Math.min(52, (w - 48) / rows.length - 10);
      body.selectAll('rect').data(rows).join('rect')
        .attr('x', (r, i) => x(i) - bw / 2).attr('width', bw)
        .attr('y', (r) => y(r.disparity)).attr('height', (r) => h - y(r.disparity))
        .attr('fill', 'var(--primary)').attr('opacity', 0.85)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.6);
      body.selectAll('text.lab').data(rows).join('text').attr('class', 'lab')
        .attr('x', (r, i) => x(i)).attr('y', (r) => y(r.disparity) - 6)
        .attr('text-anchor', 'middle').style('font-size', '11.5px')
        .style('font-family', 'var(--font-mono)').attr('fill', 'var(--squidink)')
        .text((r) => r.disparity);
      body.selectAll('text.cfg').data(rows).join('text').attr('class', 'cfg')
        .attr('x', (r, i) => x(i)).attr('y', h + 16)
        .attr('text-anchor', 'middle').style('font-size', '11px')
        .style('font-family', 'var(--font-mono)').attr('fill', 'var(--squidink)')
        .text((r) => `${r.hidden}h ${r.epochs}e`);
      body.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(R.isomap)).attr('y2', y(R.isomap))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
      body.append('text').attr('x', w - 4).attr('y', y(R.isomap) - 8)
        .attr('text-anchor', 'end').attr('class', 'chart-annotation')
        .style('font-size', '11.5px').style('text-transform', 'none')
        .text(`Isomap on the same points: ${R.isomap}`);
      wrapLabel(title, `${rows.length} attempts, each pushed the way that would help if capacity `
        + 'or training length were the problem', w - 8);
    }

    readout.innerHTML = `
      <table class="ae-table">
        <tr><th>method</th><th>distance from the true chart</th><th>what it needed</th></tr>
        <tr><td>Isomap</td><td><span class="value">${R.isomap}</span></td>
          <td>a neighbourhood graph and one eigenproblem</td></tr>
        <tr><td>a two unit bottleneck</td><td><span class="value">${R.disparity}</span></td>
          <td>${R.best.hidden} hidden units and ${R.best.epochs.toLocaleString('en-US')} epochs, the best of ${R.attempts.length}</td></tr>
        <tr><td>locally linear embedding</td><td>${R.lle}</td><td>local weights and one eigenproblem</td></tr>
        <tr><td>a self-organising map</td><td>${R.som}</td><td>a grid dragged into the data</td></tr>
        <tr><td>PCA</td><td>${R.pca}</td><td>one eigenproblem</td></tr>
      </table>
      <div class="ae-note">
        ${R.disparity > 0.5
    ? `A bottleneck is not a manifold learner. It has no notion of distance along the sheet and
        nothing in its loss rewards finding one: it is asked to rebuild each point from two numbers
        and it does that (${R.mse} of squared error), by whatever two numbers are convenient. The
        best of ${R.attempts.length} attempts lands ${R.disparity} from the true chart against
        Isomap's <span class="bold">${R.isomap}</span>, which is the same lesson the self-organising
        map taught in the <a href="../manifolds/">previous article</a> and worth repeating because
        the two methods look nothing alike: <span class="bold">a low reconstruction error is not
        evidence that the right two numbers were found.</span>`
    : `Here the bottleneck does recover the sheet, at ${R.disparity} against Isomap's ${R.isomap}.`}
      </div>`;
  }

  render();
}
