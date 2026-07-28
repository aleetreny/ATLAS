/* Why a plane was never going to be enough, in two pictures.
 *
 * Both are facts about geometry rather than about any algorithm, which is the
 * point of putting them before the algorithm.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initCrowdingWidget(data) {
  const C = data.crowding;
  const chart = makeChart('#crowding-chart', {
    width: 760, height: 340, margin: { top: 60, right: 24, bottom: 58, left: 74 },
  });
  const { g, w, h } = chart;
  const buttons = document.querySelector('#crowding-buttons');
  const readout = document.querySelector('#crowding-readout');
  const st = { key: 'simplex' };

  const btns = {};
  for (const [key, text] of [['simplex', 'points that are all the same distance apart'],
    ['volume', 'how far apart anything is']]) {
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
    .attr('x', w / 2).attr('y', -42).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');

  function render() {
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== st.key);
    body.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').remove();

    if (st.key === 'simplex') {
      const rows = C.simplex;
      const x = d3.scalePoint().domain(rows.map((r) => r.k)).range([16, w - 16]);
      const y = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.spread) * 1.15])
        .range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 0, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: rows.length, yTicks: 5, yFmt: d3.format('.0%') });
      axisLabels(g, w, h, { x: 'how many points, all the same distance apart', y: 'spread once squeezed into a plane' });
      const bw = Math.min(30, (w - 32) / rows.length - 6);
      body.selectAll('rect').data(rows).join('rect')
        .attr('x', (r) => x(r.k) - bw / 2).attr('width', bw)
        .attr('y', (r) => y(r.spread)).attr('height', (r) => h - y(r.spread))
        .attr('fill', (r) => (r.spread < 1e-9 ? 'var(--stone)' : 'var(--primary)'))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.6);
      body.selectAll('text.n').data(rows).join('text').attr('class', 'n')
        .attr('x', (r) => x(r.k)).attr('y', (r) => y(r.spread) - 6)
        .attr('text-anchor', 'middle').style('font-size', '11.5px')
        .style('font-family', 'var(--font-mono)').attr('fill', 'var(--squidink)')
        .text((r) => (r.ratio > 1.001 ? `${r.ratio.toFixed(1)}x` : 'exact'));
      wrapLabel(title, 'k points can only be mutually equidistant in k-1 dimensions; the label '
        + 'is how many times the longest distance beats the shortest once they are on a plane',
      w - 8);
    } else {
      const rows = C.volume;
      const x = d3.scaleLog().domain([1.8, 60]).range([0, w]);
      const y = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.cv) * 1.15]).range([h, 0]);
      const ticks = [2, 3, 5, 10, 20, 30, 50];
      drawGrid(g, x, y, w, h, { xValues: ticks, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xValues: ticks, xFmt: d3.format('d'), yTicks: 5, yFmt: d3.format('.0%') });
      axisLabels(g, w, h, { x: 'dimensions', y: 'spread of the pairwise distances' });
      body.append('path').datum(rows).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6)
        .attr('d', d3.line().x((r) => x(r.d)).y((r) => y(r.cv)));
      body.selectAll('circle').data(rows).join('circle')
        .attr('cx', (r) => x(r.d)).attr('cy', (r) => y(r.cv)).attr('r', 4.5)
        .attr('fill', 'var(--primary)').attr('stroke', 'white').attr('stroke-width', 1.4);
      body.selectAll('text.v').data(rows).join('text').attr('class', 'v')
        .attr('x', (r) => x(r.d)).attr('y', (r) => y(r.cv) - 11)
        .attr('text-anchor', 'middle').style('font-size', '11.5px')
        .style('font-family', 'var(--font-mono)').attr('fill', 'var(--squidink)')
        .text((r) => `${(r.cv * 100).toFixed(0)}%`);
      wrapLabel(title, '1,500 points spread evenly over a sphere: as the dimension grows they all '
        + 'end up about equally far from each other', w - 8);
    }

    const big = C.simplex[C.simplex.length - 1];
    const hi = C.volume[C.volume.length - 1];
    const lo = C.volume[0];
    readout.innerHTML = `
      <div class="nb-note">
        ${st.key === 'simplex'
    ? `Three points can be mutually equidistant on a page and four cannot: a regular tetrahedron `
      + `needs three dimensions. Squeeze ${big.k} equidistant points, which need `
      + `${big.dims_needed}, onto a plane and their distances end up spread by `
      + `<span class="bold">${(big.spread * 100).toFixed(2)}%</span>, with the longest `
      + `<span class="bold">${big.ratio}</span> times the shortest. Nothing has gone wrong and no `
      + `algorithm could do better: the arrangement being asked for does not exist. Any method `
      + `whose objective is "reproduce these distances" is being asked to do something impossible, `
      + `and what it does instead is a compromise nobody chose.`
    : `And the other half. Points spread over a sphere in ${lo.d} dimensions have distances `
      + `averaging ${lo.mean} give or take ${lo.sd}, a spread of `
      + `<span class="bold">${(lo.cv * 100).toFixed(1)}%</span>. In ${hi.d} dimensions the same `
      + `experiment gives ${hi.mean} give or take ${hi.sd}, a spread of `
      + `<span class="bold">${(hi.cv * 100).toFixed(1)}%</span>: everything is nearly the same `
      + `distance from everything else. That is what a high-dimensional distance table looks like, `
      + `and it is why "get the distances right" stops being useful advice long before it stops `
      + `being possible.`}
      </div>`;
  }

  render();
}
