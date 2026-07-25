/* The spatial payoff: a Gaussian process over California, showing its own
 * prediction next to its own ignorance. Both surfaces are precomputed, because
 * a 400-point GP over a 60x60 grid is not a thing to refit on every hover. */
import { makeChart, drawAxes, drawGrid, axisLabels, tooltip } from '../../assets/js/chart.js';

export function initSpatialWidget(spatial) {
  const { lon, lat, mean, std } = spatial.grid;
  const nx = lon.length;
  const ny = lat.length;

  const { g, w, h } = makeChart('#spatial-chart', {
    width: 620,
    height: 560,
    margin: { top: 36, right: 24, bottom: 62, left: 62 },
  });
  const x = d3.scaleLinear().domain([d3.min(lon), d3.max(lon)]).range([0, w]);
  const y = d3.scaleLinear().domain([d3.min(lat), d3.max(lat)]).range([h, 0]);

  const cellW = w / (nx - 1);
  const cellH = h / (ny - 1);
  const cells = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      cells.push({ i, j, mean: mean[j * nx + i], std: std[j * nx + i] });
    }
  }

  const heat = g.append('g');
  const dotsG = g.append('g');
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'longitude', y: 'latitude' });

  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle');

  const meanScale = d3.scaleSequential(d3.interpolateViridis).domain(d3.extent(mean));
  const stdScale = d3.scaleSequential(d3.interpolateMagma).domain(d3.extent(std).reverse());

  const tip = tooltip();
  const state = { mode: 'std' };

  function render() {
    const scale = state.mode === 'mean' ? meanScale : stdScale;
    const key = state.mode;
    heat.selectAll('rect').data(cells).join('rect')
      .attr('x', (c) => x(lon[c.i]) - cellW / 2)
      .attr('y', (c) => y(lat[c.j]) - cellH / 2)
      .attr('width', cellW + 1)
      .attr('height', cellH + 1)
      .attr('fill', (c) => scale(c[key]));

    dotsG.selectAll('circle').data(spatial.points).join('circle')
      .attr('cx', (p) => x(p.lon)).attr('cy', (p) => y(p.lat))
      .attr('r', 2.2)
      .attr('fill', state.mode === 'std' ? 'white' : 'none')
      .attr('stroke', state.mode === 'std' ? 'none' : 'white')
      .attr('stroke-width', 0.8)
      .attr('opacity', 0.85)
      .on('mousemove', (event, p) => tip.show(`$${(p.v * 100).toFixed(0)}k`, event))
      .on('mouseleave', () => tip.hide());

    title.text(state.mode === 'std'
      ? 'posterior uncertainty · dark means the model is guessing'
      : 'posterior mean · predicted median house value');

    document.querySelector('#spatial-caption').innerHTML = state.mode === 'std'
      ? `<div class="slider-label" style="line-height:1.45">Uncertainty collapses wherever the white dots cluster, along the coast and around the two urban corridors, and blooms over the empty interior and the ocean. Nothing told the model where California ends. It inferred the shape of its own ignorance from where the data is not.</div>`
      : `<div class="slider-label" style="line-height:1.45">The predicted surface alone looks authoritative everywhere, including out at sea. That is exactly the failure mode of every other method on this page, and the reason the other view matters more than this one.</div>`;
  }

  document.querySelector('#spatial-std').addEventListener('click', () => {
    state.mode = 'std';
    document.querySelector('#spatial-std').classList.remove('ghost');
    document.querySelector('#spatial-mean').classList.add('ghost');
    render();
  });
  document.querySelector('#spatial-mean').addEventListener('click', () => {
    state.mode = 'mean';
    document.querySelector('#spatial-mean').classList.remove('ghost');
    document.querySelector('#spatial-std').classList.add('ghost');
    render();
  });

  render();
}
