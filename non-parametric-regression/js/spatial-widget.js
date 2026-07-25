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
    /* Room above the plot for the title and, under it, the colour key. */
    margin: { top: 62, right: 24, bottom: 62, left: 62 },
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
    .attr('x', w / 2).attr('y', -42).attr('text-anchor', 'middle');

  const meanScale = d3.scaleSequential(d3.interpolateViridis).domain(d3.extent(mean));
  const stdScale = d3.scaleSequential(d3.interpolateMagma).domain(d3.extent(std).reverse());

  const tip = tooltip();
  const state = { mode: 'std' };

  /* A heat map with no key is a decoration. Neither view could be read: yellow
   * might have been $76k or $431k and the reader had no way to tell. */
  const LEG_W = 132;
  const LEG_H = 9;
  const legend = g.append('g').attr('transform', `translate(${w - LEG_W},${-24})`);
  const legStops = d3.range(0, 1.001, 1 / 40);
  legend.selectAll('rect').data(legStops).join('rect')
    .attr('x', (d) => d * LEG_W).attr('y', 0)
    .attr('width', LEG_W / 40 + 0.6).attr('height', LEG_H);
  const legLo = legend.append('text').attr('class', 'chart-annotation')
    .attr('x', 0).attr('y', LEG_H + 11).style('font-size', '0.56rem').style('text-transform', 'none');
  const legHi = legend.append('text').attr('class', 'chart-annotation')
    .attr('x', LEG_W).attr('y', LEG_H + 11).attr('text-anchor', 'end')
    .style('font-size', '0.56rem').style('text-transform', 'none');

  function render() {
    const scale = state.mode === 'mean' ? meanScale : stdScale;
    const key = state.mode;
    heat.selectAll('rect').data(cells).join('rect')
      .attr('x', (c) => x(lon[c.i]) - cellW / 2)
      .attr('y', (c) => y(lat[c.j]) - cellH / 2)
      .attr('width', cellW + 1)
      .attr('height', cellH + 1)
      .attr('fill', (c) => scale(c[key]))
      .on('mousemove', (event, c) => tip.show(
        state.mode === 'mean' ? `$${(c.mean * 100).toFixed(0)}k predicted` : `±$${(2 * c.std * 100).toFixed(0)}k, two sigma`,
        event
      ))
      .on('mouseleave', () => tip.hide());

    const dom = state.mode === 'mean' ? d3.extent(mean) : d3.extent(std);
    legend.selectAll('rect').attr('fill', (d) => scale(dom[0] + d * (dom[1] - dom[0])));
    legLo.text(state.mode === 'mean' ? `$${(dom[0] * 100).toFixed(0)}k` : `±$${(2 * dom[0] * 100).toFixed(0)}k`);
    legHi.text(state.mode === 'mean' ? `$${(dom[1] * 100).toFixed(0)}k` : `±$${(2 * dom[1] * 100).toFixed(0)}k`);

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
      ? `<div class="slider-label" style="line-height:1.45">Uncertainty collapses to a tight halo around each white dot and sits at the ceiling everywhere else: 3,035 of the 3,600 cells are within a percent of the maximum. That is not the picture the marketing usually shows, and it is the honest one. Maximum likelihood chose a correlation length shorter than the gap between neighbouring districts, so the model is saying it will not extrapolate past the district it was told about. Nothing gave it a map. It worked out the shape of its own ignorance from where the data is not.</div>`
      : `<div class="slider-label" style="line-height:1.45">The predicted surface looks authoritative everywhere, including out at sea, and it is the same model that just admitted it knows almost nothing away from its dots. That is exactly the failure mode of every other method on this page, and the reason the other view matters more than this one.</div>`;
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
