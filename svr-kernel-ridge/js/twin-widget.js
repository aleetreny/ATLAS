/* The identity: kernel ridge and the Gaussian process posterior mean are the
 * same curve, to thirteen decimal places, because they are the same linear
 * system wearing different philosophies. The widget draws both (measured
 * offline), lets the reader pry them apart with an artificial offset to
 * prove there really are two curves, and states what each philosophy adds.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';

export function initTwinWidget(data) {
  const I = data.identity;
  const grid = data.grid;
  const { g, w, h } = makeChart('#twin-chart', {
    width: 620,
    height: 420,
    margin: { top: 30, right: 26, bottom: 54, left: 56 },
  });
  const x = d3.scaleLinear().domain(d3.extent(grid)).nice().range([0, w]);
  const y = d3.scaleLinear().domain([5, 50]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'horsepower', y: 'miles per gallon' });

  const xr = data.x_train.map((v) => v * data.x_sd + data.x_mu);
  g.append('g').selectAll('circle').data(xr).join('circle')
    .attr('cx', (v) => x(v)).attr('cy', (v, i) => y(data.y_train[i]))
    .attr('r', 2.4).attr('fill', 'var(--stone)').attr('opacity', 0.5);

  const line = (curve, off) => d3.line().x((v, i) => x(grid[i])).y((v) => y(v + off))(curve);
  const krP = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 4);
  const gpP = g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)')
    .attr('stroke-width', 2).attr('stroke-dasharray', '7 5');
  const labKr = g.append('text').attr('class', 'chart-annotation').style('font-size', '0.56rem')
    .style('text-transform', 'none').attr('fill', 'var(--primary)').text('kernel ridge');
  const labGp = g.append('text').attr('class', 'chart-annotation').style('font-size', '0.56rem')
    .style('text-transform', 'none').attr('fill', 'var(--cosmos)').text('GP posterior mean');

  const slider = document.querySelector('#twin-off');
  const readout = document.querySelector('#twin-readout');

  function render() {
    const off = +slider.value;
    document.querySelector('#twin-off-label').textContent = off ? `+${off} mpg` : 'none';
    krP.attr('d', line(I.kr_curve, 0));
    gpP.attr('d', line(I.gp_mean, off));
    labKr.attr('x', x(grid[6])).attr('y', y(I.kr_curve[6]) - 12);
    labGp.attr('x', x(grid[6])).attr('y', y(I.gp_mean[6] + off) + (off >= 0 ? -26 : 20));
    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5">Same RBF kernel (&gamma; ${I.gamma}), same ` +
      `ridge &lambda; ${I.lambda} on the diagonal, same zero-mean prior: the largest difference between ` +
      `the two curves anywhere on this axis is <span class="value">${I.max_gap.toExponential(1)}</span> mpg, ` +
      `which is floating-point dust. ${
        off === 0
          ? 'They are drawn on top of each other right now; drag the slider to pry the dashed one loose ' +
            'and confirm there really are two.'
          : 'The separation you see is an artificial offset from the slider, not a disagreement.'
      } What differs is everything around the solve: the Gaussian process view buys the error bands of ` +
      `article 4 and a principled way to tune &gamma; and &lambda; by likelihood; the ridge view buys ` +
      `bluntness: one solve, no story, same curve.</div>`;
  }

  slider.addEventListener('input', render);
  render();
}
