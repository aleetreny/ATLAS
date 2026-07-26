/* The lottery, drawn as the pair of distributions it is.
 *
 * 300 measured final inertias per method arrive from the generator; the widget
 * draws them as strip plots on a log axis so the bad-basin outliers are
 * visible next to the pile at the optimum. The second view animates k-means++
 * actually choosing, with the D^2 weights shown as point size, because "the
 * guesses repel each other" is one of those sentences that a picture settles.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { initPP, rng } from './kmeanskit.js';
import { drawClusteredPoints, drawCenters } from './clusterdraw.js';

export function initInitWidget(data) {
  const runs = data.init_runs;
  const pts = data.blobs.points;
  const ref = data.blobs.ref_inertia;

  const { g, w, h } = makeChart('#init-chart', {
    width: 700,
    height: 430,
    margin: { top: 40, right: 30, bottom: 58, left: 120 },
  });

  const layers = { strip: g.append('g'), demo: g.append('g') };
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -18).attr('text-anchor', 'middle').style('font-size', '0.66rem');
  const readout = document.querySelector('#init-readout');
  const btnBoth = document.querySelector('#init-both');
  const btnDemo = document.querySelector('#init-demo');
  let demoTimer = null;

  function showStrips() {
    if (demoTimer) { demoTimer.stop(); demoTimer = null; }
    layers.demo.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').attr('opacity', 1);

    const all = [...runs.random, ...runs.kmeanspp];
    const x = d3.scaleLog().domain([d3.min(all) * 0.95, d3.max(all) * 1.1]).range([0, w]);
    const y = d3.scaleBand().domain(['random', 'kmeanspp']).range([0, h]).padding(0.42);
    drawGrid(g, x, y, w, h, { xTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, xFmt: d3.format('~s') });
    axisLabels(g, w, h, { x: 'final inertia after convergence, 300 restarts each (log scale)' });
    g.select('g.axis.y').call(d3.axisLeft(y).tickSize(0))
      .selectAll('text').style('font-family', 'var(--font-mono)').style('font-size', '11px')
      .text((k) => (k === 'kmeanspp' ? 'k-means++' : 'random'));

    /* Jittered strips: identical inertias pile up, so vertical jitter is the
     * only way 300 points per row stay countable by eye. Seeded, stable. */
    const jit = rng(3);
    const rows = [
      ...runs.random.map((v) => ({ v, k: 'random' })),
      ...runs.kmeanspp.map((v) => ({ v, k: 'kmeanspp' })),
    ];
    layers.strip.selectAll('circle').data(rows).join('circle')
      .attr('cx', (d) => x(d.v))
      .attr('cy', (d) => y(d.k) + y.bandwidth() / 2 + (jit() - 0.5) * y.bandwidth() * 0.85)
      .attr('r', 3)
      .attr('fill', (d) => (d.v < ref * 1.01 ? 'var(--primary)' : 'var(--cosmos)'))
      .attr('stroke', 'white').attr('stroke-width', 0.5)
      .attr('opacity', 0.55);

    layers.strip.append('line')
      .attr('x1', x(ref)).attr('x2', x(ref)).attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');
    layers.strip.append('text').attr('class', 'chart-annotation')
      .attr('x', x(ref)).attr('y', -4).attr('text-anchor', 'middle')
      .style('font-size', '0.58rem').style('text-transform', 'none')
      .text(`the good optimum, ${ref}`);

    const badR = runs.random.filter((v) => v >= ref * 1.01).length;
    const badP = runs.kmeanspp.filter((v) => v >= ref * 1.01).length;
    title.text('every dot is one complete, converged run of the algorithm');
    btnBoth.classList.remove('ghost');
    btnDemo.classList.add('ghost');
    readout.innerHTML =
      `<table class="km-table">
         <tr><th></th><th>found the optimum</th><th>stuck in a worse basin</th><th>worst inertia</th></tr>
         <tr><td>random init</td><td>${300 - badR} of 300</td>
           <td><span class="value" style="color:var(--cosmos)">${badR}</span></td>
           <td>${d3.max(runs.random).toLocaleString('en-US')}</td></tr>
         <tr><td>k-means++</td><td>${300 - badP} of 300</td>
           <td><span class="value">${badP}</span></td>
           <td>${d3.max(runs.kmeanspp).toLocaleString('en-US')}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">` +
      `Every one of these 600 runs converged. ${badR} of the random ones converged to the wrong answer, ` +
      `the worst of them <span class="value">${(d3.max(runs.random) / ref).toFixed(1)}×</span> above the optimum. ` +
      `Convergence is a property of the loop. Correctness is a property of the start.</div>`;
  }

  function showDemo() {
    layers.strip.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').attr('opacity', 0);

    const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);

    const trace = [];
    initPP(pts, 3, rng(11), trace);

    const dotG = layers.demo.append('g');
    const centreG = layers.demo.append('g');
    let stage = 0;

    const draw = () => {
      const t = trace[Math.min(stage, trace.length - 1)];
      const wts = t.weights;
      const maxW = wts ? Math.max(...wts) : 1;
      dotG.selectAll('circle').data(pts).join('circle')
        .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
        .attr('r', (p, i) => (wts ? 1.5 + 6.5 * Math.sqrt(wts[i] / maxW) : 3))
        .attr('fill', 'var(--stone)')
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.6)
        .attr('opacity', 0.75);
      drawCenters(centreG, t.centers, x, y);
      title.text(wts
        ? `centre ${t.centers.length + 1}: each point's size is its chance of being picked (distance² from the nearest centre)`
        : 'three centres chosen, one from each cluster, no luck required');
    };

    draw();
    demoTimer = d3.timer((elapsed) => {
      const next = Math.floor(elapsed / 1600);
      if (next !== stage) {
        stage = next;
        draw();
        if (stage >= trace.length - 1 && demoTimer) {
          demoTimer.stop();
          demoTimer = null;
        }
      }
    });

    btnBoth.classList.add('ghost');
    btnDemo.classList.remove('ghost');
    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5">` +
      `The first centre is a uniformly random point. From then on, the far side of the data literally ` +
      `grows: a point's chance of becoming the next centre is proportional to its squared distance from ` +
      `the nearest centre already placed, so an unclaimed cluster is the likeliest place for the next ` +
      `guess to land. The spread that random initialisation only gets by luck, k-means++ gets by ` +
      `construction.</div>`;
  }

  btnBoth.addEventListener('click', showStrips);
  btnDemo.addEventListener('click', showDemo);
  showStrips();
}
