/* The same digits, two planes, and the question of where a sample lands.
 *
 * The left plane belongs to the autoencoders article and is drawn from that
 * article's own published codes; the right one belongs to the network trained
 * here. Both decoders run in this page, so the picture beside the plane is
 * always the answer of whichever model is selected, at whatever point the
 * reader put the crosshair, including points neither model ever saw.
 *
 * The button that matters is the one that stops the reader choosing the point:
 * it draws from the proposal each model is entitled to (a gaussian fitted to
 * the codes for the autoencoder, which has no prior at all, and the standard
 * gaussian the variational one was trained towards) and shows what comes back.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { toRows } from '../../assets/js/embed.js';
import { fitGaussian, gaussianDraws, unitDraws, nearest } from './vaekit.js';

const DIGIT = d3.scaleOrdinal(d3.schemeCategory10).domain(d3.range(10));

export function initPlaneWidget(data, vae, ae) {
  const side = data.meta.side;
  const N = data.net;
  const AEZ = toRows(N.ae_latent);
  const VZ = toRows(N.mu);
  const trainIdx = N.is_test.map((t, i) => (t ? -1 : i)).filter((i) => i >= 0);
  const aeFit = fitGaussian(trainIdx.map((i) => Array.from(AEZ[i])));

  const chart = makeChart('#plane-chart', {
    width: 470, height: 430, margin: { top: 58, right: 20, bottom: 52, left: 54 },
  });
  const { svg, g, w, h } = chart;
  svg.style('max-width', '480px').style('display', 'inline-block')
    .style('vertical-align', 'top');

  const cv = makeCanvas(document.querySelector('#plane-canvas'), side * 12, side * 12);
  const caption = d3.select('#plane-caption');
  const readout = document.querySelector('#plane-readout');
  const buttons = document.querySelector('#plane-buttons');

  const st = { model: 'vae', z: [0, 0], drawn: [], seed: 4101 };

  const btns = {};
  const add = (key, text, onClick) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', onClick);
    buttons.appendChild(b);
    btns[key] = b;
    return b;
  };
  add('ae', 'the autoencoder', () => { st.model = 'ae'; st.drawn = []; st.z = [0, 0]; render(); });
  add('vae', 'the variational one', () => {
    st.model = 'vae'; st.drawn = []; st.z = [0, 0]; render();
  });
  const drawBtn = add('draw', 'draw a point from the proposal', () => {
    st.seed += 977;
    const p = st.model === 'ae'
      ? gaussianDraws(aeFit, 1, st.seed)[0]
      : unitDraws(2, 1, st.seed)[0];
    st.z = p;
    st.drawn.push(p.slice());
    if (st.drawn.length > 24) st.drawn.shift();
    render();
  });
  drawBtn.className = 'atlas-btn ghost';

  const dots = g.append('g');
  const rings = g.append('g');
  const marks = g.append('g');
  const cross = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -38).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  /* One square domain for both planes, wide enough for the wider of the two
   * clouds, because two planes drawn on two different scales cannot be
   * compared by eye and comparing them is the entire point. */
  const all = AEZ.concat(VZ);
  let lim = 0;
  all.forEach((p) => { lim = Math.max(lim, Math.abs(p[0]), Math.abs(p[1])); });
  lim = Math.min(lim, 6) * 1.04;
  const x = d3.scaleLinear().domain([-lim, lim]).range([0, w]);
  const y = d3.scaleLinear().domain([-lim, lim]).range([h, 0]);

  function ellipsePath(mean, cov, scale) {
    /* the level set of a two dimensional gaussian, as a polygon */
    const a = cov[0][0];
    const b = cov[0][1];
    const c = cov[1][1];
    const l11 = Math.sqrt(Math.max(a, 1e-12));
    const l21 = b / l11;
    const l22 = Math.sqrt(Math.max(c - l21 * l21, 1e-12));
    const pts = [];
    for (let i = 0; i <= 72; i++) {
      const t = (i / 72) * 2 * Math.PI;
      const u = Math.cos(t) * scale;
      const v = Math.sin(t) * scale;
      pts.push([x(mean[0] + l11 * u), y(mean[1] + l21 * u + l22 * v)]);
    }
    return `M${pts.map((p) => p.join(',')).join('L')}Z`;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => {
      if (k === 'draw') return;
      b.classList.toggle('ghost', k !== st.model);
    });
    const isAe = st.model === 'ae';
    const Z = isAe ? AEZ : VZ;
    const net = isAe ? ae : vae;

    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(g, w, h, { x: 'first number', y: 'second number' });

    dots.selectAll('circle').data(Z).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 2)
      .attr('fill', (d, i) => DIGIT(N.labels[i])).attr('opacity', 0.6);

    const shape = isAe
      ? [aeFit.mean, aeFit.cov]
      : [[0, 0], [[1, 0], [0, 1]]];
    rings.selectAll('path').data([1, 2]).join('path')
      .attr('d', (s) => ellipsePath(shape[0], shape[1], s))
      .attr('fill', 'none').attr('stroke', 'var(--squidink)')
      .attr('stroke-dasharray', '5 4').attr('stroke-width', 1.4).attr('opacity', 0.85);

    marks.selectAll('path').data(st.drawn).join('path')
      .attr('d', d3.symbol().type(d3.symbolCross).size(70))
      .attr('transform', (d) => `translate(${x(d[0])},${y(d[1])}) rotate(45)`)
      .attr('fill', 'var(--primary)').attr('opacity', 0.9);

    cross.selectAll('*').remove();
    cross.append('line').attr('x1', x(st.z[0])).attr('x2', x(st.z[0]))
      .attr('y1', 0).attr('y2', h).attr('stroke', 'var(--squidink)').attr('stroke-width', 1);
    cross.append('line').attr('y1', y(st.z[1])).attr('y2', y(st.z[1]))
      .attr('x1', 0).attr('x2', w).attr('stroke', 'var(--squidink)').attr('stroke-width', 1);
    cross.append('circle').attr('cx', x(st.z[0])).attr('cy', y(st.z[1])).attr('r', 8)
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3);

    wrapLabel(title, isAe
      ? `the codes the autoencoders article published, and the gaussian anyone would fit to them`
      : `the codes this article's encoder produces, and the prior it was trained towards`,
    w - 8);

    const px = net.decode(st.z);
    const gray = new Float64Array(px.length);
    for (let i = 0; i < px.length; i++) gray[i] = 1 - px[i];
    paintGray(cv.ctx, gray, side, side, { zoom: 12 });
    caption.text(isAe
      ? 'the autoencoders article’s decoder, at that point'
      : 'this article’s decoder, at that point');

    const dist = nearest(st.z, Z);
    const L = data.compare.ladder.find((r) => r.k === data.meta.k_show);
    const arm = isAe ? L.ae : L.vae;
    const other = isAe ? L.vae : L.ae;
    const mmd = isAe ? arm.mmd2 : arm.prior_mmd2;
    const floor = data.hole.floor;
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>crosshair</th>
          <th>nearest code</th>
          <th>rebuilding</th>
          <th>${data.hole.draws} draws against real digits</th>
          <th>separable at</th>
        </tr>
        <tr>
          <td><span class="value">${st.z[0].toFixed(2)}, ${st.z[1].toFixed(2)}</span></td>
          <td><span class="value">${dist.toFixed(3)}</span>, typical ${data.hole.code_spacing}</td>
          <td>${arm.mse}</td>
          <td><span class="value">${mmd}</span>, floor ${floor.mmd2}</td>
          <td><span class="value">${isAe ? arm.c2st : arm.prior_c2st}</span>, floor ${floor.c2st}</td>
        </tr>
      </table>
      <div class="gen-note">
        ${st.drawn.length === 0
    ? `Drag the crosshair anywhere, including where no digit ever was, and switch models to see
       the same square filled two different ways. Then stop choosing: the button draws a point
       from ${isAe
    ? 'the gaussian fitted to these codes, which is the only proposal a model with no prior can be given'
    : 'the standard gaussian this model was trained towards'}, which is what generating actually means.`
    : `${st.drawn.length} point${st.drawn.length === 1 ? '' : 's'} drawn so far, marked with
       crosses. The current one sits <span class="value">${dist.toFixed(3)}</span> from the
       nearest of the ${data.meta.n.toLocaleString('en-US')} codes, where a code's own nearest
       neighbour is typically ${data.hole.code_spacing} away.`}
        ${isAe
    ? `Over ${data.hole.draws} draws this model's pictures sit ${arm.mmd2} from real digits in
       the kernel statistic, against ${other.prior_mmd2} for the variational one from its prior,
       with a floor of ${floor.mmd2} measured on real digits split in two. At two numbers the
       autoencoder is not the loser here, which is the first thing on this page that did not go
       the way the textbook says.`
    : `At ${data.meta.k_show} numbers this model's draws sit ${arm.prior_mmd2} from real digits
       against the autoencoder's ${other.mmd2}. The difference between the two planes is not
       which one samples better at this size, it is that one of them has a proposal by
       construction and the other one has to be handed one.`}
      </div>`;
  }

  const surface = g.append('rect').attr('x', 0).attr('y', 0).attr('width', w).attr('height', h)
    .attr('fill', 'transparent').style('cursor', 'crosshair');
  const move = (event) => {
    const [px, py] = d3.pointer(event, g.node());
    st.z = [
      Math.max(x.domain()[0], Math.min(x.domain()[1], x.invert(px))),
      Math.max(y.domain()[1], Math.min(y.domain()[0], y.invert(py))),
    ];
    render();
  };
  surface.call(d3.drag().on('start drag', move));
  surface.on('click', move);

  render();
  return st;
}
