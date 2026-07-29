/* The same scene as a pile of ellipsoids, rasterised here.
 *
 * The outline view is the one that earns its place: it shows what the picture
 * is made of, and it shows the thing nobody puts in a paper figure, which is
 * that a blob is only ever constrained where a photograph looked at it. Drag
 * around and the outlines stay coherent; the picture they add up to is what
 * was optimised, and the outlines are merely a way of getting there.
 *
 * The slider is a measurement, not a decoration: the blobs are ranked by how
 * much weight they actually carry in this frame, and it composites the top few
 * of them, so "how many blobs is this picture really" has an answer.
 */
import { renderTruthFull, camera, psnr, depthCompare } from './truthkit.js';
import { project, rasterise, topBlobs, ellipse } from './splatkit.js';
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';
import { makeFrame, attachOrbit, drawTiles, errorImage, seriesLabel, fmt } from './panels.js';

const RES = 64;
const TRUTH_STEPS = 96;

export function initBlobWidget(data, scene, cloud) {
  const meta = data.meta;
  const host = document.querySelector('#blob-frames');
  host.innerHTML = '';
  const truthF = makeFrame(host, 'the scene', { max: 200 });
  const blobF = makeFrame(host, 'the blobs', { max: 200 });
  const outF = makeFrame(host, 'their outlines, at 2σ', { max: 200 });
  const errF = makeFrame(host, 'difference, ×6', { max: 200 });

  const KS = [5, 10, 25, 50, 100, 200, 400, 700, 1000, cloud.n].filter((k, i, a) => k <= cloud.n && a.indexOf(k) === i);
  const slider = document.querySelector('#blob-k');
  const kval = document.querySelector('#blob-k-value');
  const readout = document.querySelector('#blob-readout');
  slider.max = String(KS.length - 1);
  slider.value = String(KS.length - 1);

  /* Same quarter-turn opening as the field widget, and for the same reason:
     the shortcut button must have somewhere to go. */
  const state = { az: data.cameras.inside[0].az + 40, el: data.cameras.inside[0].el, kIdx: KS.length - 1, fill: true };

  const drawOutlines = (proj, mask) => {
    const size = 200;
    const cv = outF.canvas;
    cv.width = size;
    cv.height = size;
    const ctx = outF.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    /* the same paper the other three frames sit on, or this one reads as a
       white card among three pictures */
    ctx.fillStyle = '#f1f3f3';
    ctx.fillRect(0, 0, size, size);
    const s = size / RES;
    ctx.lineWidth = 0.8;
    proj.order.forEach((g) => {
      if (mask && !mask[g]) return;
      const e = ellipse(proj, g, 2);
      if (!Number.isFinite(e.rx) || e.rx > RES * 2) return;
      ctx.save();
      ctx.translate(e.cx * s, e.cy * s);
      ctx.rotate(e.theta);
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.max(e.rx * s, 0.4), Math.max(e.ry * s, 0.4), 0, 0, Math.PI * 2);
      const a = Math.min(0.85, 0.12 + proj.alpha[g] * 0.75);
      if (state.fill) {
        ctx.fillStyle = `rgba(${Math.round(proj.cr[g] * 255)},${Math.round(proj.cg[g] * 255)},`
          + `${Math.round(proj.cb[g] * 255)},${a * 0.55})`;
        ctx.fill();
      }
      ctx.strokeStyle = `rgba(35,47,62,${0.15 + a * 0.4})`;
      ctx.stroke();
      ctx.restore();
    });
  };

  const render = () => {
    const cam = camera(state.az, state.el, meta);
    const k = KS[state.kIdx];
    kval.textContent = fmt.n0(k);
    const T = renderTruthFull(scene, cam, RES, TRUTH_STEPS);
    const truth = T.rgb;
    const proj = project(cloud, cam, RES, meta.fov_deg);
    const top = topBlobs(cloud, proj, RES, k, { bg: meta.bg });
    const full = k >= cloud.n;
    const got = rasterise(cloud, proj, RES, { bg: meta.bg, only: full ? null : top.mask });
    truthF.draw(truth, RES);
    blobF.draw(got.rgb, RES);
    drawOutlines(proj, full ? null : top.mask);
    errF.draw(errorImage(got.rgb, truth), RES);
    const geo = depthCompare(T.depth, T.alpha, got.depth, got.alpha);
    const visible = proj.order.length;
    readout.innerHTML = `Compositing the <span class="bold">${fmt.n0(Math.min(k, visible))}</span> blobs `
      + `that carry the most of this frame, out of ${fmt.n0(cloud.n)} in the model and `
      + `${fmt.n0(visible)} in front of the camera. They account for `
      + `<span class="bold">${fmt.pc(top.share)}</span> of the weight the full cloud lays down, and score `
      + `<span class="bold">${fmt.db(psnr(got.rgb, truth))}</span> against the closed form. `
      + `The raster took <span class="bold">${fmt.ms(got.ms)}</span> for ${fmt.n0(RES * RES)} pixels, `
      + `touching <span class="bold">${got.perPixel.toFixed(1)}</span> blobs per pixel on average, `
      + `every one of them cut off at three standard deviations. `
      + `They put the surface <span class="bold">${geo.mean.toFixed(4)}</span> scene units from where it `
      + `is, and disagree with the scene about whether there is anything there at all on `
      + `<span class="bold">${fmt.pc(geo.frac)}</span> of the frame.`;
  };

  attachOrbit(host, state, () => render());
  slider.addEventListener('input', () => { state.kIdx = +slider.value; render(); });

  const toggles = document.querySelector('#blob-toggles');
  toggles.innerHTML = '';
  const b = document.createElement('button');
  b.type = 'button';
  const paint = () => {
    b.textContent = state.fill ? 'outlines filled' : 'outlines empty';
    b.className = state.fill ? 'atlas-btn' : 'atlas-btn ghost';
  };
  b.addEventListener('click', () => { state.fill = !state.fill; paint(); render(); });
  paint();
  toggles.appendChild(b);
  [['a held-out camera', data.cameras.inside[0]], ['below the band', data.cameras.outside[0]]]
    .forEach(([label, c]) => {
      const j = document.createElement('button');
      j.type = 'button';
      j.className = 'atlas-btn ghost';
      j.textContent = `go to ${label}`;
      j.addEventListener('click', () => { state.az = c.az; state.el = c.el; render(); });
      toggles.appendChild(j);
    });

  render();
}

/* The storage question, asked properly: a full covariance is six numbers where
   a sphere is one, so the fair comparison is not blob against blob but number
   against number. */
export function initAnisoWidget(data, sheets) {
  const S = data.splat;
  const node = document.querySelector('#aniso-chart');
  d3.select(node).selectAll('*').remove();
  /* The right margin holds three direct labels instead of a legend box, so it
     is sized by the longest of them: at 136 the last "t" of "spheres, same
     budget" fell 12.7px outside the svg and was cut off by the container. */
  const c = makeChart(node, { width: 620, height: 400, margin: { top: 34, right: 158, bottom: 56, left: 66 } });
  const byKey = Object.fromEntries(S.ablations.map((a) => [a.key, a]));
  const pts = S.counts.map((r) => ({ x: r.dof, y: r.inside, n: r.n }));
  const extras = [
    { key: 'iso', label: 'spheres, same count', colour: 'var(--cosmos)' },
    { key: 'iso_matched', label: 'spheres, same budget', colour: 'var(--anchor)' },
    { key: 'no_viewdep', label: 'no direction', colour: 'var(--smile)' },
  ].map((e) => ({ ...e, ...byKey[e.key] }));
  const xs = pts.map((p) => p.x).concat(extras.map((e) => e.dof));
  const ys = pts.map((p) => p.y).concat(extras.map((e) => e.inside));
  const x = d3.scaleLog().domain([Math.min(...xs) * 0.8, Math.max(...xs) * 1.25]).range([0, c.w]);
  const y = d3.scaleLinear().domain([Math.min(...ys) - 1, Math.max(...ys) + 1]).range([c.h, 0]).nice();
  /* The whole ladder spans less than a decade. An exponent-rounded tick list
     collapses to the same value three times over, and d3's own log ticks put
     "50,000" and "60,000" on top of each other, so the values are the 1-2-5
     ladder filtered to the domain: never crowded, always round. */
  const [xa, xb] = x.domain();
  const xv = [];
  for (let e = -1; e <= 9; e++) {
    [1, 2, 5].forEach((m) => {
      const v = m * 10 ** e;
      if (v >= xa && v <= xb) xv.push(v);
    });
  }
  drawGrid(c.g, x, y, c.w, c.h, { xValues: xv, yTicks: 5 });
  drawAxes(c.g, x, y, c.w, c.h, { xValues: xv, xFmt: d3.format(','), yTicks: 5 });
  axisLabels(c.g, c.w, c.h, { x: 'numbers stored', y: 'held-out PSNR' });
  c.g.append('path').datum(pts.map((p) => [x(p.x), y(p.y)])).attr('d', d3.line())
    .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
  c.g.selectAll('circle.p').data(pts).join('circle').attr('class', 'p')
    .attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y)).attr('r', 5).attr('fill', 'var(--primary)');
  /* seriesLabel rather than annotate: a font size set as an attribute on a
     node whose class already declares one does nothing at all */
  pts.forEach((p) => {
    seriesLabel(c.g, x(p.x), y(p.y) - 10, `${p.n}`, { anchor: 'middle', colour: 'var(--primary)' });
  });
  seriesLabel(c.g, x(pts[pts.length - 1].x), y(pts[pts.length - 1].y) + 24, 'ellipsoids',
    { anchor: 'end', colour: 'var(--primary)' });
  extras.forEach((e) => {
    c.g.append('path').attr('d', d3.symbol(d3.symbolDiamond, 110))
      .attr('transform', `translate(${x(e.dof)},${y(e.inside)})`)
      .attr('fill', 'none').attr('stroke', e.colour).attr('stroke-width', 2.4);
  });
  spread(extras.map((e) => ({ y: y(e.inside) + 4, label: e.label, colour: e.colour, yMax: c.h })), 14)
    .forEach((m) => {
      seriesLabel(c.g, c.w + 6, m.y, m.label, { colour: m.colour });
    });
  c.svg.append('text').attr('class', 'chart-title')
    .attr('x', c.width / 2).attr('y', 20).attr('text-anchor', 'middle')
    .text('quality against storage, not against blob count');

  const cfg = data.sheets.blobs;
  drawTiles(document.querySelector('#aniso-sheet'), sheets.blobs.tiles, cfg.labels, cfg.tile, { max: 100 });

  const iso = byKey.iso;
  const isom = byKey.iso_matched;
  const aniso = byKey.aniso;
  const rec = byKey.no_recycle;
  /* Every one of these is a signed difference, so every one of them is a
     direction as well as a number and gets said in words rather than left as a
     minus sign in front of a quantity that was introduced as a gain. */
  const lead = (a, b, an, bn) => (a >= b
    ? `${an} ahead by ${(a - b).toFixed(2)} dB`
    : `${bn} ahead by ${(b - a).toFixed(2)} dB`);
  const worth = (v) => (Math.abs(v) < 0.005
    ? 'changes nothing at all'
    : v > 0 ? `is worth ${v.toFixed(2)} dB` : `costs ${(-v).toFixed(2)} dB`);
  document.querySelector('#aniso-readout').innerHTML =
    `At the same count, ${fmt.n0(aniso.n)} blobs, ellipsoids score ${fmt.db(aniso.inside)} and spheres `
    + `${fmt.db(iso.inside)}: <span class="bold">${lead(aniso.inside, iso.inside, 'ellipsoids', 'spheres')}</span>, `
    + `for ${fmt.n0(aniso.dof - iso.dof)} extra numbers. Given those numbers back as `
    + `${fmt.n0(isom.n - iso.n)} more spheres instead, the sphere cloud reaches `
    + `<span class="bold">${fmt.db(isom.inside)}</span> from ${fmt.n0(isom.dof)} numbers against the `
    + `ellipsoids' ${fmt.n0(aniso.dof)}, so at equal storage it is `
    + `<span class="bold">${lead(aniso.inside, isom.inside, 'ellipsoids', 'spheres')}</span>. `
    + `Recycling the transparent blobs ${worth(aniso.inside - rec.inside)}, and keeping the `
    + `direction-dependent colour term ${worth(aniso.inside - byKey.no_viewdep.inside)}.`;
}
