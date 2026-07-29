/* The reader's turn to try every angle. One slider sweeps the projection
 * direction through 180 degrees; for each position the page projects all 210
 * training points, finds the single most charitable threshold, scores the
 * held-out points, and keeps a record of the reader's best attempt. The
 * point of the widget is losing: the measured ceiling for ANY angle is
 * 0.878, and the dial makes that theorem personal.
 */
import { drawGrid, drawAxes, makeChart } from '../../assets/js/chart.js';
import { project, bestCut, histo } from './kdakit.js';

const C0 = 'var(--aqua)';
const C1 = 'var(--cosmos)';

export function initDialWidget(data) {
  const X = data.x_train;
  const y = data.y_train;
  const Xt = data.x_test;
  const yt = data.y_test;

  const wrap = d3.select('#dial-chart');
  wrap.style('display', 'grid')
    .style('grid-template-columns', 'minmax(0,1fr) minmax(0,1fr)')
    .style('gap', '0.5rem');
  const left = wrap.append('div').node();
  const right = wrap.append('div').node();

  const L = makeChart(left, { width: 360, height: 340, margin: { top: 24, right: 14, bottom: 40, left: 42 } });
  const lx = d3.scaleLinear().domain(d3.extent(X, (p) => p[0])).nice().range([0, L.w]);
  const ly = d3.scaleLinear().domain(d3.extent(X, (p) => p[1])).nice().range([L.h, 0]);
  drawGrid(L.g, lx, ly, L.w, L.h, { xTicks: 4, yTicks: 4 });
  drawAxes(L.g, lx, ly, L.w, L.h, { xTicks: 4, yTicks: 4 });
  L.g.append('g').selectAll('circle').data(X).join('circle')
    .attr('cx', (p) => lx(p[0])).attr('cy', (p) => ly(p[1]))
    .attr('r', 3).attr('fill', (p, i) => (y[i] ? C1 : C0))
    .attr('stroke', 'white').attr('stroke-width', 0.5).attr('opacity', 0.8);
  const dirLine = L.g.append('line')
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 2.6).attr('opacity', 0.85);
  const cutLine = L.g.append('line')
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4').attr('opacity', 0.7);

  const R = makeChart(right, { width: 360, height: 340, margin: { top: 24, right: 14, bottom: 40, left: 20 } });
  const histG = R.g.append('g');
  /* two 360-unit panels share the row on a phone, so 0.58rem rendered at
     4.3px; 0.7rem stays above the floor, and the shorter title fits it */
  const rTitle = R.g.append('text').attr('class', 'chart-annotation')
    .attr('x', R.w / 2).attr('y', -8).attr('text-anchor', 'middle').style('font-size', '0.7rem');

  const slider = document.querySelector('#angle-slider');
  const readout = document.querySelector('#angle-readout');
  let readerBest = 0;

  function render() {
    const deg = +slider.value;
    const theta = deg * Math.PI / 180;
    document.querySelector('#angle-label').textContent = deg;

    const z = project(X, theta);
    const cut = bestCut(z, y);
    const zt = project(Xt, theta);
    const predT = zt.map((v) => {
      const side = v > cut.cut ? 1 : 0;
      return cut.flip ? 1 - side : side;
    });
    let ok = 0;
    for (let i = 0; i < yt.length; i++) ok += predT[i] === yt[i] ? 1 : 0;
    const accT = ok / yt.length;
    readerBest = Math.max(readerBest, accT);

    /* direction through the data's centre, plus the perpendicular cut */
    const cx = d3.mean(X, (p) => p[0]);
    const cy = d3.mean(X, (p) => p[1]);
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    dirLine
      .attr('x1', lx(cx - 3.4 * c)).attr('y1', ly(cy - 3.4 * s))
      .attr('x2', lx(cx + 3.4 * c)).attr('y2', ly(cy + 3.4 * s));
    /* the cut is the line { p : p . d = cut } */
    const px = cut.cut * c;
    const py = cut.cut * s;
    cutLine
      .attr('x1', lx(px - 3.4 * -s)).attr('y1', ly(py - 3.4 * c))
      .attr('x2', lx(px + 3.4 * -s)).attr('y2', ly(py + 3.4 * c));

    const lo = Math.min(...z);
    const hi = Math.max(...z) + 1e-9;
    const NB = 30;
    const { b0, b1 } = histo(z, y, lo, hi, NB);
    const hx = d3.scaleLinear().domain([0, NB]).range([0, R.w]);
    const maxC = Math.max(...b0, ...b1, 1);
    const hyy = d3.scaleLinear().domain([0, maxC]).range([R.h, 0]);
    const bw = R.w / NB;
    histG.selectAll('rect.c0').data(b0).join('rect').attr('class', 'c0')
      .attr('x', (v, i) => hx(i) + 1).attr('width', bw - 2)
      .attr('y', (v) => hyy(v)).attr('height', (v) => R.h - hyy(v))
      .attr('fill', C0).attr('opacity', 0.62);
    histG.selectAll('rect.c1').data(b1).join('rect').attr('class', 'c1')
      .attr('x', (v, i) => hx(i) + 1).attr('width', bw - 2)
      .attr('y', (v) => hyy(v)).attr('height', (v) => R.h - hyy(v))
      .attr('fill', C1).attr('opacity', 0.62);
    const cutX = hx((cut.cut - lo) / (hi - lo) * NB);
    histG.selectAll('line.cut').data([1]).join('line').attr('class', 'cut')
      .attr('x1', cutX).attr('x2', cutX).attr('y1', 0).attr('y2', R.h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4');
    rTitle.text(`at ${deg}°, most charitable cut`);

    const ceiling = data.best_line.acc_test;
    readout.innerHTML =
      `<table class="kda-table">
         <tr><td>this angle, held-out accuracy</td><td><span class="value">${accT.toFixed(3)}</span></td>
             <td>your best so far</td><td><span class="value">${readerBest.toFixed(3)}</span></td>
             <td>measured ceiling (all 721 angles)</td><td><span class="value">${ceiling.toFixed(3)}</span></td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        Math.abs(accT - ceiling) < 0.006
          ? `You have found the ceiling. It is still ${ceiling.toFixed(3)}, because the moons interleave: ` +
            `every straight projection folds some of each crescent onto the other. No angle exists that ` +
            `does better, and that is not an opinion, it is a sweep. The kernel's 0.989 is on the other ` +
            `side of a different idea, not a better angle.`
          : `Every position projects all 210 points onto the black direction and grants the projection ` +
            `its best possible threshold (dashed). Chase the ceiling if you like: the sweep already ` +
            `measured all 721 angles and none beats ${ceiling.toFixed(3)}.`
      }</div>`;
  }

  slider.addEventListener('input', render);
  render();
}
