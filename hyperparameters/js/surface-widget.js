/* The 961 cells, all of them, on both scores.
 *
 * Three views and they are not decoration: the first is what a search can see,
 * the second is what nobody sees, and the third is the difference between them,
 * which is the quantity the last section of the article is about. The band of
 * cells that are within the noise floor of the best is outlined rather than
 * shaded, because the point of it is a boundary and not a region: it says how
 * much of this surface is, as far as this measurement can tell, the answer.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';

const ACCENT = () => getComputedStyle(document.documentElement)
  .getPropertyValue('--primary').trim() || '#171799';

export function initSurfaceWidget(data) {
  const S = data.surface;
  const F = data.floor;
  const buttons = document.querySelector('#surface-buttons');
  const readout = document.querySelector('#surface-readout');
  const st = { view: 'cv', pick: S.best.at_cv.slice() };

  const chart = makeChart('#surface-chart', {
    width: 640, height: 430, margin: { top: 44, right: 120, bottom: 56, left: 66 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['cv', 'cross validated'], ['test', 'held out'],
    ['gap', 'the difference']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const nC = S.c.length;
  const nG = S.g.length;
  const cell = (i, j) => (st.view === 'cv' ? S.cv[i][j]
    : st.view === 'test' ? S.test[i][j] : S.cv[i][j] - S.test[i][j]);

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    const x = d3.scaleLinear().domain([S.g[0], S.g[nG - 1]]).range([0, w]);
    const y = d3.scaleLinear().domain([S.c[0], S.c[nC - 1]]).range([h, 0]);
    const bw = w / nG;
    const bh = h / nC;

    const vals = [];
    for (let i = 0; i < nC; i++) for (let j = 0; j < nG; j++) vals.push(cell(i, j));
    const lo = d3.min(vals);
    const hi = d3.max(vals);
    const colour = st.view === 'gap'
      ? d3.scaleDiverging(d3.interpolateRdBu).domain([hi, 0, lo])
      : d3.scaleSequential(d3.interpolateLab('#f1f3f3', ACCENT())).domain([lo, hi]);

    for (let i = 0; i < nC; i++) {
      for (let j = 0; j < nG; j++) {
        layer.append('rect')
          .attr('x', x(S.g[j]) - bw / 2).attr('y', y(S.c[i]) - bh / 2)
          .attr('width', bw + 0.6).attr('height', bh + 0.6)
          .attr('fill', colour(cell(i, j)))
          .style('cursor', 'pointer')
          .on('click', () => { st.pick = [i, j]; render(); });
      }
    }

    /* the cells this measurement cannot tell apart from the best one */
    if (st.view !== 'gap') {
      const src = st.view === 'cv' ? S.cv : S.test;
      const floor = st.view === 'cv' ? F.cv_sd : F.test_sd;
      const top = d3.max(src.map((row) => d3.max(row)));
      for (let i = 0; i < nC; i++) {
        for (let j = 0; j < nG; j++) {
          if (src[i][j] < top - floor) continue;
          layer.append('rect')
            .attr('x', x(S.g[j]) - bw / 2).attr('y', y(S.c[i]) - bh / 2)
            .attr('width', bw).attr('height', bh)
            .attr('fill', 'none').attr('stroke', 'var(--smile)').attr('stroke-width', 1.1);
        }
      }
    }

    const [pi, pj] = st.pick;
    layer.append('circle').attr('cx', x(S.g[pj])).attr('cy', y(S.c[pi])).attr('r', 6)
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2);

    drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 6 });
    axisLabels(g, w, h, { x: 'log2 of gamma', y: 'log2 of C' });

    /* the ramp lives outside the plot: a legend inside a heatmap competes with
       the data it is explaining */
    const steps = 40;
    const lx = w + 22;
    for (let s = 0; s < steps; s++) {
      const t = s / (steps - 1);
      layer.append('rect').attr('x', lx).attr('y', h - 20 - t * (h - 60))
        .attr('width', 14).attr('height', (h - 60) / steps + 1)
        .attr('fill', colour(lo + t * (hi - lo)));
    }
    [[hi, h - 20 - (h - 60)], [lo, h - 20]].forEach(([v, yy]) => {
      layer.append('text').attr('class', 'chart-note').attr('x', lx + 18)
        .attr('y', yy + 4).style('font-size', '11px')
        .text(st.view === 'gap' ? v.toFixed(3) : v.toFixed(3));
    });
    /* The ramp caption sits above the ramp and not beside it, and it is two
       words rather than a phrase: `.chart-note` runs about six units per
       character at this size, so "cv minus held out" is a hundred units in a
       right margin of a hundred and twenty, and the sweep caught it hanging out
       of the element even though it was inside the viewBox. */
    layer.append('text').attr('class', 'chart-note').attr('x', lx)
      .attr('y', h - 20 - (h - 60) - 16).style('font-size', '11px')
      .text(st.view === 'gap' ? 'difference' : 'accuracy');

    const cv = S.cv[pi][pj];
    const te = S.test[pi][pj];
    const bestCv = S.best.cv;
    const near = Math.abs(cv - bestCv) <= F.cv_sd;
    readout.innerHTML = `<div class="gen-note">C = 2<sup>${S.c[pi].toFixed(1)}</sup>, `
      + `gamma = 2<sup>${S.g[pj].toFixed(1)}</sup>. Cross validated `
      + `<span class="bold">${cv.toFixed(4)}</span>, held out `
      + `<span class="bold">${te.toFixed(4)}</span>, and the two differ by `
      + `${(cv - te >= 0 ? '+' : '')}${(cv - te).toFixed(4)}. The best cell on the `
      + `cross validated surface scores ${bestCv.toFixed(4)}, so this one is `
      + `${(bestCv - cv).toFixed(4)} below it, `
      + (near
        ? `which is inside the ${F.cv_sd.toFixed(4)} that the same cell moves when the `
          + `folds are drawn again (over ${F.repeats} redraws of one probed cell the score `
          + `ran from ${F.rows[0].cv_lo.toFixed(4)} to ${F.rows[0].cv_hi.toFixed(4)}): this `
          + `configuration and the winner are the same configuration as far as this `
          + `measurement can say.`
        : `against a fold to fold wobble of ${F.cv_sd.toFixed(4)}, so the difference `
          + `is real.`)
      + ` Click any cell to move the ring.</div>`;
  }

  render();
  return { render, state: st };
}
