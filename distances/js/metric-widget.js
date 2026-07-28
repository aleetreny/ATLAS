/* Six definitions of "how far apart", one dataset, and the question of which of
 * them describe a geometry at all.
 *
 * Both panels are computed here, from the 178 by 13 matrix the page downloads:
 * the distance matrix, its double centring, its full 178-eigenvalue spectrum
 * and the two-dimensional map. About a fifth of a second per metric, so each
 * answer is cached the first time its button is pressed.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';
import { pairwise, classicalMDS, negativeMass, stress1, toRows } from '../../assets/js/embed.js';

const CULTIVAR = ['var(--primary)', 'var(--aqua)', 'var(--smile)'];

/* Lingoes' constant: add it to every squared distance and the most negative
 * eigenvalue arrives at zero, because adding c off the diagonal shifts every
 * eigenvalue of B by c/2 except the one the centring pins there. */
const lingoes = (values) => Math.max(0, -2 * values[values.length - 1]);

export function metricSummary(X, key) {
  const D = pairwise(X, key);
  const { coords, values } = classicalMDS(D, 2);
  let pos = 0;
  for (let i = 0; i < values.length; i++) if (values[i] > 0) pos += values[i];
  let dsum = 0;
  let cnt = 0;
  for (let i = 0; i < D.length; i++) {
    for (let j = i + 1; j < D.length; j++) {
      dsum += D[i][j];
      cnt += 1;
    }
  }
  return {
    D,
    coords,
    values,
    negMass: negativeMass(values),
    negCount: Array.from(values).filter((v) => v < -1e-9).length,
    stress: stress1(D, coords),
    evr2: (Math.max(values[0], 0) + Math.max(values[1], 0)) / pos,
    lingoes: lingoes(values),
    dMean: dsum / cnt,
  };
}

export function initMetricWidget(data, cache) {
  const rows = data.metrics.rows;
  const X = toRows(data.wine.matrix);
  const classes = data.wine.classes;

  const A = makeChart('#metric-chart', {
    width: 760, height: 348, margin: { top: 52, right: 18, bottom: 50, left: 58 },
  });
  const B = makeChart('#metric-chart', {
    width: 760, height: 274, margin: { top: 52, right: 18, bottom: 76, left: 58 },
  });
  const buttons = document.querySelector('#metric-buttons');
  const readout = document.querySelector('#metric-readout');
  const state = { key: 'euclidean' };

  const btns = {};
  rows.forEach((r) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (r.key === state.key ? '' : ' ghost');
    b.textContent = r.key;
    b.addEventListener('click', () => {
      state.key = r.key;
      render();
    });
    buttons.appendChild(b);
    btns[r.key] = b;
  });

  const dots = A.g.append('g');
  const titleA = A.g.append('text').attr('class', 'chart-annotation')
    .attr('x', A.w / 2).attr('y', -30).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  const bars = B.g.append('g');
  const titleB = B.g.append('text').attr('class', 'chart-annotation')
    .attr('x', B.w / 2).attr('y', -30).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  /* The legend lives under the axis, not above it: the caption above is two
     lines wide and the two were landing on top of each other. */
  const legend = B.g.append('g');

  function render() {
    const key = state.key;
    if (!cache[key]) cache[key] = metricSummary(X, key);
    const S = cache[key];
    const ref = rows.find((r) => r.key === key);

    /* ------------------------------------------------------------ the map */
    const { x, y } = equalScales(S.coords, A.w, A.h, { pad: 1.1 });
    drawGrid(A.g, x, y, A.w, A.h, { xTicks: 5, yTicks: 5 });
    drawAxes(A.g, x, y, A.w, A.h, { xTicks: 5, yTicks: 5 });
    axisLabels(A.g, A.w, A.h, { x: 'first coordinate', y: 'second coordinate' });
    dots.selectAll('circle').data(S.coords).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3.6)
      .attr('fill', (d, i) => CULTIVAR[classes[i]]).attr('opacity', 0.82);
    wrapLabel(titleA, `the 178 bottles placed by ${key} distance alone, `
      + `${(S.evr2 * 100).toFixed(1)}% of the positive eigenvalue mass`, A.w - 8);

    /* --------------------------------------------------- the whole spectrum */
    const K = 40;
    const vals = Array.from(S.values.slice(0, K - 8))
      .concat(Array.from(S.values.slice(S.values.length - 8)));
    const xs = d3.scalePoint().domain(d3.range(vals.length)).range([14, B.w - 14]);
    const lo = Math.min(0, d3.min(vals));
    const hi = Math.max(0, d3.max(vals));
    const ys = d3.scaleLinear().domain([lo * 1.15 - hi * 0.02, hi * 1.06]).range([B.h, 0]);
    const bw = Math.min(14, (B.w - 28) / vals.length - 2);
    drawGrid(B.g, xs, ys, B.w, B.h, { xTicks: 0, yTicks: 4 });
    drawAxes(B.g, xs, ys, B.w, B.h, { xTicks: 0, yTicks: 4 });
    axisLabels(B.g, B.w, B.h, { y: 'eigenvalue' });
    B.g.select('text.axis-label.x').remove();
    bars.selectAll('rect').data(vals).join('rect')
      .attr('x', (d, i) => xs(i) - bw / 2).attr('width', bw)
      .attr('y', (d) => (d >= 0 ? ys(d) : ys(0)))
      .attr('height', (d) => Math.max(1, Math.abs(ys(d) - ys(0))))
      .attr('fill', (d) => (d < -1e-9 ? 'var(--cosmos)' : 'var(--primary)'))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.5);
    let zero = bars.select('line.zero');
    if (zero.empty()) {
      zero = bars.append('line').attr('class', 'zero')
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1);
    }
    zero.attr('x1', 0).attr('x2', B.w).attr('y1', ys(0)).attr('y2', ys(0));
    /* The x axis here is not a number line: it is "the 32 biggest, then the 8
       smallest". Saying so beats leaving a tick-free axis to be guessed at. */
    B.g.selectAll('text.gap-note').data([0]).join('text').attr('class', 'gap-note chart-annotation')
      .attr('x', B.w / 2).attr('y', B.h + 46).attr('text-anchor', 'middle')
      .style('font-size', '11.5px').style('text-transform', 'none')
      .text('the 32 largest, then the 8 smallest, of 178');
    wrapLabel(titleB, S.negCount > 0
      ? `${S.negCount} of the 178 eigenvalues are negative: no set of points anywhere has these distances`
      : 'every eigenvalue is zero or positive: these distances really are somebody\'s coordinates',
    B.w - 8);

    legend.selectAll('*').remove();
    [['var(--primary)', 'positive'], ['var(--cosmos)', 'negative']].forEach(([c, t], i) => {
      legend.append('rect').attr('x', 8 + i * 96).attr('y', B.h + 60)
        .attr('width', 10).attr('height', 10).attr('fill', c);
      legend.append('text').attr('x', 22 + i * 96).attr('y', B.h + 69)
        .style('font-size', '11.5px').style('font-family', 'var(--font-mono)')
        .attr('fill', 'var(--squidink)').text(t);
    });

    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== key);

    const euclid = cache.euclidean || (cache.euclidean = metricSummary(X, 'euclidean'));
    const verdict = S.negMass < 1e-9
      ? `Nothing negative. This dissimilarity <span class="bold">is</span> a Euclidean distance, `
        + `so classical scaling is not approximating anything: it is inverting an exact recipe, and `
        + `the only loss is the ${(100 - S.evr2 * 100).toFixed(1)}% of the geometry that will not fit `
        + `on a page.`
      : `<span class="bold">${(S.negMass * 100).toFixed(2)}%</span> of the eigenvalue mass is negative, `
        + `spread over ${S.negCount} eigenvalues. There is no arrangement of 178 points, in two `
        + `dimensions or in a million, whose ${key} distances are these. Adding `
        + `<span class="bold">${S.lingoes.toFixed(2)}</span> to every squared distance would be the `
        + `smallest repair that makes one exist, against a mean distance of ${S.dMean.toFixed(2)}.`;

    readout.innerHTML = `
      <table class="dist-table">
        <tr><th>distance</th><th>negative mass</th><th>negative eigenvalues</th><th>stress in 2-D</th></tr>
        <tr>
          <td>${key}</td>
          <td><span class="value">${S.negMass < 1e-9 ? '0' : (S.negMass * 100).toFixed(2) + '%'}</span></td>
          <td>${S.negCount} of 178</td>
          <td><span class="value">${S.stress.toFixed(4)}</span></td>
        </tr>
      </table>
      <div class="dist-note">${ref ? '' : ''}${verdict}
      ${key === 'euclidean' ? '' : ` Its two-dimensional stress is ${S.stress.toFixed(4)} against the `
        + `Euclidean ${euclid.stress.toFixed(4)}, and if that is `
        + `${S.stress < euclid.stress ? 'lower' : 'higher'} it is worth noticing that a picture can `
        + `fit a table better precisely because the table was never a picture in the first place.`}
      </div>`;
  }

  render();
}
