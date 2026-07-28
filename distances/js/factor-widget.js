/* One variable with a volume knob, and two methods that react to it very
 * differently.
 *
 * Eight variables, two hidden factors, three hundred rows, and a truth that is
 * known because this file's generator wrote it. The slider changes exactly one
 * number: how loud the private noise on the first variable is. Everything else
 * (the shared part, the noise draw, the other seven noise levels) is fixed, so
 * whatever moves in the panels is caused by that one number and nothing else.
 *
 * Both fits run here. PCA is an eigendecomposition of the covariance; factor
 * analysis is the EM loop in fakit.js, which is the same twelve lines the
 * generator runs in numpy.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { covariance, toRows } from '../../assets/js/embed.js';
import { pcaLoadings, faEM, subspaceGap, buildDataset } from './fakit.js';

const rowNorm = (L, i) => Math.sqrt(L[i].reduce((s, v) => s + v * v, 0));

export function factorFit(data, sd) {
  const F = data.factor;
  const common = toRows(F.common);
  const noise = toRows(F.noise);
  const { X } = buildDataset(common, noise, F.base_psi_sd, F.loud, sd);
  const S = covariance(X);
  const truth = toRows(F.true_loadings);
  const { loadings: Lp } = pcaLoadings(X, F.k);
  const fa = faEM(S, F.k);
  const p = F.p;
  let totP = 0;
  let totF = 0;
  for (let i = 0; i < p; i++) {
    totP += rowNorm(Lp, i) ** 2;
    totF += rowNorm(fa.loadings, i) ** 2;
  }
  const psiTrue = Float64Array.from(F.base_psi_sd, (v, j) => (j === F.loud ? sd : v) ** 2);
  return {
    S,
    pca: Lp,
    fa: fa.loadings,
    psi: fa.psi,
    psiTrue,
    iters: fa.iters,
    pcaGap: subspaceGap(truth, Lp),
    faGap: subspaceGap(truth, fa.loadings),
    pcaLoudShare: (rowNorm(Lp, F.loud) ** 2) / totP,
    faLoudShare: (rowNorm(fa.loadings, F.loud) ** 2) / totF,
    trueNorms: Array.from({ length: p }, (_, i) => rowNorm(truth, i)),
    pcaNorms: Array.from({ length: p }, (_, i) => rowNorm(Lp, i)),
    faNorms: Array.from({ length: p }, (_, i) => rowNorm(fa.loadings, i)),
  };
}

export function initFactorWidget(data, cache) {
  const F = data.factor;
  const sds = F.sds;
  const chart = makeChart('#factor-chart', {
    width: 760, height: 400, margin: { top: 66, right: 24, bottom: 56, left: 96 },
  });
  const { g, w, h } = chart;
  const viewButtons = document.querySelector('#factor-view-buttons');
  const slider = document.querySelector('#factor-sd');
  const sdLabel = document.querySelector('#factor-sd-label');
  const readout = document.querySelector('#factor-readout');
  const state = { view: 'loadings', i: 0 };

  const btns = {};
  for (const [key, text] of [['loadings', 'what each method thinks the variables carry'],
    ['error', 'the price of the assumption']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === state.view ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      state.view = key;
      render();
    });
    viewButtons.appendChild(b);
    btns[key] = b;
  }

  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -46).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  const legend = g.append('g');

  function setLegend(items) {
    legend.selectAll('*').remove();
    items.forEach(([c, txt, kind], i) => {
      const x0 = 2 + i * 150;
      if (kind === 'ring') {
        legend.append('circle').attr('cx', x0 + 5).attr('cy', -9).attr('r', 5)
          .attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2);
      } else {
        legend.append('rect').attr('x', x0).attr('y', -14).attr('width', 10).attr('height', 10)
          .attr('fill', c);
      }
      legend.append('text').attr('x', x0 + 15).attr('y', -5)
        .style('font-size', '11.5px').style('font-family', 'var(--font-mono)')
        .attr('fill', 'var(--squidink)').text(txt);
    });
  }

  function render() {
    const sd = sds[state.i];
    if (!cache[sd]) cache[sd] = factorFit(data, sd);
    const R = cache[sd];
    sdLabel.textContent = sd.toFixed(1);
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== state.view);
    body.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').remove();

    if (state.view === 'loadings') {
      const names = Array.from({ length: F.p }, (_, i) => `variable ${i + 1}`);
      const y = d3.scalePoint().domain(names).range([6, h - 6]).padding(0.6);
      const maxV = Math.max(d3.max(R.pcaNorms), d3.max(R.faNorms), d3.max(R.trueNorms)) * 1.12;
      const x = d3.scaleLinear().domain([0, maxV]).range([0, w]);
      drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 0 });
      drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: F.p });
      axisLabels(g, w, h, { x: 'how much of this variable the model says is shared' });
      g.select('g.axis.y').selectAll('text')
        .style('font-size', '12px')
        .style('font-weight', (d, i) => (i === F.loud ? '700' : '400'));
      const bh = 9;
      names.forEach((nm, i) => {
        body.append('rect').attr('x', 0).attr('y', y(nm) - bh - 1)
          .attr('width', x(R.pcaNorms[i])).attr('height', bh)
          .attr('fill', 'var(--anchor)').attr('opacity', 0.9);
        body.append('rect').attr('x', 0).attr('y', y(nm) + 1)
          .attr('width', x(R.faNorms[i])).attr('height', bh)
          .attr('fill', 'var(--primary)').attr('opacity', 0.9);
        body.append('line').attr('x1', x(R.trueNorms[i])).attr('x2', x(R.trueNorms[i]))
          .attr('y1', y(nm) - bh - 4).attr('y2', y(nm) + bh + 4)
          .attr('stroke', 'var(--squidink)').attr('stroke-width', 2);
      });
      body.append('rect').attr('x', -92).attr('y', y(names[F.loud]) - 15)
        .attr('width', w + 92).attr('height', 30).attr('fill', 'var(--cosmos)')
        .attr('opacity', 0.08).lower();
      setLegend([['var(--anchor)', 'PCA'], ['var(--primary)', 'factor analysis'],
        ['var(--squidink)', 'the truth']]);
      wrapLabel(title, `variable 1 has its private noise turned up to ${sd.toFixed(1)}; `
        + 'the other seven have not moved', w - 8);
    } else {
      const x = d3.scalePoint().domain(sds).range([10, w - 10]);
      const all = data.factor.sweep.flatMap((r) => [r.pca_gap, r.fa_gap]);
      const lo = Math.max(1e-4, d3.min(all) * 0.7);
      const hi = d3.max(all) * 1.4;
      const y = d3.scaleLog().domain([lo, hi]).range([h, 0]);
      const ticks = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3].filter((v) => v >= lo && v <= hi);
      drawGrid(g, x, y, w, h, { xTicks: 0, yValues: ticks });
      drawAxes(g, x, y, w, h, { xTicks: sds.length, yValues: ticks, yFmt: d3.format('.3f') });
      axisLabels(g, w, h, { x: 'noise on variable 1', y: 'distance from the true factors' });
      const line = (key) => d3.line().x((r) => x(r.sd)).y((r) => y(Math.max(r[key], lo)));
      body.append('path').datum(data.factor.sweep).attr('fill', 'none')
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 2.4).attr('d', line('pca_gap'));
      body.append('path').datum(data.factor.sweep).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2.4).attr('d', line('fa_gap'));
      body.append('circle').attr('cx', x(sd)).attr('cy', y(Math.max(R.pcaGap, lo))).attr('r', 6)
        .attr('fill', 'var(--anchor)').attr('stroke', 'white').attr('stroke-width', 1.6);
      body.append('circle').attr('cx', x(sd)).attr('cy', y(Math.max(R.faGap, lo))).attr('r', 6)
        .attr('fill', 'var(--primary)').attr('stroke', 'white').attr('stroke-width', 1.6);
      setLegend([['var(--anchor)', 'PCA'], ['var(--primary)', 'factor analysis']]);
      wrapLabel(title, 'lower is better: how far each method\'s plane is from the '
        + 'one the data was built on', w - 8);
    }

    const ratio = R.pcaGap / Math.max(R.faGap, 1e-12);
    readout.innerHTML = `
      <table class="dist-table">
        <tr><th></th><th>distance from the truth</th><th>loading mass on variable 1</th><th>private noise it allows</th></tr>
        <tr><td>PCA</td>
          <td><span class="value">${R.pcaGap.toFixed(5)}</span></td>
          <td><span class="value">${(R.pcaLoudShare * 100).toFixed(2)}%</span></td>
          <td>none: there is no such thing in the model</td></tr>
        <tr><td>factor analysis</td>
          <td><span class="value">${R.faGap.toFixed(5)}</span></td>
          <td><span class="value">${(R.faLoudShare * 100).toFixed(2)}%</span></td>
          <td>estimates ${R.psi[F.loud].toFixed(2)} against a true ${R.psiTrue[F.loud].toFixed(2)}</td></tr>
      </table>
      <div class="dist-note">
        ${ratio > 2
    ? `PCA is now <span class="bold">${ratio.toFixed(1)} times</span> further from the truth than `
          + `factor analysis is, and the reason is in the second column: it has handed `
          + `${(R.pcaLoudShare * 100).toFixed(1)}% of its loading mass to a variable that is almost `
          + `entirely noise, because that variable has the most variance and variance is the only `
          + `thing PCA has ever looked at.`
    : `With the noise this quiet the two answers are almost the same (${R.pcaGap.toFixed(5)} against `
          + `${R.faGap.toFixed(5)}), which is what should happen: when every variable is about equally `
          + `noisy there is nothing for a noise model to know that a rotation does not.`}
        The EM took ${R.iters} iteration${R.iters === 1 ? '' : 's'} to converge.
      </div>`;
  }

  slider.addEventListener('input', () => {
    state.i = +slider.value;
    render();
  });
  render();
}
