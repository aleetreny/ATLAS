/* The same 178 bottles with every measurement replaced by which third of its
 * range it falls in. Thirteen categorical columns, no arithmetic left, and
 * three ways to cluster them.
 *
 * The button is the argument. Renumbering the categories is a decision nobody
 * remembers making, and it cannot change what the data means. K-modes cannot
 * see it: same partition, same cost, every time. K-means on the codes sees
 * nothing else.
 */
import { makeChart, drawAxes, drawGrid } from '../../assets/js/chart.js';
import { kmodes, kmeans, ari } from './relaxkit.js';

const LEVELS = 3;

export function initModesWidget(data) {
  const M = data.modes;
  const truth = data.wine.true;
  const codes = M.codes;
  const nCols = codes[0].length;

  const { g, w, h } = makeChart('#modes-chart', {
    width: 640, height: 300, margin: { top: 30, right: 30, bottom: 48, left: 170 },
  });
  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  const y = d3.scaleBand().domain(['kmodes', 'km_codes', 'km_onehot']).range([0, h]).padding(0.35);
  drawGrid(g, x, y, w, h, { xTicks: 5 });
  drawAxes(g, x, y, w, h, {
    xTicks: 5,
    yFmt: (k) => ({ kmodes: 'k-modes', km_codes: 'k-means on the codes', km_onehot: 'k-means on one-hot' }[k]),
  });
  g.append('text').attr('class', 'axis-label x')
    .attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle')
    .text('ari against the cultivars');

  /* the range each method covered over the 40 relabellings measured offline */
  const spans = {
    kmodes: [Math.min(...M.perm_rows.map((r) => r.modes)), Math.max(...M.perm_rows.map((r) => r.modes))],
    km_codes: M.km_range,
    km_onehot: [M.km_onehot.ari, M.km_onehot.ari],
  };
  g.append('g').selectAll('rect.span').data(Object.entries(spans)).join('rect')
    .attr('class', 'span')
    .attr('x', (d) => x(d[1][0])).attr('width', (d) => Math.max(2, x(d[1][1]) - x(d[1][0])))
    .attr('y', (d) => y(d[0]) - 5).attr('height', y.bandwidth() + 10)
    .attr('fill', 'var(--squidink)').attr('opacity', 0.1);

  const barG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const btn = document.querySelector('#modes-renumber');
  const btnReset = document.querySelector('#modes-reset');
  const readout = document.querySelector('#modes-readout');
  const state = { perm: -1, seen: [] };

  const oneHot = (rows) => rows.map((r) => {
    const v = new Array(nCols * LEVELS).fill(0);
    r.forEach((c, j) => { v[j * LEVELS + c] = 1; });
    return v;
  });

  function render() {
    const P = state.perm < 0 ? null : M.perms[state.perm];
    const rows = P ? codes.map((r) => r.map((c, j) => P[j][c])) : codes;

    const km = kmodes(rows, 3);
    /* thirty restarts, matching the generator: with ten, this fit lands in a
       different basin often enough to print a number outside the range the
       article quotes two paragraphs above */
    const codesRun = kmeans(rows.map((r) => r.map(Number)), 3, { restarts: 30 });
    const ohRun = kmeans(oneHot(rows), 3, { restarts: 30 });
    const vals = {
      kmodes: ari(truth, km.labels),
      km_codes: ari(truth, codesRun.labels),
      km_onehot: ari(truth, ohRun.labels),
    };
    if (!state.seen.some((s) => s.perm === state.perm)) {
      state.seen.push({ perm: state.perm, ...vals });
    }

    barG.selectAll('rect').data(Object.entries(vals)).join('rect')
      .attr('x', (d) => x(Math.min(0, d[1]))).attr('y', (d) => y(d[0]))
      .attr('width', (d) => Math.abs(x(d[1]) - x(0))).attr('height', y.bandwidth())
      .attr('fill', (d) => (d[0] === 'kmodes' ? 'var(--primary)' : 'var(--squidink)'))
      .attr('opacity', 0.85);
    barG.selectAll('text').data(Object.entries(vals)).join('text')
      .attr('class', 'chart-annotation')
      .attr('x', (d) => x(d[1]) + 8).attr('y', (d) => y(d[0]) + y.bandwidth() / 2 + 4)
      .style('font-size', '0.64rem')
      .text((d) => d[1].toFixed(3));

    title.text(state.perm < 0
      ? 'categories numbered low = 0, middle = 1, high = 2'
      : `relabelling ${state.perm + 1} of ${M.perms.length}: the same data, different numbers`);

    const seenCodes = state.seen.map((s) => s.km_codes);
    const seenModes = state.seen.map((s) => s.kmodes);
    readout.innerHTML =
      `<table class="rel-table">
        <tr><th>method</th><th>ARI now</th><th>over your ${state.seen.length} numbering${state.seen.length === 1 ? '' : 's'}</th>
            <th>over the 40 measured offline</th></tr>
        <tr><td>k-modes</td><td><span class="value">${vals.kmodes.toFixed(3)}</span></td>
            <td>${range(seenModes)}</td><td>${spans.kmodes[0].toFixed(3)} to ${spans.kmodes[1].toFixed(3)}</td></tr>
        <tr><td>k-means, codes</td><td><span class="value">${vals.km_codes.toFixed(3)}</span></td>
            <td>${range(seenCodes)}</td><td>${spans.km_codes[0].toFixed(3)} to ${spans.km_codes[1].toFixed(3)}</td></tr>
        <tr><td>k-means, one-hot</td><td><span class="value">${vals.km_onehot.toFixed(3)}</span></td>
            <td>${range(state.seen.map((s) => s.km_onehot))}</td><td>invariant by construction</td></tr>
      </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">` +
      `K-modes' total mismatch is <span class="bold">${km.cost}</span>, and it is `
      + `${km.cost === M.kmodes.cost ? 'the same number it was before you renumbered anything' : 'different, which would mean this page has a bug'}: `
      + `Hamming distance can tell you two bottles differ in a column, never by how much, `
      + `so the numbers themselves are not in the objective at all. `
      + `${state.perm < 0
        ? `There are ${M.n_relabellings.toLocaleString('en-US')} ways to number these 13 columns, and the one that came `
          + `with the data happens to be ordered, which is why k-means on the codes is doing so well right now. `
          + `Press renumber.`
        : `K-means on the codes is now measuring distance along an ordering somebody invented.`}` +
      `</div>`;
  }

  const range = (arr) => (arr.length
    ? `${Math.min(...arr).toFixed(3)} to ${Math.max(...arr).toFixed(3)}`
    : '');

  btn.addEventListener('click', () => {
    state.perm = (state.perm + 1) % M.perms.length;
    render();
  });
  btnReset.addEventListener('click', () => {
    state.perm = -1;
    render();
  });
  render();
}
