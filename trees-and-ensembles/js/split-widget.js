/* Every candidate threshold, with the impurity it removes.
 *
 * The reader can hunt for a better split than the one the tree chose, and will
 * not find it, which is a better way of establishing "greedy but locally
 * optimal" than saying so.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initSplitWidget(data) {
  const D = data.two_d;
  const byF = [D.candidates.filter((c) => c.f === 0), D.candidates.filter((c) => c.f === 1)];

  const { g, w, h } = makeChart('#split-chart', {
    width: 580,
    height: 440,
    margin: { top: 32, right: 26, bottom: 56, left: 66 },
  });

  const x = d3.scaleLinear().range([0, w]);
  /* Padded below zero so the "0.00" of this axis does not sit on the "0.00" of
   * the other one, right where the two meet. */
  const yTop = d3.max(D.candidates, (c) => c.drop) * 1.12;
  const y = d3.scaleLinear().domain([-yTop * 0.04, yTop]).range([h, 0]);

  const clip = 'split-clip';
  g.append('clipPath').attr('id', clip).append('rect').attr('x', -1).attr('y', -1)
    .attr('width', w + 2).attr('height', h + 2);
  const plot = g.append('g').attr('clip-path', `url(#${clip})`);

  const halo = plot.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6);
  const curve = plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3);
  const rule = plot.append('line').attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.8).attr('stroke-dasharray', '5 4');
  const dot = plot.append('circle').attr('r', 6)
    .attr('fill', 'var(--cosmos)').attr('stroke', 'white').attr('stroke-width', 2);
  const bestMark = plot.append('circle').attr('r', 8.5)
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2);
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const slider = document.querySelector('#split-i');
  const statsEl = document.querySelector('#split-stats');
  const btns = [document.querySelector('#split-f0'), document.querySelector('#split-f1')];
  const state = { f: 1 };

  const line = d3.line().x((c) => x(c.thr)).y((c) => y(c.drop));

  function render() {
    const rows = byF[state.f];
    slider.max = rows.length - 1;
    const i = Math.min(+slider.value, rows.length - 1);
    const c = rows[i];
    document.querySelector('#split-i-label').textContent = i;

    x.domain(d3.extent(rows, (r) => r.thr)).nice();
    drawGrid(g, x, y, w, h, { yTicks: 5 });
    drawAxes(g, x, y, w, h, { yTicks: 5, yFmt: d3.format('.2f') });
    axisLabels(g, w, h, {
      x: `${D.features[state.f].replace(/_/g, ' ')}, candidate threshold`,
      y: 'impurity removed, ΔG',
    });

    halo.attr('d', line(rows));
    curve.attr('d', line(rows));
    rule.attr('x1', x(c.thr)).attr('x2', x(c.thr));
    dot.attr('cx', x(c.thr)).attr('cy', y(c.drop));

    const best = rows.reduce((a, b) => (b.drop > a.drop ? b : a));
    const globalBest = D.candidates.reduce((a, b) => (b.drop > a.drop ? b : a));
    bestMark.attr('cx', x(best.thr)).attr('cy', y(best.drop));

    btns.forEach((b, k) => b.classList.toggle('ghost', k !== state.f));
    title.text(`${rows.length} candidate thresholds on this feature`);

    const behind = best.drop - c.drop;
    statsEl.innerHTML =
      `<table class="tr-table">
         <tr><th></th><th>left</th><th>right</th></tr>
         <tr><td>patients</td><td>${c.nl}</td><td>${c.nr}</td></tr>
         <tr><td>gini</td><td>${c.gl.toFixed(4)}</td><td>${c.gr.toFixed(4)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem">parent gini ` +
      `<span class="value">${D.first_split.gini_parent.toFixed(4)}</span> · this split removes ` +
      `<span class="value">${c.drop.toFixed(4)}</span></div>` +
      `<div class="slider-label" style="margin-top:.5rem;line-height:1.45">${
        Math.abs(c.drop - globalBest.drop) < 1e-9
          ? `Nothing on either feature does better. This is the question the tree asks first.`
          : behind < 1e-9
            ? `Best on this feature, but the other one offers ` +
              `<span class="value">${globalBest.drop.toFixed(4)}</span>, so the tree looks there instead.`
            : `<span class="value">${behind.toFixed(4)}</span> short of the best on this feature, and ` +
              `<span class="value">${(globalBest.drop - c.drop).toFixed(4)}</span> short of the best anywhere.`
      }</div>`;
  }

  slider.addEventListener('input', render);
  btns.forEach((b, k) =>
    b.addEventListener('click', () => {
      state.f = k;
      slider.value = Math.min(+slider.value, byF[k].length - 1);
      render();
    })
  );
  document.querySelector('#split-best').addEventListener('click', () => {
    const gb = D.candidates.reduce((a, b) => (b.drop > a.drop ? b : a));
    state.f = gb.f;
    slider.value = byF[gb.f].indexOf(gb);
    render();
  });
  render();
}
