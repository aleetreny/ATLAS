/* Every pairwise slope, drawn. Seeing the distribution is the explanation:
 * contaminated pairs are numerous and far away, and the median does not care. */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { theilSen, median } from './mathkit.js';

export function initTheilSen(stars) {
  const pts = stars.points;

  const { g, w, h } = makeChart('#theilsen-chart', {
    width: 700,
    height: 300,
    margin: { top: 42, right: 26, bottom: 58, left: 30 },
  });

  const x = d3.scaleLinear().domain([-12, 12]).range([0, w]);
  const y = d3.scaleLinear().range([h - 30, 8]);

  const ticksG = g.append('g');
  const histG = g.append('g');
  const medianG = g.append('g');
  const titleT = g
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', w / 2)
    .attr('y', -18)
    .attr('text-anchor', 'middle');

  drawGrid(g, x, y, w, h - 30, { yTicks: 0 });
  axisLabels(g, w, h, { x: 'slope of the line through a pair of stars' });

  const statsEl = document.querySelector('#theilsen-stats');
  const state = { highlight: true, dropGiants: false };

  function pairsFor(active) {
    const out = [];
    for (let i = 0; i < active.length; i++) {
      for (let j = i + 1; j < active.length; j++) {
        const a = active[i];
        const b = active[j];
        if (a.x === b.x) continue;
        out.push({ m: (b.y - a.y) / (b.x - a.x), tainted: a.giant || b.giant });
      }
    }
    return out;
  }

  function render() {
    const active = state.dropGiants ? pts.filter((d) => !d.giant) : pts;
    const pairs = pairsFor(active);
    const ts = theilSen(active.map((d) => d.x), active.map((d) => d.y));

    // histogram, so the shape of the distribution is readable at 1000+ pairs
    const bins = d3.bin().domain(x.domain()).thresholds(90).value((d) => d.m)(
      pairs.filter((p) => p.m >= x.domain()[0] && p.m <= x.domain()[1])
    );
    y.domain([0, d3.max(bins, (b) => b.length) || 1]);

    histG
      .selectAll('rect')
      .data(bins)
      .join('rect')
      .attr('x', (b) => x(b.x0) + 0.5)
      .attr('width', (b) => Math.max(1, x(b.x1) - x(b.x0) - 1))
      .attr('y', (b) => y(b.length))
      .attr('height', (b) => y(0) - y(b.length))
      .attr('fill', (b) => {
        const bad = b.filter((p) => p.tainted).length;
        if (!state.highlight || !bad) return 'var(--stone)';
        return bad / b.length > 0.5 ? 'var(--smile)' : 'var(--stone)';
      })
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 0.4);

    // the median tick, the actual estimate
    medianG.selectAll('*').remove();
    medianG
      .append('line')
      .attr('x1', x(ts.slope)).attr('x2', x(ts.slope))
      .attr('y1', 0).attr('y2', y(0))
      .attr('stroke', 'var(--primary)').attr('stroke-width', 3);
    medianG
      .append('text')
      .attr('class', 'chart-annotation')
      .attr('x', x(ts.slope))
      .attr('y', -4)
      .attr('text-anchor', 'middle')
      .style('font-size', '0.66rem')
      .text(`median ${ts.slope.toFixed(2)}`);

    drawAxes(g, x, y, w, h - 30, { yTicks: 0 });
    g.select('g.axis.y').attr('opacity', 0);

    const tainted = pairs.filter((p) => p.tainted).length;
    titleT.text(`${d3.format(',')(pairs.length)} pairwise slopes from ${active.length} stars`);
    statsEl.innerHTML =
      `<div class="slider-label">pairs touching a giant: <span class="value">${d3.format(',')(tainted)}</span>` +
      ` of ${d3.format(',')(pairs.length)} (<span class="value">${((tainted / pairs.length) * 100).toFixed(0)}%</span>)</div>` +
      `<div class="slider-label">giants are <span class="value">${(pts.filter((d) => d.giant).length / pts.length * 100).toFixed(0)}%</span> of the stars, ` +
      `but they contaminate far more of the pairs. The median survives because that share is still under half.</div>` +
      `<div class="slider-label" style="margin-top:.4rem">theil-sen slope: <span class="value">${ts.slope.toFixed(3)}</span>` +
      (state.dropGiants ? '' : ` · least squares on the same data: <span class="value" style="color:var(--cosmos)">${stars.ref.ols.slope.toFixed(3)}</span>`) +
      `</div>`;
  }

  document.querySelector('#ts-highlight').addEventListener('change', (e) => {
    state.highlight = e.target.checked;
    render();
  });
  const dropBtn = document.querySelector('#ts-drop');
  dropBtn.addEventListener('click', () => {
    state.dropGiants = !state.dropGiants;
    dropBtn.textContent = state.dropGiants ? 'Put the giants back' : 'Drop the giants';
    render();
  });

  render();
}
