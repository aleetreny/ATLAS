/* Regret against pulls, with the floor the theory puts under it.
 *
 * The straight line is not a fit: it is the exact Lai and Robbins constant for
 * these five machines, recomputed in the page from their probabilities, times
 * the logarithm of the pull count.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { boundConstant } from './ucbkit.js';

const LABEL = { ucb: 'upper confidence bound', thompson: 'Thompson sampling',
  eps: 'explore 10% of the time' };

export function initRegretWidget(data) {
  const R = data.regret;
  const keys = Object.keys(R);
  const readout = document.querySelector('#regret-readout');
  const controls = d3.select('#regret-controls');
  let pick = 'thompson';

  const chart = makeChart('#regret-chart', {
    width: 640, height: 370, margin: { top: 48, right: 124, bottom: 54, left: 64 },
  });
  const { g, w, h, margin } = chart;
  const stride = R[keys[0]].stride;
  const T = data.meta.horizon;
  const x = d3.scaleLog().domain([stride, T]).range([0, w]);
  const y = d3.scaleLinear().domain([0, Math.max(...keys.map((k) => R[k].final)) * 1.1])
    .range([h, 0]);
  const xt = [100, 1000, 10000].filter((v) => v >= stride && v <= T);
  drawGrid(g, x, y, w, h, { xValues: xt, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xValues: xt, yTicks: 5, xFmt: d3.format(',d') });
  axisLabels(g, w, h, { x: 'pulls', y: 'regret so far', leftBudget: margin.left });

  const C = boundConstant(data.meta.arms);   // recomputed, not read
  const bound = [];
  for (let t = stride; t <= T; t *= 1.1) bound.push([t, C * Math.log(t)]);
  bound.push([T, C * Math.log(T)]);
  g.append('path').attr('fill', 'none').attr('stroke', '#8e9aa6').attr('stroke-width', 1.8)
    .attr('stroke-dasharray', '5 4')
    .attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(bound));

  const line = (arr) => d3.line().x((_, i) => x(Math.max((i + 1) * stride, stride)))
    .y((d) => y(d))(arr);
  keys.forEach((k) => {
    g.append('path').attr('class', `curve ${k}`).attr('fill', 'none')
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4).attr('opacity', 0.32)
      .attr('d', line(R[k].curve));
  });
  const front = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 3);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 4)`);
  key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 6).attr('y2', 6)
    .attr('stroke', '#8e9aa6').attr('stroke-width', 1.8).attr('stroke-dasharray', '5 4');
  ['the eventual', 'floor for any', 'policy'].forEach((t, i) => {
    key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', 20 + i * 16)
      .style('font-size', '10px').text(t);
  });

  function render() {
    front.attr('d', line(R[pick].curve));
    g.selectAll('path.curve').attr('opacity', function op() {
      return d3.select(this).classed(pick) ? 0 : 0.3;
    });
    const slope = data.slope.rows.find((d) => d.policy === pick);
    title.text(`${LABEL[pick]}: regret ${R[pick].final} after ${T.toLocaleString('en-US')} pulls`);
    sub.text(`its slope against the logarithm is ${slope.slope}, `
      + `${slope.ratio} times the floor`);

    const sorted = keys.slice().sort((a, b) => R[a].final - R[b].final);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>policy</th><th>regret</th><th>spread</th><th>pulls on the best</th>
            <th>slope, in units of the floor</th></tr>
        ${sorted.map((k) => {
    const s = data.slope.rows.find((d) => d.policy === k);
    return `<tr><td>${k === pick ? '<span class="value">' : ''}${LABEL[k]}${k === pick ? '</span>' : ''}</td>
              <td>${R[k].final}</td><td>${R[k].std}</td>
              <td>${(R[k].share_best * 100).toFixed(1)}%</td>
              <td>${s.ratio}</td></tr>`;
  }).join('')}
      </table>
      <div class="gen-note">
        Thompson sampling regrets <span class="value">${R.thompson.final}</span>
        where the confidence bound regrets ${R.ucb.final}, a factor of
        ${(R.ucb.final / R.thompson.final).toFixed(1)}, and the reason is in the
        panel above this one rather than in this one. The dashed line is the
        exact Lai and Robbins constant for these five machines,
        ${C.toFixed(4)}, times the logarithm of the pull count: no reasonable
        policy can stay below it forever. Thompson's measured slope is
        ${data.slope.rows.find((d) => d.policy === 'thompson').ratio} times that
        constant, which is not a contradiction and is worth being careful about:
        the bound is a statement about the limit, and at
        ${T.toLocaleString('en-US')} pulls a policy can still be under it.
      </div>`;
  }

  controls.selectAll('button').data(keys).join('button')
    .attr('class', (d) => `atlas-btn${d === pick ? '' : ' ghost'}`)
    .text((d) => LABEL[d])
    .on('click', (_, d) => {
      pick = d;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q === pick ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render };
}
