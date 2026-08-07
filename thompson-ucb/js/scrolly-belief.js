/* The two internal states, at four moments of the same run.
 *
 * Left: what UCB believes, as an estimate with a bar on top of it. Right: what
 * Thompson believes, as a curve over every value the machine could have. Both
 * are recomputed here from the counts the generator recorded, so the bars and
 * the curves are the policy's actual state and not an illustration of one.
 */
import { makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { ucbIndex, betaPdf } from './ucbkit.js';

export function initBeliefScrolly(data) {
  const S = data.snapshots;
  const arms = S.arms;
  const k = arms.length;

  const chart = makeChart('#belief-chart', {
    width: 640, height: 420, margin: { top: 52, right: 24, bottom: 46, left: 96 },
  });
  const { g, w, h } = chart;
  const half = w / 2 - 18;
  const rowH = h / k;
  const x = d3.scaleLinear().domain([0, 1]).range([0, half]);
  const xr = d3.scaleLinear().domain([0, 1]).range([w - half, w]);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -32)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -14)
    .style('font-size', '11.5px');
  g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', h + 16)
    .style('font-size', '10.5px').text('estimate and bound');
  g.append('text').attr('class', 'chart-note').attr('x', w - half).attr('y', h + 16)
    .style('font-size', '10.5px').text('posterior over the truth');
  /* two lines: one line of "machine 3, truth 0.45" is 140 units against a left
     margin of 96, so it hangs off the canvas */
  arms.forEach((a, i) => {
    [`machine ${i + 1}`, `truth ${a}`].forEach((line, j) => {
      g.append('text').attr('class', 'chart-note').attr('x', -8)
        .attr('y', i * rowH + rowH / 2 - 2 + j * 16).attr('text-anchor', 'end')
        .style('font-size', '10.5px').text(line);
    });
  });

  const left = g.append('g');
  const right = g.append('g');

  function draw(t) {
    const U = S.ucb[String(t)];
    const T = S.thompson[String(t)];
    left.selectAll('*').remove();
    right.selectAll('*').remove();
    arms.forEach((truth, i) => {
      const y0 = i * rowH + 6;
      const bh = rowH - 20;
      /* UCB: the estimate as a bar, the bonus as a whisker on the end of it */
      const idx = ucbIndex(U.means[i], U.counts[i], t, data.constant.proved_c);
      left.append('rect').attr('x', 0).attr('y', y0).attr('width', x(U.means[i]))
        .attr('height', bh).attr('fill', 'var(--primary)').attr('fill-opacity', 0.75);
      left.append('line').attr('x1', x(U.means[i])).attr('x2', x(Math.min(idx, 1)))
        .attr('y1', y0 + bh / 2).attr('y2', y0 + bh / 2)
        .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2);
      left.append('line').attr('x1', x(Math.min(idx, 1))).attr('x2', x(Math.min(idx, 1)))
        .attr('y1', y0 + 2).attr('y2', y0 + bh - 2)
        .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2);
      left.append('line').attr('x1', x(truth)).attr('x2', x(truth))
        .attr('y1', y0).attr('y2', y0 + bh)
        .attr('stroke', '#8e9aa6').attr('stroke-width', 1.6).attr('stroke-dasharray', '3 3');
      left.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', y0 + bh + 9)
        .style('font-size', '9.5px').text(`${U.counts[i]} pulls`);

      /* Thompson: the density of the posterior, drawn to a common height */
      const pts = [];
      for (let s = 0; s <= 100; s++) {
        const v = s / 100;
        pts.push([v, betaPdf(v, T.alpha[i], T.beta[i])]);
      }
      const top = Math.max(...pts.map((p) => p[1]), 1e-9);
      right.append('path').attr('fill', 'var(--primary)').attr('fill-opacity', 0.3)
        .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6)
        .attr('d', d3.area().x((p) => xr(p[0]))
          .y0(y0 + bh).y1((p) => y0 + bh - (p[1] / top) * bh)(pts));
      right.append('line').attr('x1', xr(truth)).attr('x2', xr(truth))
        .attr('y1', y0).attr('y2', y0 + bh)
        .attr('stroke', '#8e9aa6').attr('stroke-width', 1.6).attr('stroke-dasharray', '3 3');
      right.append('text').attr('class', 'chart-note').attr('x', w - half + 2)
        .attr('y', y0 + bh + 9).style('font-size', '9.5px').text(`${T.counts[i]} pulls`);
    });
    title.text(`after ${t} pulls`);
    const uWide = arms.map((_, i) => ucbIndex(U.means[i], U.counts[i], t,
      data.constant.proved_c) - U.means[i]);
    sub.text(`the widest bound is ${Math.max(...uWide).toFixed(2)} and the narrowest `
      + `${Math.min(...uWide).toFixed(2)}`);
  }

  const handlers = {};
  S.marks.forEach((t, i) => { handlers[i] = () => draw(t); });
  draw(S.marks[0]);
  const ctrl = scrolly('#belief-scrolly .step', handlers);
  return { ...ctrl, draw };
}
