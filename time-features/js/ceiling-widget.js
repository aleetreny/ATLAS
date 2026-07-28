/* What a tree can and cannot say.
 *
 * A tree predicts the average of the training targets in a leaf, so its output
 * is bounded by the range of targets it was trained on. On a series that
 * trends, the twelve months being forecast contain values above everything it
 * has ever seen, and no amount of boosting can reach them: the line goes flat
 * at the ceiling. Fitting the differences instead moves the ceiling to a
 * quantity that is not trending, and the same model walks straight past it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initCeilingWidget(data) {
  const E = data.extrapolate;
  const labels = data.series.labels;
  const ARMS = [
    { key: 'levels', label: 'trained on the level' },
    { key: 'diff', label: 'trained on the change' },
    { key: 'both', label: 'both' },
  ];
  let current = 0;
  const readout = document.querySelector('#ceiling-readout');
  const controls = d3.select('#ceiling-controls');
  const chart = makeChart('#ceiling-chart', {
    width: 640, height: 380, margin: { top: 46, right: 116, bottom: 48, left: 64 },
  });
  const { g, w, h, margin } = chart;

  const nh = E.history.length;
  const nf = E.truth.length;
  const x = d3.scaleLinear().domain([0, nh + nf - 1]).range([0, w]);
  const all = E.history.concat(E.truth, E.levels.path, E.diff.path, [E.ceiling]);
  const y = d3.scaleLinear().domain([d3.min(all) - 0.8, d3.max(all) + 0.8]).range([h, 0]);
  const xt = [0, 12, 24, 36, 48, nh, nh + nf - 1];
  drawGrid(g, x, y, w, h, { xValues: xt, yTicks: 5 });
  drawAxes(g, x, y, w, h, {
    xValues: xt, yTicks: 5, xFmt: (i) => labels[E.origin - nh + i].slice(0, 4),
  });
  axisLabels(g, w, h, { y: 'parts per million', leftBudget: margin.left });

  g.append('path').datum(E.history).attr('fill', 'none').attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 1.5)
    .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d)));
  g.selectAll('circle.truth').data(E.truth).join('circle').attr('class', 'truth')
    .attr('cx', (d, i) => x(nh + i)).attr('cy', (d) => y(d)).attr('r', 3)
    .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
  g.append('line').attr('x1', 0).attr('x2', w + 6).attr('y1', y(E.ceiling)).attr('y2', y(E.ceiling))
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.4).attr('stroke-dasharray', '6 4');
  g.append('text').attr('class', 'chart-note').attr('x', w + 10).attr('y', y(E.ceiling) + 4)
    .style('font-size', '11px').attr('fill', 'var(--cosmos)').text('the ceiling');
  g.append('line').attr('x1', x(nh - 0.5)).attr('x2', x(nh - 0.5)).attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--squidink)').attr('stroke-dasharray', '4 4').attr('opacity', 0.6);
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');

  function render() {
    const arm = ARMS[current];
    const show = arm.key === 'both' ? ['levels', 'diff'] : [arm.key];
    const colours = { levels: 'var(--primary)', diff: 'var(--anchor)' };
    layer.selectAll('*').remove();
    show.forEach((k) => {
      const path = [E.history[nh - 1]].concat(E[k].path);
      layer.append('path').datum(path).attr('fill', 'none').attr('stroke', colours[k])
        .attr('stroke-width', 2.6)
        .attr('d', d3.line().x((d, i) => x(nh - 1 + i)).y((d) => y(d)));
      layer.selectAll(`circle.${k}`).data(E[k].path).join('circle').attr('class', k)
        .attr('cx', (d, i) => x(nh + i)).attr('cy', (d) => y(d)).attr('r', 3.1)
        .attr('fill', colours[k]);
      layer.append('text').attr('class', 'chart-note')
        .attr('x', w + 10).attr('y', y(E[k].path[nf - 1]) + 4).style('font-size', '11px')
        .attr('fill', colours[k]).text(k === 'levels' ? 'on the level' : 'on the change');
    });
    wrapLabel(title, `the twelve months after ${E.label}, from a boosted tree `
      + `${arm.label}`, w - 8);

    const maxLevels = Math.max(...E.levels.path);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>one month</th><th>a year</th><th>averaged over twelve</th></tr>
        <tr><td>trained on the level</td>
            <td>${E.levels.mase[0]}</td><td>${E.levels.mase[11]}</td>
            <td><span class="value">${E.levels.mase_mean}</span></td></tr>
        <tr><td>trained on the change</td>
            <td>${E.diff.mase[0]}</td><td>${E.diff.mase[11]}</td>
            <td><span class="value">${E.diff.mase_mean}</span></td></tr>
      </table>
      <div class="gen-note">
        The dashed red line is the largest value in the training data,
        ${E.ceiling} parts per million, and no tree trained on levels can print
        a number above it: the highest this forecast reaches is
        ${maxLevels.toFixed(4)}. ${E.over_ceiling} of the ${nf} months being
        forecast are above the ceiling, so those ${E.over_ceiling} are
        unreachable before the model has learned anything at all. Trained on the
        change instead, the same model with the same columns scores
        ${E.diff.mase_mean} against ${E.levels.mase_mean} over
        ${E.n_origins} origins, a factor of
        ${(E.levels.mase_mean / E.diff.mase_mean).toFixed(2)}, because the thing
        it is now bounded by is the size of a monthly change, and that is not
        going anywhere.
      </div>`;
  }

  controls.selectAll('button').data(ARMS).join('button')
    .attr('class', (d, i) => `atlas-btn${i === current ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (evt, d) => {
      current = ARMS.indexOf(d);
      controls.selectAll('button')
        .attr('class', (dd) => `atlas-btn${dd === d ? '' : ' ghost'}`);
      render();
    });

  render();
  return { render };
}
