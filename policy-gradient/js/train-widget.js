/* Four training runs and one sweep, on the same world.
 *
 * The four differ by one thing each, so the gap between two of them is what
 * that thing is worth: a baseline, reusing the batch, and clipping the reuse.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const VIEWS = [
  { key: 'curves', label: 'four runs' },
  { key: 'entropy', label: 'the entropy bonus' },
];
const LABEL = {
  reinforce: 'plain',
  baseline: 'minus each state\'s value',
  repeat: 'and reuse each batch eight times',
  ppo: 'and clip the reuse',
};
/* the legend is 120px wide, so its lines are written out rather than cut from
   LABEL by word count, which used to push "each batch eight times" off the svg */
const KEY_LINES = {
  reinforce: ['plain'],
  baseline: ['minus the value', 'of the state'],
  repeat: ['and reuse each', 'batch eight times'],
  ppo: ['and clip the reuse'],
};
const HEAD = {
  reinforce: 'plain',
  baseline: 'baseline',
  repeat: 'reuse',
  ppo: 'clip',
};

export function initTrainWidget(data) {
  const T = data.training;
  const readout = document.querySelector('#train-readout');
  const controls = d3.select('#train-controls');
  let view = 'curves';

  const chart = makeChart('#train-chart', {
    width: 640, height: 350, margin: { top: 48, right: 128, bottom: 56, left: 70 },
  });
  const { g, w, h, margin } = chart;
  const plot = g.append('g');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');
  const keys = Object.keys(T);

  function renderCurves() {
    const every = T[keys[0]].record_every * T[keys[0]].batch;
    const len = T[keys[0]].curve.length;
    const x = d3.scaleLinear().domain([0, len * every]).range([0, w]);
    const all = keys.flatMap((k) => T[k].curve).concat([data.optimal.J]);
    const y = d3.scaleLinear().domain([Math.min(...all) * 1.1, Math.max(...all) * 1.15])
      .range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, yFmt: d3.format('.2f') });
    axisLabels(g, w, h, { x: 'episodes', y: 'what the policy is worth',
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    plot.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(data.optimal.J))
      .attr('y2', y(data.optimal.J)).attr('stroke', '#8e9aa6').attr('stroke-width', 1.6)
      .attr('stroke-dasharray', '5 4');
    plot.append('text').attr('class', 'chart-note').attr('x', w - 2)
      .attr('y', y(data.optimal.J) + 14).attr('text-anchor', 'end')
      .style('font-size', '10.5px').text(`the best possible: ${data.optimal.J}`);
    const colours = ['#c9ced0', 'var(--anchor)', 'var(--primary)', 'var(--cosmos)'];
    /* plain and baseline land on top of each other, which is the result; drawn
       at equal width the first would simply vanish, so it is drawn wider and
       the thinner one sits inside it */
    const widths = [5.4, 2.2, 2.4, 2.4];
    keys.forEach((k, i) => {
      plot.append('path').attr('fill', 'none').attr('stroke', colours[i])
        .attr('stroke-width', widths[i])
        .attr('d', d3.line().x((_, j) => x(j * every)).y((d) => y(d))(T[k].curve));
    });
    key.selectAll('*').remove();
    keys.forEach((k, i) => {
      key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 40 + 6).attr('y2', i * 40 + 6)
        .attr('stroke', colours[i]).attr('stroke-width', widths[i]);
      KEY_LINES[k].forEach((t, j) => {
        key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 40 + 20 + j * 11)
          .style('font-size', '10px').text(t);
      });
    });
    title.text(`four runs, ${T[keys[0]].seeds} seeds each, batches of ${T[keys[0]].batch}, `
      + `${T[keys[0]].episodes} episodes`);
    sub.text(`the value of the policy, computed exactly at every point, not sampled`);

    /* one column per run rather than one row: five long headers across do not
       fit the width the readout gets */
    const cells = (f) => keys.map((k) => `<td>${f(T[k])}</td>`).join('');
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>each adds one thing</th>${keys.map((k) => `<th>${HEAD[k]}</th>`).join('')}</tr>
        <tr><td>value it reaches</td>${cells((d) => d.J)}</tr>
        <tr><td>spread between seeds</td>${cells((d) => d.J_spread)}</tr>
        <tr><td>samples clipped</td>${cells((d) => `${(d.clipped * 100).toFixed(1)}%`)}</tr>
        <tr><td>typical policy move</td>${cells((d) => d.kl)}</tr>
        <tr><td>worst policy move</td>${cells((d) => d.kl_max)}</tr>
      </table>
      <div class="gen-note">
        Read the gaps rather than the numbers. Adding a baseline moves the value
        from ${T.reinforce.J} to ${T.baseline.J}, which is nothing at this budget,
        and cuts the spread between seeds from ${T.reinforce.J_spread} to
        ${T.baseline.J_spread}, which is most of what it was for. Reusing each
        batch eight times instead of throwing it away moves it to
        <span class="value">${T.repeat.J}</span>, against a ceiling of
        ${data.optimal.J}: that is the entire jump. And clipping that reuse costs
        ${(T.repeat.J - T.ppo.J).toFixed(5)} here while cutting the worst policy
        jump from ${T.repeat.kl_max} to ${T.ppo.kl_max},
        ${(T.repeat.kl_max / T.ppo.kl_max).toFixed(1)} times smaller, by touching
        only ${(T.ppo.clipped * 100).toFixed(1)}% of the samples. On a world this
        small the insurance is not worth much; the thing it insures, reuse, is
        worth everything.
      </div>`;
  }

  function renderEntropy() {
    const rows = data.entropy.rows;
    const x = d3.scalePoint().domain(rows.map((d) => String(d.beta))).range([0, w]).padding(0.5);
    const y = d3.scaleLinear().domain([Math.min(...rows.map((d) => d.J)) * 1.2,
      Math.max(...rows.map((d) => d.J)) * 1.15]).range([h, 0]);
    const y2 = d3.scaleLinear().domain([0, data.entropy.max_entropy * 1.06]).range([h, 0]);
    drawGrid(g, x, y, w, h, { yTicks: 5 });
    drawAxes(g, x, y, w, h, { yTicks: 5, yFmt: d3.format('.2f') });
    axisLabels(g, w, h, { x: 'how much the bonus is worth', y: 'what the policy is worth',
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    const gT = plot.append('g').attr('transform', `translate(${w},0)`);
    gT.call(d3.axisRight(y2).ticks(4)).attr('class', 'axis y');
    /* the second axis takes the room a legend would want, so the two lines are
       labelled where they start instead, each in its own colour */
    plot.append('text').attr('class', 'chart-note').attr('transform',
      `translate(${w + 44}, ${h / 2}) rotate(90)`).attr('text-anchor', 'middle')
      .style('font-size', '10.5px').text('how spread the policy stays');
    /* the ceiling of the right axis, drawn rather than mentioned, and in the
       colour of the line it belongs to: a grey reference here would read as
       belonging to the left axis, which is the other scale */
    const yr = y2(data.entropy.max_entropy);
    plot.append('line').attr('x1', 0).attr('x2', w).attr('y1', yr).attr('y2', yr)
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.2).attr('stroke-opacity', 0.45)
      .attr('stroke-dasharray', '2 4');
    plot.append('text').attr('class', 'chart-note').attr('x', w - 2).attr('y', yr - 5)
      .attr('text-anchor', 'end').attr('fill', 'var(--anchor)').attr('fill-opacity', 0.75)
      .style('font-size', '10.5px')
      .text(`picking at random: ${data.entropy.max_entropy}`);
    plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2.6)
      .attr('d', d3.line().x((d) => x(String(d.beta))).y((d) => y(d.J))(rows));
    plot.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)')
      .attr('stroke-width', 2).attr('stroke-dasharray', '5 4')
      .attr('d', d3.line().x((d) => x(String(d.beta))).y((d) => y2(d.entropy))(rows));
    [['var(--primary)', 'what the policy is worth', y(rows[0].J) + 17],
      ['var(--anchor)', 'how spread it stays', y2(rows[0].entropy) - 10]]
      .forEach(([col, label, yy]) => {
        plot.append('text').attr('class', 'chart-note').attr('x', 6).attr('y', yy)
          .attr('fill', col).style('font-size', '10.5px').text(label);
      });
    key.selectAll('*').remove();
    title.text('paying the policy to stay undecided');
    sub.text(`${rows[0].seeds} seeds each, ${data.entropy.episodes} episodes per run, `
      + 'clipped reuse throughout');
    const best = rows.reduce((a, b) => (b.J > a.J ? b : a), rows[0]);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>bonus</th>${rows.map((d) => `<th>${d.beta}</th>`).join('')}</tr>
        <tr><td>value reached</td>
            ${rows.map((d) => `<td>${d === best ? '<span class="value">' : ''}${d.J}${d === best ? '</span>' : ''}</td>`).join('')}</tr>
        <tr><td>spread between seeds</td>${rows.map((d) => `<td>${d.spread}</td>`).join('')}</tr>
        <tr><td>policy spread</td>${rows.map((d) => `<td>${d.entropy}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        Here the bonus buys nothing and costs steadily: the best value is at
        <span class="value">${best.beta}</span> and every step up from there
        gives some of it away, down to ${rows[rows.length - 1].J} at
        ${rows[rows.length - 1].beta}, while the policy's spread climbs from
        ${rows[0].entropy} to ${rows[rows.length - 1].entropy} against
        ${data.entropy.max_entropy} for picking at random. That is a real result
        about this world and not a general one, and the reason is worth saying:
        this world has one goal, no local optimum worth escaping, and a policy
        that commits early commits to the right thing. The bonus is insurance
        against the opposite case, and insurance you did not need looks exactly
        like this.
      </div>`;
  }

  function render() {
    g.selectAll('g.axis').remove();
    g.selectAll('g.grid').remove();
    g.selectAll('text.axis-label').remove();
    plot.selectAll('*').remove();
    if (view === 'curves') renderCurves(); else renderEntropy();
  }

  controls.selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (_, d) => {
      view = d.key;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render };
}
