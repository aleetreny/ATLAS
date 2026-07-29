/* What happens when the machines change halfway through.
 *
 * The best and the worst swap places at the halfway mark. Nothing tells the
 * policies. The interesting column is not the total, it is the second half.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const LABEL = {
  ucb: 'confidence bound',
  thompson: 'Thompson sampling',
  ucb_window: 'confidence bound, last 500 pulls only',
  thompson_window: 'Thompson, last 500 pulls only',
};

export function initDriftWidget(data) {
  const D = data.drift;
  const readout = document.querySelector('#drift-readout');
  const controls = d3.select('#drift-controls');
  let pick = 'thompson';

  const chart = makeChart('#drift-chart', {
    width: 640, height: 360, margin: { top: 48, right: 128, bottom: 54, left: 64 },
  });
  const { g, w, h, margin } = chart;
  const stride = D.rows[0].stride;
  const x = d3.scaleLinear().domain([0, D.horizon]).range([0, w]);
  const y = d3.scaleLinear().domain([0, Math.max(...D.rows.map((r) => r.final)) * 1.12])
    .range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  axisLabels(g, w, h, { x: 'pulls', y: 'regret so far', leftBudget: margin.left });
  g.append('line').attr('x1', x(D.switch)).attr('x2', x(D.switch)).attr('y1', 0).attr('y2', h)
    .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.8);
  g.append('text').attr('class', 'chart-note').attr('x', x(D.switch) - 6).attr('y', 14)
    .attr('text-anchor', 'end').style('font-size', '10.5px').attr('fill', 'var(--cosmos)')
    .text('they swap here');

  const line = (arr) => d3.line().x((_, i) => x(i * stride)).y((d) => y(d))(arr);
  D.rows.forEach((r) => {
    g.append('path').attr('class', `curve ${r.policy}`).attr('fill', 'none')
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.4).attr('opacity', 0.3)
      .attr('d', line(r.curve));
  });
  const front = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 3);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 4)`);
  D.rows.forEach((r, i) => {
    key.append('line').attr('x1', 0).attr('x2', 12).attr('y1', i * 32 + 4).attr('y2', i * 32 + 4)
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6).attr('opacity', 0.5);
    const words = LABEL[r.policy].split(' ');
    [words.slice(0, 2).join(' '), words.slice(2).join(' ')].filter(Boolean)
      .forEach((t, j) => {
        key.append('text').attr('class', 'chart-note').attr('x', 0)
          .attr('y', i * 32 + 16 + j * 10).style('font-size', '10px').text(t);
      });
  });

  function render() {
    const row = D.rows.find((r) => r.policy === pick);
    front.attr('d', line(row.curve));
    g.selectAll('path.curve').attr('opacity', function op() {
      return d3.select(this).classed(pick) ? 0 : 0.28;
    });
    title.text(`${LABEL[pick]}: ${row.before} before the swap, ${row.after} after`);
    sub.text(`the best and the worst machine trade places at pull ${D.switch.toLocaleString('en-US')}`);

    const plain = D.rows.filter((r) => !r.policy.includes('window'));
    const win = D.rows.filter((r) => r.policy.includes('window'));
    const worst = plain.reduce((a, b) => (b.after > a.after ? b : a), plain[0]);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>policy</th><th>first half</th><th>second half</th><th>total</th></tr>
        ${D.rows.map((r) => `
          <tr><td>${r.policy === pick ? '<span class="value">' : ''}${LABEL[r.policy]}${r.policy === pick ? '</span>' : ''}</td>
              <td>${r.before}</td><td>${r.after}</td><td>${r.final}</td></tr>`).join('')}
      </table>
      <div class="gen-note">
        Read the second column and then the third. Before the swap the two
        policies are in the order the last panel put them; after it, the one that
        was winning is the one that suffers most,
        <span class="value">${worst.after}</span> against
        ${plain.find((r) => r !== worst).after}. The cause is the same thing that
        made it win: it stopped exploring, so it has no way to notice. Keeping
        only the last ${D.window} pulls costs
        ${(win[0].before / plain[0].before).toFixed(1)} to
        ${(win[1].before / plain[1].before).toFixed(1)} times more regret in the
        first half, where nothing is changing, and repays it in the second. Which
        half you are in is not something the algorithm can tell you.
      </div>`;
  }

  controls.selectAll('button').data(D.rows.map((r) => r.policy)).join('button')
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
