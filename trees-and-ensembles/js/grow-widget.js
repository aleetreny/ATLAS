/* Adding trees, and the curve that never turns back up.
 *
 * The readout states the contrast with boosting explicitly, because that is the
 * point of the section and the reader met the opposite curve four articles ago.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, tooltip } from '../../assets/js/chart.js';

export function initGrowWidget(data) {
  const rows = data.grow;
  const { g, w, h } = makeChart('#grow-chart', {
    width: 660,
    height: 380,
    margin: { top: 34, right: 90, bottom: 58, left: 84 },
  });

  const x = d3.scaleLog().domain([0.9, 460]).range([0, w]);
  const y = d3.scaleLinear().range([h, 0]);
  const tip = tooltip();

  const halo = g.append('path').attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 6.5);
  const curve = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3.2);
  const dotG = g.append('g');
  const endLabel = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w + 8).style('font-size', '0.62rem').style('text-transform', 'none')
    .attr('fill', 'var(--primary)');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const readout = document.querySelector('#grow-readout');
  const btnL = document.querySelector('#grow-logloss');
  const btnA = document.querySelector('#grow-acc');
  const state = { metric: 'logloss' };

  function render() {
    const ll = state.metric === 'logloss';
    const val = (r) => (ll ? r.logloss : r.test);
    const ext = d3.extent(rows, val);
    const pad = (ext[1] - ext[0]) * 0.12;
    y.domain(ll ? [0, ext[1] * 1.05] : [ext[0] - pad, Math.min(1, ext[1] + pad)]);

    /* Explicit tick values: a log axis starting at 0.9 renders that end as
     * "900m", which is both meaningless here and wide enough to collide with
     * the y axis. These are tree counts and should read as tree counts. */
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, yFmt: d3.format(ll ? '.2f' : '.3f') });
    g.select('g.axis.x').call(d3.axisBottom(x).tickValues([1, 3, 10, 30, 100, 400]).tickFormat(d3.format('d')).tickSizeOuter(0));
    axisLabels(g, w, h, {
      x: 'trees in the forest',
      y: ll ? 'log loss on the held-out biopsies' : 'accuracy on the held-out biopsies',
    });

    const line = d3.line().x((r) => x(r.n)).y((r) => y(val(r)));
    halo.attr('d', line(rows));
    curve.attr('d', line(rows));
    dotG.selectAll('circle').data(rows).join('circle')
      .attr('cx', (r) => x(r.n)).attr('cy', (r) => y(val(r))).attr('r', 4.4)
      .attr('fill', 'var(--primary)').attr('stroke', 'white').attr('stroke-width', 1.4)
      .on('mousemove', (event, r) => tip.show(
        `${r.n} tree${r.n === 1 ? '' : 's'}<br/>log loss ${r.logloss.toFixed(4)}<br/>accuracy ${r.test.toFixed(4)}`,
        event
      ))
      .on('mouseleave', () => tip.hide());
    endLabel.attr('y', y(val(rows[rows.length - 1])) + 4).text(ll ? 'log loss' : 'accuracy');

    title.text(ll ? 'it falls, it flattens, and it stays flat' : 'accuracy against the number of trees');
    btnL.classList.toggle('ghost', !ll);
    btnA.classList.toggle('ghost', ll);

    /* The claim is that the curve never turns back up. Measure the worst
     * upturn after the halfway point rather than asserting it. */
    const half = Math.floor(rows.length / 2);
    const tail = rows.slice(half);
    const worstUp = ll
      ? d3.max(tail, (r, i) => (i ? r.logloss - tail[i - 1].logloss : 0))
      : d3.max(tail, (r, i) => (i ? tail[i - 1].test - r.test : 0));
    const first = rows[0];
    const last = rows[rows.length - 1];
    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5">` +
      `From <span class="value">${ll ? first.logloss.toFixed(3) : first.test.toFixed(4)}</span> at one tree to ` +
      `<span class="value">${ll ? last.logloss.toFixed(4) : last.test.toFixed(4)}</span> at ${last.n}. ` +
      `Over the second half of this range the largest step in the wrong direction is ` +
      `<span class="value">${worstUp.toFixed(4)}</span>. ` +
      `A boosting curve on the same axes turns back upward and keeps going, because there every tree is ` +
      `fitted to the last one's mistakes. Here they are independent draws, so B is a compute budget and not ` +
      `a hyperparameter: pick the largest number you can afford and stop thinking about it.</div>`;
  }

  btnL.addEventListener('click', () => {
    state.metric = 'logloss';
    render();
  });
  btnA.addEventListener('click', () => {
    state.metric = 'acc';
    render();
  });
  render();
}
