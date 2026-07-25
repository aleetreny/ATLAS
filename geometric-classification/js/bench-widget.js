/* Six models on thirty real features, with the fold spread drawn.
 *
 * The bars are coloured by whether the boundary is allowed to curve, because
 * that is the only comparison the section is making, and the readout states the
 * gap between the best of each family next to the noise it has to clear.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initBenchWidget(data) {
  const rows = data.benchmark;
  const { g, w, h } = makeChart('#gb-chart', {
    width: 700,
    height: 330,
    margin: { top: 34, right: 70, bottom: 60, left: 140 },
  });

  const y = d3.scaleBand().domain(rows.map((r) => r.name)).range([0, h]).padding(0.3);
  const x = d3.scaleLinear().range([0, w]);

  const barsG = g.append('g');
  const cvG = g.append('g');
  const labelG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');
  g.append('g').attr('class', 'axis y');

  const state = { metric: 'acc' };
  const readout = document.querySelector('#gb-readout');
  const btnAcc = document.querySelector('#gb-acc');
  const btnTime = document.querySelector('#gb-time');

  const colour = (r) => (r.linear ? 'var(--primary)' : 'var(--cosmos)');

  function render() {
    const acc = state.metric === 'acc';
    const val = (r) => (acc ? r.cv_mean : r.seconds * 1000);
    const fmt = acc ? d3.format('.4f') : d3.format('.1f');
    const lo = acc ? Math.min(0.95, d3.min(rows, (r) => r.cv_mean - r.cv_sd) - 0.004) : 0;
    const hi = d3.max(rows, (r) => (acc ? r.cv_mean + r.cv_sd : r.seconds * 1000));
    x.domain([lo, hi * (acc ? 1.001 : 1.12)]);

    drawGrid(g, x, y, w, h, { xTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, xFmt: fmt });
    axisLabels(g, w, h, {
      x: acc ? 'accuracy, five-fold cross validation over all 569 biopsies' : 'milliseconds to fit once on 398 rows',
    });
    g.select('g.axis.y').call(d3.axisLeft(y).tickSize(0))
      .selectAll('text').style('font-family', 'var(--font-mono)').style('font-size', '11px');

    barsG.selectAll('rect').data(rows, (r) => r.name).join('rect')
      .attr('y', (r) => y(r.name))
      .attr('height', y.bandwidth())
      .attr('fill', colour)
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 0.9)
      .transition('bar').duration(420)
      .attr('x', x(lo))
      .attr('width', (r) => Math.max(1.5, x(val(r)) - x(lo)));

    /* The whisker only means anything on the accuracy view: there is no
     * cross-validated spread for a single timing, and drawing one would be
     * inventing a measurement. */
    cvG.selectAll('g.whisker').data(acc ? rows : [], (r) => r.name).join(
      (e) => {
        const grp = e.append('g').attr('class', 'whisker');
        grp.append('line').attr('class', 'bar');
        grp.append('line').attr('class', 'capL');
        grp.append('line').attr('class', 'capR');
        return grp;
      },
      (u) => u,
      (ex) => ex.remove()
    ).each(function (r) {
      const cy = y(r.name) + y.bandwidth() / 2;
      const a = x(Math.max(lo, r.cv_mean - r.cv_sd));
      const b = x(r.cv_mean + r.cv_sd);
      const s = d3.select(this);
      s.select('line.bar').transition('bar').duration(420)
        .attr('x1', a).attr('x2', b).attr('y1', cy).attr('y2', cy)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6);
      s.select('line.capL').transition('bar').duration(420)
        .attr('x1', a).attr('x2', a).attr('y1', cy - 5).attr('y2', cy + 5)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6);
      s.select('line.capR').transition('bar').duration(420)
        .attr('x1', b).attr('x2', b).attr('y1', cy - 5).attr('y2', cy + 5)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6);
    });

    labelG.selectAll('text').data(rows, (r) => r.name).join('text')
      .attr('class', 'chart-annotation')
      .attr('y', (r) => y(r.name) + y.bandwidth() / 2 + 4)
      .style('font-size', '0.64rem').style('text-transform', 'none')
      .attr('fill', 'var(--squidink)')
      .transition('bar').duration(420)
      .attr('x', (r) => x(val(r)) + 8)
      .text((r) => fmt(val(r)) + (acc ? '' : ' ms'));

    title.text(acc
      ? 'green: the boundary must be flat.  pink: it may curve.'
      : 'one fit on 398 rows and 30 features, in milliseconds');
    btnAcc.classList.toggle('ghost', !acc);
    btnTime.classList.toggle('ghost', acc);

    const lin = rows.filter((r) => r.linear);
    const ker = rows.filter((r) => !r.linear);
    const bestLin = lin.reduce((a, b) => (b.cv_mean > a.cv_mean ? b : a));
    const bestKer = ker.reduce((a, b) => (b.cv_mean > a.cv_mean ? b : a));
    const gap = bestKer.cv_mean - bestLin.cv_mean;
    const noise = Math.max(bestLin.cv_sd, bestKer.cv_sd);
    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5">` +
      `Best flat boundary: <span class="bold">${bestLin.name}</span> at ` +
      `<span class="value">${bestLin.cv_mean.toFixed(4)}</span>. ` +
      `Best curved one: <span class="bold">${bestKer.name}</span> at ` +
      `<span class="value">${bestKer.cv_mean.toFixed(4)}</span>. ` +
      `The kernel is worth <span class="value">${(gap * 100).toFixed(2)}</span> accuracy points, ` +
      `against a fold-to-fold standard deviation of <span class="value">${(noise * 100).toFixed(2)}</span> points. ` +
      `${gap < noise
        ? 'The gap is smaller than the noise, which is the honest way of saying there is no gap.'
        : 'The gap clears the noise, so on this data the curve is buying something real.'}` +
      `</div>`;
  }

  btnAcc.addEventListener('click', () => {
    state.metric = 'acc';
    render();
  });
  btnTime.addEventListener('click', () => {
    state.metric = 'time';
    render();
  });
  render();
}
