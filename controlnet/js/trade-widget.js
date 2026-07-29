/* Obedience and realism on one pair of axes, because the whole point is that
 * they do not peak in the same place.
 *
 * Two quantities with nothing in common get their own vertical scale, which is
 * a chart crime often enough that it is worth saying why it is not one here:
 * the reader is being asked where each curve turns, not how they compare in
 * size. The second view drops the dual axis entirely and plots one against the
 * other, which is the same data with no scaling decision in it at all.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initTradeWidget(data) {
  const S = data.sweep;
  const buttons = document.querySelector('#trade-buttons');
  const readout = document.querySelector('#trade-readout');
  const st = { view: 'both' };

  const chart = makeChart('#trade-chart', {
    width: 640, height: 360, margin: { top: 52, right: 78, bottom: 58, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const btns = {};
  for (const [key, text] of [['both', 'against the dial'],
    ['against', 'one against the other']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const bestObey = S.reduce((a, r) => (r.obeys > a.obeys ? r : a));
  const bestReal = S.reduce((a, r) => (Math.abs(r.mmd2) < Math.abs(a.mmd2) ? r : a));

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();
    g.selectAll('g.axis.y2').remove();
    g.selectAll('text.axis-label.y2').remove();

    if (st.view === 'both') {
      const x = d3.scaleLinear().domain([-0.4, d3.max(S, (r) => r.scale) + 0.4]).range([0, w]);
      const yo = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);
      const mvals = S.map((r) => Math.abs(r.mmd2));
      const ym = d3.scaleLog().domain([Math.min(...mvals) * 0.7, Math.max(...mvals) * 1.5])
        .range([h, 0]);
      drawGrid(g, x, yo, w, h, { xValues: S.map((r) => r.scale), yTicks: 5 });
      /* the scales include halves, and the default tick format rounds them, so
         the axis printed "1" twice for 0.5 and 1 */
      drawAxes(g, x, yo, w, h, {
        xValues: S.map((r) => r.scale), yTicks: 5, yFmt: d3.format('.0%'),
        xFmt: (v) => (Number.isInteger(v) ? String(v) : String(v)),
      });
      axisLabels(g, w, h, { x: 'guidance scale', y: 'makes what was asked for' });

      /* the second scale, on the right, drawn by hand and labelled in its own
         colour so nobody has to guess which curve it belongs to */
      const g2 = g.append('g').attr('class', 'axis y2').attr('transform', `translate(${w},0)`)
        .call(d3.axisRight(ym).tickValues([1e-3, 3e-3, 1e-2].filter(
          (v) => v >= ym.domain()[0] && v <= ym.domain()[1])).tickFormat(d3.format('.0e')));
      g2.selectAll('text').attr('fill', 'var(--squidink)');

      [['obeys', (r) => r.obeys, yo, 'var(--primary)'],
        ['looks real', (r) => Math.abs(r.mmd2), ym, 'var(--squidink)']]
        .forEach(([label, f, scale, colour], si) => {
          layer.append('path')
            .attr('d', d3.line().x((r) => x(r.scale)).y((r) => scale(f(r)))(S))
            .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
            .attr('stroke-dasharray', si === 1 ? '6 4' : null);
          layer.selectAll(`circle.s${si}`).data(S).join('circle').attr('class', `s${si}`)
            .attr('cx', (r) => x(r.scale)).attr('cy', (r) => scale(f(r))).attr('r', 4.5)
            .attr('fill', colour);
        });
      /* the two peaks, which are the finding */
      [[bestObey, (r) => yo(r.obeys), 'var(--primary)', 'most obedient'],
        [bestReal, (r) => ym(Math.abs(r.mmd2)), 'var(--squidink)', 'most real']]
        .forEach(([r, fy, colour, name], si) => {
          layer.append('line').attr('x1', x(r.scale)).attr('x2', x(r.scale))
            .attr('y1', fy(r)).attr('y2', h)
            .attr('stroke', colour).attr('stroke-width', 1.2).attr('stroke-dasharray', '3 3')
            .attr('opacity', 0.75);
          layer.append('text').attr('class', 'chart-note')
            .attr('x', x(r.scale) + (si === 0 ? 7 : -7)).attr('y', h - 8 - si * 16)
            .attr('text-anchor', si === 0 ? 'start' : 'end')
            .style('font-size', '11.5px').attr('fill', colour).text(`${name}: ${r.scale}`);
        });
      /* what real digits score on the same obedience measure */
      layer.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', yo(data.real.obeys)).attr('y2', yo(data.real.obeys))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.5).attr('stroke-dasharray', '7 4');
      /* below its own line and on the left, where the obedience curve is still
         near the floor: above it lands on that curve as it climbs */
      layer.append('text').attr('class', 'chart-note')
        .attr('x', 2).attr('y', yo(data.real.obeys) + 15).style('font-size', '11.5px')
        .attr('fill', 'var(--anchor)').text('the judge on REAL digits');
      wrapLabel(title, bestObey.scale === bestReal.scale
        ? 'both peak at the same setting on this model'
        : 'the two peaks are not the same setting', w - 8);
    } else {
      const x = d3.scaleLog()
        .domain([d3.min(S, (r) => Math.abs(r.mmd2)) * 0.7, d3.max(S, (r) => Math.abs(r.mmd2)) * 1.5])
        .range([0, w]);
      const y = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 4, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 4, yTicks: 5, yFmt: d3.format('.0%'), xFmt: d3.format('.0e') });
      axisLabels(g, w, h, { x: 'distance from real digits', y: 'makes what was asked for' });
      layer.append('path')
        .attr('d', d3.line().x((r) => x(Math.abs(r.mmd2))).y((r) => y(r.obeys))(S))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2);
      layer.selectAll('circle').data(S).join('circle')
        .attr('cx', (r) => x(Math.abs(r.mmd2))).attr('cy', (r) => y(r.obeys)).attr('r', 5)
        .attr('fill', 'var(--primary)');
      S.forEach((r) => {
        layer.append('text').attr('class', 'chart-note')
          .attr('x', x(Math.abs(r.mmd2))).attr('y', y(r.obeys) - 11).attr('text-anchor', 'middle')
          .style('font-size', '11px').attr('fill', 'var(--primary)').text(r.scale);
      });
      wrapLabel(title, 'the same seven settings, with no second scale to argue about', w - 8);
    }

    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>guidance</th><th>obeys</th><th>distance from real</th>
          <th>variety</th><th>inside the floor</th>
        </tr>
        ${S.map((r) => `<tr>
          <td>${r.scale === bestObey.scale || r.scale === bestReal.scale
    ? '<span class="bold">' + r.scale + '</span>' : r.scale}</td>
          <td><span class="value">${(r.obeys * 100).toFixed(1)}%</span></td>
          <td><span class="value">${r.mmd2}</span></td>
          <td>${r.variety}</td>
          <td>${r.resolves ? 'no' : 'yes'}</td>
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        ${bestObey.scale === bestReal.scale
    ? `On this model both peak at ${bestObey.scale}, so there is no trade to draw and the page
       says so rather than inventing one.`
    : `The most obedient setting is <span class="value">${bestObey.scale}</span> at
       ${(bestObey.obeys * 100).toFixed(1)}%, and the most realistic is
       <span class="value">${bestReal.scale}</span> at ${bestReal.mmd2}. They are not the same
       setting, and everything past the second one buys obedience with realism: by
       ${S[S.length - 1].scale} the samples are
       ${(Math.abs(S[S.length - 1].mmd2) / Math.abs(bestReal.mmd2)).toFixed(1)} times further from
       real digits than at their best.`}
        ${bestObey.obeys > data.real.obeys
    ? `And one number in this table should stop you: at ${bestObey.scale} the model produces the
       requested digit <span class="bold">${(bestObey.obeys * 100).toFixed(1)}%</span> of the
       time, while the same judge names the right digit for a REAL held out one only
       ${(data.real.obeys * 100).toFixed(1)}% of the time. Obeying better than reality is not a
       good sign: it means drawing the textbook version of each digit, and the variety column
       agrees, ${bestObey.variety} against ${data.real.variety} for real ones.`
    : `The best obedience here, ${(bestObey.obeys * 100).toFixed(1)}%, stays under what the judge
       scores on real digits (${(data.real.obeys * 100).toFixed(1)}%).`}
      </div>`;
  }

  render();
  return { render };
}
