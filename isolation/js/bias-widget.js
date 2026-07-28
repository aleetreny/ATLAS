/* Where the random cuts leave a mark.
 *
 * One gaussian blob, nothing else in the picture, and the forest scored at the
 * same distance from the centre in seven directions. Every cut is vertical or
 * horizontal, so a point straight out along an axis is shielded by everything
 * the blob has at that height, and a point out along the diagonal is not. If
 * the detector had no preferred direction, each curve here would be flat.
 */
import { makeChart, drawAxes, axisLabels, drawGrid, spread } from '../../assets/js/chart.js';
import { seriesLabel, legend } from './draw.js';

export function initBiasWidget(data) {
  const R = data.forest.radial;
  let idx = R.rows.findIndex((r) => r.r === 4.0);
  if (idx < 0) idx = R.rows.length - 1;

  const chart = makeChart('#bias-chart', {
    width: 640, height: 330, margin: { top: 60, right: 110, bottom: 56, left: 62 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#bias-readout');
  const slider = d3.select('#bias-r')
    .attr('min', 0).attr('max', R.rows.length - 1).attr('step', 1).property('value', idx);

  const x = d3.scaleLinear().domain([0, 90]).range([0, w]);
  const allScores = R.rows.flatMap((r) => r.scores);
  const y = d3.scaleLinear()
    .domain([Math.min(...allScores) - 0.02, Math.max(...allScores) + 0.02]).range([h, 0]);
  const line = d3.line().x((d, i) => x(R.angles[i])).y((d) => y(d));

  function render() {
    idx = +slider.property('value');
    const row = R.rows[idx];
    g.selectAll('*').remove();
    drawGrid(g, x, y, w, h, { xValues: R.angles, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: R.angles, yTicks: 5 });
    axisLabels(g, w, h, { x: 'direction from the centre, in degrees', y: 'the forest\'s score' });

    R.rows.forEach((r, i) => {
      if (i === idx) return;
      g.append('path').attr('d', line(r.scores))
        .attr('fill', 'none').attr('stroke', 'var(--squidink)')
        .attr('stroke-width', 1).attr('opacity', 0.18);
    });
    g.append('path').attr('d', line(row.scores))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
    g.selectAll('circle.pt').data(row.scores).join('circle').attr('class', 'pt')
      .attr('cx', (d, i) => x(R.angles[i])).attr('cy', (d) => y(d)).attr('r', 4)
      .attr('fill', 'var(--primary)').attr('stroke', 'var(--paper)').attr('stroke-width', 1.2);

    /* The two directions the sentence below is about, named on the picture and
       pushed apart vertically: at the small radii the curve is nearly flat, so
       both labels want the same height and their boxes cross. */
    const marks = [[0, 'straight out along an axis'], [45, 'out along the diagonal']]
      .map(([a, label]) => ({ a, label, v: row.scores[R.angles.indexOf(a)] }))
      .map((d) => ({ ...d, y: y(d.v) - 12, yMax: h }));
    spread(marks, 17).forEach((d) => {
      seriesLabel(g, x(d.a) + (d.a === 0 ? 8 : 0), d.y, d.label,
        { size: 10.5, anchor: d.a === 0 ? 'start' : 'middle', opacity: 0.9 });
    });
    legend(g, [
      { label: `at ${row.r} from the centre`, shape: 'line', colour: 'var(--primary)' },
      { label: 'the other distances', shape: 'line', colour: 'var(--stone)' },
    ], { x0: 2, y0: -34, gap: 230 });

    d3.select('#bias-r-label').text(row.r);
    const flat = R.rows.filter((r) => r.r >= 4.0);
    readout.html(
      `<span class="slider-label">distance from the centre <span class="value">${row.r}</span></span> `
      + `At this distance the forest scores ${row.axis.toFixed(5)} straight out along an axis and `
      + `${row.diag.toFixed(5)} along the diagonal, `
      + (Math.abs(row.gap) < 0.005
        ? `which is as near to the same as makes no difference.`
        : `so the ${row.gap > 0 ? 'diagonal' : 'axis'} is ahead by `
          + `${Math.abs(row.gap).toFixed(5)} at the same distance from the same blob.`)
      + ` Push the slider out and the two part company for good: from `
      + `${flat[0].r} onwards the axis score does not move at all `
      + `(${R.axis_flat.toFixed(5)} in total across ${flat.length} distances) while the diagonal `
      + `climbs another ${R.diag_climb.toFixed(5)}. `
      + `<span class="bold">A point that walks away along an axis stops looking any stranger.</span> `
      + `That is not a property of the data, which is a plain round gaussian cloud of `
      + `${R.n.toLocaleString('en-US')} points: it is the shape of the cuts.`
    );
  }

  slider.on('input', render);
  render();
  return { render };
}
