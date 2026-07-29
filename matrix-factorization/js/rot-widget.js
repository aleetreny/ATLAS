/* Rotate the first two factors and watch nothing happen.
 *
 * The cloud of books does not move, because rotating the plane is a change of
 * axes and not a change of the model: every dot product between two books, and
 * therefore every prediction the model makes, is exactly what it was. What
 * does move is the list on the right, which is what a "factor 1 is young adult
 * fantasy" claim is made of. The readout prints the largest change in any of
 * the pairwise scores, which stays at the size of a rounding error the whole
 * way round.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, equalScales } from '../../assets/js/chart.js';

export function initRotWidget(data) {
  const items = data.widget.items;
  const V = data.widget.full;
  const slider = document.querySelector('#rot-slider');
  const label = document.querySelector('#rot-value');
  const readout = document.querySelector('#rot-readout');
  const note = document.querySelector('#rot-note');

  const chart = makeChart('#rot-chart', {
    width: 640, height: 420, margin: { top: 40, right: 30, bottom: 56, left: 62 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const pts = V.map((row) => [row[0], row[1]]);
  const { x, y } = equalScales(pts, w, h);

  /* the pairwise scores that must not move, sampled once so that the number in
     the readout is the same experiment at every angle */
  const sample = [];
  for (let i = 0; i < 60; i++) {
    for (let j = i + 1; j < 60; j++) sample.push([i, j]);
  }
  const baseline = sample.map(([i, j]) => V[i].reduce((a, v, c) => a + v * V[j][c], 0));

  function rotated(theta) {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    return V.map((row) => {
      const out = row.slice();
      out[0] = c * row[0] - s * row[1];
      out[1] = s * row[0] + c * row[1];
      return out;
    });
  }

  function render() {
    const deg = Number(slider.value);
    const theta = (deg * Math.PI) / 180;
    label.textContent = `${deg}°`;
    const R = rotated(theta);

    /* the largest change in any pairwise score, computed here rather than
       asserted: a rotation is orthogonal, so this is float noise or a bug */
    let worst = 0;
    sample.forEach(([i, j], n) => {
      const dot = R[i].reduce((a, v, c) => a + v * R[j][c], 0);
      worst = Math.max(worst, Math.abs(dot - baseline[n]));
    });

    clear();
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    plot.selectAll(null).data(V).enter().append('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 2.4)
      .attr('fill', 'var(--stone)');

    /* the moving axis: the direction the rotated first factor now points in */
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const span = Math.max(Math.abs(x.domain()[0]), Math.abs(x.domain()[1]));
    plot.append('line')
      .attr('x1', x(-span * c)).attr('y1', y(-span * s))
      .attr('x2', x(span * c)).attr('y2', y(span * s))
      .attr('stroke', 'var(--primary)').attr('stroke-width', 2);

    /* the five books furthest along it, which is what "this factor means X"
       is actually built from */
    const proj = V.map((row) => row[0] * c + row[1] * s);
    const top = d3.range(V.length).sort((a, b) => proj[b] - proj[a]).slice(0, 5);
    top.forEach((i) => {
      plot.append('circle').attr('cx', x(V[i][0])).attr('cy', y(V[i][1])).attr('r', 4)
        .attr('fill', 'var(--primary)');
    });

    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('.1f'), yFmt: d3.format('.1f') });
    axisLabels(g, w, h, { x: 'first factor', y: 'second factor' });

    const names = top.map((i) => items[i].title);
    readout.innerHTML =
      `At <span class="value">${deg}°</span>, the five books furthest along the moving axis are `
      + `<span class="value">${names.join('</span>, <span class="value">')}</span>. `
      + `The largest change in any of the ${sample.length.toLocaleString('en-US')} pairwise scores `
      + `between these books is <span class="value">${worst.toExponential(2)}</span>, which is the `
      + `size of a rounding error and not the size of an effect: the model has not changed at all. `
      + `Only the description of it has.`;

    note.textContent = deg === 0
      ? 'This is the factorisation as the fit left it. The axes are wherever the initialisation '
        + 'and the update order happened to put them.'
      : 'Same model, same predictions, different story about what the axes mean.';
  }

  slider.addEventListener('input', render);
  render();
  return { render, rotated };
}
