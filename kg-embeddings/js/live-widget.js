/* One complex dimension of a trained model, turned by hand.
 *
 * RotatE puts every entity on the plane and every relation is a rotation about
 * the origin. That is a picture, and it is also the whole model: the score of
 * a triple is how far the head lands from the tail after the turn. The slider
 * is the angle, the marked position is the angle the fit chose, and the
 * distance printed underneath is the score function itself.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, equalScales } from '../../assets/js/chart.js';

export function initLiveWidget(data) {
  const W = data.widget;
  if (!W) return null;
  const slider = document.querySelector('#live-slider');
  const label = document.querySelector('#live-value');
  const buttons = document.querySelector('#live-buttons');
  const readout = document.querySelector('#live-readout');
  const note = document.querySelector('#live-note');

  const rels = W.relations;
  const st = { rel: 0 };

  const btns = [];
  rels.forEach((rel, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = rel.relation.replace(/_/g, ' ');
    b.addEventListener('click', () => { st.rel = i; render(); });
    buttons.appendChild(b);
    btns.push(b);
  });

  const chart = makeChart('#live-chart', {
    width: 640, height: 420, margin: { top: 40, right: 30, bottom: 56, left: 62 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const pts = Object.values(W.coords);
  const { x, y } = equalScales(pts.concat([[0, 0]]), w, h);

  function render() {
    btns.forEach((b, i) => b.classList.toggle('ghost', i !== st.rel));
    const rel = rels[st.rel];
    const steps = Number(slider.value);
    const theta = (steps / 24) * 2 * Math.PI - Math.PI;
    label.textContent = `${(theta * 180 / Math.PI).toFixed(0)}°`;

    clear();
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    plot.append('line').attr('x1', x(x.domain()[0])).attr('x2', x(x.domain()[1]))
      .attr('y1', y(0)).attr('y2', y(0)).attr('stroke', 'var(--stone)');
    plot.append('line').attr('x1', x(0)).attr('x2', x(0))
      .attr('y1', y(y.domain()[0])).attr('y2', y(y.domain()[1])).attr('stroke', 'var(--stone)');

    const rows = rel.pairs.slice(0, 4).filter(([hd, tl]) => W.coords[hd] && W.coords[tl]);
    let dist = 0;
    let bestDist = 0;
    rows.forEach(([hd, tl]) => {
      const H = W.coords[hd];
      const T = W.coords[tl];
      const c = Math.cos(theta);
      const s = Math.sin(theta);
      const R = [H[0] * c - H[1] * s, H[0] * s + H[1] * c];
      const cf = Math.cos(rel.angle);
      const sf = Math.sin(rel.angle);
      const F = [H[0] * cf - H[1] * sf, H[0] * sf + H[1] * cf];
      dist += Math.hypot(R[0] - T[0], R[1] - T[1]);
      bestDist += Math.hypot(F[0] - T[0], F[1] - T[1]);

      plot.append('path').attr('fill', 'none').attr('stroke', 'var(--stone)')
        .attr('stroke-width', 1.2).attr('stroke-dasharray', '3 3')
        .attr('d', d3.line()(d3.range(0, 41).map((k) => {
          const a = (k / 40) * theta;
          return [x(H[0] * Math.cos(a) - H[1] * Math.sin(a)),
            y(H[0] * Math.sin(a) + H[1] * Math.cos(a))];
        })));
      plot.append('circle').attr('cx', x(H[0])).attr('cy', y(H[1])).attr('r', 4)
        .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
      plot.append('circle').attr('cx', x(T[0])).attr('cy', y(T[1])).attr('r', 4.5)
        .attr('fill', 'var(--anchor)');
      plot.append('circle').attr('cx', x(R[0])).attr('cy', y(R[1])).attr('r', 4.5)
        .attr('fill', 'var(--primary)');
      plot.append('line').attr('x1', x(R[0])).attr('y1', y(R[1]))
        .attr('x2', x(T[0])).attr('y2', y(T[1]))
        .attr('stroke', 'var(--primary)').attr('stroke-width', 1).attr('stroke-dasharray', '2 2');
    });

    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('.2f'), yFmt: d3.format('.2f') });
    axisLabels(g, w, h, { x: 'real part', y: 'imaginary part' });
    g.append('text').attr('class', 'chart-note')
      .attr('x', w / 2).attr('y', -22).attr('text-anchor', 'middle').style('font-size', '12px')
      .text(`${rows.length} triples of "${rel.relation.replace(/_/g, ' ')}", in the first complex dimension`);
    [['hollow: where the head starts', 'var(--squidink)'],
      ['filled: where the turn lands it', 'var(--primary)'],
      ['dark: where the tail actually is', 'var(--anchor)']].forEach(([txt, colour], i) => {
      g.append('text').attr('class', 'chart-note')
        .attr('x', 4).attr('y', 14 + i * 13).style('font-size', '10.5px').style('fill', colour)
        .text(txt);
    });

    const fitted = (rel.angle * 180 / Math.PI).toFixed(0);
    const symmetric = (rel.symmetry ?? 0) > 0.5;
    const pcSym = (100 * (rel.symmetry ?? 0)).toFixed(0);
    const selfInverse = Math.abs(rel.angle) < 0.35
      || Math.abs(Math.abs(rel.angle) - Math.PI) < 0.35;
    readout.innerHTML =
      `Turning by <span class="value">${(theta * 180 / Math.PI).toFixed(0)}°</span> leaves the `
      + `heads <span class="value">${(dist / Math.max(rows.length, 1)).toFixed(3)}</span> from `
      + `their tails on average in this dimension. The angle the fit chose for `
      + `<span class="value">${rel.relation.replace(/_/g, ' ')}</span> is `
      + `<span class="value">${fitted}°</span>, which leaves them at `
      + `<span class="value">${(bestDist / Math.max(rows.length, 1)).toFixed(3)}</span>. `
      /* whether the relation is symmetric is measured in the graph and travels
         in the file; whether the angle came out self inverse is read off the
         fit. They are two separate facts and the sentence used to infer the
         first from the second, which made it right by construction. */
      + `${symmetric
        ? `This relation is symmetric in the data (${pcSym}% of its triples come with their `
          + `mirror), and a rotation that undoes itself has to be a half turn or none at all. `
          + `${selfInverse
            ? 'The angle it settled on is one of those.'
            : 'The angle it settled on is not, in this dimension: the prediction is about the '
              + 'whole rotation and there are more planes than the one on screen.'}`
        : `This relation is not symmetric (${pcSym}% of its triples come with their mirror), so `
          + `applying it twice must not bring you home. `
          + `${selfInverse
            ? 'The angle here is nonetheless close to zero or a half turn, which one plane out of '
              + 'sixteen is entitled to be.'
            : 'The angle it settled on is nowhere near zero or a half turn.'}`}`;

    note.textContent = 'The model has ' + (data.meta.dim / 2) + ' of these planes and turns each one '
      + 'by its own angle; this is the first. A score is the total distance over all of them, so a '
      + 'relation that needs to move things differently in different respects has room to.';
  }

  slider.addEventListener('input', render);
  render();
  return { render, state: st };
}
