/* The two dials of a distillation loss, swept instead of guessed.
 *
 * Temperature decides how much of the teacher's distribution is visible to a
 * gradient; the mixing weight decides how much of the true label survives next
 * to it. Both are usually copied from a paper, and both are one sweep each.
 * The entropy of the teacher at each temperature is drawn on the same panel,
 * because it is the quantity the temperature is actually moving.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initTempWidget(data) {
  const D = data.distil;
  const buttons = document.querySelector('#temp-buttons');
  const readout = document.querySelector('#temp-readout');
  const st = { view: 'temp' };

  const chart = makeChart('#temp-chart', {
    width: 640, height: 360, margin: { top: 52, right: 74, bottom: 58, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['temp', 'the temperature'],
    ['mix', 'how much of the true label to keep']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const temps = D.temps;
  const tempRows = temps.map((t) => {
    const a = D.arms[`soft T=${t}`];
    return { t, mean: a.mean, sd: a.sd, entropy: D.entropies[String(t)] };
  });

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();
    const isTemp = st.view === 'temp';
    const rows = isTemp ? tempRows : D.mixes;
    const xv = (r) => (isTemp ? r.t : r.weight);

    const x = isTemp
      ? d3.scaleLog().domain([temps[0] * 0.85, temps[temps.length - 1] * 1.2]).range([0, w])
      : d3.scaleLinear().domain([0, 1]).range([0, w]);
    const vals = rows.flatMap((r) => [r.mean - r.sd, r.mean + r.sd]);
    const y = d3.scaleLinear().domain([d3.min(vals) - 0.006, d3.max(vals) + 0.006]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: rows.map(xv), yTicks: 5 });
    drawAxes(g, x, y, w, h, {
      xValues: rows.map(xv), yTicks: 5, yFmt: d3.format('.3f'),
      xFmt: isTemp ? (v) => String(v) : d3.format('.2f'),
    });
    axisLabels(g, w, h, {
      x: isTemp ? 'temperature' : 'weight on the soft target',
      y: 'student accuracy',
    });

    layer.append('path')
      .attr('d', d3.line().x((r) => x(xv(r))).y((r) => y(r.mean))(rows))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
    rows.forEach((r) => {
      layer.append('line').attr('x1', x(xv(r))).attr('x2', x(xv(r)))
        .attr('y1', y(r.mean - r.sd)).attr('y2', y(r.mean + r.sd))
        .attr('stroke', 'var(--primary)').attr('stroke-width', 1.3);
      layer.append('circle').attr('cx', x(xv(r))).attr('cy', y(r.mean)).attr('r', 4)
        .attr('fill', 'var(--primary)');
    });

    if (isTemp) {
      /* the entropy of what the teacher is saying, on its own axis on the right,
         because a second series on a hidden scale is decoration */
      const ye = d3.scaleLinear()
        .domain([0, d3.max(tempRows.map((r) => r.entropy)) * 1.1]).range([h, 0]);
      layer.append('path')
        .attr('d', d3.line().x((r) => x(r.t)).y((r) => ye(r.entropy))(tempRows))
        .attr('fill', 'none').attr('stroke', 'var(--aqua)').attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '5 3');
      const axis = g.append('g').attr('class', 'axis y')
        .attr('transform', `translate(${w},0)`);
      ye.ticks(4).forEach((v) => {
        axis.append('text').attr('x', 8).attr('y', ye(v) + 4)
          .attr('class', 'chart-note').style('font-size', '10.5px')
          .attr('fill', 'var(--aqua)').text(v.toFixed(1));
      });
      layer.append('text').attr('class', 'chart-note').attr('x', w + 8).attr('y', -8)
        .style('font-size', '11px').attr('fill', 'var(--aqua)').text('nats');
      layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -34)
        .style('font-size', '11px').attr('fill', 'var(--aqua)')
        .text('dashed: entropy of what the teacher says, right axis');
    }

    const best = rows.reduce((a, b) => (b.mean > a.mean ? b : a));
    const worst = rows.reduce((a, b) => (b.mean < a.mean ? b : a));
    const span = best.mean - worst.mean;
    const floor = best.sd + worst.sd;
    readout.innerHTML = `<div class="gen-note">`
      + (isTemp
        ? `Across temperatures ${temps.join(', ')}, the best is `
          + `<span class="bold">T = ${best.t}</span> at ${best.mean.toFixed(4)} and the worst `
          + `is T = ${worst.t} at ${worst.mean.toFixed(4)}. The entropy of the teacher's `
          + `output goes from ${tempRows[0].entropy.toFixed(3)} nats to `
          + `${tempRows[tempRows.length - 1].entropy.toFixed(3)}, so the dial is doing what `
          + `it says it does. `
        : `Across mixtures, the best is <span class="bold">${best.weight.toFixed(2)}</span> `
          + `on the soft target at ${best.mean.toFixed(4)} and the worst is `
          + `${worst.weight.toFixed(2)} at ${worst.mean.toFixed(4)}. Weight zero is training `
          + `on labels alone and weight one is training on the teacher alone. `)
      + (span <= floor
        ? `The whole sweep spans ${span.toFixed(4)}, inside the ${floor.toFixed(4)} of seed `
          + `spread at its two ends: on this pair of models the dial does not decide `
          + `anything, and a paper's recommended value would have been unfalsifiable here.`
        : `The sweep spans ${span.toFixed(4)} against ${floor.toFixed(4)} of seed spread at `
          + `its two ends, so the choice is real.`)
      + `</div>`;
  }

  render();
  return { render, state: st };
}
