/* The bias in "the best score I found", separated into its two halves.
 *
 * The chart carries three curves for the same picked configurations. The first
 * is the score that picked them. The second is a SECOND cross validation of the
 * same cells with different folds, so the difference between one and two is
 * selection and nothing else. The third is the held out score, which adds a
 * bias in the opposite direction, because cross validation trains on fewer rows
 * than the final fit does. Drawing only the first and the third, which is what
 * everyone does, mixes two effects that point opposite ways.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const SERIES = [
  ['cv', 'the score that picked it', 'var(--primary)', null],
  ['cv2', 'the same cells, folds redrawn', 'var(--cosmos)', '5 3'],
  ['test', 'held out', 'var(--squidink)', '2 3'],
];

export function initCurseWidget(data) {
  const C = data.curse;
  const buttons = document.querySelector('#curse-buttons');
  const readout = document.querySelector('#curse-readout');
  const st = { view: 'levels' };

  const chart = makeChart('#curse-chart', {
    width: 640, height: 390, margin: { top: 60, right: 30, bottom: 58, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['levels', 'the three scores'],
    ['gaps', 'the two biases']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    const bs = C.rows.map((r) => r.b);
    const x = d3.scaleLog().domain([bs[0], bs[bs.length - 1]]).range([0, w]);
    const levels = st.view === 'levels';
    const all = levels
      ? C.rows.flatMap((r) => [r.cv, r.cv2, r.test])
      : C.rows.flatMap((r) => [r.selection, r.gap]);
    const y = d3.scaleLinear()
      .domain(levels ? [d3.min(all) - 0.01, d3.max(all) + 0.005]
        : [Math.min(0, d3.min(all)) - 0.001, Math.max(0, d3.max(all)) + 0.001])
      .range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: bs, yTicks: 5 });
    drawAxes(g, x, y, w, h, {
      xValues: bs, yTicks: 5, xFmt: (v) => String(v),
      yFmt: levels ? d3.format('.3f') : d3.format('+.4f'),
    });
    axisLabels(g, w, h, {
      x: 'configurations the search was allowed to try',
      y: levels ? 'accuracy of the one it picked' : 'how much the score overstates',
    });

    if (!levels) {
      layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('opacity', 0.5);
    }

    const draw = (key, colour, dash) => {
      layer.append('path')
        .attr('d', d3.line().x((r) => x(r.b)).y((r) => y(r[key]))(C.rows))
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
        .attr('stroke-dasharray', dash);
      C.rows.forEach((r) => layer.append('circle').attr('cx', x(r.b)).attr('cy', y(r[key]))
        .attr('r', 3.6).attr('fill', colour));
    };
    const legend = levels
      ? SERIES
      : [['selection', 'the winner’s curse alone', 'var(--cosmos)', null],
        ['gap', 'against held out, both biases together', 'var(--squidink)', '2 3']];
    legend.forEach(([key, , colour, dash]) => draw(key, colour, dash));
    legend.forEach(([, name, colour, dash], i) => {
      const ly = -44 + i * 15;
      layer.append('text').attr('class', 'chart-note').attr('x', 30).attr('y', ly)
        .style('font-size', '11px').attr('fill', colour).text(name);
      layer.append('line').attr('x1', 0).attr('x2', 24).attr('y1', ly - 4).attr('y2', ly - 4)
        .attr('stroke', colour).attr('stroke-width', 2.4).attr('stroke-dasharray', dash);
    });

    const wide = C.rows[C.rows.length - 1];
    const mid = C.rows.find((r) => r.b === 32) || C.rows[Math.floor(C.rows.length / 2)];
    readout.innerHTML = `<div class="gen-note">Try ${wide.b} configurations and keep the `
      + `best: its cross validated score averages ${wide.cv.toFixed(4)}, and the same `
      + `configurations scored again with different folds average `
      + `${wide.cv2.toFixed(4)}. That difference, `
      + `<span class="bold">${wide.selection >= 0 ? '+' : ''}${wide.selection.toFixed(4)}</span>, `
      + `is the winner's curse with nothing else mixed in: it is the part of the winning `
      + `score that was luck in the fold draw. At ${mid.b} configurations it is `
      + `${mid.selection >= 0 ? '+' : ''}${mid.selection.toFixed(4)}. Taken over all the `
      + `cells rather than over the winners it averages `
      + `${C.honest_selection >= 0 ? '+' : ''}${C.honest_selection.toFixed(4)}, which is `
      + `what it should be, with a spread of ${C.pair_sd.toFixed(4)} cell to cell: the `
      + `selection is not adding noise, it is picking the top of it. `
      + `Against the held out set the same pick shows `
      + `${wide.gap >= 0 ? '+' : ''}${wide.gap.toFixed(4)} with a spread of `
      + `${wide.gap_sd.toFixed(4)} over the ${C.trials.toLocaleString('en-US')} draws, `
      + (wide.gap < 0
        ? `which has the opposite sign, because cross validation trains on fewer rows than `
          + `the final fit and that makes it pessimistic by `
          + `${Math.abs(C.honest_gap).toFixed(4)} on an average cell. The two biases fight, `
          + `and on this problem the pessimism wins.`
        : `so here the two biases point the same way.`)
      + `</div>`;
  }

  render();
  return { render, state: st };
}
