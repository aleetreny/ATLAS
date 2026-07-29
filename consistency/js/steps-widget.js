/* Quality against step count, for all three, on one pair of axes.
 *
 * This is the chart the article exists for, and it has one design decision in
 * it worth stating: the horizontal axis is the number of NETWORK EVALUATIONS,
 * not the number of steps, and for these three samplers those happen to be the
 * same number. They did not in the previous article, where guidance made every
 * step cost two, and they will not in general. Saying evaluations keeps the
 * axis honest when someone carries the chart somewhere else.
 *
 * The band along the bottom is what real digits score against other real
 * digits. A point inside it is not "good": it is a point this measurement
 * cannot tell apart from real data, which is a different and weaker claim.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

const SERIES = [
  ['rectified flow', 'var(--squidink)', '7 4'],
  ['after reflow', 'var(--primary)', null],
];

export function initStepsWidget(data) {
  const A = data.arms;
  const F = data.setup.floor;
  const B = data.meta.budgets;
  const buttons = document.querySelector('#steps-buttons');
  const readout = document.querySelector('#steps-readout');
  const st = { view: 'mmd' };

  const chart = makeChart('#steps-chart', {
    width: 640, height: 380, margin: { top: 50, right: 30, bottom: 58, left: 76 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -30).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const btns = {};
  for (const [key, text] of [['mmd', 'distance from real digits'],
    ['c2st', 'how easily a classifier tells them apart']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const student = A.find((r) => r.arm === 'distilled student');
  const rowsOf = (name) => A.filter((r) => r.arm === name).sort((a, b) => a.steps - b.steps);
  const best = {};
  SERIES.forEach(([name]) => {
    best[name] = rowsOf(name).reduce((a, r) => (Math.abs(r.mmd2) < Math.abs(a.mmd2) ? r : a));
  });

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();
    const mmd = st.view === 'mmd';
    const val = (r) => (mmd ? Math.abs(r.mmd2) : r.c2st);
    const floorVal = mmd ? Math.abs(F.mmd2_max) : F.c2st_max;

    const x = d3.scaleLog().domain([0.85, 120]).range([0, w]);
    const all = A.map(val).concat([floorVal]);
    const y = mmd
      ? d3.scaleLog().domain([Math.min(...all) * 0.65, Math.max(...all) * 1.6]).range([h, 0])
      : d3.scaleLinear().domain([0.45, 1.02]).range([h, 0]);

    /* decades rather than d3's own log ticks, which came out as 2e-1, 9e-2,
       4e-2, 8e-3, 3e-3, 7e-4: legible one at a time and unreadable as a set */
    const yv = mmd
      ? [1e-3, 1e-2, 1e-1].filter((v) => v >= y.domain()[0] && v <= y.domain()[1])
      : null;
    drawGrid(g, x, y, w, h, { xValues: B, yValues: yv, yTicks: 5 });
    drawAxes(g, x, y, w, h, {
      xValues: B, yValues: yv, yTicks: 5, xFmt: (v) => String(v),
      yFmt: mmd ? d3.format('.0e') : d3.format('.2f'),
    });
    axisLabels(g, w, h, {
      x: 'network evaluations per sample',
      y: mmd ? 'distance from real digits' : 'separable at',
    });

    /* the band this measurement cannot see into */
    layer.append('rect').attr('x', 0).attr('width', w)
      .attr('y', y(floorVal)).attr('height', Math.max(0, h - y(floorVal)))
      .attr('fill', 'var(--anchor)').attr('opacity', 0.1);
    layer.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', y(floorVal)).attr('y2', y(floorVal))
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.4).attr('stroke-dasharray', '6 4');
    layer.append('text').attr('class', 'chart-note').attr('x', w - 2)
      .attr('y', y(floorVal) - 6).attr('text-anchor', 'end').style('font-size', '11px')
      .attr('fill', 'var(--anchor)').text('real against real: nothing below this is a result');

    SERIES.forEach(([name, colour, dash]) => {
      const rows = rowsOf(name);
      layer.append('path')
        .attr('d', d3.line().x((r) => x(r.steps)).y((r) => y(val(r)))(rows))
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
        .attr('stroke-dasharray', dash);
      rows.forEach((r) => {
        layer.append('circle').attr('cx', x(r.steps)).attr('cy', y(val(r))).attr('r', 4.2)
          .attr('fill', colour);
      });
    });

    /* A legend rather than labels on the curves, because these two curves MEET:
       the whole finding is that they end up in the same place, so the right end
       is the one region of this chart where two series labels cannot both fit.
       They were there in the first draft and the sweep caught them overlapping
       each other and the floor caption. Top right is empty in both views,
       because both curves are low on the right. */
    SERIES.concat([['distilled student', 'var(--secondary)', 'diamond']])
      .forEach(([name, colour, dash], i) => {
        const ly = 10 + i * 18;
        /* the swatch is placed from the label's MEASURED box rather than from a
           guessed character width: "distilled student" at eleven pixels is not
           the same number of pixels in every font that might have loaded, and a
           guess put the first draft of this legend off the right edge */
        const t = layer.append('text').attr('class', 'chart-note')
          .attr('x', w - 2).attr('y', ly).attr('text-anchor', 'end')
          .style('font-size', '11px').attr('fill', colour).text(name);
        let left = w - 2 - 90;
        try {
          const b = t.node().getBBox();
          if (b.width > 0) left = b.x;
        } catch {
          /* a node not yet laid out has no box; the fallback is close enough */
        }
        if (dash === 'diamond') {
          layer.append('path').attr('d', d3.symbol().type(d3.symbolDiamond).size(64)())
            .attr('transform', `translate(${left - 16},${ly - 4})`).attr('fill', colour);
        } else {
          layer.append('line').attr('x1', left - 28).attr('x2', left - 6)
            .attr('y1', ly - 4).attr('y2', ly - 4)
            .attr('stroke', colour).attr('stroke-width', 2.4).attr('stroke-dasharray', dash);
        }
      });

    /* the student is one point, and it is drawn as a different shape so nobody
       reads it as the end of a curve that was never walked */
    const sx = x(1);
    const sy = y(val(student));
    layer.append('path')
      .attr('d', d3.symbol().type(d3.symbolDiamond).size(150)())
      .attr('transform', `translate(${sx},${sy})`)
      .attr('fill', 'var(--secondary)').attr('stroke', 'var(--white)').attr('stroke-width', 1.4);
    /* no label beside the diamond: the reflowed curve runs through exactly that
       space, so an inline caption strikes through its own data. The legend
       names it instead. */

    /* whether the reflowed curve may be called flat is not this widget's
       opinion: it is the redraw measurement in the next section */
    const budgetPair = data.spread.pairs.find(
      (p) => (p.a === 'after reflow x1' && p.b === 'after reflow x100')
        || (p.a === 'after reflow x100' && p.b === 'after reflow x1'));
    const flat = budgetPair && !budgetPair.resolved;
    wrapLabel(title, mmd
      ? (flat
        ? 'after reflow the curve is flat to within what redrawing the noise does to it'
        : 'the two curves fall at different speeds, which is the point of reflow')
      : 'the same three arms scored by a classifier instead, which needs no kernel choice',
    w - 8);

    const arms = ['rectified flow', 'after reflow', 'distilled student'];
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>evaluations</th>${B.map((b) => `<th>${b}</th>`).join('')}</tr>
        ${arms.map((name) => `<tr>
          <td>${name}</td>
          ${B.map((b) => {
    const r = A.find((q) => q.arm === name && q.steps === b);
    if (!r) return '<td>&middot;</td>';
    const inside = !r.resolves;
    return `<td><span class="${inside ? '' : 'value'}">${
      mmd ? r.mmd2 : r.c2st.toFixed(3)}</span></td>`;
  }).join('')}
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        ${(() => {
    const oneFlow = A.find((r) => r.arm === 'rectified flow' && r.steps === 1);
    const oneRe = A.find((r) => r.arm === 'after reflow' && r.steps === 1);
    const bestRe = best['after reflow'];
    const inFloor = A.filter((r) => !r.resolves).length;
    return `At a single evaluation the plain rectified flow scores
      <span class="value">${oneFlow.mmd2}</span>, and it needs ${best['rectified flow'].steps}
      to come down to <span class="value">${best['rectified flow'].mmd2}</span>. After reflow
      the same single evaluation scores <span class="value">${oneRe.mmd2}</span>, against
      <span class="value">${bestRe.mmd2}</span> at ${bestRe.steps}. The distilled student, which
      never had a chain to shorten, scores <span class="value">${student.mmd2}</span>.
      ${mmd
        ? (inFloor
          ? `${inFloor} of these ${A.length} entries sit inside the real against real floor
             (${F.mmd2_max}) and are printed without emphasis.`
          : `Every entry is outside the real against real floor (${F.mmd2_max}), so every one of
             them is distinguishable from real data.`)
        : `Note where the floor is on this view: ${F.c2st_max}. Every arm except the collapsed
           one is at or under it, so by this statistic they have already stopped being
           separable from real digits, while the kernel statistic still separates them. Two
           measurements, two answers, and neither is the wrong one.`}
      Whether the gaps between the last three of those numbers may be read as an ordering is a
      different question from whether they are outside the floor, and it is the one the next
      section answers: on this model, ${data.spread.resolved} of the
      ${data.spread.pairs.length} comparisons survive a redraw test, and they all involve the
      arm that collapsed.`;
  })()}
      </div>`;
  }

  render();
  return { render };
}
