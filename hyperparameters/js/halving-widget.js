/* What a cheap look tells you about an expensive one.
 *
 * Every line is one configuration, drawn across the four training budgets it
 * was scored at. Halving is exactly the claim that these lines do not cross
 * much, so drawing them is drawing the assumption; the rank correlation in the
 * readout is the same claim as a number.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initHalvingWidget(data) {
  const H = data.halving;
  const buttons = document.querySelector('#halving-buttons');
  const readout = document.querySelector('#halving-readout');
  const st = { view: 'score' };

  const chart = makeChart('#halving-chart', {
    width: 640, height: 400, margin: { top: 62, right: 30, bottom: 58, left: 70 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['score', 'the scores'], ['rank', 'the ranks']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const nF = H.fracs.length;
  const nC = H.configs;
  const full = H.scores[nF - 1];
  const bestCfg = full.indexOf(Math.max(...full));

  /* ranks recomputed here rather than shipped: one array of numbers is the
     source, and a second copy of it in the file is a second thing to keep true */
  function ranksAt(fi) {
    const order = H.scores[fi].map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    const out = new Array(nC);
    order.forEach(([, i], k) => { out[i] = k + 1; });
    return out;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    const x = d3.scalePoint().domain(H.rows.map((r) => r.n)).range([0, w]).padding(0.5);
    const isScore = st.view === 'score';
    const flat = H.scores.flat();
    const y = isScore
      ? d3.scaleLinear().domain([Math.min(...flat) - 0.02, Math.max(...flat) + 0.01]).range([h, 0])
      : d3.scaleLinear().domain([nC + 0.5, 0.5]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
    drawAxes(g, x, y, w, h, { yTicks: 5, yFmt: isScore ? d3.format('.2f') : d3.format('d') });
    axisLabels(g, w, h, {
      x: 'rows the candidate was allowed to train on',
      y: isScore ? 'cross validated accuracy' : 'rank among the 24 candidates',
    });

    const ranks = H.fracs.map((_, fi) => ranksAt(fi));
    for (let c = 0; c < nC; c++) {
      const pts = H.rows.map((row, fi) => [x(row.n), y(isScore ? H.scores[fi][c] : ranks[fi][c])]);
      const win = c === bestCfg;
      layer.append('path')
        .attr('d', d3.line()(pts))
        .attr('fill', 'none')
        .attr('stroke', win ? 'var(--primary)' : 'var(--squidink)')
        .attr('stroke-width', win ? 2.6 : 1)
        .attr('opacity', win ? 1 : 0.28);
      if (win) {
        pts.forEach(([px, py]) => layer.append('circle').attr('cx', px).attr('cy', py)
          .attr('r', 3.6).attr('fill', 'var(--primary)'));
      }
    }
    const pick = H.halving_pick;
    const ptsPick = H.rows.map((row, fi) => [x(row.n),
      y(isScore ? H.scores[fi][pick] : ranks[fi][pick])]);
    layer.append('path').attr('d', d3.line()(ptsPick))
      .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.2)
      .attr('stroke-dasharray', '5 3');

    /* The legend goes above the plot rather than beside it. Twenty four faint
       lines fill this panel from edge to edge, so there is no empty corner
       inside it, and a right hand margin wide enough for "the best at full
       size" would be a quarter of the chart. */
    [['the best at full size', 'var(--primary)', null],
      ['what halving kept', 'var(--cosmos)', '5 3']].forEach(([t, colour, dash], i) => {
      const ly = -44 + i * 16;
      layer.append('text').attr('class', 'chart-note').attr('x', 30).attr('y', ly)
        .style('font-size', '11px').attr('fill', colour).text(t);
      layer.append('line').attr('x1', 0).attr('x2', 24)
        .attr('y1', ly - 4).attr('y2', ly - 4)
        .attr('stroke', colour).attr('stroke-width', 2.4).attr('stroke-dasharray', dash);
    });

    const cheap = H.rows[0];
    const same = H.halving_pick === bestCfg;
    const lost = full[bestCfg] - full[H.halving_pick];
    readout.innerHTML = `<div class="gen-note">Twenty four candidates, scored at `
      + `${H.rows.map((r) => r.n).join(', ')} training rows. At the cheapest budget the `
      + `order already agrees with the expensive one to a rank correlation of `
      + `<span class="bold">${cheap.spearman.toFixed(3)}</span>, and `
      + `${(cheap.top_half_kept * 100).toFixed(0)}% of the top half survives. `
      + `Halving spends the equivalent of ${H.cost_halving.toFixed(0)} full evaluations `
      + `instead of ${H.cost_full.toFixed(0)}, a factor of ${H.saving.toFixed(2)}, and `
      + (same
        ? `keeps the candidate that turns out to be best.`
        : `keeps a candidate that ends ${lost.toFixed(4)} below the best one, against a `
          + `fold to fold wobble of ${data.floor.cv_sd.toFixed(4)}: `
          + (lost <= data.floor.cv_sd
            ? `a gap this measurement cannot separate from zero.`
            : `a real gap.`))
      + `</div>`;
  }

  render();
  return { render, state: st };
}
