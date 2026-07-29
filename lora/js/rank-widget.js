/* Two questions about rank that are usually run together.
 *
 * The first view is a comparison nobody publishes: take the update that full
 * fine tuning actually produced, truncate it to rank r with a singular value
 * decomposition, and apply it. That is the best rank r approximation OF THAT
 * UPDATE, which is not the same thing as the best rank r adapter, and the
 * difference between those two is the point. A trained adapter is free to go
 * looking for a different low rank update rather than copy this one, so it can
 * and does come out above the curve.
 *
 * The second view puts every cheap adaptation method on one axis, the number of
 * parameters allowed to move, which is the axis they all claim to compete on.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initRankWidget(data) {
  const buttons = document.querySelector('#rank-buttons');
  const readout = document.querySelector('#rank-readout');
  const st = { view: 'ceiling' };

  const chart = makeChart('#rank-chart', {
    width: 640, height: 380, margin: { top: 60, right: 34, bottom: 58, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['ceiling', 'against the best possible adapter'],
    ['budget', 'against the other cheap methods']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const loras = data.arms.filter((a) => a.rank != null).sort((a, b) => a.rank - b.rank);
  /* the placement arms belong on this axis too: "which layers get an adapter" is a
     different decision from "how big is it", and it buys parameters on the same
     scale. "both" is the rank r run already on the curve, so only the two partial
     placements are added */
  const placed = data.where.rows.filter((r) => r.arm !== 'both')
    .map((r) => ({ ...r, arm: `${r.arm} layer only, r=${data.where.rank}` }));
  const others = data.arms.filter((a) => a.rank == null).concat(placed);

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    if (st.view === 'ceiling') {
      const ceil = data.ceiling.filter((r) => r.rank > 0);
      const x = d3.scaleLog().domain([ceil[0].rank * 0.8, ceil[ceil.length - 1].rank * 1.25])
        .range([0, w]);
      const all = ceil.map((r) => r.acc).concat(loras.map((a) => a.mean))
        .concat([data.spectra.tuned_acc, data.spectra.base_acc]);
      const y = d3.scaleLinear().domain([d3.min(all) - 0.01, d3.max(all) + 0.008]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: ceil.map((r) => r.rank), yTicks: 5 });
      drawAxes(g, x, y, w, h, {
        xValues: ceil.map((r) => r.rank), yTicks: 5,
        xFmt: (v) => String(v), yFmt: d3.format('.3f'),
      });
      axisLabels(g, w, h, { x: 'rank of the adapter', y: 'accuracy on the new task' });

      layer.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(data.spectra.tuned_acc)).attr('y2', y(data.spectra.tuned_acc))
        .attr('stroke', 'var(--smile)').attr('stroke-width', 1.4).attr('stroke-dasharray', '6 4');
      layer.append('text').attr('class', 'chart-note').attr('x', w - 2)
        .attr('y', y(data.spectra.tuned_acc) + 14).attr('text-anchor', 'end')
        .style('font-size', '11px').attr('fill', 'var(--smile)')
        .text(`full fine tuning, ${data.spectra.tuned_acc.toFixed(4)}`);

      [['the measured update, truncated to rank r', ceil.map((r) => [r.rank, r.acc]), 'var(--anchor)'],
        ['what an adapter of rank r finds for itself', loras.map((a) => [a.rank, a.mean]), 'var(--primary)']]
        .forEach(([name, pts, colour], i) => {
          layer.append('path')
            .attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(pts))
            .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2);
          pts.forEach((p) => layer.append('circle').attr('cx', x(p[0])).attr('cy', y(p[1]))
            .attr('r', 4).attr('fill', colour));
          layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -40 + i * 15)
            .style('font-size', '11px').attr('fill', colour).text(name);
        });
    } else {
      /* the named arms carry the only labels here, and at a narrow width two of
         them land close enough to touch, so they are stacked by rank of x */
      const ordered = others.slice().sort((a, b) => a.trainable - b.trainable);
      const pts = loras.map((a) => ({ ...a, label: `r=${a.rank}` }))
        .concat(ordered.map((a, i) => ({ ...a, label: a.arm, lift: 11 + (i % 2) * 15 })));
      const x = d3.scaleLog()
        .domain([d3.min(pts.map((p) => p.trainable)) * 0.5,
          d3.max(pts.map((p) => p.trainable)) * 2]).range([0, w]);
      const y = d3.scaleLinear()
        .domain([d3.min(pts.map((p) => p.mean)) - 0.012, d3.max(pts.map((p) => p.mean)) + 0.01])
        .range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      drawAxes(g, x, y, w, h, {
        xTicks: 5, yTicks: 5, yFmt: d3.format('.3f'),
        xFmt: (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1000 ? `${Math.round(v / 1000)}k` : String(v)),
      });
      axisLabels(g, w, h, { x: 'parameters allowed to move', y: 'accuracy on the new task' });

      layer.append('path')
        .attr('d', d3.line().x((p) => x(p.trainable)).y((p) => y(p.mean))(
          loras.map((a) => ({ trainable: a.trainable, mean: a.mean }))))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
      pts.forEach((p) => {
        const isLora = p.rank != null;
        layer.append('circle').attr('cx', x(p.trainable)).attr('cy', y(p.mean)).attr('r', 5)
          .attr('fill', isLora ? 'var(--primary)' : 'none')
          .attr('stroke', isLora ? 'none' : 'var(--cosmos)').attr('stroke-width', 2.2);
        layer.append('line').attr('x1', x(p.trainable)).attr('x2', x(p.trainable))
          .attr('y1', y(p.mean - p.sd)).attr('y2', y(p.mean + p.sd))
          .attr('stroke', isLora ? 'var(--primary)' : 'var(--cosmos)').attr('stroke-width', 1.2);
        if (!isLora) {
          layer.append('text').attr('class', 'chart-note').attr('x', x(p.trainable))
            .attr('y', y(p.mean) - p.lift)
            .attr('text-anchor', x(p.trainable) > w * 0.6 ? 'end' : 'middle')
            .style('font-size', '11px').text(p.label);
        }
      });
      layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -40)
        .style('font-size', '11px').attr('fill', 'var(--primary)')
        .text(`filled: low rank adapters, rank ${loras[0].rank} to ${loras[loras.length - 1].rank}`);
      layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -25)
        .style('font-size', '11px').attr('fill', 'var(--cosmos)')
        .text('hollow: everything else, named');
    }

    const ceilRows = data.ceiling.filter((r) => r.rank > 0);
    const pairs = loras.map((a) => {
      const c = ceilRows.find((r) => r.rank === a.rank);
      return c ? { rank: a.rank, got: a.mean, best: c.acc, kept: c.kept } : null;
    }).filter(Boolean);
    const worst = pairs.reduce((a, b) => ((b.best - b.got) > (a.best - a.got) ? b : a), pairs[0]);
    const full = data.arms.find((a) => a.arm === 'everything');
    const r8 = loras.find((a) => a.rank === 8) || loras[Math.floor(loras.length / 2)];
    readout.innerHTML = `<table class="gen-table"><thead><tr><th>rank</th>`
      + `<th>energy kept</th><th>best possible</th><th>what training found</th>`
      + `<th>trainable</th></tr></thead><tbody>`
      + pairs.map((p) => {
        const a = loras.find((q) => q.rank === p.rank);
        return `<tr><td>${p.rank}</td><td>${(p.kept * 100).toFixed(1)}%</td>`
          + `<td>${p.best.toFixed(4)}</td><td>${p.got.toFixed(4)}</td>`
          + `<td>${a.trainable.toLocaleString('en-US')}</td></tr>`;
      }).join('')
      + `</tbody></table><div class="gen-note">Rank ${r8.rank} moves `
      + `${r8.trainable.toLocaleString('en-US')} parameters, `
      + `${(r8.trainable / full.trainable * 100).toFixed(2)}% of the `
      + `${full.trainable.toLocaleString('en-US')} in the network, and reaches `
      + `${r8.mean.toFixed(4)} against ${full.mean.toFixed(4)} for moving all of them. `
      + `The widest gap between the truncated update and what a trained adapter of that rank `
      + `reaches is at rank ${worst.rank}: `
      + `${Math.abs(worst.best - worst.got).toFixed(4)} in favour of `
      + `${worst.best > worst.got ? 'the truncation' : 'the adapter'}. Where the adapter is `
      + `ahead it is not copying this update at all, which is why "the update is low rank" `
      + `explains less about why LoRA works than it looks like it does.`
      + `</div>`;
  }

  render();
  return { render, state: st };
}
