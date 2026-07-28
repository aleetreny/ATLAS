/* Where the latent goes in, drawn.
 *
 * Four idempotent states over one diagram: the ordinary generator from the
 * previous article, the mapping network in front of it, the styles arriving at
 * every block instead of only at the input, and the per-pixel noise. The block
 * sizes and channel counts are read from the export rather than typed, so the
 * picture is of the network this page runs.
 */
import { makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';

export function initArchScrolly(data) {
  const M = data.meta;
  const c = makeChart('#arch-chart', {
    width: 660, height: 460, margin: { top: 20, right: 16, bottom: 20, left: 16 },
  });
  const g = c.g;
  const boxW = 96;
  const boxH = 34;
  const colX = 200;
  const top = 40;
  const gapY = (c.h - top - 40) / (M.blocks.length - 1);

  const box = (parent, x, y, w, h, label, cls, fill) => {
    const grp = parent.append('g').attr('class', cls);
    grp.append('rect').attr('x', x).attr('y', y).attr('width', w).attr('height', h)
      .attr('fill', fill).attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
    grp.append('text').attr('x', x + w / 2).attr('y', y + h / 2 + 4)
      .attr('text-anchor', 'middle').attr('class', 'chart-annotation')
      .style('font-size', '11px').style('letter-spacing', '0.3px').text(label);
    return grp;
  };
  const arrow = (x1, y1, x2, y2, cls, dash = null) => g.append('line')
    .attr('class', cls).attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2)
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2)
    .attr('stroke-dasharray', dash).attr('marker-end', 'url(#arch-arrow)');

  c.svg.append('defs').append('marker').attr('id', 'arch-arrow')
    .attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
    .attr('markerWidth', 5).attr('markerHeight', 5).attr('orient', 'auto')
    .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', 'var(--squidink)');

  /* the synthesis column, which both architectures share */
  const blocks = M.blocks.map((b, i) => {
    const y = top + i * gapY;
    const grp = box(g, colX, y, boxW + 40, boxH, `block ${i + 1}, ${b.res}x${b.res}`, 'block',
      'var(--paper)');
    if (i) arrow(colX + (boxW + 40) / 2, y - gapY + boxH, colX + (boxW + 40) / 2, y - 3, 'flow');
    return { grp, y };
  });
  box(g, colX, top + (M.blocks.length - 1) * gapY + gapY * 0.62, boxW + 40, boxH,
    `${M.res}x${M.res} image`, 'out', 'var(--paper)');
  arrow(colX + (boxW + 40) / 2, top + (M.blocks.length - 1) * gapY + boxH,
    colX + (boxW + 40) / 2, top + (M.blocks.length - 1) * gapY + gapY * 0.62 - 3, 'flow');

  /* the plain path: z straight into a dense layer at the top */
  const plain = g.append('g');
  plain.append('rect').attr('x', colX - 4).attr('y', top - 30).attr('width', boxW + 48)
    .attr('height', 24).attr('fill', 'var(--anchor)').attr('fill-opacity', 0.18)
    .attr('stroke', 'var(--anchor)');
  plain.append('text').attr('x', colX + (boxW + 40) / 2).attr('y', top - 13)
    .attr('text-anchor', 'middle').attr('class', 'chart-annotation').style('font-size', '11px')
    .style('letter-spacing', '0.3px').text(`z (${M.z}) into a dense layer`);

  /* the mapping network */
  const mapg = g.append('g').attr('opacity', 0);
  const zTop = top + gapY * 1.2;
  box(mapg, 10, zTop, boxW, boxH, `z (${M.z})`, 'zbox', 'var(--white)');
  box(mapg, 10, zTop + 56, boxW, boxH * 1.6, 'mapping, 4 layers', 'mapbox', 'var(--white)');
  box(mapg, 10, zTop + 56 + boxH * 1.6 + 22, boxW, boxH, `w (${M.w})`, 'wbox', 'var(--white)');
  [
    [10 + boxW / 2, zTop + boxH, 10 + boxW / 2, zTop + 53],
    [10 + boxW / 2, zTop + 56 + boxH * 1.6, 10 + boxW / 2, zTop + 56 + boxH * 1.6 + 19],
  ].forEach(([a, b, d, e]) => {
    mapg.append('line').attr('x1', a).attr('y1', b).attr('x2', d).attr('y2', e)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2)
      .attr('marker-end', 'url(#arch-arrow)');
  });

  /* styles from w into every block */
  const styles = g.append('g').attr('opacity', 0);
  const wy = zTop + 56 + boxH * 1.6 + 22 + boxH / 2;
  blocks.forEach((b) => {
    styles.append('path')
      .attr('d', `M${10 + boxW},${wy} C${colX - 50},${wy} ${colX - 50},${b.y + boxH / 2} `
        + `${colX - 3},${b.y + boxH / 2}`)
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 1.6)
      .attr('marker-end', 'url(#arch-arrow)');
  });
  styles.append('text').attr('x', colX - 54).attr('y', top - 6).attr('text-anchor', 'end')
    .attr('class', 'chart-annotation').style('font-size', '11px').style('letter-spacing', '0.4px')
    .style('fill', 'var(--primary)').text('a style per block');

  /* noise into every block, from the right */
  const noise = g.append('g').attr('opacity', 0);
  blocks.forEach((b) => {
    noise.append('line').attr('x1', c.w - 10).attr('y1', b.y + boxH / 2)
      .attr('x2', colX + boxW + 43).attr('y2', b.y + boxH / 2)
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.4)
      .attr('stroke-dasharray', '4 3').attr('marker-end', 'url(#arch-arrow)');
  });
  noise.append('text').attr('x', c.w - 8).attr('y', top - 6).attr('text-anchor', 'end')
    .attr('class', 'chart-annotation').style('font-size', '11px').style('letter-spacing', '0.4px')
    .style('fill', 'var(--cosmos)').text('one noise map per block');

  const constBox = g.append('g').attr('opacity', 0);
  constBox.append('rect').attr('x', colX - 4).attr('y', top - 30).attr('width', boxW + 48)
    .attr('height', 24).attr('fill', 'var(--primary)').attr('fill-opacity', 0.18)
    .attr('stroke', 'var(--primary)');
  constBox.append('text').attr('x', colX + (boxW + 40) / 2).attr('y', top - 13)
    .attr('text-anchor', 'middle').attr('class', 'chart-annotation').style('font-size', '11px')
    .style('letter-spacing', '0.3px').text('a learned constant');

  const show = (sel, on) => sel.transition('fade').duration(400).attr('opacity', on ? 1 : 0);
  const states = {
    0: () => {
      show(plain, true); show(mapg, false); show(styles, false); show(noise, false);
      show(constBox, false);
    },
    1: () => {
      show(plain, true); show(mapg, true); show(styles, false); show(noise, false);
      show(constBox, false);
    },
    2: () => {
      show(plain, false); show(mapg, true); show(styles, true); show(noise, false);
      show(constBox, true);
    },
    3: () => {
      show(plain, false); show(mapg, true); show(styles, true); show(noise, true);
      show(constBox, true);
    },
  };
  const sc = scrolly('#arch-scrolly .step', states);
  states[0]();
  return sc;
}
