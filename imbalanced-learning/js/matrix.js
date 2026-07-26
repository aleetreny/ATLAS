/* The confusion matrix as a drawn object, shared by the scrolly and the
 * threshold widget.
 *
 * Four boxes, fixed positions, only the numbers and the shading move. Shading
 * is per-row (share of the actual class), because "how much of the truth ended
 * up here" is the reading that stays meaningful when the classes differ by
 * a factor of a hundred.
 */

const CELLS = [
  { key: 'tp', row: 0, col: 0, label: 'found', tone: 'var(--primary)' },
  { key: 'fn', row: 0, col: 1, label: 'missed', tone: 'var(--cosmos)' },
  { key: 'fp', row: 1, col: 0, label: 'false alarm', tone: 'var(--cosmos)' },
  { key: 'tn', row: 1, col: 1, label: 'all-clear', tone: 'var(--stone)' },
];

export function makeMatrix(parent, x0, y0, w, h, opts = {}) {
  const numSize = opts.numSize || 15;
  const g = parent.append('g').attr('transform', `translate(${x0},${y0})`);
  const cw = w / 2;
  const ch = h / 2;

  g.append('text').attr('class', 'chart-annotation').attr('x', -8).attr('y', ch / 2 + 4)
    .attr('text-anchor', 'end').style('font-size', '0.56rem').text('tree');
  g.append('text').attr('class', 'chart-annotation').attr('x', -8).attr('y', ch + ch / 2 + 4)
    .attr('text-anchor', 'end').style('font-size', '0.56rem').text('not');
  g.append('text').attr('class', 'chart-annotation').attr('x', cw / 2).attr('y', -8)
    .attr('text-anchor', 'middle').style('font-size', '0.56rem').text('said tree');
  g.append('text').attr('class', 'chart-annotation').attr('x', cw + cw / 2).attr('y', -8)
    .attr('text-anchor', 'middle').style('font-size', '0.56rem').text('said not');

  const boxes = CELLS.map((c) => {
    const bg = g.append('g').attr('transform', `translate(${c.col * cw},${c.row * ch})`);
    bg.append('rect').attr('width', cw - 4).attr('height', ch - 4).attr('x', 2).attr('y', 2)
      .attr('fill', c.tone).attr('fill-opacity', 0.12)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
    const num = bg.append('text').attr('x', cw / 2).attr('y', ch / 2 - 2)
      .attr('text-anchor', 'middle')
      .style('font-family', 'var(--font-mono)').style('font-size', `${numSize}px`).style('font-weight', '700')
      .attr('fill', 'var(--squidink)');
    bg.append('text').attr('class', 'chart-annotation').attr('x', cw / 2).attr('y', ch / 2 + 15)
      .attr('text-anchor', 'middle').style('font-size', '0.52rem').text(c.label);
    return { ...c, rect: bg.select('rect'), num };
  });

  return {
    update(m) {
      /* m: {tp, fp, fn, tn} */
      const rowTotal = { 0: m.tp + m.fn, 1: m.fp + m.tn };
      for (const b of boxes) {
        const v = m[b.key];
        b.num.text(v.toLocaleString('en-US'));
        b.rect.attr('fill-opacity', 0.08 + 0.55 * (v / Math.max(rowTotal[b.row], 1)));
      }
    },
    hide() { g.attr('opacity', 0); },
    show() { g.attr('opacity', 1); },
  };
}
