/* Drawing shared by module 4.1, where every picture is about a sequence of
 * words rather than a cloud of points.
 *
 * The site's `chart.js` covers axes, grids and scales. What it does not cover,
 * because until this module nothing needed it, is the handful of marks that
 * come up in every one of these seven articles: a row of tokens you can point
 * at, a matrix of attention or confusion, and a bar chart whose bars are
 * labelled with words rather than numbers.
 *
 * Two rules from earlier modules are baked in here rather than left to each
 * caller, because both were paid for once already:
 *
 *  - font sizes are set with `.style`, never `.attr`. A presentation attribute
 *    loses to any CSS rule for the class, and `.chart-annotation` sets one.
 *  - nothing goes through `.chart-annotation` for a series name. That class is
 *    uppercase with 2px of letter-spacing, which is for shouting two words,
 *    and it turns a lowercase sigma into a capital one.
 */

export function seriesLabel(g, x, y, text, { anchor = 'start', size = 11.5,
  opacity = 0.85, colour = 'var(--squidink)', weight = null } = {}) {
  const t = g.append('text')
    .attr('x', x).attr('y', y)
    .attr('text-anchor', anchor)
    .style('font-family', 'var(--font-mono)')
    .style('font-size', `${size}px`)
    .style('letter-spacing', '0.4px')
    .style('fill', colour)
    .style('opacity', opacity)
    .text(text);
  if (weight) t.style('font-weight', weight);
  return t;
}

/**
 * Place a label and then shorten it until it actually fits.
 *
 * The need for this is specific and keeps coming back: a label built from data
 * (the heading words of a topic, the neighbours of a word) has no length you
 * can plan a margin around, and it changes while the reader watches. Estimating
 * "about seven units per letter" works until a topic turns up headed by
 * "agreement exchange also government". Placing it, asking
 * `getComputedTextLength` and dropping trailing words until it fits costs two
 * lines and cannot be wrong.
 */
export function fitText(g, x, y, text, maxWidth, opts = {}) {
  const node = seriesLabel(g, x, y, text, opts);
  if (node.node().getComputedTextLength() <= maxWidth) return node;
  const parts = String(text).split(' ');
  while (parts.length > 1) {
    parts.pop();
    node.text(`${parts.join(' ')}...`);
    if (node.node().getComputedTextLength() <= maxWidth) return node;
  }
  /* One word that does not fit on its own: cut characters rather than leave it
   * hanging off the canvas. */
  let s = String(text);
  while (s.length > 1 && node.node().getComputedTextLength() > maxWidth) {
    s = s.slice(0, -1);
    node.text(`${s}...`);
  }
  return node;
}

export function legend(g, items, { x0 = 0, y0 = -14, gap = 150, size = 10.5,
  cols = 0 } = {}) {
  const sel = g.selectAll('g.legend-chip').data(items);
  sel.exit().remove();
  const enter = sel.enter().append('g').attr('class', 'legend-chip');
  enter.append('path').attr('class', 'mark');
  enter.append('text').attr('x', 12).attr('dy', '0.32em')
    .style('font-family', 'var(--font-mono)')
    .style('letter-spacing', '0.4px')
    .style('fill', 'var(--squidink)');
  const merged = enter.merge(sel);
  const perRow = cols || items.length;
  merged.attr('transform', (d, i) => `translate(${x0 + (i % perRow) * gap},${y0 + Math.floor(i / perRow) * 17})`);
  merged.select('path.mark')
    .attr('d', (d) => (d.shape === 'ring'
      ? 'M-5,0 A5,5 0 1,0 5,0 A5,5 0 1,0 -5,0'
      : d.shape === 'line'
        ? 'M-7,0 L7,0'
        : d.shape === 'dash'
          ? 'M-7,0 L-2,0 M2,0 L7,0'
          : d.shape === 'square'
            ? 'M-4,-4 L4,-4 L4,4 L-4,4 Z'
            : 'M-4,0 A4,4 0 1,0 4,0 A4,4 0 1,0 -4,0'))
    .attr('fill', (d) => (['ring', 'line', 'dash'].includes(d.shape) ? 'none' : d.colour))
    .attr('stroke', (d) => d.colour)
    .attr('stroke-width', 2.2);
  merged.select('text').style('font-size', `${size}px`).text((d) => d.label);
  return merged;
}

/**
 * Horizontal bars with a word on the left and a number on the right.
 *
 * The one thing it does that a hand-rolled version keeps getting wrong: the
 * value label goes inside the bar when the bar is long enough to hold it and
 * outside when it is not, measured with `getComputedTextLength` rather than
 * estimated from the character count. Seven letters per unit works until a
 * word is "money-supply".
 */
export function wordBars(g, rows, w, h, {
  colour = 'var(--primary)', gap = 4, labelWidth = 96, valueFmt = (v) => v.toFixed(3),
  max = null, size = 11.5, highlight = () => false,
} = {}) {
  g.selectAll('*').remove();
  if (!rows.length) return;
  const hi = max === null ? Math.max(...rows.map((r) => r.value)) : max;
  const band = h / rows.length;
  const bh = Math.max(4, band - gap);
  const x = d3.scaleLinear().domain([0, hi || 1]).range([0, Math.max(10, w - labelWidth - 54)]);

  rows.forEach((r, i) => {
    const y = i * band;
    const row = g.append('g').attr('transform', `translate(0,${y})`);
    row.append('text')
      .attr('x', labelWidth - 8).attr('y', bh / 2).attr('dy', '0.34em')
      .attr('text-anchor', 'end')
      .style('font-family', 'var(--font-mono)')
      .style('font-size', `${size}px`)
      .style('fill', 'var(--squidink)')
      .style('opacity', highlight(r, i) ? 1 : 0.8)
      .style('font-weight', highlight(r, i) ? 600 : 400)
      .text(r.label);
    const len = Math.max(0, x(r.value));
    row.append('rect')
      .attr('x', labelWidth).attr('y', 0)
      .attr('width', len).attr('height', bh)
      .attr('fill', r.colour || colour)
      .attr('opacity', highlight(r, i) ? 1 : 0.72);
    const inside = len > 46;
    row.append('text')
      .attr('x', labelWidth + len + (inside ? -6 : 6))
      .attr('y', bh / 2).attr('dy', '0.34em')
      .attr('text-anchor', inside ? 'end' : 'start')
      .style('font-family', 'var(--font-mono)')
      .style('font-size', `${size - 0.5}px`)
      .style('fill', inside ? 'var(--paper)' : 'var(--squidink)')
      .style('opacity', inside ? 0.95 : 0.75)
      .text(valueFmt(r.value));
  });
}

/**
 * A run of tokens laid out as wrapping chips.
 *
 * Returns the placed rows so a caller can point at token i without measuring
 * anything itself. Widths come from `getComputedTextLength`, so a long word
 * pushes the rest along instead of overlapping it.
 */
export function tokenStrip(g, tokens, w, {
  size = 12, padX = 5, padY = 4, lineGap = 6, fill = () => 'transparent',
  textFill = () => 'var(--squidink)', opacity = () => 1, title = () => null,
} = {}) {
  g.selectAll('*').remove();
  const rowH = size + 2 * padY;
  let x = 0;
  let y = 0;
  const placed = [];
  tokens.forEach((tok, i) => {
    const t = g.append('text')
      .style('font-family', 'var(--font-mono)')
      .style('font-size', `${size}px`)
      .text(tok);
    const tw = t.node().getComputedTextLength();
    const cw = tw + 2 * padX;
    if (x + cw > w && x > 0) { x = 0; y += rowH + lineGap; }
    t.remove();
    const cell = g.append('g').attr('transform', `translate(${x},${y})`)
      .attr('class', 'token-chip').attr('data-token', i);
    cell.append('rect')
      .attr('width', cw).attr('height', rowH).attr('rx', 3)
      .attr('fill', fill(tok, i)).attr('opacity', opacity(tok, i));
    cell.append('text')
      .attr('x', padX).attr('y', rowH / 2).attr('dy', '0.34em')
      .style('font-family', 'var(--font-mono)')
      .style('font-size', `${size}px`)
      .style('fill', textFill(tok, i))
      .text(tok);
    const tip = title(tok, i);
    if (tip) cell.append('title').text(tip);
    placed.push({ i, x, y, w: cw, h: rowH, cell });
    x += cw + 3;
  });
  return { rows: placed, height: y + rowH };
}

/**
 * A matrix as a grid of squares: attention, confusion, co-occurrence.
 *
 * `scale` maps a value to a colour. There is no default divergent ramp on
 * purpose: an earlier article shipped one whose two branches were both red,
 * which is fine for a magnitude and useless for a sign, so every caller states
 * what its colours mean.
 */
export function matrix(g, values, n, m, w, h, { scale, stroke = null,
  title = null } = {}) {
  g.selectAll('*').remove();
  const cw = w / m;
  const ch = h / n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const v = values[i * m + j];
      const cell = g.append('rect')
        .attr('x', j * cw).attr('y', i * ch)
        .attr('width', Math.ceil(cw) + 0.5).attr('height', Math.ceil(ch) + 0.5)
        .attr('fill', scale(v, i, j));
      if (stroke) cell.attr('stroke', stroke).attr('stroke-width', 0.4);
      if (title) cell.append('title').text(title(v, i, j));
    }
  }
}

/** A monochrome ramp in the article's accent, for a non-negative quantity. */
export function accentRamp(t) {
  const u = Math.max(0, Math.min(1, t));
  return d3.interpolateLab('#f1f3f3', 'var(--primary)')(u);
}

/**
 * Ticks for a logarithmic axis that crosses one, built by hand.
 *
 * `d3.ticks` on the exponent prints 316.228, and the scale's own ticks print
 * eighteen overlapping labels. Powers of ten and their triples, filtered to
 * the domain, is what actually reads.
 */
export function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    [1, 3].forEach((m) => {
      const v = m * 10 ** e;
      if (v >= lo && v <= hi) out.push(v);
    });
  }
  return out;
}

/** Compact axis numbers: 12,000 as 12k, 1,200,000 as 1.2M. */
export function shortNum(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  if (a >= 1) return `${v}`;
  return v.toPrecision(2);
}
