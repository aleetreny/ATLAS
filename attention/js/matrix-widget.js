/* Every attention matrix in the model, computed here, with three switches.
 *
 * The switches are the article. Turning off the scaling makes the rows collapse
 * onto single cells, which is what the second figure predicts from the
 * dimension alone. Turning off the positions makes the model blind to order,
 * which is a statement about the whole function and not just this sentence.
 * Knocking out a head shows what that head was contributing, if anything.
 *
 * All three change numbers rather than pictures: the tags along the bottom are
 * re-predicted on every switch, so the reader can see a change of setting cost
 * the model an actual answer.
 */
import { makeChart } from '../../assets/js/chart.js';
import { seriesLabel, legend, accentRamp } from '../../assets/js/textdraw.js';
import { forward, meanEntropy, meanPeak } from './attnkit.js';

export function initMatrixWidget(data, model) {
  const sents = model.sentences;
  let si = 0;
  let layer = 0;
  let head = 0;
  let scale = true;
  let positional = true;
  let dropped = false;

  const chart = makeChart('#matrix-chart', {
    width: 640, height: 430, margin: { top: 40, right: 30, bottom: 108, left: 108 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#matrix-readout');

  const heads = [];
  for (let l = 0; l < model.layers; l++) {
    for (let hd = 0; hd < model.heads; hd++) heads.push({ l, hd });
  }

  d3.select('#matrix-heads').selectAll('button').data(heads).join('button')
    .attr('class', (d) => `atlas-btn${(d.l === layer && d.hd === head) ? '' : ' ghost'}`)
    .text((d) => `layer ${d.l + 1} head ${d.hd + 1}`)
    .on('click', (event, d) => { layer = d.l; head = d.hd; sync(); render(); });

  d3.select('#matrix-knobs').selectAll('button')
    .data([{ id: 'scale', label: 'divided by the root of the head size' },
      { id: 'pos', label: 'with positions' },
      { id: 'drop', label: 'this head knocked out' }])
    .join('button')
    .attr('class', 'atlas-btn ghost')
    .text((d) => d.label)
    .on('click', (event, d) => {
      if (d.id === 'scale') scale = !scale;
      else if (d.id === 'pos') positional = !positional;
      else dropped = !dropped;
      sync();
      render();
    });

  const slider = d3.select('#matrix-sentence')
    .attr('min', 0).attr('max', sents.length - 1).attr('step', 1).property('value', si)
    .on('input', () => { si = +slider.property('value'); render(); });

  function sync() {
    d3.select('#matrix-heads').selectAll('button')
      .attr('class', (d) => `atlas-btn${(d.l === layer && d.hd === head) ? '' : ' ghost'}`);
    d3.select('#matrix-knobs').selectAll('button')
      .attr('class', (d) => {
        const on = d.id === 'scale' ? scale : d.id === 'pos' ? positional : dropped;
        return `atlas-btn${on ? '' : ' ghost'}`;
      })
      .text((d) => {
        if (d.id === 'scale') return scale ? 'divided by the root of the head size' : 'not divided';
        if (d.id === 'pos') return positional ? 'with positions' : 'without positions';
        return dropped ? 'this head knocked out' : 'every head working';
      });
  }

  function render() {
    const s = sents[si];
    const opts = { scale, positional, drop: dropped ? [layer, head] : null };
    const res = forward(model, s.ids, opts);
    const ref = forward(model, s.ids, { scale: true, positional: true, drop: null });
    const map = res.maps[layer][head];
    const n = s.words.length;

    d3.select('#matrix-sentence-label').text(`${si + 1} of ${sents.length}`);
    g.selectAll('*').remove();
    const cw = w / n;
    const ch = h / n;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        g.append('rect')
          .attr('x', j * cw).attr('y', i * ch)
          .attr('width', cw + 0.4).attr('height', ch + 0.4)
          .attr('fill', accentRamp(map[i][j]))
          .append('title')
          .text(`"${s.words[i]}" attends to "${s.words[j]}": ${map[i][j].toFixed(4)}`);
      }
    }

    /* Rows are the word doing the looking, columns the word being looked at,
     * and that is worth writing down: half of all confusion about these
     * pictures is which way round they are. */
    s.words.forEach((word, i) => {
      seriesLabel(g, -8, i * ch + ch / 2 + 3.5, word, { anchor: 'end', size: 10, opacity: 0.85 });
      const t = seriesLabel(g, j2x(i, cw), h + 8, word, { anchor: 'end', size: 8, opacity: 0.85 });
      t.attr('transform', `rotate(-85 ${j2x(i, cw)} ${h + 8})`);
    });
    seriesLabel(g, -8, -12, 'looking', { anchor: 'end', size: 10, opacity: 0.6 });
    seriesLabel(g, 0, -12, 'looked at', { anchor: 'start', size: 10, opacity: 0.6 });

    const ent = meanEntropy([[map]]);
    const peak = meanPeak([[map]]);
    const changed = res.tags.map((t, i) => (t === ref.tags[i] ? null : i)).filter((v) => v !== null);
    const right = res.tags.filter((t, i) => model.tags[t] === s.gold[i]).length;

    readout.html(
      `<div>Layer ${layer + 1}, head ${head + 1}. Every row sums to one: it is where that word `
      + `looked. The mean entropy of these rows is `
      + `<span class="value">${ent.toFixed(3)}</span> against `
      + `<span class="value">${Math.log(n).toFixed(3)}</span> for looking everywhere equally, and `
      + `the average largest weight in a row is `
      + `<span class="value">${peak.toFixed(3)}</span>.</div>`
      + `<div>With these settings the model gets <span class="bold">${right} of ${n}</span> tags `
      + `right on this sentence`
      + (changed.length
        ? `, and changes its answer at ${changed.length} `
          + `position${changed.length === 1 ? '' : 's'} compared with the model as trained: `
          + `${changed.map((i) => `<span class="bold">${s.words[i]}</span>`).join(', ')}.`
        : `, the same answers as the model as trained.`)
      + `</div>`
      + (!positional
        ? `<div>Without positions this is no longer a function of a sentence, it is a function of `
          + `a <span class="bold">bag</span>: shuffle the words and every output moves with its `
          + `own word and nothing else changes. That is measured exactly rather than asserted, `
          + `and the number is in the paragraph below.</div>`
        : '')
      + (!scale
        ? `<div>Without dividing by the root of the head size, the rows concentrate. A row that `
          + `has collapsed onto one cell has almost no gradient left to give, which is the whole `
          + `reason the division is in the formula.</div>`
        : '')
    );
  }

  function j2x(i, cw) {
    return i * cw + cw / 2;
  }

  sync();
  render();
  return { render };
}
