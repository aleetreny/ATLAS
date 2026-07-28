/* The same twenty-four words, as each of the three objectives receives them.
 *
 * The whole article rests on one sentence, "what separates them is a mask and
 * a target and nothing else", and a sentence like that should be shown rather
 * than asserted. So this widget draws the actual training example: the tokens
 * the model gets as input, which of them were destroyed on the way in, and
 * which positions it is graded on afterwards.
 *
 * The three corrupted examples come out of the generator, produced by the same
 * `corrupt_mlm` and `corrupt_spans` the training loop calls. Re-implementing
 * the masking here in JavaScript would have been easier and would have turned
 * a measurement into an illustration of one, which is the trade this site
 * never makes.
 *
 * Two channels, and they are deliberately different marks rather than two
 * shades of the accent:
 *
 *   a filled chip    the model does NOT see the real word here
 *   a bar underneath this position is graded
 *
 * Which is what makes the contrast readable at a glance: the masked arm has a
 * few of each, the causal arm has a bar under every single token and not one
 * filled chip, and the span arm keeps its grading in a second sequence
 * entirely.
 */
import { makeChart } from '../../assets/js/chart.js';
import { seriesLabel, tokenStrip } from '../../assets/js/textdraw.js';

const ARMS = [
  { key: 'bert', label: 'masked (BERT)' },
  { key: 'gpt', label: 'causal (GPT)' },
  { key: 't5', label: 'span corruption (T5)' },
];

export function initObjectiveWidget(data) {
  const E = data.examples;
  if (!E) return null;
  let arm = 'bert';

  const chart = makeChart('#objective-chart', {
    width: 640, height: 260, margin: { top: 34, right: 16, bottom: 16, left: 16 },
  });
  const { svg, g, w, margin } = chart;
  const readout = d3.select('#objective-readout');

  d3.select('#objective-arms').selectAll('button').data(ARMS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === arm ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      arm = d.key;
      d3.select('#objective-arms').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === arm ? '' : ' ghost'}`);
      render();
    });

  /* A strip plus the graded bars under it, returning the height it used so the
     caller can stack the next thing below it. */
  function strip(parent, y, tokens, { graded = [], replaced = [], caption }) {
    const gradedSet = new Set(graded);
    const replacedSet = new Set(replaced);
    const holder = parent.append('g').attr('transform', `translate(0,${y})`);
    seriesLabel(holder, 0, -8, caption, { size: 11, opacity: 0.75 });
    const body = holder.append('g');
    const laid = tokenStrip(body, tokens, w, {
      size: 12,
      fill: (t, i) => (replacedSet.has(i) ? 'var(--primary)' : 'transparent'),
      textFill: (t, i) => (replacedSet.has(i) ? 'var(--paper)' : 'var(--squidink)'),
      opacity: (t, i) => (replacedSet.has(i) ? 0.92 : 1),
      title: (t, i) => (replacedSet.has(i)
        ? 'the model does not see the original word here'
        : (gradedSet.has(i) ? 'graded on this position' : null)),
    });
    laid.rows.forEach((r) => {
      if (!gradedSet.has(r.i)) return;
      body.append('rect')
        .attr('x', r.x).attr('y', r.y + r.h + 1.5)
        .attr('width', r.w).attr('height', 2.5)
        .attr('fill', 'var(--secondary)').attr('opacity', 0.85);
    });
    return laid.height + 8;
  }

  function render() {
    g.selectAll('*').remove();
    const ex = E[arm];
    let y = 6;
    y += strip(g, y, ex.input, {
      graded: ex.graded,
      replaced: ex.replaced,
      caption: 'what the model reads',
    });
    if (ex.target) {
      y += 16;
      y += strip(g, y, ex.target, {
        /* From the second token on. The decoder is fed the target shifted by
         * one, so it is graded on every position except the one it is given to
         * start with, and that is also the count the readout prints. */
        graded: ex.target.map((t, i) => i).slice(1),
        replaced: ex.target.map((t, i) => (t.startsWith('<x') ? i : -1)).filter((i) => i >= 0),
        caption: 'and what it has to produce, as a second sequence',
      });
    }
    svg.attr('viewBox', `0 0 ${chart.width} ${y + margin.top + margin.bottom}`);

    const shown = ex.input.length;
    const graded = ex.target ? ex.target.length - 1 : ex.graded.length;
    const hidden = ex.replaced.length;
    /* What this particular draw happens to contain, rather than what the recipe
     * says a draw contains. Eighty-ten-ten on five masked tokens produces a
     * random substitution about half the time, so a sentence pointing at one
     * that is not there sends the reader hunting for it. */
    const swapped = ex.replaced.filter((i) => ex.input[i] !== '<mask>').length;
    const gradedUntouched = ex.graded.filter((i) => !ex.replaced.includes(i)).length;
    const legend = '<div><span class="bold">Filled</span> means the model does not see the real '
      + 'word there. <span class="bold">Underlined</span> means it is graded on that position.</div>';
    const drawSays = (swapped === 0 && gradedUntouched === 0)
      ? 'the draw put the placeholder on every one of them, so the other two cases are not on '
        + 'screen. This is one fixed example rather than a fresh sample, so the counts stay put'
      : `${swapped} of the ${hidden} hidden tokens ${swapped === 1 ? 'was' : 'were'} replaced by `
        + `a different word rather than the placeholder, and ${gradedUntouched} graded `
        + `${gradedUntouched === 1 ? 'position was' : 'positions were'} left untouched`;
    readout.html(
      `<div>Of the <span class="value">${shown}</span> tokens it reads here, `
      + (ex.target
        /* The span arm counts differently and saying otherwise would be wrong
         * rather than merely clumsy: a sentinel is not a destroyed token, it is
         * a whole run of them gone, and the grading is not in this sequence. */
        ? `<span class="bold">${hidden}</span> ${hidden === 1 ? 'is a sentinel standing' : 'are sentinels standing'} `
          + `in for a run of words that is gone entirely, and the grading happens on the `
          + `<span class="bold">${graded}</span> tokens of the second sequence.</div>`
        : `<span class="bold">${hidden}</span> ${hidden === 1 ? 'was' : 'were'} destroyed on the `
          + `way in and it is graded on <span class="bold">${graded}</span>.</div>`)
      + legend
      + (arm === 'bert'
        ? '<div>Not every hidden token becomes the placeholder. The recipe from the paper is '
          + 'eighty-ten-ten: eight in ten of the chosen positions get the placeholder, one in ten '
          + 'gets a random word instead, and one in ten is left exactly as it was and graded '
          + 'anyway. It exists so the model never learns that the placeholder is the only position '
          + 'worth thinking about, which matters because it will never see a placeholder again '
          + `after pretraining. In this passage ${drawSays}.</div>`
        : arm === 'gpt'
          ? '<div>Nothing is filled, because the causal objective destroys nothing. It hides the '
            + 'future with a mask inside the attention instead, and grades every position at once '
            + 'against the next word. That is why the bar runs the whole length of the sentence '
            + 'and why this arm gets so much more supervision out of the same text.</div>'
          : '<div>The deleted runs are gone from the input entirely, each replaced by a single '
            + 'sentinel, so the model cannot even count how many words are missing. The grading '
            + 'happens in the second sequence, which the decoder produces one token at a time; '
            + 'that decoder is why this arm buys fewer layers per side at the same budget.</div>')
    );
  }

  render();
  return { render };
}
