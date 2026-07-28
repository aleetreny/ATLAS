/* Three objectives on one budget, on whichever axis you want to compare them.
 *
 * The three views are not three ways of drawing the same thing. Each of them
 * is a different question, and the reason the widget exists is that the three
 * questions have three different winners:
 *
 *   the probe asks what the representation knows, frozen;
 *   the gap asks how well it can fill in a word it cannot see;
 *   the signal asks how many supervised predictions the objective extracted
 *   from each token it read, which is the one nobody prints and the one that
 *   explains the shape of the last five years.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { seriesLabel, wordBars } from '../../assets/js/textdraw.js';

const VIEWS = [
  { key: 'probe', label: 'what the frozen representation knows', fmt: (v) => `${(100 * v).toFixed(2)}%` },
  { key: 'cloze', label: 'filling in a word it cannot see', fmt: (v) => `${(100 * v).toFixed(2)}%` },
  { key: 'signal_per_token', label: 'supervised predictions per token read', fmt: (v) => v.toFixed(3) },
];

export function initArmsWidget(data) {
  let view = 'probe';
  const chart = makeChart('#arms-chart', {
    width: 640, height: 260, margin: { top: 40, right: 24, bottom: 26, left: 12 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#arms-readout');

  d3.select('#arms-views').selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      view = d.key;
      d3.select('#arms-views').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });

  function render() {
    const V = VIEWS.find((v) => v.key === view);
    const rows = data.arms.map((a) => ({ label: a.model, value: a[view], arm: a }));
    const best = rows.reduce((p, c) => (c.value > p.value ? c : p), rows[0]);
    g.selectAll('*').remove();
    wordBars(g, rows, w, h - 10, {
      labelWidth: 190,
      size: 12,
      valueFmt: V.fmt,
      highlight: (r) => r.label === best.label,
    });
    seriesLabel(g, 0, -16, V.label, { anchor: 'start', size: 11 });

    const others = rows.filter((r) => r.label !== best.label);
    const gap = best.value - Math.max(...others.map((r) => r.value));
    readout.html(
      `<div>On this measure <span class="bold">${best.label}</span> leads with `
      + `<span class="bold">${V.fmt(best.value)}</span>, ahead of the next by `
      + `<span class="value">${V.fmt(gap)}</span>.</div>`
      + (view === 'probe'
        ? `<div>The probe is a single linear layer trained on top of a completely frozen model, `
          + `on the part-of-speech task the two previous articles used. Nothing inside the model `
          + `moves, so what is being measured is what the pretraining left behind and not how much `
          + `capacity the model has.</div>`
        : view === 'cloze'
          ? `<div>Filling a gap in the middle of a passage is the task that <span class="bold">`
            + `favours the bidirectional objectives by construction</span>, and saying so is part `
            + `of reporting it: the causal model is being asked to guess a word using only what `
            + `came before it, which is the one thing it was built to do and the harder version of `
            + `this question.</div>`
          : `<div>This is the column nobody prints. Each objective read the same number of tokens `
            + `and took the same number of optimiser steps; what differs is how many of those `
            + `tokens it was actually graded on. A causal objective is graded on every position; a `
            + `masked one only on the positions it hid. Three quarters of the difference in how `
            + `these three families scaled over the last five years is in this ratio.</div>`)
    );
  }

  render();
  return { render };
}
