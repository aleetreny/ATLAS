/* The sampler, running here, with the answer key drawn next to it.
 *
 * Every other picture of LDA shows the finished topics. This one shows the
 * chain: 600 documents down the side, sorted into their true categories with
 * a rule between each, and the eight topics across. A cell is how much of that
 * document the model currently assigns to that topic. At sweep zero every
 * token has a topic picked out of a hat and the panel is uniform grey. Blocks
 * appear or they do not, and whether they line up with the rules is the whole
 * question of the article.
 *
 * The sweeps run synchronously in the click handler rather than on an
 * animation frame. A frame callback never fires when the browser panel is not
 * compositing, which leaves the control dead for any check running headless
 * and looking, correctly, like a broken widget.
 */
import { makeChart } from '../../assets/js/chart.js';
import { seriesLabel, fitText } from '../../assets/js/textdraw.js';
import { makeCorpus, newSampler, sweep, phiOf, thetaOf, topWords, jointLoglik, alignScore }
  from './ldakit.js';

const RUNS = [
  { n: 1, label: 'one sweep' },
  { n: 10, label: 'ten more' },
  { n: 50, label: 'fifty more' },
];

export function initSamplerWidget(data) {
  const S = data.sampler;
  const corpus = makeCorpus(S);
  const cats = S.label_names || data.meta.categories;
  let state = newSampler(corpus, { k: S.k, alpha: S.alpha, beta: S.beta, seed: S.seed });
  let trace = [{ sweep: 0, loglik: jointLoglik(state) }];

  /* Documents grouped by their true category, which is the comparison the
   * panel exists to make. The order is fixed once so the rows do not move
   * between sweeps. */
  const order = corpus.labels
    .map((l, i) => ({ i, c: cats.indexOf(l) }))
    .sort((a, b) => (a.c - b.c) || (a.i - b.i));
  const bounds = [];
  order.forEach((o, r) => {
    if (r === 0 || o.c !== order[r - 1].c) bounds.push({ row: r, cat: cats[o.c] });
  });

  const chart = makeChart('#sampler-chart', {
    width: 640, height: 430, margin: { top: 36, right: 196, bottom: 34, left: 96 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#sampler-readout');

  d3.select('#sampler-runs').selectAll('button').data(RUNS).join('button')
    .attr('class', 'atlas-btn ghost')
    .text((d) => d.label)
    .on('click', (event, d) => {
      for (let i = 0; i < d.n; i++) sweep(state);
      trace.push({ sweep: state.sweeps, loglik: jointLoglik(state) });
      render();
    });

  d3.select('#sampler-reset').selectAll('button').data([0]).join('button')
    .attr('class', 'atlas-btn ghost')
    .text('back to sweep zero')
    .on('click', () => {
      state = newSampler(corpus, { k: S.k, alpha: S.alpha, beta: S.beta, seed: S.seed });
      trace = [{ sweep: 0, loglik: jointLoglik(state) }];
      render();
    });

  function render() {
    const theta = thetaOf(state);
    const phi = phiOf(state);
    const tops = topWords(phi, corpus.words, 4);
    const al = alignScore(theta, corpus.labels, cats);

    g.selectAll('*').remove();
    const cw = w / state.k;
    const ch = h / order.length;

    /* One rect per document per topic. 600 x 8 is 4,800 nodes, which is more
     * than this site usually draws, so they go in as one path-free block of
     * rects with no per-cell listeners: the tooltip would cost more than the
     * picture is worth and the picture is the point. */
    const cells = g.append('g');
    order.forEach((o, r) => {
      const row = theta[o.i];
      for (let t = 0; t < state.k; t++) {
        cells.append('rect')
          .attr('x', t * cw).attr('y', r * ch)
          .attr('width', cw + 0.4).attr('height', Math.max(0.9, ch + 0.4))
          .attr('fill', 'var(--primary)')
          .attr('opacity', Math.min(1, row[t]));
      }
    });

    /* the answer key: a rule and a name at each category boundary */
    bounds.forEach((b) => {
      g.append('line')
        .attr('x1', -6).attr('x2', w).attr('y1', b.row * ch).attr('y2', b.row * ch)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('opacity', 0.55);
      seriesLabel(g, -10, b.row * ch + 11, b.cat, { anchor: 'end', size: 10.5, opacity: 0.9 });
    });

    /* The topics themselves, to the right. Their length is data and changes
       while the reader watches, so each one is placed and then trimmed to the
       margin that actually exists rather than to one guessed in advance. */
    const room = chart.margin.right - 14;
    tops.forEach((t, i) => {
      fitText(g, w + 10, i * 15 + 12, `${i + 1}. ${t.map((x) => x.word).join(' ')}`, room,
        { anchor: 'start', size: 10 });
    });
    fitText(g, w + 10, -8, 'heaviest words per topic', room,
      { anchor: 'start', size: 10, opacity: 0.7 });
    seriesLabel(g, 0, -8, `topics 1 to ${state.k}`, { anchor: 'start', size: 10.5 });
    seriesLabel(g, w, h + 22, `sweep ${state.sweeps}`, { anchor: 'end', size: 11 });

    const first = trace[0].loglik;
    const now = trace[trace.length - 1].loglik;
    readout.html(
      (state.sweeps === 0
        ? `<div>Every one of the `
          + `<span class="value">${corpus.nTok.toLocaleString('en-US')}</span> tokens has just been `
          + `given a topic out of a hat. The joint log likelihood of that is `
          + `<span class="value">${now.toLocaleString('en-US', { maximumFractionDigits: 0 })}</span>, `
          + `and every sweep from here has to improve on it.</div>`
        : `<div>After <span class="value">${state.sweeps}</span> `
          + `sweep${state.sweeps === 1 ? '' : 's'} over `
          + `<span class="value">${corpus.nTok.toLocaleString('en-US')}</span> tokens, the joint `
          + `log likelihood is <span class="value">${now.toLocaleString('en-US', { maximumFractionDigits: 0 })}</span>, `
          + `up from <span class="value">${first.toLocaleString('en-US', { maximumFractionDigits: 0 })}</span> `
          + `at the random start.</div>`)
      + `<div>Assigning each document to its heaviest topic and scoring that against the `
      + `${cats.length} hand-written categories: adjusted Rand index `
      + `<span class="bold">${al.ari.toFixed(4)}</span>, purity `
      + `<span class="bold">${al.purity.toFixed(4)}</span>, and the best one-to-one matching of `
      + `topics to categories gets <span class="bold">${(100 * al.oneToOne).toFixed(1)}%</span> `
      + `of documents right.</div>`
      + (state.sweeps === 0
        ? `<div>That is the score of pure noise, and it is the number every later sweep has to `
          + `be read against.</div>`
        : `<div>The model has never seen a category name. Whether the blocks line up with the `
          + `rules on the left is the only question this article is asking.</div>`)
    );
  }

  render();
  return { render, state: () => state };
}
