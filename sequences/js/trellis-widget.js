/* Viterbi drawn as what it is: every path considered, one path kept.
 *
 * Twelve tags down the side, one column per word. Every cell holds the score
 * of the best path that ends in that tag at that word, and every cell keeps a
 * pointer back to the tag it came from, so the picture is the algorithm's own
 * table rather than an illustration of it.
 *
 * The comparison the widget exists for is the faint path against the solid
 * one. The faint path is greedy: at each word take whichever tag looks best
 * right there. The solid path is Viterbi, which will accept a worse tag at one
 * word to make a much better one possible three words later. Where the two
 * come apart is the entire value of the algorithm, on that sentence, and it is
 * a thing to look at rather than a claim to accept.
 */
import { makeChart } from '../../assets/js/chart.js';
import { seriesLabel, legend } from '../../assets/js/textdraw.js';
import { viterbi, greedy, scorePath } from './hmmkit.js';

export function initTrellisWidget(data, hmm) {
  /* Las frases viajan con el modelo, con sus indices ya remapeados al
   * vocabulario recortado que se envia, asi que aqui no se busca nada. */
  const examples = data.model.sentences;
  let ei = 0;
  let showGreedy = true;

  const chart = makeChart('#trellis-chart', {
    width: 640, height: 400, margin: { top: 40, right: 20, bottom: 44, left: 58 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#trellis-readout');

  d3.select('#trellis-views').selectAll('button')
    .data([{ k: true, label: 'with the greedy path' }, { k: false, label: 'Viterbi alone' }])
    .join('button')
    .attr('class', (d) => `atlas-btn${d.k === showGreedy ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      showGreedy = d.k;
      d3.select('#trellis-views').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.k === showGreedy ? '' : ' ghost'}`);
      render();
    });

  const slider = d3.select('#trellis-sentence')
    .attr('min', 0).attr('max', examples.length - 1).attr('step', 1).property('value', ei)
    .on('input', () => { ei = +slider.property('value'); render(); });

  function render() {
    const ex = examples[ei];
    const ids = ex.ids;
    const V = viterbi(hmm, ids);
    const G = greedy(hmm, ids);
    const n = ids.length;
    const T = hmm.T;

    d3.select('#trellis-sentence-label').text(`${ei + 1} of ${examples.length}`);
    g.selectAll('*').remove();

    const x = d3.scalePoint().domain(d3.range(n)).range([0, w]).padding(0.5);
    const y = d3.scalePoint().domain(d3.range(T)).range([0, h - 22]).padding(0.5);

    /* Cell shading is the score relative to the best in its own column. An
     * absolute scale would be useless: the scores fall by a few nats per word,
     * so by the end of a sentence every cell is dark and the column structure,
     * which is what the algorithm is choosing between, disappears. */
    const cells = g.append('g');
    for (let i = 0; i < n; i++) {
      let best = -Infinity;
      let worst = Infinity;
      for (let t = 0; t < T; t++) {
        const v = V.delta[i * T + t];
        if (Number.isFinite(v)) { best = Math.max(best, v); worst = Math.min(worst, v); }
      }
      for (let t = 0; t < T; t++) {
        const v = V.delta[i * T + t];
        const u = Number.isFinite(v) ? (v - worst) / Math.max(1e-9, best - worst) : 0;
        cells.append('circle')
          .attr('cx', x(i)).attr('cy', y(t)).attr('r', 3 + 5 * u)
          .attr('fill', 'var(--primary)')
          .attr('opacity', 0.13 + 0.65 * u)
          .append('title')
          .text(`${hmm.tags[t]} at "${ex.words[i]}": ${v.toFixed(2)}`);
      }
    }

    const line = (path, colour, width, dash, opacity) => {
      const p = d3.path();
      for (let i = 0; i < n; i++) {
        if (i === 0) p.moveTo(x(i), y(path[i]));
        else p.lineTo(x(i), y(path[i]));
      }
      g.append('path').attr('d', p.toString())
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', width)
        .attr('stroke-dasharray', dash).attr('opacity', opacity);
    };
    if (showGreedy) line(G, 'var(--cosmos)', 2, '5 4', 0.9);
    line(V.path, 'var(--squidink)', 2.4, null, 0.95);

    hmm.tags.forEach((t, i) => {
      seriesLabel(g, -8, y(i) + 4, t.toLowerCase(), { anchor: 'end', size: 10.5, opacity: 0.8 });
    });
    ex.words.forEach((word, i) => {
      seriesLabel(g, x(i), h - 4, word, { anchor: 'middle', size: 10.5, opacity: 0.9 });
    });

    legend(g, [
      { label: 'Viterbi', colour: 'var(--squidink)', shape: 'line' },
    ].concat(showGreedy
      ? [{ label: 'greedy, one word at a time', colour: 'var(--cosmos)', shape: 'dash' }]
      : []), { x0: 0, y0: -20, gap: 130, size: 10.5 });

    const same = Array.from(V.path).every((v, i) => v === G[i]);
    const diffs = Array.from(V.path).map((v, i) => (v === G[i] ? null : i)).filter((v) => v !== null);
    const goldPath = ex.gold.map((t) => hmm.tags.indexOf(t));
    const vHit = Array.from(V.path).filter((v, i) => v === goldPath[i]).length;
    const gHit = Array.from(G).filter((v, i) => v === goldPath[i]).length;

    readout.html(
      `<div>"${ex.words.join(' ')}"</div>`
      + `<div>Viterbi scores this tagging at `
      + `<span class="value">${V.score.toFixed(2)}</span>; the greedy path scores `
      + `<span class="value">${scorePath(hmm, ids, G).toFixed(2)}</span>, and a lower number is `
      + `a worse explanation of the same sentence. `
      + (same
        ? `On this sentence the two agree everywhere.`
        : `They disagree at ${diffs.length} word${diffs.length === 1 ? '' : 's'}: `
          + `${diffs.map((i) => `<span class="bold">${ex.words[i]}</span>`).join(', ')}.`)
      + `</div>`
      + `<div>Against the hand-assigned tags: Viterbi gets `
      + `<span class="bold">${vHit} of ${n}</span>, greedy gets `
      + `<span class="bold">${gHit} of ${n}</span>. Every circle is a partial path the algorithm `
      + `is keeping alive, and its size is how good that path is relative to the others in the `
      + `same column.</div>`
    );
  }

  render();
  return { render };
}
