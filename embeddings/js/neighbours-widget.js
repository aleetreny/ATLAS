/* Pick a word, get its ten nearest words, in whichever of the three models.
 *
 * All three are asked the same question about the same word at the same time,
 * which is the only way the comparison means anything: three separate figures
 * of three cherry-picked words would prove whatever the author wanted.
 *
 * The part-of-speech tag next to each neighbour is doing real work. The claim
 * this article tests is that what "similar" means is decided by the window
 * rather than by the algorithm, and the tag is what turns that from an
 * impression into a count: the readout says how many of the ten share the
 * query's own tag.
 */
import { seriesLabel } from '../../assets/js/textdraw.js';
import { neighbours, posAgreement } from './veckit.js';

export function initNeighboursWidget(data, space) {
  /* Words worth opening on: frequent enough to have been learned properly,
   * spread across parts of speech, and none of them a function word, because
   * the neighbours of "the" are not an interesting first impression. */
  const probes = pickProbes(space);
  let qi = 0;

  const host = d3.select('#neighbours-chart');
  const readout = d3.select('#neighbours-readout');

  const slider = d3.select('#neighbours-word')
    .attr('min', 0).attr('max', probes.length - 1).attr('step', 1).property('value', qi)
    .on('input', () => { qi = +slider.property('value'); render(); });

  function render() {
    const i = probes[qi];
    const word = space.words[i];
    d3.select('#neighbours-word-label').text(word);

    host.html('');
    const table = host.append('table').attr('class', 'nb-table');
    const head = table.append('thead').append('tr');
    head.append('th').text('#');
    space.names.forEach((n) => head.append('th').text(n));
    const body = table.append('tbody');

    const per = space.names.map((n) => neighbours(space, n, i, 10));
    for (let r = 0; r < 10; r++) {
      const tr = body.append('tr');
      tr.append('td').attr('class', 'rank').text(r + 1);
      per.forEach((list) => {
        const nb = list[r];
        const td = tr.append('td');
        if (!nb) return;
        td.append('span')
          .attr('class', nb.tag === space.tags[i] ? 'same-pos' : 'other-pos')
          .text(nb.word);
        td.append('span').attr('class', 'tag').text(` ${nb.tag.toLowerCase()}`);
        td.append('span').attr('class', 'cos').text(` ${nb.cos.toFixed(2)}`);
      });
    }

    const agree = space.names.map((n) => ({ n, ...posAgreement(space, n, i, 10) }));
    const shared = sharedCount(per);
    readout.html(
      `<div><span class="bold">${word}</span> occurs `
      + `<span class="value">${space.counts[i].toLocaleString('en-US')}</span> times in the `
      + `corpus and is tagged <span class="bold">${space.tags[i].toLowerCase()}</span> in `
      + `<span class="value">${(100 * space.tagPurity[i]).toFixed(0)}%</span> of them.</div>`
      + `<div>Neighbours sharing that tag: `
      + `${agree.map((a) => `<span class="bold">${a.n}</span> ${a.hits} of ${a.of}`).join(', ')}. `
      + `Words in the accent share the tag; the rest do not.</div>`
      + `<div>Only <span class="value">${shared}</span> word`
      + `${shared === 1 ? '' : 's'} appear in all three top tens. The three models saw the same `
      + `${data.meta.tokens.toLocaleString('en-US')} words of the same corpus and disagree about `
      + `what this one is like.</div>`
    );
  }

  render();
  return { render };
}

function sharedCount(per) {
  if (!per.length) return 0;
  const sets = per.map((l) => new Set(l.map((x) => x.word)));
  return [...sets[0]].filter((w) => sets.every((s) => s.has(w))).length;
}

/* Deterministic from the shipped data: the most frequent word carrying each of
 * the open-class tags, then filled out with more of the same, so the slider
 * runs across nouns, verbs, adjectives and adverbs rather than across ten
 * near-synonyms. */
function pickProbes(space) {
  const WANT = ['NOUN', 'VERB', 'ADJ', 'ADV'];
  const out = [];
  const perTag = 3;
  WANT.forEach((tag) => {
    let taken = 0;
    for (let i = 0; i < space.n && taken < perTag; i++) {
      if (space.tags[i] !== tag) continue;
      if (space.tagPurity[i] < 0.75) continue;
      if (space.words[i].length < 4) continue;
      out.push(i);
      taken += 1;
    }
  });
  return out.length ? out.sort((a, b) => a - b) : [0];
}
