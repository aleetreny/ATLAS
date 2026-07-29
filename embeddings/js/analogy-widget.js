/* The famous trick, run here, with the line that makes it work left visible.
 *
 * "a is to b as c is to what" is answered by finding the word nearest to
 * b - a + c. The detail that decides the reported accuracy of every analogy
 * benchmark ever published is whether a, b and c are removed from the search
 * before taking the nearest. They usually are, and the reason is not stated:
 * without the exclusion the answer is very often c itself, because c is the
 * nearest thing to something that is almost c.
 *
 * So the button here toggles exactly that one line, and the readout says what
 * happened. It is the only widget in this article that can make the models
 * look better by doing less.
 */
import { seriesLabel } from '../../assets/js/textdraw.js';
import { analogy } from './veckit.js';

const SETS = {
  'singular to plural': [['year', 'years', 'day', 'days'], ['man', 'men', 'child', 'children'],
    ['car', 'cars', 'book', 'books'], ['hand', 'hands', 'eye', 'eyes']],
  'present to past': [['is', 'was', 'are', 'were'], ['say', 'said', 'come', 'came'],
    ['take', 'took', 'give', 'gave'], ['go', 'went', 'see', 'saw']],
  'man to woman': [['he', 'she', 'him', 'her'], ['man', 'woman', 'boy', 'girl'],
    ['father', 'mother', 'son', 'daughter']],
};

export function initAnalogyWidget(data, space) {
  let setName = Object.keys(SETS)[0];
  let exclude = true;

  const host = d3.select('#analogy-chart');
  const readout = d3.select('#analogy-readout');

  d3.select('#analogy-sets').selectAll('button').data(Object.keys(SETS)).join('button')
    .attr('class', (d) => `atlas-btn${d === setName ? '' : ' ghost'}`)
    .text((d) => d)
    .on('click', (event, d) => {
      setName = d;
      d3.select('#analogy-sets').selectAll('button')
        .attr('class', (q) => `atlas-btn${q === setName ? '' : ' ghost'}`);
      render();
    });

  d3.select('#analogy-models').selectAll('button')
    .data([{ k: true, label: 'excluding the three given words' },
      { k: false, label: 'searching all words' }])
    .join('button')
    .attr('class', (d) => `atlas-btn${d.k === exclude ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      exclude = d.k;
      d3.select('#analogy-models').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.k === exclude ? '' : ' ghost'}`);
      render();
    });

  function render() {
    const quads = SETS[setName];
    host.html('');
    const table = host.append('table').attr('class', 'an-table');
    const head = table.append('thead').append('tr');
    head.append('th').text('question');
    head.append('th').text('wanted');
    space.names.forEach((n) => head.append('th').text(n));
    const body = table.append('tbody');

    let asked = 0;
    const hits = Object.fromEntries(space.names.map((n) => [n, 0]));
    const echoes = Object.fromEntries(space.names.map((n) => [n, 0]));

    quads.forEach(([a, b, c, d]) => {
      const known = [a, b, c, d].every((w) => space.index.has(w));
      const tr = body.append('tr');
      tr.append('td').text(`${a} : ${b} :: ${c} : ?`);
      tr.append('td').text(known ? d : 'not in vocabulary');
      if (!known) {
        space.names.forEach(() => tr.append('td').text('-'));
        return;
      }
      asked += 1;
      space.names.forEach((n) => {
        const res = analogy(space, n, a, b, c, { exclude, k: 1 });
        const got = res && res[0] ? res[0].word : '-';
        const td = tr.append('td');
        td.attr('class', got === d ? 'hit' : 'miss').text(got);
        if (got === d) hits[n] += 1;
        if (got === a || got === b || got === c) echoes[n] += 1;
      });
    });

    const summary = space.names
      .map((n) => `<span class="bold">${n}</span> ${hits[n]} of ${asked}`)
      .join(', ');
    const echoed = space.names.reduce((acc, n) => acc + echoes[n], 0);
    readout.html(
      `<div>On these ${asked} questions: ${summary}.</div>`
      + (exclude
        ? `<div>The three given words are excluded from the search, which is what every published `
          + `accuracy on this task does and almost none of them mention. Turn it off and watch.</div>`
        : `<div>With the exclusion off, `
          + `<span class="bold">${echoed}</span> of the ${asked * space.names.length} answers are `
          + `just one of the three words in the question handed back. That is what the exclusion `
          + `is hiding, and it is the reason the convention exists.</div>`)
      + `<div>Trained on ${data.meta.tokens.toLocaleString('en-US')} words, which is small. `
      + `Analogy accuracy at this scale is not a claim about the method, it is a claim about this `
      + `corpus, and the number the table below gives over every question in every set is the one `
      + `to quote rather than these ${asked}.</div>`
    );
  }

  render();
  return { render };
}
