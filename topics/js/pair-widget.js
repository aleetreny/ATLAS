/* Two runs of the same model on the same data, matched up and put side by side.
 *
 * The pairing matters. A topic model has no canonical order, so topic 3 of one
 * run is not topic 3 of another, and laying two tables next to each other by
 * position exaggerates the difference enormously: it measures the permutation.
 * These are matched by the optimal assignment on cosine similarity first, so
 * what is left on screen is the disagreement that survives giving the two runs
 * every chance to line up.
 *
 * Words shared between a matched pair are drawn in the accent, words that
 * appear in only one are drawn faint. The rows are sorted by how well they
 * matched, so the top of the list is the model at its most reproducible and
 * the bottom is what a reader would actually be surprised by.
 */
import { seriesLabel } from '../../assets/js/textdraw.js';

const VIEWS = [
  { key: 'best', label: 'the pairs that agreed most' },
  { key: 'worst', label: 'the pairs that agreed least' },
];

export function initPairWidget(data) {
  const M = data.stability.matched;
  const seeds = data.stability.seeds;
  let view = 'best';

  const host = d3.select('#pair-chart');
  const readout = d3.select('#pair-readout');

  d3.select('#pair-views').selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      view = d.key;
      d3.select('#pair-views').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });

  function render() {
    const rows = view === 'best' ? M.slice(0, 4) : M.slice(-4).reverse();
    host.html('');
    const table = host.append('table').attr('class', 'pair-table');
    const head = table.append('thead').append('tr');
    head.append('th').text('match');
    head.append('th').text(`run with seed ${seeds[0]}`);
    head.append('th').text(`run with seed ${seeds[1]}`);
    const body = table.append('tbody');
    rows.forEach((m) => {
      const tr = body.append('tr');
      tr.append('td').attr('class', 'sim').text(m.cosine.toFixed(3));
      [m.a, m.b].forEach((words) => {
        const td = tr.append('td');
        words.forEach((word, i) => {
          td.append('span')
            .attr('class', m.shared.includes(word) ? 'shared' : 'only')
            .text(word);
          if (i < words.length - 1) td.append('span').attr('class', 'sep').text(' ');
        });
      });
    });

    const shown = rows;
    const meanShared = shown.reduce((a, m) => a + m.shared.length, 0) / shown.length;
    readout.html(
      `<div>Four of the ${M.length} matched pairs, `
      + `${view === 'best' ? 'the ones that agreed most' : 'the ones that agreed least'}. `
      + `Words in the accent appear in both runs; faint words appear in only one. These four `
      + `share <span class="value">${meanShared.toFixed(1)}</span> of eight heading words on `
      + `average, with matched similarity from `
      + `<span class="value">${Math.min(...shown.map((m) => m.cosine)).toFixed(3)}</span> to `
      + `<span class="value">${Math.max(...shown.map((m) => m.cosine)).toFixed(3)}</span>.</div>`
      + `<div>Both runs used the same corpus, the same number of topics and the same priors. The `
      + `only difference between them is the number the random number generator started from, `
      + `which is not a modelling decision and is almost never reported.</div>`
    );
  }

  render();
  return { render };
}
