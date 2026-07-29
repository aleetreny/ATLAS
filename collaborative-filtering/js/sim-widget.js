/* Pick a book, pick a definition of similar, and read the neighbours.
 *
 * This is the widget the article exists for. The four definitions are all
 * defensible, all one line of arithmetic, and they return visibly different
 * books: the two that work on the ones and zeros hand back whatever everybody
 * reads, and the one that works on what the bias model left over hands back
 * the same series and the same author. The popularity column is there so that
 * this is a measurement and not an impression.
 */
import { makeSim } from './simkit.js';

const FLAVOUR_ORDER = ['count', 'jaccard', 'rating', 'centred'];
const LABELS = {
  count: 'who read both',
  jaccard: 'both over either',
  rating: 'raw stars',
  centred: 'centred stars',
};
const LAMBDAS = [0, 1, 2, 5, 10, 25, 50, 100, 250];

export function initSimWidget(data) {
  const sim = makeSim(data.widget);
  const picker = document.querySelector('#sim-book');
  const buttons = document.querySelector('#sim-buttons');
  const slider = document.querySelector('#sim-shrink');
  const shrinkLabel = document.querySelector('#sim-shrink-value');
  const list = document.querySelector('#sim-list');
  const readout = document.querySelector('#sim-readout');
  const note = document.querySelector('#sim-note');

  /* The default is not the first book in the file and not the most popular
     one: it is a book from a long series, where a reader can tell a good
     neighbour from a popular one at a glance. */
  const preferred = sim.items.findIndex((it) => /Harry Potter|Hunger Games|Twilight/.test(it.title));
  const st = { book: preferred >= 0 ? preferred : 0, flavour: 'count', lam: 0 };

  sim.items.forEach((it, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `${it.title}, ${it.author}`;
    picker.appendChild(o);
  });
  picker.value = String(st.book);
  picker.addEventListener('change', () => { st.book = Number(picker.value); render(); });

  const btns = {};
  FLAVOUR_ORDER.forEach((key) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = LABELS[key];
    b.addEventListener('click', () => { st.flavour = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  });

  slider.addEventListener('input', () => { st.lam = LAMBDAS[Number(slider.value)]; render(); });

  const pct = (v, d = 0) => `${(100 * v).toFixed(d)}%`;

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.flavour));
    shrinkLabel.textContent = st.lam === 0 ? 'nothing' : String(st.lam);

    const rows = sim.neighbours(st.book, st.flavour, st.lam, 8);
    const me = sim.items[st.book];

    const html = ['<table class="neigh-table"><thead><tr>'
      + '<th>Neighbour</th><th class="num">similarity</th>'
      + '<th class="num">shared readers</th><th class="num">its rank in the catalogue</th>'
      + '</tr></thead><tbody>'];
    rows.forEach((row) => {
      const it = sim.items[row.j];
      html.push(`<tr><td class="title">${it.title}<br />`
        + `<span class="author">${it.author}</span></td>`
        + `<td class="num">${row.s.toFixed(3)}</td>`
        + `<td class="num">${row.co.toLocaleString('en-US')}</td>`
        + `<td class="num">#${it.rank.toLocaleString('en-US')}</td></tr>`);
    });
    html.push('</tbody></table>');
    list.innerHTML = html.join('');

    const meanPct = rows.reduce((a, row) => a + sim.items[row.j].pct, 0) / rows.length;
    const meanCo = rows.reduce((a, row) => a + row.co, 0) / rows.length;
    const meanRank = rows.reduce((a, row) => a + sim.items[row.j].rank, 0) / rows.length;
    const sameAuthor = rows.filter((row) => sim.items[row.j].author === me.author).length;

    readout.innerHTML =
      `Neighbours of <span class="value">${me.title}</span> `
      + `(#${me.rank.toLocaleString('en-US')} in the catalogue, `
      + `${me.n.toLocaleString('en-US')} ratings) under `
      + `<span class="value">${LABELS[st.flavour]}</span>`
      + `${st.lam ? `, shrunk by ${st.lam}` : ''}. `
      + `The eight it returns sit at rank <span class="value">${Math.round(meanRank).toLocaleString('en-US')}</span> `
      + `on average, at the <span class="value">${(100 * meanPct).toFixed(1)}th</span> percentile of `
      + `popularity, on <span class="value">${Math.round(meanCo).toLocaleString('en-US')}</span> `
      + `shared readers each. `
      + `${sameAuthor > 0
        ? `<span class="value">${sameAuthor}</span> of the eight ${sameAuthor === 1 ? 'is' : 'are'} `
          + `by the same author, and nothing here knows what an author is.`
        : 'None of them is by the same author, and nothing here knows what an author is.'}`;

    note.innerHTML = st.flavour === 'centred'
      ? 'Centred means the bias model has already taken out the reader who likes everything, '
        + 'the book everyone likes, and the fact that the average rating on this site is four '
        + 'stars. What is left is disagreement, and two books are similar when the same people '
        + 'disagree about them in the same direction.'
      : st.flavour === 'rating'
        ? 'Cosine on the stars themselves looks like it uses the ratings, and mostly does not: '
          + 'four fifths of these ratings are four or five stars, so the vector is nearly the '
          + 'indicator of who read it with a constant on top.'
        : 'These two only ever see whether somebody rated a book, never what they said. That is '
          + 'not a weakness by itself, and for a site with no stars it is all there is.';
  }

  render();
  return { sim, render, state: st };
}
