/* The retrieval half of a two tower model, running here.
 *
 * Pick a few books, the page averages their vectors into a query and scores
 * the whole catalogue with one dot product each. That is not a demonstration
 * of the idea, it is the idea: the item side was computed once and frozen, the
 * user side is one small computation, and the scoring is a matrix vector
 * product that a database can do with an index. The guard checks the top five
 * for a fixed set of picks against the generator's own.
 */
export function initTowerWidget(data) {
  const items = data.widget.items;
  const V = data.widget.vectors;
  const bias = data.widget.bias;
  const picks = document.querySelector('#tower-picks');
  const buttons = document.querySelector('#tower-buttons');
  const list = document.querySelector('#tower-list');
  const readout = document.querySelector('#tower-readout');
  const note = document.querySelector('#tower-note');

  /* the pickable shelf is the first thirty of the catalogue by popularity,
     which is what a reader will recognise; the scoring runs over all of it */
  const shelf = items.map((it, i) => i).slice(0, 30);
  const st = { chosen: new Set([shelf[0], shelf[3], shelf[7]]), withBias: true };

  shelf.forEach((i) => {
    const b = document.createElement('button');
    b.textContent = items[i].title;
    b.addEventListener('click', () => {
      if (st.chosen.has(i)) st.chosen.delete(i);
      else st.chosen.add(i);
      render();
    });
    picks.appendChild(b);
  });

  const btns = {};
  for (const [key, text] of [['with', 'with the popularity term'], ['without', 'without it']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.withBias = key === 'with'; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function score(chosen, withBias) {
    const k = V[0].length;
    const q = new Float64Array(k);
    chosen.forEach((i) => { for (let c = 0; c < k; c++) q[c] += V[i][c]; });
    if (chosen.size) for (let c = 0; c < k; c++) q[c] /= chosen.size;
    return V.map((row, i) => {
      let s = 0;
      for (let c = 0; c < k; c++) s += row[c] * q[c];
      return { i, s: s + (withBias ? bias[i] : 0) };
    });
  }

  function render() {
    btns.with.classList.toggle('ghost', !st.withBias);
    btns.without.classList.toggle('ghost', st.withBias);
    [...picks.children].forEach((b, n) => b.classList.toggle('on', st.chosen.has(shelf[n])));

    const scored = score(st.chosen, st.withBias)
      .filter((d) => !st.chosen.has(d.i))
      .sort((a, b) => b.s - a.s)
      .slice(0, 8);

    const html = ['<table class="rank-table"><thead><tr><th>What it would show next</th>'
      + '<th class="num">score</th><th class="num">rank in the catalogue</th></tr></thead><tbody>'];
    scored.forEach((d) => {
      const it = items[d.i];
      html.push(`<tr><td class="title">${it.title}<br /><span class="author">${it.author}</span>`
        + `${it.cold ? ' <span class="author">(no ratings in training)</span>' : ''}</td>`
        + `<td class="num">${d.s.toFixed(3)}</td>`
        + `<td class="num">#${it.rank.toLocaleString('en-US')}</td></tr>`);
    });
    html.push('</tbody></table>');
    list.innerHTML = html.join('');

    const meanRank = Math.round(scored.reduce((a, d) => a + items[d.i].rank, 0) / scored.length);
    const chosenNames = [...st.chosen].map((i) => items[i].title);
    const cold = scored.filter((d) => items[d.i].cold).length;
    readout.innerHTML = st.chosen.size === 0
      ? 'Nothing is selected, so the query vector is zero and the list is whatever the popularity '
        + 'term alone prefers. Pick a book or three above.'
      : `Query built from <span class="value">${st.chosen.size}</span> book`
        + `${st.chosen.size === 1 ? '' : 's'} (${chosenNames.slice(0, 3).join(', ')}`
        + `${chosenNames.length > 3 ? `, and ${chosenNames.length - 3} more` : ''}), `
        + `scored against all <span class="value">${items.length}</span> item vectors with one dot `
        + `product each. The eight it returns sit at rank `
        + `<span class="value">${meanRank.toLocaleString('en-US')}</span> in the catalogue on `
        + `average${cold ? `, and <span class="value">${cold}</span> of them had every training `
          + 'rating deleted, so their vector comes from their author and their title alone' : ''}. `
        + `${st.withBias
          ? 'The popularity term is on, which is what a retrieval model is usually deployed with.'
          : 'With the popularity term off, the list is what the query vector alone prefers, which '
            + 'is more personal and less defensible.'}`;

    note.textContent = 'The item vectors were computed once by the generator and frozen. Nothing '
      + 'here retrains anything: adding a book to the query changes one average and eight hundred '
      + 'multiplications, which is exactly the property the architecture was chosen for.';
  }

  render();
  return { render, score, state: st };
}
