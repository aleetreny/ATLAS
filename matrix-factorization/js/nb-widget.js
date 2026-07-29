/* Neighbours in factor space, computed here, from two different fits.
 *
 * The previous article found neighbours by comparing columns of the ratings
 * table directly. These are found by comparing the short vectors the
 * factorisation left behind, which is a different operation on the same data,
 * and the two lists are worth putting side by side. The second column is the
 * implicit fit, which never sees a star: it is trained on whether a book was
 * rated at all, weighted by confidence, and it is the one whose output is a
 * list rather than a number.
 */
const LABELS = {
  explicit: 'from the ratings fit',
  implicit: 'from the implicit fit',
};

export function initNbWidget(data) {
  const items = data.widget.items;
  const spaces = { explicit: data.widget.full, implicit: data.widget.implicit };
  const picker = document.querySelector('#nb-book');
  const buttons = document.querySelector('#nb-buttons');
  const lists = document.querySelector('#nb-lists');
  const readout = document.querySelector('#nb-readout');

  const preferred = items.findIndex((it) => /Harry Potter|Hunger Games|Gone Girl/.test(it.title));
  const st = { book: preferred >= 0 ? preferred : 0, both: true };

  items.forEach((it, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `${it.title}, ${it.author}`;
    picker.appendChild(o);
  });
  picker.value = String(st.book);
  picker.addEventListener('change', () => { st.book = Number(picker.value); render(); });

  const norms = {};
  Object.entries(spaces).forEach(([key, M]) => {
    norms[key] = M.map((row) => Math.sqrt(row.reduce((a, v) => a + v * v, 0)));
  });

  function neighbours(key, i, n = 6) {
    const M = spaces[key];
    const out = [];
    for (let j = 0; j < M.length; j++) {
      if (j === i) continue;
      let d = 0;
      for (let c = 0; c < M[j].length; c++) d += M[i][c] * M[j][c];
      out.push({ j, s: d / Math.max(norms[key][i] * norms[key][j], 1e-9) });
    }
    out.sort((a, b) => b.s - a.s);
    return out.slice(0, n);
  }

  const btns = {};
  for (const [key, text] of [['both', 'both fits'], ['explicit', 'ratings fit only']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.both = key === 'both'; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function table(key) {
    const rows = neighbours(key, st.book);
    const html = [`<div><p class="list-head">${LABELS[key]}</p>`,
      '<table class="rank-table"><thead><tr><th>Neighbour</th><th class="num">cosine</th>'
      + '<th class="num">rank</th></tr></thead><tbody>'];
    rows.forEach((row) => {
      const it = items[row.j];
      html.push(`<tr><td class="title">${it.title}<br /><span class="author">${it.author}</span></td>`
        + `<td class="num">${row.s.toFixed(3)}</td>`
        + `<td class="num">#${it.rank.toLocaleString('en-US')}</td></tr>`);
    });
    html.push('</tbody></table></div>');
    return { html: html.join(''), rows };
  }

  function render() {
    btns.both.classList.toggle('ghost', !st.both);
    btns.explicit.classList.toggle('ghost', st.both);
    const a = table('explicit');
    const b = table('implicit');
    lists.innerHTML = st.both ? a.html + b.html : a.html;

    const me = items[st.book];
    const meanRank = (rows) => Math.round(rows.reduce((s, r) => s + items[r.j].rank, 0) / rows.length);
    const shared = a.rows.filter((r) => b.rows.some((q) => q.j === r.j)).length;
    readout.innerHTML =
      `Neighbours of <span class="value">${me.title}</span> `
      + `(#${me.rank.toLocaleString('en-US')} in the catalogue), by cosine between the vectors the `
      + `fit produced. The ratings fit returns books at rank `
      + `<span class="value">${meanRank(a.rows).toLocaleString('en-US')}</span> on average`
      + `${st.both
        ? ` and the implicit fit at <span class="value">${meanRank(b.rows).toLocaleString('en-US')}</span>, `
          + `with <span class="value">${shared}</span> of six books in common. The implicit fit has `
          + `never seen a star: its input is whether a book was rated at all.`
        : '. The vectors are the same ones the model predicts with, so this list is what the model '
          + 'means by similar rather than a separate similarity computed on top.'}`;
  }

  render();
  return { render, neighbours, state: st };
}
