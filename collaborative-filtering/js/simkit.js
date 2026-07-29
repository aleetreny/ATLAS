/* The four similarities, computed here rather than replayed.
 *
 * A similarity between two books is a function of four numbers: how many
 * people rated each, how many rated both, and the dot product of the two
 * columns over the people who rated both. Ship those and the browser can
 * compute every flavour on the page, including ones the generator never
 * exported, which is what makes the shrinkage slider a computation instead of
 * a lookup. The generator's own cosine is checked against this file's at load
 * time, so the two definitions are one definition.
 *
 * The pair arrays are the upper triangle of a symmetric matrix, in row major
 * order: pair(i, j) with i < j sits at i*n - i*(i+1)/2 + j - i - 1.
 */
export function makeSim(widget) {
  const items = widget.items;
  const n = items.length;
  const idx = (i, j) => {
    const [a, b] = i < j ? [i, j] : [j, i];
    return (a * n - (a * (a + 1)) / 2 + b - a - 1);
  };

  const co = (i, j) => (i === j ? items[i].n : widget.co[idx(i, j)]);
  const dotRaw = (i, j) => (i === j ? items[i].norm_raw ** 2 : widget.dot_raw[idx(i, j)]);
  const dotC = (i, j) => (i === j ? items[i].norm_c ** 2 : widget.dot_c[idx(i, j)]);

  /* cosine on the ones and zeros: who read both, over who read either, in the
     geometric mean sense. Jaccard is the same counts arranged differently. */
  const cosine = (i, j) => co(i, j) / Math.sqrt(items[i].n * items[j].n);
  const jaccard = (i, j) => co(i, j) / (items[i].n + items[j].n - co(i, j));
  /* cosine on the ratings themselves, which is what a first implementation
     writes and which mostly measures how many people rated both */
  const rating = (i, j) => dotRaw(i, j) / (items[i].norm_raw * items[j].norm_raw);
  /* and on what is left after the bias model has taken out the reader, the
     book and the fact that everybody gives four stars */
  const centred = (i, j) => dotC(i, j) / (items[i].norm_c * items[j].norm_c);

  const FLAVOURS = {
    count: { fn: cosine, name: 'who read both', axis: 'cosine on the ones and zeros' },
    jaccard: { fn: jaccard, name: 'jaccard', axis: 'both over either' },
    rating: { fn: rating, name: 'raw ratings', axis: 'cosine on the stars themselves' },
    centred: { fn: centred, name: 'centred', axis: 'cosine on what the bias model left' },
  };

  /* Shrinkage. A similarity computed from four shared readers is not a small
     similarity, it is an unknown one, and multiplying by n/(n+lambda) is the
     cheap way of saying so: it costs nothing where n is large and pulls the
     four reader pairs to the floor. */
  function neighbours(i, flavour, lambda, k = 8) {
    const { fn } = FLAVOURS[flavour];
    const out = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const c = co(i, j);
      const raw = fn(i, j);
      const s = lambda > 0 ? (raw * c) / (c + lambda) : raw;
      out.push({ j, s, raw, co: c });
    }
    out.sort((a, b) => b.s - a.s);
    return out.slice(0, k);
  }

  return { items, n, co, cosine, jaccard, rating, centred, FLAVOURS, neighbours };
}
