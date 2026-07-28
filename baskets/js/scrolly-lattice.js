/* The opening story: five items, thirty-one subsets, and one property that
 * makes the other 2^169 tractable.
 *
 * Every support drawn is counted here from the shipped bitsets, so the numbers
 * on the little lattice are the real supports of real baskets.
 */
import { makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { enumerateAll } from './basketkit.js';
import { latticeLayout, legend, seriesLabel } from './draw.js';

const K = 5;

export function initScrollyLattice(data, db) {
  const all = enumerateAll(db, K);
  const keys = [...all.keys()];
  const minCount = Math.ceil(0.02 * db.n);

  const chart = makeChart('#lattice-scrolly-chart', {
    width: 640, height: 460, margin: { top: 60, right: 30, bottom: 40, left: 116 },
  });
  const { g, w, h } = chart;
  const { pos } = latticeLayout(keys, K, w, h, { top: 14, bottom: 14 });
  const short = (i) => db.names[i].split('/')[0].split(' ')[0];

  const gLinks = g.append('g');
  const gNodes = g.append('g');
  const gNote = g.append('g');

  /* an edge from every set to every set one item larger that contains it */
  const links = [];
  keys.forEach((key) => {
    const idx = key.split(',').map(Number);
    keys.forEach((other) => {
      const o = other.split(',').map(Number);
      if (o.length !== idx.length + 1) return;
      if (idx.every((v) => o.includes(v))) links.push({ a: key, b: other });
    });
  });

  function render(state) {
    const freq = (key) => all.get(key) >= minCount;
    /* a set is "cut" when some subset of it is already infrequent: that is the
       property the whole algorithm rests on, so it is computed rather than
       assumed */
    const cut = (key) => {
      const idx = key.split(',').map(Number);
      for (let d = 0; d < idx.length; d++) {
        const sub = idx.filter((_, i) => i !== d).join(',');
        if (sub && (!freq(sub) || cut(sub))) return true;
      }
      return false;
    };

    gLinks.selectAll('line').data(state.links ? links : []).join('line')
      .attr('x1', (d) => pos.get(d.a).x).attr('y1', (d) => pos.get(d.a).y)
      .attr('x2', (d) => pos.get(d.b).x).attr('y2', (d) => pos.get(d.b).y)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.6).attr('opacity', 0.2);

    const marks = keys.map((key) => ({ key, ...pos.get(key), count: all.get(key) }));
    const sel = gNodes.selectAll('g.node').data(marks, (d) => d.key);
    sel.exit().remove();
    const enter = sel.enter().append('g').attr('class', 'node');
    enter.append('circle');
    enter.append('text').attr('class', 'cnt').attr('dy', -8).attr('text-anchor', 'middle');
    const merged = enter.merge(sel);
    merged.attr('transform', (d) => `translate(${d.x},${d.y})`);
    merged.select('circle')
      .attr('r', 6)
      .attr('fill', (d) => {
        if (!state.colour) return 'var(--squidink)';
        if (state.prune && cut(d.key)) return 'none';
        return freq(d.key) ? 'var(--primary)' : 'var(--stone)';
      })
      .attr('stroke', (d) => (state.prune && cut(d.key) ? 'var(--stone)' : 'var(--paper)'))
      .attr('stroke-width', (d) => (state.prune && cut(d.key) ? 1.2 : 0.8))
      .attr('stroke-dasharray', (d) => (state.prune && cut(d.key) ? '2 2' : null))
      .attr('opacity', (d) => (state.colour && !freq(d.key) ? 0.5 : 0.95));
    merged.select('text.cnt')
      .style('font-family', 'var(--font-mono)').style('font-size', '9px')
      .style('fill', 'var(--squidink)')
      .attr('opacity', (d) => (state.counts && d.key.split(',').length <= 2 ? 0.8 : 0))
      .text((d) => d.count);

    /* the row labels: one per size, plus the names of the five items */
    gNote.selectAll('*').remove();
    for (let size = 1; size <= K; size++) {
      const y = 14 + ((h - 28) * (size - 1)) / (K - 1);
      seriesLabel(gNote, -10, y + 4, size === 1 ? 'one item' : `${size} items`,
        { anchor: 'end', size: 10.5, opacity: 0.7 });
    }
    if (state.names) {
      keys.filter((k2) => k2.split(',').length === 1).forEach((key) => {
        const p = pos.get(key);
        seriesLabel(gNote, p.x, p.y + 20, short(Number(key)),
          { anchor: 'middle', size: 10, opacity: 0.85 });
      });
    }
    if (state.legend) {
      legend(gNote, [
        { label: `at least ${minCount} baskets`, shape: 'dot', colour: 'var(--primary)' },
        { label: 'fewer', shape: 'dot', colour: 'var(--stone)' },
        { label: 'never counted', shape: 'ring', colour: 'var(--stone)' },
      ], { x0: 2, y0: -36, gap: 200 });
    }
    if (state.note) seriesLabel(gNote, 2, -14, state.note, { size: 11.5, opacity: 0.95 });
  }

  const nFreq = keys.filter((k2) => all.get(k2) >= minCount).length;
  const cutCount = (() => {
    const freq = (key) => all.get(key) >= minCount;
    const cut = (key) => {
      const idx = key.split(',').map(Number);
      for (let d = 0; d < idx.length; d++) {
        const sub = idx.filter((_, i) => i !== d).join(',');
        if (sub && (!freq(sub) || cut(sub))) return true;
      }
      return false;
    };
    return keys.filter(cut).length;
  })();

  const STATES = [
    { names: true, note: '' },
    { names: true, counts: true, note: 'the support of each single item, and of each pair' },
    { names: true, counts: true, links: true, note: 'every set, joined to the sets one item larger' },
    { names: true, counts: true, links: true, colour: true, legend: true,
      note: `${nFreq} of the ${keys.length} reach the threshold` },
    { names: true, links: true, colour: true, prune: true, legend: true,
      note: `${cutCount} of the ${keys.length} need never be counted` },
  ];

  render(STATES[0]);
  const handlers = {};
  STATES.forEach((s, i) => {
    handlers[String(i)] = () => render(s);
  });
  return {
    scrolly: scrolly('#lattice-scrolly .step', handlers),
    facts: {
      k: K,
      subsets: keys.length,
      frequent: nFreq,
      cut: cutCount,
      minCount,
      names: keys.filter((k2) => k2.split(',').length === 1).map((k2) => db.names[Number(k2)]),
      singles: keys.filter((k2) => k2.split(',').length === 1).map((k2) => all.get(k2)),
    },
  };
}
