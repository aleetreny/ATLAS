/* The other trick: compress the database instead of pruning the search.
 *
 * The tree is built here, transaction by transaction, in whichever item order
 * the buttons choose. The node count in the readout is for the whole database;
 * the picture is the first few levels, because a real FP-tree is tens of
 * thousands of nodes and the shape near the root is the part that matters.
 */
import { makeChart } from '../../assets/js/chart.js';
import { transactions, fpTree } from './basketkit.js';
import { treeLayout, legend, seriesLabel } from './draw.js';

const STEPS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, null];

export function initTreeWidget(data, db) {
  const txs = transactions(db);
  const ORDERS = [
    { key: 'freq', label: 'commonest first' },
    { key: 'rare', label: 'rarest first' },
    { key: 'alpha', label: 'alphabetical' },
  ];
  let order = 'freq';
  let si = 3;

  const chart = makeChart('#tree-chart', {
    width: 640, height: 400, margin: { top: 56, right: 20, bottom: 30, left: 20 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#tree-readout');
  const slider = d3.select('#tree-n')
    .attr('min', 0).attr('max', STEPS.length - 1).attr('step', 1).property('value', si);

  d3.select('#tree-orders').selectAll('button').data(ORDERS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === order ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      order = d.key;
      d3.select('#tree-orders').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === order ? '' : ' ghost'}`);
      render();
    });

  /* db.names arrives sorted by frequency, so the three orders are three
     permutations of the same indices and nothing has to be recounted */
  function ranking(key) {
    const idx = db.names.map((_, i) => i);
    let ordered;
    if (key === 'freq') ordered = idx;
    else if (key === 'rare') ordered = idx.slice().reverse();
    else ordered = idx.slice().sort((a, b) => db.names[a].localeCompare(db.names[b]));
    const rank = {};
    ordered.forEach((v, i) => { rank[v] = i; });
    return rank;
  }

  /* the whole-database node count for each order, once */
  const FULL = Object.fromEntries(ORDERS.map((o) => {
    const t = fpTree(txs, ranking(o.key));
    return [o.key, t.nodes];
  }));
  const SLOTS = db.counts.reduce((a, b) => a + b, 0);

  function render() {
    si = +slider.property('value');
    const limit = STEPS[si];
    const rank = ranking(order);
    const built = fpTree(txs, rank, { limit });

    g.selectAll('*').remove();
    const MAX_DEPTH = 4;
    const { nodes, links, drawn } = treeLayout(built.root, w, h - 16, { maxDepth: MAX_DEPTH });
    g.selectAll('line.link').data(links).join('line').attr('class', 'link')
      .attr('x1', (d) => d.from.x).attr('y1', (d) => d.from.y)
      .attr('x2', (d) => d.to.x).attr('y2', (d) => d.to.y)
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', (d) => Math.max(0.4, Math.min(4, 4 * (d.to.count / built.inserted))))
      .attr('opacity', 0.35);
    g.selectAll('circle.node').data(nodes).join('circle').attr('class', 'node')
      .attr('cx', (d) => d.x).attr('cy', (d) => d.y)
      .attr('r', (d) => (d.depth === 0 ? 6 : Math.max(1.6, Math.min(7, 7 * Math.sqrt(d.count / built.inserted)))))
      .attr('fill', (d) => (d.depth === 0 ? 'var(--squidink)' : 'var(--primary)'))
      .attr('opacity', 0.9);

    /* the busiest child of the root, named, because that is the item the
       ordering put nearest the top and the whole argument is about which one
       that is */
    const top = nodes.filter((n) => n.depth === 1).sort((a, b) => b.count - a.count)[0];
    if (top) {
      /* Away from the root, and then measured rather than guessed: with the
         rarest item first the busiest child sits a few units from the left edge
         and a label anchored to its left runs off the canvas. Place it, ask how
         wide it came out, and put it where it fits. */
      const label = seriesLabel(g, 0, top.y - 10, `${db.names[top.item]} (${top.count})`,
        { anchor: 'start', size: 10.5 });
      const wid = label.node().getComputedTextLength();
      let lx = top.x - 10 - wid;
      if (lx < 0) lx = Math.min(top.x + 10, w - wid);
      label.attr('x', Math.max(0, lx));
    }
    legend(g, [
      { label: 'a prefix shared by many baskets', shape: 'dot', colour: 'var(--primary)' },
      { label: `first ${MAX_DEPTH} levels drawn`, shape: 'line', colour: 'var(--squidink)' },
    ], { x0: 2, y0: -32, gap: 300 });

    d3.select('#tree-n-label').text(limit === null ? db.n.toLocaleString('en-US')
      : limit.toLocaleString('en-US'));
    const best = Math.min(...Object.values(FULL));
    const mine = FULL[order];
    readout.html(
      `<span class="slider-label">baskets offered <span class="value">`
      + `${(limit === null ? db.n : limit).toLocaleString('en-US')}</span></span>. `
      + `<span class="bold">${built.inserted.toLocaleString('en-US')}</span> of them contain at `
      + `least one of the ${db.names.length} commonest items, which are the ones this page ships, `
      + `and those go into the tree: it has `
      + `<span class="bold">${built.nodes.toLocaleString('en-US')}</span> nodes, of which the `
      + `picture shows the ${drawn.toLocaleString('en-US')} in the first ${MAX_DEPTH} levels. `
      + `Over all ${db.n.toLocaleString('en-US')} baskets, which between them hold `
      + `${SLOTS.toLocaleString('en-US')} slots of those ${db.names.length} items, this ordering `
      + `needs `
      + `<span class="bold">${mine.toLocaleString('en-US')}</span> nodes, which is `
      + `${(SLOTS / mine).toFixed(3)} slots per node. `
      + (mine === best
        ? `That is the smallest of the three, and it is not a coincidence: putting the commonest `
          + `items nearest the root is what makes baskets share prefixes at all.`
        : `The commonest-first ordering needs ${best.toLocaleString('en-US')}, so this one costs `
          + `<span class="bold">${(mine / best).toFixed(3)} times</span> as many nodes for exactly `
          + `the same data.`)
      + ` Nothing is lost either way: every basket is still in there, and the tree can be read `
      + `back. What changes is how much of it there is to read.`
    );
  }

  slider.on('input', render);
  render();
  return { render, full: FULL };
}
