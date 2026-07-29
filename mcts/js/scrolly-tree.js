/* The four steps of one simulation, drawn on the tree that is actually there.
 *
 * The tree is warmed up with a handful of simulations and then one more is run
 * with its trace recorded, so what the figure highlights is what that
 * simulation did: which child the formula picked and why, which node was added,
 * how the play out ended, and which nodes changed on the way back up.
 */
import { makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { makeGame, lcg, makeTree, simulate, stateFromGrid } from './gamekit.js';

const WARM = 24;

export function initTreeScrolly(data) {
  const M = data.meta;
  const game = makeGame(M.rows, M.cols, M.connect);
  const state = stateFromGrid(data.tree.grid);
  const tree = makeTree(game, state, data.tree.who);
  const rand = lcg(90210);
  for (let i = 0; i < WARM; i++) simulate(game, tree, rand, {});
  /* the state of the tree BEFORE the traced simulation, so the counts drawn in
     the first three steps are the ones the formula actually saw */
  const before = { ...tree.N };
  const trace = simulate(game, tree, rand, { trace: true });

  const chart = makeChart('#tree-chart', {
    width: 640, height: 420, margin: { top: 54, right: 24, bottom: 24, left: 24 },
  });
  const { g, w, h } = chart;
  const link = g.append('g');
  const node = g.append('g');
  const note = g.append('g');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -34)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -15)
    .style('font-size', '11.5px');

  const rootKids = tree.kids[tree.rootKey];
  const chosen = trace.path[1];
  const deep = trace.path.length > 2 ? trace.path[2] : null;
  const deepKids = deep ? tree.kids[trace.path[1]] || [] : [];
  /* three rows, and the play out has to fit under the third one: the label at
     the end of the chain used to hang off the bottom of the drawing */
  const rowY = [34, 140, 240];
  const CHAIN = 6;
  const place = (i, n) => (w / (n + 1)) * (i + 1);

  function draw(step) {
    link.selectAll('*').remove();
    node.selectAll('*').remove();
    note.selectAll('*').remove();
    const showDeep = step >= 1 && deep;
    const onPath = new Set(trace.path);

    const dot = (x, y, r, fill, stroke) => node.append('circle').attr('cx', x).attr('cy', y)
      .attr('r', r).attr('fill', fill).attr('stroke', stroke || 'none').attr('stroke-width', 2.5);
    const label = (x, y, t, size = 10.5, fill = null) => node.append('text')
      .attr('class', 'chart-note').attr('x', x).attr('y', y).attr('text-anchor', 'middle')
      .style('font-size', `${size}px`).attr('fill', fill).text(t);

    const rootX = w / 2;
    /* the root, and one child per legal column */
    rootKids.forEach((ck, i) => {
      const x = place(i, rootKids.length);
      const hot = step === 0 ? ck === chosen : onPath.has(ck) && step >= 3;
      link.append('line').attr('x1', rootX).attr('y1', rowY[0] + 16).attr('x2', x)
        .attr('y2', rowY[1] - 16)
        .attr('stroke', hot ? 'var(--primary)' : '#c9ced0')
        .attr('stroke-width', hot ? 3 : 1.4);
      const n = step === 0 ? before[ck] : tree.N[ck];
      dot(x, rowY[1], 17, ck === chosen && step >= 0 ? 'var(--primary)' : '#c9ced0',
        step === 3 && onPath.has(ck) ? 'var(--anchor)' : null);
      label(x, rowY[1] + 4, n, 11, ck === chosen ? '#ffffff' : null);
      label(x, rowY[1] + 34, `column ${tree.move[ck]}`, 10);
      if (step === 0 && trace.scored.length) {
        const opt = trace.scored[0].options.find((o) => o.key === ck);
        label(x, rowY[1] - 26, opt.q.toFixed(2), 10);
        label(x, rowY[1] - 15, `+${opt.u.toFixed(2)}`, 10, '#8e9aa6');
      }
    });
    dot(rootX, rowY[0], 20, 'var(--anchor)');
    label(rootX, rowY[0] + 4, step === 0 ? before[tree.rootKey] : tree.N[tree.rootKey], 11, '#ffffff');
    label(rootX, rowY[0] - 28, 'the position on the board', 10.5);

    if (showDeep) {
      const cx = place(rootKids.indexOf(chosen), rootKids.length);
      deepKids.forEach((ck, i) => {
        const x = place(i, deepKids.length);
        const hot = ck === deep;
        link.append('line').attr('x1', cx).attr('y1', rowY[1] + 17).attr('x2', x)
          .attr('y2', rowY[2] - 14)
          .attr('stroke', hot ? 'var(--primary)' : '#dfe3e5')
          .attr('stroke-width', hot ? 3 : 1.2)
          .attr('stroke-dasharray', hot ? null : '3 3');
        dot(x, rowY[2], 14, hot ? 'var(--primary)' : '#e7eaec',
          hot && step === 3 ? 'var(--anchor)' : null);
        if (hot) label(x, rowY[2] + 4, tree.N[ck], 10.5, '#ffffff');
      });
      const dx = place(deepKids.indexOf(deep), deepKids.length);
      if (step >= 2) {
        /* the play out: a chain that leaves the tree entirely */
        for (let j = 0; j < Math.min(trace.plies, CHAIN); j++) {
          link.append('line').attr('x1', dx).attr('y1', rowY[2] + 16 + j * 11)
            .attr('x2', dx).attr('y2', rowY[2] + 24 + j * 11)
            .attr('stroke', '#8e9aa6').attr('stroke-width', 1.4);
        }
        const endY = rowY[2] + 20 + Math.min(trace.plies, CHAIN) * 11;
        /* the chain can hang under any column, so the label goes on whichever
           side of it has room */
        const right = dx < w / 2;
        note.append('text').attr('class', 'chart-note')
          .attr('x', dx + (right ? 8 : -8)).attr('y', endY + 4)
          .attr('text-anchor', right ? 'start' : 'end').style('font-size', '10.5px')
          .text(`${trace.plies} random moves, ending in `
            + `${trace.ended === 0.5 ? 'a draw' : `a win for player ${trace.ended}`}`);
      }
    }

    if (step === 3) {
      /* the sign belongs to the number being shown, not to the raw result: the
         first version put a plus in front of a minus one */
      const signed = (v) => `${v > 0 ? '+' : ''}${v.toFixed(0)}`;
      const atRoot = trace.path.length % 2 === 1 ? trace.value : -trace.value;
      note.append('text').attr('class', 'chart-note').attr('x', rootX + 34)
        .attr('y', rowY[0] + 4).style('font-size', '10.5px').attr('fill', 'var(--anchor)')
        .text(`${signed(atRoot)} here`);
      const cx = place(rootKids.indexOf(chosen), rootKids.length);
      note.append('text').attr('class', 'chart-note').attr('x', cx + 26)
        .attr('y', rowY[1] + 4).style('font-size', '10.5px').attr('fill', 'var(--anchor)')
        .text(`${signed(-atRoot)} here`);
    }

    const titles = [
      'step one: pick the child with the largest score',
      'step two: add one node the tree has never had',
      'step three: play the rest out at random',
      'step four: carry the result back up, flipping sign',
    ];
    const subs = [
      'top: what it has been worth. grey: how little it has been tried',
      'the new node starts with no visits and no wins, which is why it was picked',
      'no formula, no policy: legal moves at random until somebody wins or the board fills',
      'a draw for one is a draw for the other, and a win for one is a loss',
    ];
    title.text(titles[step]);
    sub.text(subs[step]);
  }

  const handlers = { 0: () => draw(0), 1: () => draw(1), 2: () => draw(2), 3: () => draw(3) };
  handlers[0]();
  const ctrl = scrolly('#tree-scrolly .step', handlers);
  return { ...ctrl, trace, tree };
}
