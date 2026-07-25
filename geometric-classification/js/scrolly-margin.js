/* Six steps in two acts.
 *
 * Act one is a genuinely separable problem, because "every one of these lines is
 * perfect on the training set" is only a true sentence about separable data, and
 * the whole first step rests on it. Act two adds one point and watches the hard
 * margin collapse by a factor of 153, which is the argument for ever wanting a
 * soft one.
 *
 * Handlers are idempotent state declarations, so scrolling back works for free.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { lineInBox, streetPolygon } from './marginkit.js';

export function initScrollyMargin(data) {
  const M = data.margin;
  const all = M.points;
  const ROGUE = M.rogue_index;
  const clean = all.filter((p, i) => i !== ROGUE);
  const byC = Object.fromEntries(M.runs.map((r) => [r.C, r]));

  const { g, w, h } = makeChart('#margin-chart', {
    width: 620,
    height: 560,
    margin: { top: 34, right: 26, bottom: 56, left: 62 },
  });

  const xe = d3.extent(all, (p) => p.x);
  const ye = d3.extent(all, (p) => p.y);
  const padX = (xe[1] - xe[0]) * 0.1;
  const padY = (ye[1] - ye[0]) * 0.1;
  const x = d3.scaleLinear().domain([xe[0] - padX, xe[1] + padX]).range([0, w]);
  const y = d3.scaleLinear().domain([ye[0] - padY, ye[1] + padY]).range([h, 0]);
  const BOX = [x.domain(), y.domain()];

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'feature 1', y: 'feature 2' });

  const clip = 'mg-clip';
  g.append('clipPath').attr('id', clip).append('rect').attr('x', -1).attr('y', -1)
    .attr('width', w + 2).attr('height', h + 2);
  const plot = g.append('g').attr('clip-path', `url(#${clip})`);

  const streetG = plot.append('g');
  const altG = plot.append('g');
  const kerbG = plot.append('g');
  const lineG = plot.append('g');
  const dotsG = g.append('g');
  const ringG = g.append('g');
  const rogueG = g.append('g');

  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const seg = (d) => (d ? `M${x(d[0][0])},${y(d[0][1])}L${x(d[1][0])},${y(d[1][1])}` : null);
  const poly = (p) => (p ? `M${p.map((q) => `${x(q[0])},${y(q[1])}`).join('L')}Z` : null);

  /* The rogue gets its own group so it can be absent for the first three steps
   * without disturbing the join keys of everything else. */
  const rogue = all[ROGUE];
  rogueG.append('circle')
    .attr('cx', x(rogue.x)).attr('cy', y(rogue.y)).attr('r', 7)
    .attr('fill', 'var(--aqua)').attr('stroke', 'var(--squidink)').attr('stroke-width', 2.4)
    .attr('opacity', 0);

  dotsG.selectAll('circle').data(clean).join('circle')
    .attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y))
    .attr('r', 4.4)
    .attr('fill', (p) => (p.c ? 'var(--cosmos)' : 'var(--aqua)'))
    .attr('stroke', 'white').attr('stroke-width', 1)
    .attr('opacity', 0.9);

  /* Support indices come from a fit on one set or the other, so they have to be
   * translated into positions in whichever array is on screen. */
  const svPoints = (run, withRogue) => {
    const src = withRogue ? all : clean;
    return run.support.map((i) => src[i]).filter(Boolean);
  };

  const show = ({ fit = null, withRogue = false, alts = false, sv = false, label = '' }) => {
    const t = (sel) => sel.transition('fade').duration(430);

    t(altG.selectAll('path').data(alts ? M.alternatives : []).join(
      (e) => e.append('path').attr('fill', 'none').attr('stroke', 'var(--squidink)')
        .attr('stroke-width', 1.8).attr('stroke-dasharray', '6 5').attr('opacity', 0)
    ).attr('d', (a) => seg(lineInBox(a.w, a.b, 0, BOX[0], BOX[1])))).attr('opacity', 0.55);

    t(streetG.selectAll('path').data(fit ? [fit] : []).join(
      (e) => e.append('path').attr('fill', 'var(--primary)').attr('opacity', 0)
    ).attr('d', (r) => poly(streetPolygon(r.w, r.b, BOX)))).attr('opacity', 0.13);

    t(kerbG.selectAll('path').data(fit ? [1, -1] : []).join(
      (e) => e.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
        .attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4').attr('opacity', 0)
    ).attr('d', (c) => seg(lineInBox(fit.w, fit.b, c, BOX[0], BOX[1])))).attr('opacity', 0.85);

    t(lineG.selectAll('path').data(fit ? [fit] : []).join(
      (e) => e.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
        .attr('stroke-width', 3.4).attr('opacity', 0)
    ).attr('d', (r) => seg(lineInBox(r.w, r.b, 0, BOX[0], BOX[1])))).attr('opacity', 1);

    t(ringG.selectAll('circle').data(sv && fit ? svPoints(fit, withRogue) : []).join(
      (e) => e.append('circle').attr('fill', 'none').attr('stroke', 'var(--squidink)')
        .attr('stroke-width', 2).attr('opacity', 0)
    ).attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y)).attr('r', 8.5)).attr('opacity', 1);

    t(rogueG.select('circle')).attr('opacity', withRogue ? 1 : 0);
    caption.text(label);
  };

  const HC = M.hard_clean;
  const HR = M.hard_rogue;
  const shrink = Math.round(HC.half_width / HR.half_width);

  scrolly('#margin-scrolly .step', {
    0: () => show({ alts: true, label: `${M.alternatives.length} boundaries, every one perfect on these ${clean.length} points` }),
    1: () => show({ fit: HC, label: `the widest street: half-width ${HC.half_width.toFixed(3)}` }),
    2: () => show({ fit: HC, sv: true, label: `${HC.n_support} support vectors hold up the whole model` }),
    3: () => show({ fit: HR, withRogue: true, sv: true, label: `one point added, and the street is ${shrink}× narrower` }),
    4: () => show({ fit: byC[0.03], withRogue: true, sv: true, label: `C = 0.03: half-width ${byC[0.03].half_width.toFixed(3)}, violations allowed` }),
    5: () => show({ fit: byC[0.01], withRogue: true, sv: true, label: `C = 0.01: ${byC[0.01].n_support} of ${all.length} points now hold it up` }),
  });

  /* Runtime checks on the two claims the steps make in numbers. Both fits must
   * separate perfectly, or "the hard margin still exists, it is just useless"
   * is not what is on screen. */
  if (HC.train_acc !== 1 || HR.train_acc !== 1) {
    console.warn(`hard margins are not separating: clean ${HC.train_acc}, rogue ${HR.train_acc}`);
  }
  const accs = new Set(M.runs.map((r) => r.train_acc));
  if (accs.size !== 1) {
    console.warn(`soft margin sweep: training accuracy is not constant, it takes ${[...accs].join(', ')}`);
  }
}
