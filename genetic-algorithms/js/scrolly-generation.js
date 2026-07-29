/* One generation, drawn as bits.
 *
 * The population is the real first generation of the reference run, produced by
 * the same code the generator ran, so the cut points and the flipped bits on
 * screen are the ones that actually happened rather than a decorative example.
 */
import { makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { runGA, fitnessOf } from './gakit.js';

const SHOW = 12;          // individuals drawn: twelve rows read, forty do not

export function initGenerationScrolly(data) {
  const P = data.probe;
  const cfg = { ...data.reference.config, gens: 1 };
  const fit = fitnessOf('trap_tight', P.block, P.shuffle);
  const run = runGA(cfg, fit, { captureFirst: true });
  const cap = run.capture;
  const n = cfg.bits;

  const chart = makeChart('#generation-chart', {
    width: 640, height: 430, margin: { top: 58, right: 92, bottom: 40, left: 34 },
  });
  const { g, w, h } = chart;
  const cw = w / n;
  const rh = Math.min(22, h / SHOW);
  const rows = d3.range(SHOW);

  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -38)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -22)
    .style('font-size', '11.5px');

  /* block dividers: the four bit groups the trap is built from */
  const blocks = g.append('g');
  for (let i = 0; i <= n; i += P.block) {
    blocks.append('line').attr('x1', i * cw).attr('x2', i * cw).attr('y1', -6)
      .attr('y2', SHOW * rh + 4).attr('stroke', '#c9ced0').attr('stroke-width', 1);
  }

  const cells = g.append('g');
  const marks = g.append('g');
  const bars = g.append('g');
  const labels = g.append('g');

  function draw(genomes, fits, { flips = [], winners = null, cuts = null, note = '' } = {}) {
    const flat = [];
    rows.forEach((i) => {
      for (let j = 0; j < n; j++) flat.push({ i, j, v: genomes[i][j] });
    });
    cells.selectAll('rect').data(flat).join('rect')
      .attr('x', (d) => d.j * cw).attr('y', (d) => d.i * rh)
      .attr('width', cw - 0.6).attr('height', rh - 2)
      .attr('fill', (d) => (d.v ? 'var(--primary)' : '#e2e6e7'));

    const flipSet = new Set(flips.filter(([i]) => i < SHOW).map(([i, j]) => `${i},${j}`));
    marks.selectAll('rect').data([...flipSet]).join('rect')
      .attr('x', (d) => Number(d.split(',')[1]) * cw - 0.6)
      .attr('y', (d) => Number(d.split(',')[0]) * rh - 1)
      .attr('width', cw + 0.6).attr('height', rh)
      .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.6);

    const maxFit = Math.max(...fits.slice(0, SHOW), 1);
    bars.selectAll('rect').data(rows).join('rect')
      .attr('x', w + 6).attr('y', (i) => i * rh)
      .attr('width', (i) => (fits[i] / maxFit) * 44).attr('height', rh - 2)
      .attr('fill', '#c9ced0');
    bars.selectAll('text').data(rows).join('text').attr('class', 'chart-note')
      .attr('x', w + 54).attr('y', (i) => i * rh + rh - 4).style('font-size', '10.5px')
      .text((i) => fits[i]);

    labels.selectAll('text').data(winners ? rows : []).join('text')
      .attr('class', 'chart-note').attr('x', -6).attr('y', (i) => i * rh + rh - 4)
      .attr('text-anchor', 'end').style('font-size', '10px')
      .text((i) => (winners ? winners[i] : ''));

    const cutMarks = cuts ? rows.map((i) => cuts[Math.floor(i / 2)]) : [];
    g.selectAll('line.cut').data(cuts ? cutMarks : []).join('line').attr('class', 'cut')
      .attr('x1', (c) => c * cw).attr('x2', (c) => c * cw)
      .attr('y1', (_, i) => i * rh - 1).attr('y2', (_, i) => i * rh + rh - 1)
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2);
    sub.text(note);
  }

  const handlers = {
    0: () => {
      draw(cap.parents, cap.fit,
        { note: `${cfg.pop} random guesses of ${cfg.bits} bits, best ${Math.max(...cap.fit)}` });
      title.text('the population, and what each guess is worth');
    },
    1: () => {
      draw(cap.afterSelect, cap.winners.map((k) => cap.fit[k]),
        { winners: cap.winners,
          /* short on purpose: the plot is 514 units wide and this line lives
             inside it, so the long version belongs in the step card */
          note: `${new Set(cap.winners).size} distinct parents fill ${cfg.pop} slots` });
      title.text(`parents chosen by contests of ${cfg.tour}`);
    },
    2: () => {
      draw(cap.afterCross, cap.afterCross.map((q) => fitnessOf('trap_tight', P.block, P.shuffle)(q)),
        { cuts: cap.cuts,
          note: 'each pair is cut at one point and the tails are swapped' });
      title.text('one cut per pair, and the tails change hands');
    },
    3: () => {
      draw(cap.children, run.history[1] ? cap.children.map(fit) : [],
        { flips: cap.flips,
          note: `${cap.flips.length} bits flipped, outlined; one in `
            + `${Math.round(1 / cfg.pmut)} on average` });
      title.text('and then a few bits are flipped at random');
    },
  };
  handlers[0]();
  const ctrl = scrolly('#generation-scrolly .step', handlers);
  return { ...ctrl, run };
}
