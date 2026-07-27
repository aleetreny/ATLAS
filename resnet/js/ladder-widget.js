/* The ladder: six depths, twice, and the number that is supposed to be
 * impossible.
 *
 * The toggle matters more than it looks. On test accuracy a deep network
 * doing worse is unremarkable and everyone has a word for it. The same shape
 * on TRAINING accuracy is the whole reason residual connections exist,
 * because a network that cannot even fit the data it was shown is not
 * overfitting, it is failing to optimise.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initLadderWidget(data) {
  const rows = data.ladder;
  const W = 700;
  const H = 400;
  const margin = { top: 30, right: 132, bottom: 54, left: 68 };
  const { g, w, h } = makeChart('#ladder-chart', { width: W, height: H, margin });

  let field = 'train_acc';

  const x = d3.scalePoint().domain(rows.map((r) => r.depth)).range([0, w]).padding(0.08);
  const y = d3.scaleLinear().range([h, 0]);

  const grid = g.append('g');
  const axes = g.append('g');
  const plot = g.append('g');
  const labels = g.append('g');
  axisLabels(g, w, h, { x: 'layers in the network', y: 'accuracy' });

  const families = [
    { key: 'plain', label: 'plain stack', colour: '#232f3e', dash: null },
    { key: 'residual', label: 'with shortcuts', colour: 'var(--primary)', dash: null },
  ];

  const line = d3.line().x((d) => x(d.depth)).y((d) => y(d.v));

  function draw() {
    const all = rows.flatMap((r) => families.map((f) => r[f.key][field]));
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const padY = Math.max((hi - lo) * 0.18, 0.004);
    y.domain([Math.max(0, lo - padY), Math.min(1, hi + padY)]);

    grid.selectAll('*').remove();
    axes.selectAll('*').remove();
    plot.selectAll('*').remove();
    labels.selectAll('*').remove();
    drawGrid(grid, x, y, w, h, { yTicks: 5 });
    drawAxes(axes, x, y, w, h, { yTicks: 5, yFmt: d3.format('.1%') });

    families.forEach((f, i) => {
      const pts = rows.map((r) => ({ depth: r.depth, v: r[f.key][field] }));
      plot.append('path')
        .datum(pts)
        .attr('fill', 'none')
        .attr('stroke', f.colour)
        .attr('stroke-width', 2.6)
        .attr('d', line);
      plot.selectAll(`.pt-${f.key}`)
        .data(pts)
        .join('circle')
        .attr('class', `pt-${f.key}`)
        .attr('cx', (d) => x(d.depth))
        .attr('cy', (d) => y(d.v))
        .attr('r', 4)
        .attr('fill', f.colour);
      labels.append('text')
        .attr('x', w + 12)
        .attr('y', 16 + i * 20)
        .attr('class', 'annotation')
        .style('font-size', '12px')
        .style('fill', f.colour)
        .text(f.label);
    });

    /* Mark the plain family's best and its deepest, because the distance
       between them is the finding. */
    const plainPts = rows.map((r) => ({ depth: r.depth, v: r.plain[field] }));
    const best = plainPts.reduce((a, b) => (b.v > a.v ? b : a));
    const last = plainPts[plainPts.length - 1];
    if (best.depth !== last.depth) {
      plot.append('line')
        .attr('x1', x(best.depth)).attr('x2', x(last.depth))
        .attr('y1', y(best.v)).attr('y2', y(best.v))
        .attr('stroke', '#8a94a6')
        .attr('stroke-dasharray', '4 4');
      plot.append('line')
        .attr('x1', x(last.depth)).attr('x2', x(last.depth))
        .attr('y1', y(best.v)).attr('y2', y(last.v))
        .attr('stroke', '#8a94a6')
        .attr('stroke-width', 1.5);
      plot.append('text')
        .attr('x', x(last.depth) - 8)
        .attr('y', (y(best.v) + y(last.v)) / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('class', 'annotation')
        .style('font-size', '11px')
        .style('fill', '#8a94a6')
        .text(`${((best.v - last.v) * 100).toFixed(1)} points`);
    }

    const what = field === 'train_acc' ? 'the data they were trained on' : 'data they never saw';
    const pBest = best;
    const rBest = rows.reduce((a, b) => (b.residual[field] > a.residual[field] ? b : a));
    const deepest = rows[rows.length - 1];

    let verdict;
    if (field === 'train_acc') {
      const drop = (pBest.v - last.v) * 100;
      /* Whether the residual family also declines is judged against the
         measured seed spread, not against a threshold picked to make the
         answer come out well. Below that floor, a wobble is a wobble. */
      const floor = data.seed_spread ? data.seed_spread.train_range * 100 : 0.5;
      const rVals = rows.map((r) => r.residual[field] * 100);
      const rRange = Math.max(...rVals) - Math.min(...rVals);
      const alsoDips = rRange > floor * 2
        ? ` The shortcuts do not abolish the effect either: that family runs from ${Math.min(...rVals).toFixed(2)}% to ${Math.max(...rVals).toFixed(2)}% across the same ladder, a spread of ${rRange.toFixed(1)} points against a ${floor.toFixed(1)}-point noise floor measured from three seeds. The shortcut buys a later and much gentler decline, not immunity.`
        : ` The residual family, by contrast, stays inside ${rRange.toFixed(1)} points across the whole ladder, which is within the ${floor.toFixed(1)}-point spread three different seeds produce on their own. Adding ${rows[rows.length - 1].depth - rows[0].depth} layers to it changed nothing measurable.`;
      verdict = (drop > 0.2
        ? `The plain family peaks at <span class="bold">${(pBest.v * 100).toFixed(2)}%</span> with ${pBest.depth} layers and then goes <i>down</i>, reaching ${(last.v * 100).toFixed(2)}% at ${last.depth}. That is ${drop.toFixed(1)} points of training accuracy thrown away by adding layers, and no amount of data would fix it: these networks are failing to fit what they were already shown.`
        : `On this run the plain family does not turn down: it peaks at <span class="bold">${(pBest.v * 100).toFixed(2)}%</span> with ${pBest.depth} layers and still reaches ${(last.v * 100).toFixed(2)}% at ${last.depth}. The degradation the paper reported is a property of a regime, and at this scale it did not appear.`) + alsoDips;
    } else {
      verdict = `With shortcuts the best is <span class="bold">${(rBest.residual[field] * 100).toFixed(2)}%</span> at ${rBest.depth} layers; the plain family's best is ${(pBest.v * 100).toFixed(2)}%. Test accuracy is the number everyone quotes, and it is the one that hides the finding: switch to the training set.`;
    }

    document.querySelector('#ladder-readout').innerHTML =
      `<span class="bold">Accuracy on ${what}.</span> At ${deepest.depth} layers the shortcuts are worth `
      + `${((deepest.residual[field] - deepest.plain[field]) * 100).toFixed(1)} points `
      + `(${(deepest.residual[field] * 100).toFixed(2)}% against ${(deepest.plain[field] * 100).toFixed(2)}%), `
      + `for ${(deepest.residual.params - deepest.plain.params).toLocaleString('en-US')} extra parameters out of `
      + `${deepest.plain.params.toLocaleString('en-US')}. ${verdict}`;
  }

  const row = document.querySelector('#ladder-buttons');
  [['train_acc', 'training set'], ['test_acc', 'test set']].forEach(([key, label], i) => {
    const b = document.createElement('button');
    b.className = `atlas-button${i === 0 ? ' is-active' : ''}`;
    b.textContent = label;
    b.addEventListener('click', () => {
      field = key;
      row.querySelectorAll('.atlas-button').forEach((n) => n.classList.remove('is-active'));
      b.classList.add('is-active');
      draw();
    });
    row.appendChild(b);
  });

  draw();
}
