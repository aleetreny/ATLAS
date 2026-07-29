/* The arms of the distillation experiment, with the two controls picked out.
 *
 * The claim being tested is "dark knowledge": that what a student gains from a
 * teacher is the shape of the wrong classes, not just a cleaner label. Two arms
 * separate those. Hardening the teacher's output to its argmax keeps the label
 * correction and throws the shape away. Scrambling the mass among the wrong
 * classes keeps the label, keeps the confidence and keeps the entropy exactly,
 * and destroys only which wrong class got which share.
 *
 * If the students trained on those two do as well as the one trained on the
 * real distribution, then the shape was not what was being learned.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initDistilWidget(data) {
  const D = data.distil;
  const buttons = document.querySelector('#distil-buttons');
  const readout = document.querySelector('#distil-readout');
  const st = { view: 'all' };

  const chart = makeChart('#distil-chart', {
    width: 640, height: 400, margin: { top: 44, right: 30, bottom: 96, left: 84 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['all', 'every arm'], ['controls', 'the three that matter']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const KEY = {
    'hard labels': 'var(--squidink)',
    'teacher argmax': 'var(--anchor)',
    'scrambled wrong classes': 'var(--cosmos)',
    'student ceiling': 'var(--smile)',
    'soft T=4 plus unlabelled': 'var(--aqua)',
  };
  const soft = (k) => k.startsWith('soft T=') && !k.includes('unlabelled');
  const bestSoft = Object.entries(D.arms).filter(([k]) => soft(k))
    .reduce((a, b) => (b[1].mean > a[1].mean ? b : a));

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();
    const names = st.view === 'all'
      ? Object.keys(D.arms)
      : ['hard labels', bestSoft[0], 'teacher argmax', 'scrambled wrong classes'];

    const x = d3.scaleBand().domain(names).range([0, w]).padding(0.28);
    const vals = names.flatMap((n) => [D.arms[n].mean - D.arms[n].sd,
      D.arms[n].mean + D.arms[n].sd]);
    const y = d3.scaleLinear().domain([d3.min(vals) - 0.008, d3.max(vals) + 0.006]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.3f') });
    axisLabels(g, w, h, { x: '', y: 'student accuracy' });

    names.forEach((n) => {
      const a = D.arms[n];
      const colour = KEY[n] || (n === bestSoft[0] ? 'var(--primary)' : 'var(--primary)');
      layer.append('rect').attr('x', x(n)).attr('width', x.bandwidth())
        .attr('y', y(a.mean)).attr('height', h - y(a.mean))
        .attr('fill', colour).attr('opacity', n === bestSoft[0] ? 1 : 0.85);
      layer.append('line').attr('x1', x(n) + x.bandwidth() / 2).attr('x2', x(n) + x.bandwidth() / 2)
        .attr('y1', y(a.mean - a.sd)).attr('y2', y(a.mean + a.sd))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.3);
      layer.append('text').attr('class', 'chart-note')
        .attr('transform', `translate(${x(n) + x.bandwidth() / 2},${h + 8}) rotate(40)`)
        .style('font-size', '10.5px').text(n);
    });

    const hard = D.arms['hard labels'];
    const best = D.arms[bestSoft[0]];
    const argmax = D.arms['teacher argmax'];
    const scr = D.arms['scrambled wrong classes'];
    const floor = best.sd + scr.sd;
    const shapeMatters = best.mean - scr.mean > floor;
    readout.innerHTML = `<div class="gen-note">The teacher is an ensemble of `
      + `${data.teacher.members} networks at ${data.teacher.ensemble_acc.toFixed(4)}; the `
      + `student is ${(D.teacher_params / D.student_params).toFixed(0)} times smaller. `
      + `Trained on labels it reaches ${hard.mean.toFixed(4)}; trained on the teacher's `
      + `distribution at its best temperature, ${best.mean.toFixed(4)}, a gain of `
      + `${(best.mean - hard.mean).toFixed(4)} against a seed spread of `
      + `${(hard.sd + best.sd).toFixed(4)}. Now the two controls. Hardening the teacher to `
      + `its argmax gives ${argmax.mean.toFixed(4)}, so `
      + `${(argmax.mean - hard.mean).toFixed(4)} of the gain is the teacher correcting `
      + `labels. Scrambling the mass among the wrong classes, which keeps the winner, the `
      + `confidence and the entropy (${D.scrambled_entropy.toFixed(4)} against `
      + `${D.entropies['4'].toFixed(4)}) and destroys only which wrong class got which `
      + `share, gives ${scr.mean.toFixed(4)}: `
      + (shapeMatters
        ? `<span class="bold">${(best.mean - scr.mean).toFixed(4)} below the real `
          + `distribution</span>, larger than the ${floor.toFixed(4)} of seed spread, so the `
          + `shape of the wrong classes is carrying something on its own.`
        : `within ${floor.toFixed(4)} of the real distribution, which this measurement `
          + `cannot separate. On this pair of models the shape of the wrong classes is not `
          + `where the gain lives.`)
      + `</div>`;
  }

  render();
  return { render, state: st };
}
