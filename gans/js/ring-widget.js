/* Eight modes, and who finds them.
 *
 * The grid of twelve panels is measured: four objectives, three seeds each,
 * trained offline by the generator and reported through a mode counter that
 * was calibrated in both directions before it was allowed to say anything.
 * The panel underneath is the same game trained here, in this tab, with the
 * same arithmetic written a second time in JavaScript, so that the thing the
 * reader watches is the thing the article measured and not an animation of it.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { makeCanvas } from '../../assets/js/imagery.js';
import { LiveGAN, modeReport } from './gankit.js';

const ARM_NAMES = {
  ns: 'non-saturating',
  sat: 'saturating',
  'wgan-clip': 'WGAN, weights clipped',
  'wgan-gp': 'WGAN, gradient penalty',
};

export function initRingGrid(data) {
  const M = data.meta;
  const host = document.querySelector('#ring-grid');
  const cell = 168;
  M.arms.forEach((arm) => {
    const row = document.createElement('div');
    row.className = 'ring-row';
    const label = document.createElement('div');
    label.className = 'ring-row-label';
    label.innerHTML = ARM_NAMES[arm] || arm;
    row.appendChild(label);
    const panels = document.createElement('div');
    panels.className = 'ring-panels';
    row.appendChild(panels);
    host.appendChild(row);

    M.seeds.forEach((seed) => {
      const run = data.runs[`${arm}/${seed}`];
      const box = document.createElement('div');
      box.className = 'ring-cell';
      panels.appendChild(box);
      const c = makeChart(box, {
        width: cell, height: cell, margin: { top: 6, right: 6, bottom: 6, left: 6 },
      });
      const s = d3.scaleLinear().domain([-2.9, 2.9]).range([0, c.w]);
      const sy = d3.scaleLinear().domain([-2.9, 2.9]).range([c.h, 0]);
      c.g.append('rect').attr('width', c.w).attr('height', c.h)
        .attr('fill', 'var(--paper)').attr('stroke', 'var(--stone)');
      M.centres.forEach((p) => {
        c.g.append('circle').attr('cx', s(p[0])).attr('cy', sy(p[1])).attr('r', 5)
          .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1);
      });
      c.g.append('g').selectAll('circle').data(run.points).enter().append('circle')
        .attr('cx', (p) => s(p[0])).attr('cy', (p) => sy(p[1])).attr('r', 1.1)
        .attr('fill', 'var(--primary)').attr('fill-opacity', 0.55);
      const cap = document.createElement('div');
      cap.className = 'ring-cap';
      cap.innerHTML = `seed ${seed}: <b>${run.final.covered}/${M.k}</b> modes, `
        + `${(run.final.quality * 100).toFixed(1)}% on the ring`;
      box.appendChild(cap);
    });
  });

  const cal = data.calib;
  document.querySelector('#ring-calib').innerHTML =
    `<table class="atlas-table"><thead><tr><th>Fed to the counter</th><th>Modes it reports</th>`
    + `<th>Share on the ring</th><th>Has to be</th></tr></thead><tbody>`
    + [['real samples', cal.real, `${M.k}`], ['one mode, real width', cal.point_mass, '1'],
      ['two modes', cal.two_modes, '2'],
      ['uniform points over the square', cal.uniform, '0 or nearly']]
      .map(([n, r, want]) => `<tr><td>${n}</td><td>${r.covered}</td>`
        + `<td>${(r.quality * 100).toFixed(1)}%</td><td>${want}</td></tr>`).join('')
    + `</tbody></table>`;
}

/* The attempts to break the result above, which is the part that turns "it did
 * not happen" into something a reader can weigh. */
export function initFalsify(data) {
  const F = data.falsify;
  const base = data.meta;
  const rows = [];
  const baseline = {};
  base.arms.forEach((arm) => {
    const cov = base.seeds.map((s) => data.runs[`${arm}/${s}`].final.covered);
    const q = base.seeds.map((s) => data.runs[`${arm}/${s}`].final.quality);
    baseline[arm] = { cov, q };
  });
  F.trials.forEach((t) => {
    const runs = t.seeds.map((s) => F.runs[`${t.name}/${s}`]).filter(Boolean);
    if (!runs.length) return;
    const before = baseline[t.arm];
    rows.push(`<tr><td>${t.why}</td>`
      + `<td>${before.cov.join(', ')}</td>`
      + `<td>${runs.map((r) => r.final.covered).join(', ')}</td>`
      + `<td>${runs.map((r) => `${(r.final.quality * 100).toFixed(1)}%`).join(', ')}</td></tr>`);
  });
  document.querySelector('#falsify-table').innerHTML =
    `<table class="atlas-table"><thead><tr><th>What was changed</th>`
    + `<th>Modes before, per seed</th><th>Modes after</th><th>Share on the ring after</th>`
    + `</tr></thead><tbody>${rows.join('')}</tbody></table>`;
}

export function initRingLive(data) {
  const M = data.meta;
  const L = data.live;
  const N = 132;                                    // heat map resolution
  const LO = -2.9;
  const HI = 2.9;
  const host = document.querySelector('#live-canvas');
  const { ctx } = makeCanvas(host, 420, 420);
  const state = { arm: 'ns', gan: null, busy: false };

  const lossChart = makeChart('#live-loss', {
    width: 420, height: 260, margin: { top: 26, right: 20, bottom: 48, left: 62 },
  });

  function reset() {
    state.gan = new LiveGAN({
      seed: 7, h: L.h, batch: L.batch, arm: state.arm, k: M.k, r: M.r, sd: M.sd,
    });
    /* Open at exactly the step count the generator recorded, rather than at
       zero: an untrained pair is the worst case rather than a starting point,
       and at this many steps the readout can say what the same run measured
       offline, which is the check the reader can make by looking. */
    state.gan.run(L.steps);
    render();
  }

  function paint() {
    const g = state.gan;
    const surf = g.surface(LO, HI, N);
    const img = ctx.createImageData(N, N);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        /* row 0 of the grid is the bottom of the plot */
        const v = surf[(N - 1 - r) * N + c];
        const p = (r * N + c) * 4;
        /* Diverging around one half rather than a ramp from zero, because the
           reading that matters is "which way is it leaning", and an untrained
           discriminator sits at one half everywhere. On a ramp that comes out
           as a solid block of mid tone, which looks like a bug; here it comes
           out white, which is the actual message: it is saying nothing yet.
           Green where it believes the data is real, blue where it believes it
           is generated. */
        const k = Math.min(Math.abs(v - 0.5) * 2, 1);
        const col = v >= 0.5 ? [63, 98, 18] : [0, 49, 129];
        img.data[p] = Math.round(255 - (255 - col[0]) * k);
        img.data[p + 1] = Math.round(255 - (255 - col[1]) * k);
        img.data[p + 2] = Math.round(255 - (255 - col[2]) * k);
        img.data[p + 3] = 255;
      }
    }
    const off = document.createElement('canvas');
    off.width = N;
    off.height = N;
    off.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, 420, 420);

    const sx = (v) => ((v - LO) / (HI - LO)) * 420;
    const sy = (v) => 420 - ((v - LO) / (HI - LO)) * 420;
    ctx.strokeStyle = '#232f3e';
    ctx.lineWidth = 1;
    M.centres.forEach((p) => {
      ctx.beginPath();
      ctx.arc(sx(p[0]), sy(p[1]), 9, 0, 2 * Math.PI);
      ctx.stroke();
    });
    const pts = g.sample(1200);
    ctx.fillStyle = 'rgba(63, 98, 18, 0.75)';
    pts.forEach((p) => {
      ctx.beginPath();
      ctx.arc(sx(p[0]), sy(p[1]), 1.6, 0, 2 * Math.PI);
      ctx.fill();
    });
    return pts;
  }

  function paintLoss() {
    const rows = state.gan.losses;
    lossChart.g.selectAll('*').remove();
    const x = d3.scaleLinear().domain([0, Math.max(rows.length, 50)]).range([0, lossChart.w]);
    const y = d3.scaleLinear().domain([0, 3]).range([lossChart.h, 0]).clamp(true);
    drawAxes(lossChart.g, x, y, lossChart.w, lossChart.h, { xTicks: 5, yTicks: 4 });
    axisLabels(lossChart.g, lossChart.w, lossChart.h, { x: 'step', y: 'loss' });
    [['discriminator', 0, 'var(--anchor)'], ['generator', 1, 'var(--primary)']]
      .forEach(([label, k, colour], i) => {
        lossChart.g.append('path').attr('fill', 'none').attr('stroke', colour)
          .attr('stroke-width', 1.8)
          .attr('d', d3.line().x((d, j) => x(j)).y((d) => y(d[k]))(rows));
        lossChart.g.append('text').attr('class', 'chart-annotation')
          .attr('x', 4).attr('y', 12 + i * 14).style('font-size', '11px')
          .style('letter-spacing', '0.4px').style('fill', colour).text(label);
      });
  }

  function render() {
    const pts = paint();
    paintLoss();
    const rep = modeReport(pts, M.centres, M.sd);
    const ref = L.runs[state.arm];
    const same = state.gan.it === L.steps;
    document.querySelector('#live-readout').innerHTML =
      `<p><span class="bold">${state.gan.it}</span> steps of the `
      + `${ARM_NAMES[state.arm]} loss, on a generator with `
      + `<span class="bold">${ref.params}</span> weights. Right now it covers `
      + `<span class="bold">${rep.covered}</span> of the ${M.k} modes, with `
      + `${(rep.quality * 100).toFixed(1)}% of its samples on the ring and its busiest mode `
      + `holding ${(rep.maxShare * 100).toFixed(1)}% of them.</p>`
      + `<p>The background is the discriminator: green where it believes the data is real, blue `
      + `where it believes it is generated, white where it cannot tell. The generator only ever `
      + `learns by climbing that surface, so wherever the surface is white it is being told `
      + `nothing at all.`
      + (same ? ` At this same step count the generator measured ${ref.report_shown.covered} modes `
        + `and ${(ref.report_shown.quality * 100).toFixed(1)}% on the ring offline, on its own `
        + `draw of noise rather than this one, so those are two samples from the same generator `
        + `and not the same number twice.` : '')
      + `</p>`;
  }

  function chunk(total) {
    if (state.busy) return;
    state.busy = true;
    let done = 0;
    const each = 50;
    const tick = () => {
      state.gan.run(Math.min(each, total - done));
      done += each;
      render();
      if (done < total) setTimeout(tick, 0);
      else state.busy = false;
    };
    tick();
  }

  const controls = document.querySelector('#live-buttons');
  const mk = (label, cls, fn) => {
    const b = document.createElement('button');
    b.className = cls;
    b.textContent = label;
    b.addEventListener('click', fn);
    controls.appendChild(b);
    return b;
  };
  mk('train 50 steps', 'atlas-btn ghost', () => chunk(50));
  mk('train 500 steps', 'atlas-btn ghost', () => chunk(500));
  mk('start over', 'atlas-btn ghost', reset);

  const armRow = document.querySelector('#live-arm');
  const armButtons = ['ns', 'sat'].map((arm) => {
    const b = document.createElement('button');
    b.className = arm === state.arm ? 'atlas-btn' : 'atlas-btn ghost';
    b.textContent = `${ARM_NAMES[arm]} loss`;
    b.addEventListener('click', () => {
      state.arm = arm;
      armButtons.forEach((o, i) => {
        o.className = ['ns', 'sat'][i] === arm ? 'atlas-btn' : 'atlas-btn ghost';
      });
      reset();
    });
    armRow.appendChild(b);
    return b;
  });

  reset();
  return { state, chunk };
}
