/* The bench: twelve bottles, one of them draggable, three rules watching.
 *
 * The slider indexes the measured grid rather than carrying a value of its own,
 * which is the module's convention and the reason nothing here can land on a
 * position the generator never scored. Every number on screen is computed from
 * the sample as it stands at that position; the generator's rows exist only so
 * that the guard can compare them.
 */
import { makeChart, drawAxes, axisLabels, drawGrid, spread } from '../../assets/js/chart.js';
import { zScores, modifiedZ, tukeyFences, zCeiling } from './statkit.js';
import { drawSample, drawBand, marker, legend, seriesLabel } from './draw.js';

export function initLineWidget(data) {
  const S = data.one_d.sub;
  const base = S.values;
  const SUS = S.suspect;
  const COMP = S.companion;
  const Z_CUT = data.one_d.z_cut;
  const MZ_CUT = data.one_d.mz_cut;
  const grid = S.grid;

  let idx = 0;
  let two = false;

  const modes = d3.select('#line-modes');
  const slider = d3.select('#line-pos').attr('min', 0).attr('max', grid.length - 1).attr('step', 1);
  const readout = d3.select('#line-readout');

  const topChart = makeChart('#line-chart', {
    width: 640, height: 280, margin: { top: 76, right: 36, bottom: 56, left: 30 },
  });
  const curveChart = makeChart('#line-curve', {
    width: 640, height: 330, margin: { top: 70, right: 128, bottom: 54, left: 58 },
  });

  const gBands = topChart.g.append('g');
  const gAxis = topChart.g.append('g');
  const gPts = topChart.g.append('g');
  const gNotes = topChart.g.append('g');
  const baseline = topChart.h - 22;

  legend(topChart.g.append('g'), [
    { label: 'three sigma', shape: 'band', colour: 'var(--anchor)' },
    { label: "Tukey's fences", shape: 'band', colour: 'var(--cosmos)' },
  ], { x0: 2, y0: -62, gap: 200, size: 10.5 });
  legend(topChart.g.append('g'), [
    { label: 'mean', shape: 'line', colour: 'var(--anchor)' },
    { label: 'median', shape: 'dash', colour: 'var(--galaxy)' },
    { label: 'the odd bottles', shape: 'dot', colour: 'var(--primary)' },
  ], { x0: 2, y0: -46, gap: 148, size: 10.5 });

  const x = d3.scaleLinear().range([0, topChart.w]);

  /* ------------------------------------------------------------- the curves */
  /* Precomputed once over the whole grid, because the shape of the curve is the
     argument and the reader should see all of it at every position, not the
     part they have walked so far. */
  function curves(withCompanion) {
    return grid.map((pos) => {
      const v = base.slice();
      v[SUS] = pos;
      if (withCompanion) v[COMP] = S.comp_at;
      const zs = zScores(v);
      const mz = modifiedZ(v);
      const f = tukeyFences(v);
      return {
        pos,
        z: zs.z[SUS],
        zComp: zs.z[COMP],
        mz: mz.z[SUS],
        mzComp: mz.z[COMP],
        mean: zs.mean,
        sd: zs.sd,
        fenceHi: f.hi,
        fenceLo: f.lo,
        nZ: zs.z.filter((q) => Math.abs(q) > Z_CUT).length,
        nMz: mz.z.filter((q) => Math.abs(q) > MZ_CUT).length,
        nFence: v.filter((q) => q < f.lo || q > f.hi).length,
      };
    });
  }
  const CURVES = { one: curves(false), two: curves(true) };

  function render() {
    const rows = two ? CURVES.two : CURVES.one;
    const row = rows[idx];
    const v = base.slice();
    v[SUS] = row.pos;
    if (two) v[COMP] = S.comp_at;
    const zs = zScores(v);
    const mz = modifiedZ(v);
    const f = tukeyFences(v);

    /* ---------------------------------------------------------- number line */
    const hi = Math.max(...grid, S.comp_at);
    const lo = Math.min(...base);
    const pad = (hi - lo) * 0.04;
    x.domain([lo - pad, hi + pad]);

    let gx = gAxis.select('g.axis.x');
    if (gx.empty()) gx = gAxis.append('g').attr('class', 'axis x');
    gx.attr('transform', `translate(0,${topChart.h})`)
      .call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
    axisLabels(gAxis, topChart.w, topChart.h, { x: 'magnesium, mg per litre' });

    drawBand(gBands, x, zs.mean - Z_CUT * zs.sd, zs.mean + Z_CUT * zs.sd, 8, baseline + 12,
      { colour: 'var(--anchor)', klass: 'zband', opacity: 0.12 });
    drawBand(gBands, x, f.lo, f.hi, 16, baseline + 4,
      { colour: 'var(--cosmos)', klass: 'fband', opacity: 0.1 });

    gAxis.selectAll('g.marker').remove();
    marker(gAxis, x(zs.mean), 2, baseline + 16, '',
      { colour: 'var(--anchor)', w: topChart.w, dash: null });
    marker(gAxis, x(mz.median), 2, baseline + 16, '',
      { colour: 'var(--galaxy)', w: topChart.w, dash: '2 3' });

    drawSample(gPts, v, x, baseline, {
      r: 6, tol: 11,
      highlight: (i) => i === SUS || (two && i === COMP),
    });

    gNotes.selectAll('*').remove();
    seriesLabel(gNotes, 0, -18,
      `mean ${zs.mean.toFixed(1)}   sd ${zs.sd.toFixed(1)}   `
      + `median ${mz.median.toFixed(1)}   MAD ${mz.mad.toFixed(1)}`,
      { size: 11.5, opacity: 0.9 });

    /* -------------------------------------------------------------- curves */
    const cx = d3.scaleLinear().domain(d3.extent(grid)).range([0, curveChart.w]);
    const maxY = Math.max(MZ_CUT + 1.6, d3.max(rows, (r) => Math.max(
      Math.abs(r.z), Math.abs(r.mz), two ? Math.abs(r.mzComp) : 0
    )) * 1.06);
    /* A root scale, for the reason the density article's reachability plot has
       one: the modified z runs to twenty while every decision on this page
       happens between three and three and a half, and on a linear axis the
       entire argument lives in the bottom fifth of the panel. The root is
       monotone, so "above the line" still means what it looks like. */
    const yVals = [0, 1, 2, 4, 6, 9, 13, 18, 25].filter((v) => v <= maxY);
    const cy = d3.scaleSqrt().domain([0, maxY]).range([curveChart.h, 0]);

    drawGrid(curveChart.g, cx, cy, curveChart.w, curveChart.h, { xTicks: 5, yValues: yVals });
    drawAxes(curveChart.g, cx, cy, curveChart.w, curveChart.h, { xTicks: 5, yValues: yVals });
    axisLabels(curveChart.g, curveChart.w, curveChart.h,
      { x: 'where the odd bottle is pushed to', y: 'score, root scale' });

    const line = d3.line().x((d) => cx(d.pos)).y((d) => cy(d.y));
    const series = [
      { key: 'z', label: 'z, pushed bottle', colour: 'var(--primary)', dash: null,
        vals: rows.map((r) => ({ pos: r.pos, y: Math.abs(r.z) })) },
      { key: 'mz', label: 'modified z, same bottle', colour: 'var(--galaxy)', dash: '5 3',
        vals: rows.map((r) => ({ pos: r.pos, y: Math.abs(r.mz) })) },
    ];
    if (two) {
      series.push({ key: 'zc', label: 'z, the one that never moved', colour: 'var(--cosmos)',
        dash: null, vals: rows.map((r) => ({ pos: r.pos, y: Math.abs(r.zComp) })) });
      series.push({ key: 'mzc', label: 'modified z, that same one', colour: 'var(--aqua)',
        dash: '5 3', vals: rows.map((r) => ({ pos: r.pos, y: Math.abs(r.mzComp) })) });
    }

    const paths = curveChart.g.selectAll('path.series').data(series, (d) => d.key);
    paths.exit().remove();
    paths.enter().append('path').attr('class', 'series').merge(paths)
      .attr('d', (d) => line(d.vals))
      .attr('fill', 'none')
      .attr('stroke', (d) => d.colour)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', (d) => d.dash)
      .attr('opacity', 0.92);

    /* the two thresholds and the ceiling, which is the line the top curve
       cannot cross however hard the slider is pushed. The three of them land
       within half a unit of each other on this axis, so the labels are pushed
       apart vertically rather than drawn where their lines are. */
    curveChart.g.selectAll('g.rule').remove();
    const rules = curveChart.g.append('g').attr('class', 'rule');
    const ceil = zCeiling(base.length, two ? 2 : 1);
    const marks = [{ v: Z_CUT, label: 'three sigma', colour: 'var(--anchor)' },
      { v: MZ_CUT, label: 'modified z', colour: 'var(--galaxy)' },
      { v: ceil, label: `ceiling ${ceil.toFixed(3)}`, colour: 'var(--squidink)' }]
      .filter((r) => r.v <= maxY)
      .map((r) => ({ ...r, y: cy(r.v) + 4, yMax: curveChart.h }));
    marks.forEach((r) => {
      rules.append('line')
        .attr('x1', 0).attr('x2', curveChart.w).attr('y1', cy(r.v)).attr('y2', cy(r.v))
        .attr('stroke', r.colour).attr('stroke-width', 1.2)
        .attr('stroke-dasharray', '3 3').attr('opacity', 0.7);
    });
    spread(marks, 18).forEach((r) => {
      rules.append('line')
        .attr('x1', curveChart.w).attr('x2', curveChart.w + 5)
        .attr('y1', cy(r.v)).attr('y2', r.y - 4)
        .attr('stroke', r.colour).attr('stroke-width', 0.9).attr('opacity', 0.55);
      seriesLabel(rules, curveChart.w + 8, r.y, r.label, { size: 10.5, opacity: 0.9 })
        .style('fill', r.colour);
    });

    /* where the slider is */
    rules.append('line')
      .attr('x1', cx(row.pos)).attr('x2', cx(row.pos))
      .attr('y1', 0).attr('y2', curveChart.h)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('opacity', 0.35);
    series.forEach((s) => {
      rules.append('circle')
        .attr('cx', cx(row.pos)).attr('cy', cy(s.vals[idx].y)).attr('r', 4)
        .attr('fill', s.colour).attr('stroke', 'var(--paper)').attr('stroke-width', 1.2);
    });

    curveChart.g.selectAll('g.serieslab').remove();
    const labs = curveChart.g.append('g').attr('class', 'serieslab');
    legend(labs, series.map((s) => ({ label: s.label, shape: s.dash ? 'dash' : 'line',
      colour: s.colour })), { x0: 2, y0: -46, gap: 224, size: 10.5 });
    if (series.length > 2) {
      labs.selectAll('g.legend-chip')
        .attr('transform', (d, i) => `translate(${2 + (i % 2) * 224},${i < 2 ? -46 : -28})`);
    }

    /* ------------------------------------------------------------- readout */
    d3.select('#line-pos-label').text(row.pos.toFixed(0));
    const flaggedZ = zs.z.map((q, i) => (Math.abs(q) > Z_CUT ? i : -1)).filter((i) => i >= 0);
    const flaggedM = mz.z.map((q, i) => (Math.abs(q) > MZ_CUT ? i : -1)).filter((i) => i >= 0);
    const flaggedF = v.map((q, i) => (q < f.lo || q > f.hi ? i : -1)).filter((i) => i >= 0);
    /* Which bottles have actually been made odd, rather than which ones the
       controls are about: at the left end of the slider nothing has moved yet,
       and a rule that flags nothing there is right, not blind. */
    const odd = [];
    if (row.pos > base[SUS] + 1e-9) odd.push(SUS);
    if (two) odd.push(COMP);
    const names = (list) => (list.length === 0 ? 'nothing'
      : list.map((i) => `${v[i].toFixed(0)}`).join(', '));
    const verdict = (list) => {
      const caught = odd.filter((i) => list.includes(i)).length;
      const extra = list.filter((i) => !odd.includes(i)).length;
      let core;
      if (odd.length === 0) core = list.length ? 'flags a bottle that was never moved' : 'nothing to catch yet';
      else if (odd.length === 1) core = caught ? 'catches it' : 'misses it';
      else if (caught === odd.length) core = 'catches both';
      else if (caught === 0) core = 'catches neither';
      else core = 'catches one of the two';
      return extra && odd.length
        ? `${core}, and ${extra} untouched bottle${extra === 1 ? '' : 's'} as well`
        : core;
    };
    readout.html(
      `<table class="odd-table"><thead><tr><th>Rule</th><th>Says</th><th>Flags</th>`
      + `<th>Verdict</th></tr></thead><tbody>`
      + `<tr><td>three sigma</td><td>|z| = ${Math.abs(zs.z[SUS]).toFixed(3)}`
      + (two ? ` and ${Math.abs(zs.z[COMP]).toFixed(3)}` : '') + `</td>`
      + `<td>${names(flaggedZ)}</td><td>${verdict(flaggedZ)}</td></tr>`
      + `<tr><td>Tukey's fences</td><td>above ${f.hi.toFixed(1)}</td>`
      + `<td>${names(flaggedF)}</td><td>${verdict(flaggedF)}</td></tr>`
      + `<tr><td>modified z</td><td>|z| = ${Math.abs(mz.z[SUS]).toFixed(3)}`
      + (two ? ` and ${Math.abs(mz.z[COMP]).toFixed(3)}` : '') + `</td>`
      + `<td>${names(flaggedM)}</td><td>${verdict(flaggedM)}</td></tr>`
      + `</tbody></table>`
    );
  }

  /* --------------------------------------------------------------- controls */
  const MODES = [
    { key: false, label: 'one odd bottle' },
    { key: true, label: 'two odd bottles' },
  ];
  modes.selectAll('button').data(MODES).join('button')
    .attr('class', (d) => `atlas-btn${d.key === two ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      two = d.key;
      modes.selectAll('button').attr('class', (q) => `atlas-btn${q.key === two ? '' : ' ghost'}`);
      render();
    });

  slider.property('value', idx).on('input', function onInput() {
    idx = +this.value;
    render();
  });

  render();
  return { render };
}
