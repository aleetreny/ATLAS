/* The opening story, on twelve bottles and one number each.
 *
 * Everything drawn here is recomputed from the sample as it stands, never read
 * off a table: the mean, the spread, the band each rule declares ordinary and
 * the z of the bottle being pushed. That is the point of the section, so a
 * pre-baked frame would be a lie in exactly the place the article is about.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { zScores, modifiedZ, tukeyFences } from './statkit.js';
import { drawSample, drawBand, marker, legend, seriesLabel } from './draw.js';

export function initScrollyLine(data) {
  const S = data.one_d.sub;
  const base = S.values;
  const SUS = S.suspect;
  const COMP = S.companion;
  const Z_CUT = data.one_d.z_cut;
  const MZ_CUT = data.one_d.mz_cut;

  /* The two stops of the walk are chosen from the measured curve, not typed in:
     the first place the classical rule fires, and the far end. */
  const firstFlag = S.one.find((r) => Math.abs(r.z) > Z_CUT) ?? S.one[Math.floor(S.one.length / 2)];
  const farEnd = S.one[S.one.length - 1];

  const STATES = [
    { pos: base[SUS], comp: false, bands: false },
    { pos: base[SUS], comp: false, bands: true },
    { pos: firstFlag.pos, comp: false, bands: true },
    { pos: farEnd.pos, comp: false, bands: true },
    { pos: farEnd.pos, comp: true, bands: true },
  ];

  const chart = makeChart('#line-scrolly-chart', {
    width: 640, height: 400, margin: { top: 78, right: 36, bottom: 62, left: 30 },
  });
  const { g, w, h } = chart;
  const baseline = h - 30;
  const bandTop = 12;
  const gBands = g.append('g');
  const gMarks = g.append('g');
  const gPoints = g.append('g');
  const gNotes = g.append('g');

  /* Two rows rather than one: five chips laid end to end are wider than the
     canvas, and the mean and the median have to be in here rather than labelled
     on their own lines, which is where they collided with the numbers below. */
  legend(g.append('g'), [
    { label: 'three sigma', shape: 'band', colour: 'var(--anchor)' },
    { label: "Tukey's fences", shape: 'band', colour: 'var(--cosmos)' },
  ], { x0: 2, y0: -64, gap: 200, size: 10.5 });
  legend(g.append('g'), [
    { label: 'mean', shape: 'line', colour: 'var(--anchor)' },
    { label: 'median', shape: 'dash', colour: 'var(--galaxy)' },
    { label: 'the bottle being pushed', shape: 'dot', colour: 'var(--primary)' },
  ], { x0: 2, y0: -48, gap: 148, size: 10.5 });

  const x = d3.scaleLinear().range([0, w]);

  function render(state, animate = true) {
    const v = base.slice();
    v[SUS] = state.pos;
    if (state.comp) v[COMP] = S.comp_at;

    const hi = Math.max(...v);
    const lo = Math.min(...v);
    const pad = Math.max(6, (hi - lo) * 0.09);
    x.domain([lo - pad, hi + pad]);

    const zs = zScores(v);
    const mz = modifiedZ(v);
    const f = tukeyFences(v);
    /* Named, because two unnamed transitions on the same nodes cancel each
       other and the band would freeze while the points moved. */
    const anim = (name) => (sel) => (animate ? sel.transition(name).duration(650) : sel);

    /* only the horizontal axis carries meaning here; the vertical direction is
       just room for repeated readings to stack in */
    let gx = gMarks.select('g.axis.x');
    if (gx.empty()) gx = gMarks.append('g').attr('class', 'axis x');
    anim('axis')(gx.attr('transform', `translate(0,${h})`))
      .call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
    axisLabels(gMarks, w, h, { x: 'magnesium, mg per litre' });

    /* the bands: what each rule is willing to call ordinary */
    if (state.bands) {
      drawBand(gBands, x, zs.mean - Z_CUT * zs.sd, zs.mean + Z_CUT * zs.sd, bandTop, baseline + 12,
        { colour: 'var(--anchor)', klass: 'zband', opacity: 0.12, anim: anim('zb') });
      drawBand(gBands, x, f.lo, f.hi, bandTop + 8, baseline + 4,
        { colour: 'var(--cosmos)', klass: 'fband', opacity: 0.1, anim: anim('fb') });
    } else {
      gBands.selectAll('rect').remove();
    }

    gMarks.selectAll('g.marker').remove();
    marker(gMarks, x(zs.mean), bandTop - 8, baseline + 16, '',
      { colour: 'var(--anchor)', w, dash: null });
    marker(gMarks, x(mz.median), bandTop - 8, baseline + 16, '',
      { colour: 'var(--galaxy)', w, dash: '2 3' });

    drawSample(gPoints, v, x, baseline, {
      r: 6,
      highlight: (i) => i === SUS || (state.comp && i === COMP),
      anim: anim('pts'),
    });

    /* the numbers, in the top margin where nothing is drawn */
    gNotes.selectAll('*').remove();
    const flagged = zs.z.filter((q) => Math.abs(q) > Z_CUT).length;
    const flaggedM = mz.z.filter((q) => Math.abs(q) > MZ_CUT).length;
    seriesLabel(gNotes, 0, -28,
      `mean ${zs.mean.toFixed(1)}   sd ${zs.sd.toFixed(1)}   `
      + `z of the pushed bottle ${zs.z[SUS].toFixed(3)}`, { size: 11.5, opacity: 0.95 });
    seriesLabel(gNotes, 0, -12,
      `three sigma flags ${flagged}   ·   the fences flag `
      + `${v.filter((q) => q < f.lo || q > f.hi).length}   ·   modified z flags ${flaggedM}`,
      { size: 11.5, opacity: 0.75 });
  }

  render(STATES[0], false);
  const handlers = {};
  STATES.forEach((s, i) => {
    handlers[String(i)] = () => render(s);
  });
  return scrolly('#line-scrolly .step', handlers);
}
