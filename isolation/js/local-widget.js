/* Local against global, on the dataset the local one was invented for.
 *
 * Both scores are computed here from the same distance table, so the comparison
 * is the definition and not two libraries. The picture is the same points twice
 * over: the mark's size is the score, and the ring is the truth.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { distanceTable, knnDistance, lof, averagePrecision, rocAuc } from './detkit.js';
import { legend, seriesLabel, accent } from './draw.js';

export function initLocalWidget(data) {
  const S = data.sets['two densities'];
  const X = S.points;
  const y = S.labels;
  const K = data.lof.k_grid;
  const D = distanceTable(X);
  const nAnom = y.reduce((a, b) => a + b, 0);

  let ki = Math.max(0, K.indexOf(data.lof.disagree.k) - 3);   /* opens below the
                                                                 k where LOF is
                                                                 perfect, so the
                                                                 slider has work */
  let mode = 'lof';

  const chart = makeChart('#local-chart', {
    width: 640, height: 470, margin: { top: 74, right: 26, bottom: 56, left: 46 },
  });
  const { g, w, h } = chart;
  const gPts = g.append('g');
  const gNote = g.append('g');
  const readout = d3.select('#local-readout');

  const slider = d3.select('#local-k')
    .attr('min', 0).attr('max', K.length - 1).attr('step', 1).property('value', ki);
  const MODES = [{ key: 'lof', label: 'local outlier factor' },
    { key: 'knn', label: 'distance to the kth neighbour' }];
  d3.select('#local-modes').selectAll('button').data(MODES).join('button')
    .attr('class', (d) => `atlas-btn${d.key === mode ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      mode = d.key;
      d3.select('#local-modes').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === mode ? '' : ' ghost'}`);
      render();
    });

  const xs = X.map((p) => p[0]);
  const ys = X.map((p) => p[1]);
  const x = d3.scaleLinear().domain([Math.min(...xs) - 0.6, Math.max(...xs) + 0.6]).range([0, w]);
  const yy = d3.scaleLinear().domain([Math.min(...ys) - 0.6, Math.max(...ys) + 0.6]).range([h, 0]);

  /* every k, once: the readout compares the two detectors at every setting and
     the reader should not have to wait for it */
  const CACHE = K.map((k) => {
    const l = lof(X, k);
    const d = knnDistance(D, k);
    return {
      k,
      lof: l.lof,
      knn: d,
      apLof: averagePrecision(y, l.lof),
      apKnn: averagePrecision(y, d),
      aucLof: rocAuc(y, l.lof),
      aucKnn: rocAuc(y, d),
    };
  });

  function render() {
    ki = +slider.property('value');
    const row = CACHE[ki];
    const s = mode === 'lof' ? row.lof : row.knn;
    const lo = Math.min(...s);
    const hi = Math.max(...s);
    const r = d3.scaleSqrt().domain([lo, hi]).range([2.2, 11]);

    drawAxes(g, x, yy, w, h, { xTicks: 6, yTicks: 6 });
    axisLabels(g, w, h, { x: 'first measurement', y: 'second' });

    const top = s.map((v, i) => i).sort((a, b) => s[b] - s[a]).slice(0, nAnom);
    const inTop = new Set(top);
    const sel = gPts.selectAll('g.pt').data(X);
    sel.exit().remove();
    const enter = sel.enter().append('g').attr('class', 'pt');
    enter.append('circle').attr('class', 'score');
    enter.append('circle').attr('class', 'truth');
    const merged = enter.merge(sel);
    merged.attr('transform', (p) => `translate(${x(p[0])},${yy(p[1])})`);
    merged.select('circle.score')
      .attr('r', (p, i) => r(s[i]))
      .attr('fill', (p, i) => (inTop.has(i) ? accent() : 'var(--squidink)'))
      .attr('opacity', (p, i) => (inTop.has(i) ? 0.85 : 0.32))
      .attr('stroke', 'var(--paper)').attr('stroke-width', 0.7);
    merged.select('circle.truth')
      .attr('r', (p, i) => (y[i] ? 12 : 0))
      .attr('fill', 'none')
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2);

    gNote.selectAll('*').remove();
    legend(gNote, [
      { label: 'planted anomaly', shape: 'ring', colour: 'var(--cosmos)' },
      { label: `the ${nAnom} highest scores`, shape: 'dot', colour: accent() },
      { label: 'size is the score', shape: 'dot', colour: 'var(--squidink)' },
    ], { x0: 2, y0: -50, gap: 178 });
    seriesLabel(gNote, 2, -24,
      `${mode === 'lof' ? 'local outlier factor' : 'distance to neighbour number ' + row.k}`
      + `, from ${lo.toFixed(3)} to ${hi.toFixed(3)}`, { size: 11 });

    d3.select('#local-k-label').text(row.k);
    const caught = top.filter((i) => y[i] === 1).length;
    const loose = top.filter((i) => i >= S.tight && i < S.loose).length;
    const other = mode === 'lof' ? row.apKnn : row.apLof;
    const mine = mode === 'lof' ? row.apLof : row.apKnn;
    readout.html(
      `<span class="slider-label">neighbours <span class="value">${row.k}</span></span> `
      + `The tight cluster has ${S.tight} points and the loose one ${S.loose - S.tight}, and the `
      + `${nAnom} planted anomalies are ringed. Taking the top ${nAnom} scores, `
      + `<span class="bold">${mode === 'lof' ? 'the local factor' : 'the plain distance'} catches `
      + `${caught} of ${nAnom}</span>`
      + (loose
        ? `, and ${loose} of the ${nAnom - caught} it gets wrong `
          + `${loose === 1 ? 'is a member' : 'are members'} of the loose cluster: ordinary points `
          + `whose only crime is living somewhere sparse.`
        : `, and none of its mistakes is a member of the loose cluster.`)
      + ` Average precision ${mine.toFixed(4)} against ${other.toFixed(4)} for the other one at `
      + `the same k`
      + (Math.abs(mine - other) < 0.005
        ? `, which is the same answer.`
        : `, a gap of ${Math.abs(mine - other).toFixed(4)} in favour of `
          + `${mine > other ? 'the one on screen' : 'the one that is not'}.`)
      + ` The local factor is a ratio, so one is its neutral value: a point as dense as the `
      + `company it keeps scores one wherever it lives.`
    );
  }

  slider.on('input', render);
  render();
  return { render, cache: CACHE };
}
