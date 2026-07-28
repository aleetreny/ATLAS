/* The one-class SVM's boundary, drawn from the model rather than sketched.
 *
 * Fitting one needs a quadratic programme, so the generator does the fitting and
 * ships what a fitted model is: the support vectors, their coefficients and the
 * offset. Everything else, including the contour, is computed here from
 * f(x) = sum alpha_i K(x_i, x) - rho, which is the whole model.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { svmDecision, averagePrecision } from './detkit.js';
import { drawPoints, legend, seriesLabel, accent } from './draw.js';

const GRID = 120;

export function initBoundaryWidget(data) {
  const S = data.sets['two densities'];
  const X = S.points;
  const y = S.labels;
  const NU = data.svm.nu_grid;
  const GAMMA = data.svm.gamma_grid;

  let ni = NU.indexOf(0.1) >= 0 ? NU.indexOf(0.1) : 1;
  let gi = GAMMA.indexOf(0.5) >= 0 ? GAMMA.indexOf(0.5) : 1;

  const chart = makeChart('#svm-chart', {
    width: 640, height: 490, margin: { top: 74, right: 26, bottom: 56, left: 46 },
  });
  const { g, w, h } = chart;
  const gContour = g.append('g');
  const gPts = g.append('g');
  const gNote = g.append('g');
  const readout = d3.select('#svm-readout');

  const nSlider = d3.select('#svm-nu')
    .attr('min', 0).attr('max', NU.length - 1).attr('step', 1).property('value', ni);
  const gSlider = d3.select('#svm-gamma')
    .attr('min', 0).attr('max', GAMMA.length - 1).attr('step', 1).property('value', gi);

  const xs = X.map((p) => p[0]);
  const ys = X.map((p) => p[1]);
  const x = d3.scaleLinear().domain([Math.min(...xs) - 1, Math.max(...xs) + 1]).range([0, w]);
  const yy = d3.scaleLinear().domain([Math.min(...ys) - 1, Math.max(...ys) + 1]).range([h, 0]);

  function render() {
    ni = +nSlider.property('value');
    gi = +gSlider.property('value');
    const nu = NU[ni];
    const gamma = GAMMA[gi];
    /* Indexed by position, never by a stringified float: Python writes 2.0 as
       "2.0" and JavaScript writes it as "2", so a key built from the numbers on
       one side cannot be found from the other, and the widget drew nothing at
       gamma 2 without saying so. */
    const key = `${ni}|${gi}`;
    const m = data.svm.models[key];
    if (!m) {
      d3.select('#svm-readout').html(`<span class="bold">No model at nu ${nu} and gamma `
        + `${gamma}.</span> The generator did not export this pair, which is a bug rather than a `
        + `finding.`);
      return;
    }
    const model = { sv: m.sv, alpha: m.alpha, rho: m.rho, gamma };

    /* the decision function on a raster, then the zero contour of it */
    const probes = [];
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        probes.push([
          x.domain()[0] + ((x.domain()[1] - x.domain()[0]) * c) / (GRID - 1),
          yy.domain()[1] - ((yy.domain()[1] - yy.domain()[0]) * r) / (GRID - 1),
        ]);
      }
    }
    const f = svmDecision(model, X, probes);
    const contours = d3.contours().size([GRID, GRID]).thresholds([0])(f);
    const px = d3.scaleLinear().domain([0, GRID - 1]).range([0, w]);
    const py = d3.scaleLinear().domain([0, GRID - 1]).range([0, h]);
    const path = d3.geoPath(d3.geoTransform({
      point(a, b) { this.stream.point(px(a), py(b)); },
    }));

    gContour.selectAll('*').remove();
    contours.forEach((c) => {
      gContour.append('path').attr('d', path(c))
        .attr('fill', accent()).attr('fill-opacity', 0.1)
        .attr('stroke', accent()).attr('stroke-width', 2);
    });

    drawAxes(g, x, yy, w, h, { xTicks: 6, yTicks: 6 });
    axisLabels(g, w, h, { x: 'first measurement', y: 'second' });

    const fx = svmDecision(model, X, X);
    const svSet = new Set(m.sv);
    drawPoints(gPts, X, y, x, yy, { r: 2.8 });
    gPts.selectAll('circle.sv').data(m.sv).join('circle').attr('class', 'sv')
      .attr('cx', (i) => x(X[i][0])).attr('cy', (i) => yy(X[i][1])).attr('r', 5.4)
      .attr('fill', 'none').attr('stroke', accent()).attr('stroke-width', 1.4)
      .attr('opacity', 0.75);

    gNote.selectAll('*').remove();
    legend(gNote, [
      { label: 'ordinary', shape: 'dot', colour: 'var(--squidink)' },
      { label: 'planted anomaly', shape: 'ring', colour: 'var(--cosmos)' },
      { label: 'support vector', shape: 'ring', colour: accent() },
      { label: 'the boundary', shape: 'line', colour: accent() },
    ], { x0: 2, y0: -50, gap: 168, cols: 2 });

    d3.select('#svm-nu-label').text(nu);
    d3.select('#svm-gamma-label').text(gamma);

    const outside = fx.filter((v) => v < 0).length;
    const fracOut = outside / X.length;
    const fracSv = m.sv.length / X.length;
    const ap = averagePrecision(y, fx.map((v) => -v));
    readout.html(
      `<span class="slider-label">nu <span class="value">${nu}</span></span> `
      + `<span class="slider-label">gamma <span class="value">${gamma}</span></span> `
      + `The boundary leaves <span class="bold">${outside}</span> of the ${X.length} points `
      + `outside, which is ${(100 * fracOut).toFixed(2)}%, and it is held up by `
      + `${m.sv.length} support vectors, which is ${(100 * fracSv).toFixed(2)}%. `
      + `The theorem says nu should sit between those two. `
      + (fracOut <= nu + 1e-9
        ? `Here it does: ${(100 * fracOut).toFixed(2)}% outside, nu ${(100 * nu).toFixed(0)}%, `
          + `${(100 * fracSv).toFixed(2)}% support vectors.`
        : `<span class="bold">Here it does not.</span> The lower half holds, as it does at every `
          + `setting on this page, but ${(100 * fracOut).toFixed(2)}% of the points are outside a `
          + `boundary that promised at most ${(100 * nu).toFixed(0)}%, which is `
          + `${(fracOut / nu).toFixed(1)} times what was asked for.`)
      + ` Average precision ${ap.toFixed(4)}`
      + (Math.abs(ap - m.ap) < 5e-4 ? '' : ` (the generator recorded ${m.ap})`)
      + `. Nothing here is a density and nothing here is a distance to a centre: the model is a `
      + `boundary, and every point inside it is equally acceptable.`
    );
  }

  nSlider.on('input', render);
  gSlider.on('input', render);
  render();
  return { render };
}
