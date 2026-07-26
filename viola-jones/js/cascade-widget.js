/* The cascade, run for real, in your browser, over every window of a photo.
 *
 * The page builds one integral image of the picture and slides a 25-pixel
 * window across all 10,816 positions, running the trained cascade at each.
 * Nothing is precomputed: the survivor counts you see are produced here and
 * checked against the generator's at load time.
 */
import { drawAxes, makeChart } from '../../assets/js/chart.js';
import { loadImage, imageToGray, makeCanvas, canvasLabel } from '../../assets/js/imagery.js';
import { integralImage, runCascade } from './vjkit.js';

export async function initCascadeWidget(data, base) {
  const sc = data.scan;
  const img = await loadImage(`${base}${sc.image}`);
  const S = sc.size;
  const { gray } = imageToGray(img, S, S);
  const win = sc.window;
  const learners = data.learners;
  const stages = data.cascade.stages;

  /* run every window, recording where each one died */
  const t0 = performance.now();
  const ii = integralImage(gray, S, S);
  const stride = S + 1;
  const results = [];
  const perStage = new Array(stages.length + 1).fill(0);
  let usedTotal = 0;
  for (let top = 0; top + win <= S; top++) {
    for (let left = 0; left + win <= S; left++) {
      const r = runCascade(ii, stride, top, left, learners, stages);
      results.push({ top, left, stage: r.stage, survived: r.survived, score: r.score });
      perStage[r.stage] += 1;
      usedTotal += r.used;
    }
  }
  const ms = performance.now() - t0;
  const meanUsed = usedTotal / results.length;

  const ZOOM = 3;
  const W = S * ZOOM + 20;
  const H = S * ZOOM + 34;
  const { ctx } = makeCanvas('#scan-chart', W, H);
  const chart = makeChart('#scan-funnel', { width: 420, height: 260, margin: { top: 26, right: 18, bottom: 46, left: 62 } });

  const slider = document.querySelector('#scan-stage');
  const readout = document.querySelector('#scan-readout');
  slider.max = String(stages.length);

  function render() {
    const upto = +slider.value;              // 0 = before any stage
    document.querySelector('#scan-stage-label').textContent =
      upto === 0 ? 'none yet' : `${upto} of ${stages.length}`;

    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img, 10, 24, S * ZOOM, S * ZOOM);
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.fillRect(10, 24, S * ZOOM, S * ZOOM);
    ctx.restore();

    /* windows still alive after `upto` stages, drawn at their centres so
     * ten thousand of them stay legible */
    const alive = results.filter((r) => r.stage >= upto);
    ctx.save();
    ctx.fillStyle = 'rgba(30,64,175,0.30)';
    for (const r of alive) {
      ctx.fillRect(10 + (r.left + win / 2) * ZOOM - 1, 24 + (r.top + win / 2) * ZOOM - 1, 2, 2);
    }
    if (upto === stages.length) {
      ctx.strokeStyle = '#b91c1c';
      ctx.lineWidth = 1.6;
      for (const r of alive) {
        ctx.strokeRect(10 + r.left * ZOOM, 24 + r.top * ZOOM, win * ZOOM, win * ZOOM);
      }
    }
    ctx.restore();
    canvasLabel(ctx, upto === 0
      ? `all ${results.length.toLocaleString('en-US')} windows, before any test`
      : `${alive.length.toLocaleString('en-US')} windows still alive after ${upto} stage${upto === 1 ? '' : 's'}`,
    W / 2, 16, { size: 11 });

    /* the funnel */
    chart.g.selectAll('*').remove();
    const counts = [results.length];
    for (let st = 0; st < stages.length; st++) {
      counts.push(results.filter((r) => r.stage > st).length);
    }
    const x = d3.scalePoint().domain(d3.range(counts.length)).range([0, chart.w]).padding(0.3);
    const y = d3.scaleLog().domain([Math.max(1, d3.min(counts) * 0.7), results.length * 1.2]).range([chart.h, 0]);
    drawAxes(chart.g, x, y, chart.w, chart.h, { yTicks: 4, yFmt: d3.format('~s') });
    chart.g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3)
      .attr('d', d3.line().x((v, i) => x(i)).y((v) => y(Math.max(v, 1)))(counts));
    chart.g.selectAll('circle').data(counts).join('circle')
      .attr('cx', (v, i) => x(i)).attr('cy', (v) => y(Math.max(v, 1)))
      .attr('r', (v, i) => (i === upto ? 6 : 4))
      .attr('fill', (v, i) => (i === upto ? 'var(--cosmos)' : 'var(--primary)'))
      .attr('stroke', 'white').attr('stroke-width', 1.4);
    chart.g.append('text').attr('class', 'chart-annotation')
      .attr('x', chart.w / 2).attr('y', -10).attr('text-anchor', 'middle')
      .style('font-size', '0.56rem').style('text-transform', 'none')
      .text('windows surviving each stage (log scale)');
    chart.g.append('text').attr('class', 'chart-annotation')
      .attr('x', chart.w / 2).attr('y', chart.h + 36).attr('text-anchor', 'middle')
      .style('font-size', '0.56rem').style('text-transform', 'none')
      .text('cascade stage');

    const stageInfo = upto > 0 ? stages[upto - 1] : null;
    readout.innerHTML =
      `<table class="vj-table">
         <tr><td>windows scanned</td><td><span class="value">${results.length.toLocaleString('en-US')}</span></td>
             <td>still alive here</td><td><span class="value">${alive.length.toLocaleString('en-US')}</span></td>
             <td>features per window, average</td><td><span class="value">${meanUsed.toFixed(2)}</span> of ${data.cascade.full_cost}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        upto === 0
          ? `Every position a 25-pixel window can take on this photograph. A detector that ran all ` +
            `${data.cascade.full_cost} features everywhere would do ` +
            `${(results.length * data.cascade.full_cost).toLocaleString('en-US')} feature evaluations. ` +
            `Push the slider and watch that bill collapse.`
          : upto === 1
            ? `The first stage costs ${stageInfo.size} feature${stageInfo.size === 1 ? '' : 's'} and it has ` +
              `already thrown away ${(results.length - alive.length).toLocaleString('en-US')} windows. Everything ` +
              `it rejected will never be looked at again, which is where the savings come from: not from ` +
              `being clever about faces, but from being fast about everything else.`
            : upto === stages.length
              ? `${alive.length} windows survive all ${stages.length} stages, drawn as boxes. The average ` +
                `window cost <span class="bold">${meanUsed.toFixed(2)}</span> feature evaluations instead of ` +
                `${data.cascade.full_cost}, a ${(data.cascade.full_cost / meanUsed).toFixed(1)}-fold saving, and ` +
                `this browser did the whole scan in ${ms.toFixed(0)} ms. Note the boxes are not all faces: a ` +
                `detector trained on a hundred crops is a toy, and the honest lesson is in the next section.`
              : `Stage ${upto} spent ${stageInfo.size} more feature${stageInfo.size === 1 ? '' : 's'} on the ` +
                `windows that survived, and the ones it kills cost the earlier stages only. Each survivor is ` +
                `getting more expensive and there are fewer of them, which is the whole design.`
      }</div>`;
  }

  slider.addEventListener('input', render);
  render();

  /* Runtime check: the browser's scan must reproduce the generator's. */
  const mine = results.filter((r) => r.survived).length;
  if (mine !== sc.survivors.length) {
    console.warn(`browser scan found ${mine} survivors, the generator recorded ${sc.survivors.length}`);
  }
  const gap = Math.abs(meanUsed - sc.mean_evaluated);
  if (gap > 0.02) {
    console.warn(`browser mean features per window ${meanUsed.toFixed(2)} vs generator ${sc.mean_evaluated}`);
  }
  return { survivors: mine, meanUsed, ms };
}
