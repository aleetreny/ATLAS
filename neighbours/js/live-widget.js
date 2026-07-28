/* The gradient descent, running here.
 *
 * Three datasets share one perplexity slider and one set of buttons: the
 * digits, three blobs whose radii and separations are known by construction,
 * and pure noise. The point of putting them behind the same control is that the
 * algorithm cannot tell them apart, and the reader can watch it not tell them
 * apart.
 *
 * Fifty iterations is about a fifth of a second at five hundred points, so the
 * step button renders synchronously and "run to the end" chunks through
 * setTimeout, which keeps the page alive and lets the map be watched forming.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';
import { toRows } from '../../assets/js/embed.js';
import { jointP, makeRun, stepRun, klNow, countBlobs, seededNormal, seededBlock } from './tsnekit.js';

const DIGIT = d3.scaleOrdinal(d3.schemeCategory10).domain(d3.range(10));
const BLOB = ['var(--primary)', 'var(--smile)', 'var(--anchor)'];

export function buildDatasets(data) {
  const N = data.meta.n_live;
  const P = data.meta.pcs;
  const blobs = [];
  const B = data.blobs;
  const centres = toRows(B.centres);
  B.seeds.forEach((sd, i) => {
    const part = seededNormal(B.per, P, sd, B.radii[i]);
    for (let r = 0; r < B.per; r++) {
      const row = new Float64Array(P);
      for (let j = 0; j < P; j++) {
        row[j] = Math.round((centres[i][j] + part[r][j]) * 1e3) / 1e3;
      }
      blobs.push(row);
    }
  });
  const noise = seededNormal(N, P, data.noise.seed, data.noise.scale)
    .map((r) => Float64Array.from(r, (v) => Math.round(v * 1e3) / 1e3));
  return {
    digits: {
      X: toRows(data.digits.scores),
      init: toRows(data.digits.init),
      colour: (i) => DIGIT(data.digits.labels[i]),
      label: 'MNIST digits',
    },
    blobs: {
      X: blobs,
      init: toRows(data.blobs.init),
      colour: (i) => BLOB[B.labels[i]],
      label: 'three blobs, radii 1, 2 and 5',
    },
    noise: {
      X: noise,
      init: toRows(data.digits.init),
      colour: () => 'var(--squidink)',
      label: 'pure noise',
    },
  };
}

export function initLiveWidget(data, sets, cache, onProgress = null) {
  const M = data.meta;
  const cfg = {
    iters: M.iters, exaggIters: M.exagg_iters, exagg: M.exagg, lr: M.lr,
  };
  const chart = makeChart('#live-chart', {
    width: 780, height: 430, margin: { top: 60, right: 22, bottom: 30, left: 40 },
  });
  const { g, w, h } = chart;
  const dataButtons = document.querySelector('#live-data-buttons');
  const slider = document.querySelector('#live-perp');
  const perpLabel = document.querySelector('#live-perp-label');
  const readout = document.querySelector('#live-readout');
  const st = {
    set: 'digits', pi: M.perplexities.indexOf(M.perplexity_show), run: null, timer: null,
  };

  const btns = {};
  ['digits', 'blobs', 'noise'].forEach((key) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === st.set ? '' : ' ghost');
    b.textContent = sets[key].label;
    b.addEventListener('click', () => {
      st.set = key;
      restart();
    });
    dataButtons.appendChild(b);
    btns[key] = b;
  });

  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -40).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');

  function key() { return `${st.set}:${M.perplexities[st.pi]}`; }

  function ensureP() {
    const k = key();
    if (!cache[k]) cache[k] = jointP(sets[st.set].X, M.perplexities[st.pi]).P;
    return cache[k];
  }

  function autorun() {
    if (st.timer) return;
    const chunk = () => {
      stepRun(st.run, 50);
      render();
      st.timer = st.run.it < cfg.iters ? setTimeout(chunk, 0) : null;
    };
    st.timer = setTimeout(chunk, 30);
  }

  /* Changing the dataset or the perplexity starts the new run going by itself:
     leaving the reader with a static cloud and three buttons to guess between
     is the state this widget exists to get past. "start again" is the one that
     stops at zero, because that is what it says. */
  function restart(auto = true) {
    if (st.timer) {
      clearTimeout(st.timer);
      st.timer = null;
    }
    st.run = makeRun(ensureP(), sets[st.set].init, cfg);
    render();
    if (auto) autorun();
  }

  function draw() {
    const Y = st.run.Y;
    const { x, y } = equalScales(Y, w, h, { pad: 1.12 });
    body.selectAll('circle').data(Y).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 3.2)
      .attr('fill', (d, i) => sets[st.set].colour(i)).attr('opacity', 0.85);
  }

  function render() {
    if (onProgress) onProgress(st);
    perpLabel.textContent = String(M.perplexities[st.pi]);
    slider.value = String(st.pi);
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== st.set);
    draw();
    const it = st.run.it;
    const perp = M.perplexities[st.pi];
    const stage = it < cfg.exaggIters
      ? `early exaggeration, every neighbour probability multiplied by ${cfg.exagg}`
      : 'the honest objective';
    wrapLabel(title, `${sets[st.set].label}, perplexity ${perp}, `
      + `${it} of ${cfg.iters} steps: ${it === 0 ? 'nothing has moved yet' : stage}`, w - 8);

    const done = it >= cfg.iters;
    const kl = it === 0 ? NaN : klNow(st.run.P, st.run.Y);
    const clumps = done ? countBlobs(st.run.Y, data.noise.counter_multiple) : null;
    const ref = st.set === 'digits' ? data.digits.runs[String(perp)]
      : (st.set === 'noise' ? data.noise.rows.find((r) => r.perplexity === perp)
        : data.blobs.rows.find((r) => r.perplexity === perp));

    let note;
    if (!done) {
      note = it === 0
        ? `Press <span class="bold">run to the end</span> and watch. For the first `
          + `${cfg.exaggIters} steps every neighbour probability is multiplied by ${cfg.exagg}, `
          + `which pulls neighbours together hard and leaves room between the groups that form; `
          + `after that the objective is the real one and the map settles.`
        : `${it} steps in. The divergence is ${kl.toFixed(4)} and falling; the shape at this point `
          + `is not the answer.`;
    } else if (st.set === 'blobs') {
      const B = data.blobs;
      note = `The three blobs were built with radii ${B.radii.join(', ')} and one gap twice the `
        + `other. In the data that is a widest-to-narrowest ratio of `
        + `<span class="bold">${B.true_radius_ratio}</span> and a far-gap-to-near-gap ratio of `
        + `<span class="bold">${B.true_sep_ratio}</span>. On this page they came out at `
        + `<span class="bold">${ref.radius_ratio}</span> and `
        + `<span class="bold">${ref.sep_ratio}</span>. Both are about 1, at every perplexity in `
        + `the slider. A t-SNE map has no scale: the sizes of its clumps and the gaps between them `
        + `are set by how many points are in them, not by anything in the data.`;
    } else if (st.set === 'noise') {
      note = `Nothing was in this. ${data.meta.n_live} points drawn independently from a gaussian `
        + `in ${M.pcs} dimensions, no clusters, no structure of any kind, and the map looks like `
        + `something. A clump counter pinned on cases with known answers finds `
        + `<span class="bold">${clumps.count}</span> here, so what the eye is reading as groups is `
        + `texture rather than clusters, but the eye reads it anyway, which is the whole reason to `
        + `have counted.`;
    } else {
      note = `Finished. The divergence is ${kl.toFixed(4)}; a five nearest neighbour vote on these `
        + `two coordinates alone gets <span class="bold">${(ref.accuracy * 100).toFixed(0)}%</span> `
        + `of the digits right, which is what the map is genuinely good for. The clump counter `
        + `finds <span class="bold">${clumps.count}</span> `
        + `group${clumps.count === 1 ? '' : 's'} here. There are ten.`;
    }

    readout.innerHTML = `
      <table class="nb-table">
        <tr><th>dataset</th><th>perplexity</th><th>steps</th><th>divergence</th><th>clumps a counter finds</th></tr>
        <tr>
          <td>${sets[st.set].label}</td>
          <td><span class="value">${perp}</span></td>
          <td>${it} of ${cfg.iters}</td>
          <td><span class="value">${it === 0 ? '-' : kl.toFixed(4)}</span></td>
          <td><span class="value">${done ? clumps.count : 'when it finishes'}</span></td>
        </tr>
      </table>
      <div class="nb-note">${note}</div>`;
  }

  document.querySelector('#live-step').addEventListener('click', () => {
    stepRun(st.run, 50);
    render();
  });
  document.querySelector('#live-reset').addEventListener('click', () => restart(false));
  document.querySelector('#live-run').addEventListener('click', autorun);
  slider.addEventListener('input', () => {
    st.pi = +slider.value;
    restart();
  });

  /* The widget animates itself into the answer on load rather than either
     freezing the page for a second or opening on a blank canvas that says
     nothing until the reader works out which button to press. The chunked
     runner is the same one the button uses. */
  st.run = makeRun(ensureP(), sets.digits.init, cfg);
  render();
  autorun();
  return st;
}
