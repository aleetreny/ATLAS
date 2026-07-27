/* The atoms: one digit, three dictionaries, and the constraint made visible.
 *
 * Nothing here is a stored picture. The page downloads 146 quantised tiles in
 * one PNG, and then fits the chosen digit's recipe itself: least squares for
 * PCA and ICA, Lee and Seung's multiplicative update for NMF. main.js checks
 * those fits against the generator's before the reader is told anything.
 *
 * Two things are drawn rather than asserted. The atoms use a diverging
 * colour scale, so PCA's and ICA's run crimson AND teal while NMF's can only
 * ever run one way: the constraint is a colour. And "impossible ink" paints
 * the pixels a reconstruction pushed below zero, which NMF cannot produce and
 * the other two cannot avoid.
 */
import { loadImage, imageToGray, makeCanvas, paintGray, canvasLabel }
  from '../../assets/js/imagery.js';
import { fitLinear, fitNonNegative, reconstruct, mse } from './linalg.js';

const METHODS = [
  ['pca', 'PCA'],
  ['ica', 'ICA'],
  ['nmf', 'NMF'],
];

/* Signed atoms as white to the article's own accent for positive and white to
 * the shared anchor blue for negative, the same two colours the recipe bars
 * underneath already use for added and subtracted.
 *
 * The shared divergingColour helper is not used here on purpose: both of its
 * legs sit in the red family (dark plum one way, bright pink the other), which
 * is fine for a filter response and not fine for the one picture in this
 * article whose entire job is that NMF has no second sign to show. */
const POSITIVE = [54, 83, 20];      // --primary  #365314
const NEGATIVE = [0, 49, 129];      // --anchor   #003181

function signColour(maxAbs) {
  const m = maxAbs > 1e-12 ? maxAbs : 1;
  return (v) => {
    const t = Math.max(-1, Math.min(1, v / m));
    const to = t >= 0 ? POSITIVE : NEGATIVE;
    const a = Math.abs(t);
    return [255 - (255 - to[0]) * a, 255 - (255 - to[1]) * a, 255 - (255 - to[2]) * a];
  };
}

/* Pull the tiles out of the sprite and undo the 8-bit quantisation. */
export function readSprite(gray, sheetW, A) {
  const t = A.tile;
  const cols = A.sprite_cols;
  const tile = (i) => {
    const r0 = Math.floor(i / cols) * t;
    const c0 = (i % cols) * t;
    const [lo, hi] = A.decode[i];
    const span = hi - lo;
    const out = new Float64Array(t * t);
    for (let y = 0; y < t; y++) {
      for (let x = 0; x < t; x++) {
        out[y * t + x] = lo + gray[(r0 + y) * sheetW + c0 + x] * span;
      }
    }
    return out;
  };
  const at = (key) => tile(A.index[key]);
  const bank = {
    mean: at('mean'),
    zero: new Float64Array(t * t),
    show: A.showcase.map((s) => tile(s.tile)),
    pca: d3.range(25).map((j) => at(`pca${j}`)),
    ica: {},
    nmf: {},
  };
  for (const r of A.ranks) {
    bank.ica[r] = d3.range(r).map((j) => at(`ica${r}_${j}`));
    bank.nmf[r] = d3.range(r).map((j) => at(`nmf${r}_${j}`));
  }
  return bank;
}

/* Atoms, offset and coefficients for one method at one rank. */
export function fitOne(method, rank, x, bank, iters) {
  const atoms = method === 'pca' ? bank.pca.slice(0, rank) : bank[method][rank];
  const offset = method === 'nmf' ? bank.zero : bank.mean;
  const coeffs = method === 'nmf'
    ? fitNonNegative(x, atoms, iters)
    : fitLinear(x, atoms, offset);
  const recon = reconstruct(coeffs, atoms, offset);
  return { atoms, offset, coeffs, recon, error: mse(x, recon) };
}

export function initAtomsWidget(data, bank) {
  const A = data.atoms;
  const T = A.tile;
  const host = document.querySelector('#atoms-canvas');
  const readout = document.querySelector('#atoms-readout');
  const slider = document.querySelector('#atoms-rank');
  const rankLabel = document.querySelector('#atoms-rank-label');
  const negBtn = document.querySelector('#atoms-negatives');
  const digitRow = document.querySelector('#atoms-digits');

  const state = { digit: 3, rankIdx: A.ranks.indexOf(16), negatives: false };
  let cols = 0;

  const digitBtns = A.showcase.map((s, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (i === state.digit ? '' : ' ghost');
    b.textContent = String(s.label);
    b.setAttribute('aria-label', `rebuild the digit ${s.label}`);
    b.addEventListener('click', () => {
      state.digit = i;
      render();
    });
    digitRow.appendChild(b);
    return b;
  });

  function render() {
    const rank = A.ranks[state.rankIdx];
    const x = bank.show[state.digit];
    const fits = {};
    for (const [key] of METHODS) fits[key] = fitOne(key, rank, x, bank, A.nmf_fit_iters);

    /* ---- layout. A fixed four-across grid shrinks its own labels to about
       4px on a phone, so below 620px it folds to two across, which doubles
       the rendered text size without changing a single font declaration. */
    const wide = host.clientWidth >= 620 || host.clientWidth === 0;
    const want = wide ? 4 : 2;
    if (want !== cols) {
      cols = want;
      host.innerHTML = '';
    }
    const panelW = 208;
    const zoom = 5;
    const imgSize = T * zoom;
    const gridRows = Math.ceil(4 / cols);

    /* The panel height is computed from what is actually drawn, and from the
       TALLEST rank rather than the current one, so the canvas does not change
       size under the reader's hand as the slider moves. Hard-coding it at 358
       left the dictionary of the 25-atom factorisation hanging 70px past the
       bottom edge, which the canvas bleed check caught and a screenshot would
       not have, because a canvas simply does not paint what falls off it. */
    const gridHeight = (r) => {
      const gc = Math.ceil(Math.sqrt(r));
      const gz = Math.max(1, Math.min(3, (imgSize - (gc - 1) * 2) / (gc * T)));
      return Math.ceil(r / gc) * (T * gz + 2);
    };
    const gridTop = 40 + imgSize + 30 + 34 + 22;
    const rowH = gridTop + Math.max(...A.ranks.map(gridHeight)) + 26;
    const width = panelW * cols + 24;
    const height = rowH * gridRows + 10;

    let ctx = host.__ctx;
    if (!ctx || host.__w !== width || host.__h !== height) {
      host.innerHTML = '';
      ctx = makeCanvas(host, width, height).ctx;
      host.__ctx = ctx;
      host.__w = width;
      host.__h = height;
    }
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    /* One diverging scale across every atom of every method, so that "NMF's
       atoms only ever go one way" is something the reader sees rather than an
       artefact of three separate scales.
     *
     * But the atoms have to be put on a common footing first, because an
     * atom's scale is arbitrary: it trades off exactly against its own
     * coefficient, so a factorisation is free to return big atoms with small
     * amounts or the reverse. Measured here, NMF's atoms peak at 13.4 while
     * PCA's peak at 0.20, a factor of 67, and on a scale set by the largest of
     * them ZERO per cent of the PCA and ICA pixels came out visibly different
     * from white. Both dictionaries rendered as blank squares. Normalising
     * each atom to unit length fixes it without touching the thing being
     * shown, since dividing by a positive number cannot change a sign. */
    const unit = (a) => {
      let n = 0;
      for (const v of a) n += v * v;
      n = Math.sqrt(n) || 1;
      return Float64Array.from(a, (v) => v / n);
    };
    const shown = {};
    let maxAbs = 0;
    for (const [key] of METHODS) {
      shown[key] = fits[key].atoms.map(unit);
      for (const a of shown[key]) for (const v of a) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
    const atomColour = signColour(maxAbs);

    const panels = [
      { title: 'the digit', sub: `MNIST, ${T}x${T} pixels`, pixels: x, fit: null },
      ...METHODS.map(([key, name]) => ({
        title: name,
        sub: `${rank} atom${rank === 1 ? '' : 's'}`,
        pixels: fits[key].recon,
        fit: fits[key],
        key,
      })),
    ];

    panels.forEach((p, i) => {
      const x0 = 12 + (i % cols) * panelW;
      const y0 = 6 + Math.floor(i / cols) * rowH;
      const imgX = x0 + (panelW - imgSize) / 2;

      canvasLabel(ctx, p.title, x0 + panelW / 2, y0 + 15, { size: 13 });
      canvasLabel(ctx, p.sub, x0 + panelW / 2, y0 + 30, { size: 11, colour: '#5a6472' });

      /* the picture */
      const neg = state.negatives && p.fit;
      paintGray(ctx, p.pixels, T, T, {
        zoom,
        x0: imgX,
        y0: y0 + 40,
        colour: neg
          ? (v) => (v < -A.errors[0].negative_threshold
            ? [201, 42, 93]
            : [Math.round(Math.max(0, Math.min(1, v)) * 255)].concat(
              [Math.round(Math.max(0, Math.min(1, v)) * 255),
                Math.round(Math.max(0, Math.min(1, v)) * 255)]))
          : null,
      });
      ctx.strokeStyle = '#d4dada';
      ctx.strokeRect(imgX + 0.5, y0 + 40.5, imgSize, imgSize);

      if (!p.fit) {
        canvasLabel(ctx, `label: ${A.showcase[state.digit].label}`, x0 + panelW / 2,
          y0 + 40 + imgSize + 20, { size: 12 });
        canvasLabel(ctx, 'the target, untouched', x0 + panelW / 2,
          y0 + 40 + imgSize + 36, { size: 11, colour: '#5a6472' });
        return;
      }

      canvasLabel(ctx, `error ${p.fit.error.toFixed(5)}`, x0 + panelW / 2,
        y0 + 40 + imgSize + 20, { size: 12 });

      /* the recipe: one signed bar per atom */
      const barTop = y0 + 40 + imgSize + 30;
      const barH = 34;
      const barW = Math.min(12, (imgSize - 2) / rank);
      const cmax = Math.max(...p.fit.coeffs.map(Math.abs), 1e-9);
      ctx.strokeStyle = '#d4dada';
      ctx.beginPath();
      ctx.moveTo(imgX, barTop + barH / 2 + 0.5);
      ctx.lineTo(imgX + imgSize, barTop + barH / 2 + 0.5);
      ctx.stroke();
      p.fit.coeffs.forEach((c, j) => {
        const bx = imgX + j * (imgSize / rank) + (imgSize / rank - barW) / 2;
        const hgt = (Math.abs(c) / cmax) * (barH / 2);
        ctx.fillStyle = c >= 0 ? '#365314' : '#003181';
        ctx.fillRect(bx, c >= 0 ? barTop + barH / 2 - hgt : barTop + barH / 2, barW, hgt);
      });
      const nneg = p.fit.coeffs.filter((c) => c < 0).length;
      canvasLabel(ctx, nneg === 0 ? 'every amount positive' : `${nneg} of ${rank} subtracted`,
        x0 + panelW / 2, barTop + barH + 14, { size: 11, colour: '#5a6472' });

      /* the dictionary itself */
      const gcols = Math.ceil(Math.sqrt(rank));
      const grows = Math.ceil(rank / gcols);
      const gz = Math.max(1, Math.min(3, (imgSize - (gcols - 1) * 2) / (gcols * T)));
      const gw = gcols * (T * gz + 2) - 2;
      const gx = x0 + (panelW - gw) / 2;
      const gy = barTop + barH + 22;
      shown[p.key].forEach((a, j) => {
        paintGray(ctx, a, T, T, {
          zoom: gz,
          x0: gx + (j % gcols) * (T * gz + 2),
          y0: gy + Math.floor(j / gcols) * (T * gz + 2),
          colour: atomColour,
        });
      });
      canvasLabel(ctx, 'its atoms', x0 + panelW / 2, gy + grows * (T * gz + 2) + 12,
        { size: 11, colour: '#5a6472' });
    });

    rankLabel.textContent = String(rank);
    digitBtns.forEach((b, i) => b.classList.toggle('ghost', i !== state.digit));
    negBtn.classList.toggle('ghost', !state.negatives);
    negBtn.textContent = state.negatives ? 'showing impossible ink' : 'show impossible ink';

    /* ---- readout, all of it derived from the fits just drawn */
    const row = A.errors.find((r) => r.rank === rank);
    const ep = fits.pca.error;
    const ei = fits.ica.error;
    const en = fits.nmf.error;
    const gap = Math.abs(ep - ei);
    const negPix = fits.pca.recon.filter((v) => v < -row.negative_threshold).length;
    const negPixI = fits.ica.recon.filter((v) => v < -row.negative_threshold).length;
    const negPixN = fits.nmf.recon.filter((v) => v < -row.negative_threshold).length;
    const zeros = fits.nmf.coeffs.filter((c) => c < 1e-6).length;

    readout.innerHTML = `
      <table class="dir-table">
        <tr><th>this digit, ${rank} atoms</th><th>error</th><th>amounts subtracted</th><th>ink below -${row.negative_threshold}</th></tr>
        <tr><td>PCA</td><td><span class="value">${ep.toFixed(5)}</span></td>
            <td>${fits.pca.coeffs.filter((c) => c < 0).length} of ${rank}</td>
            <td>${negPix} pixel${negPix === 1 ? '' : 's'}</td></tr>
        <tr><td>ICA</td><td><span class="value">${ei.toFixed(5)}</span></td>
            <td>${fits.ica.coeffs.filter((c) => c < 0).length} of ${rank}</td>
            <td>${negPixI} pixel${negPixI === 1 ? '' : 's'}</td></tr>
        <tr><td>NMF</td><td><span class="value">${en.toFixed(5)}</span></td>
            <td>${fits.nmf.coeffs.filter((c) => c < 0).length} of ${rank}</td>
            <td>${negPixN} pixel${negPixN === 1 ? '' : 's'}</td></tr>
      </table>
      <div class="dir-note">
        ${gap <= row.ica_gap_worst_image * 1.5
    ? `PCA and ICA rebuild this digit to <span class="bold">the same error</span>, `
          + `${gap.toExponential(1)} apart, and they will at every rank and for every digit. `
          + `They are not competing: ICA takes PCA's ${rank}-dimensional subspace and turns it, and `
          + `a turn cannot change how far a point is from a subspace. Their atoms look nothing alike `
          + `and their arithmetic lands in the same place. What separates the two numbers above is `
          + `not the method, it is that the atoms arrived rounded to 8 bits: before that rounding `
          + `the gap across all ${A.n_images.toLocaleString('en-US')} digits is `
          + `${row.ica_gap_unquantised.toExponential(0)}, and afterwards it is `
          + `${row.ica_gap.toExponential(1)} on average and never more than `
          + `${row.ica_gap_worst_image.toExponential(1)} on any single digit.`
    : `PCA and ICA differ by ${gap.toExponential(2)} here, more than the `
          + `${row.ica_gap_worst_image.toExponential(1)} that rounding the atoms to 8 bits can `
          + `explain. They span the same subspace, so something in the fit has gone wrong.`}
        <br /><br />
        NMF pays <span class="bold">${(en / ep).toFixed(3)} times</span> PCA's error on this digit
        (${row.nmf_ratio} times across the whole set) and gets two things back that no amount of
        variance buys. It used ${rank - zeros} of its ${rank} atoms and set the other ${zeros}
        to exactly zero, so the recipe is a short list rather than a full one. And its picture
        contains no impossible ink: every amount is positive and every atom is positive, so the
        reconstruction cannot go below zero even in principle, while PCA's went below
        -${row.negative_threshold} on ${negPix} of this digit's ${T * T} pixels.
      </div>`;
  }

  slider.addEventListener('input', () => {
    state.rankIdx = +slider.value;
    render();
  });
  negBtn.addEventListener('click', () => {
    state.negatives = !state.negatives;
    render();
  });
  window.addEventListener('resize', () => render());
  render();
  return { fitOne: (m, r, i) => fitOne(m, r, bank.show[i], bank, A.nmf_fit_iters) };
}

/* Load the sprite and hand back the decoded dictionary. */
export async function loadAtoms(data) {
  const A = data.atoms;
  const img = await loadImage('./data/atoms.png');
  const { gray, w } = imageToGray(img);
  return readSprite(gray, w, A);
}
