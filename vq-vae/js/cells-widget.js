/* Four cells, sixty four entries each, and a decoder that runs here.
 *
 * This is the widget the article exists for. A continuous code cannot be
 * turned by hand: there is nowhere to stop. A discrete one can, so pick a cell,
 * step through the alphabet, and watch what a single symbol is worth. The atlas
 * on the right is the same question asked exhaustively: every entry of the
 * codebook decoded in that cell with the other three held where they are.
 */
import { makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { uniformCodes } from './vqkit.js';

const ATLAS_COLS = 8;

export function initCellsWidget(data, vq, prior) {
  const side = data.meta.digits.pixels === 196 ? 14 : Math.round(Math.sqrt(data.meta.digits.pixels));
  const N = data.net;
  const shots = N.showcase;
  const st = { shot: 0, cell: 0, codes: shots[0].codes.slice(), source: 'digit' };

  const cv = makeCanvas(document.querySelector('#cells-canvas'), side * 13, side * 13);
  const atlas = makeCanvas(document.querySelector('#cells-atlas'),
    side * ATLAS_COLS * 2, side * (vq.k / ATLAS_COLS) * 2);
  const grid = document.querySelector('#cell-grid');
  const caption = document.querySelector('#cell-caption');
  const canvasCaption = document.querySelector('#cells-canvas-caption');
  const atlasCaption = document.querySelector('#cells-atlas-caption');
  const readout = document.querySelector('#cells-readout');
  const buttons = document.querySelector('#cells-buttons');

  const btns = {};
  const add = (key, text, fn) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', fn);
    buttons.appendChild(b);
    btns[key] = b;
  };
  add('digit', 'a held out digit', () => {
    st.source = 'digit';
    st.shot = (st.shot + 1) % shots.length;
    st.codes = shots[st.shot].codes.slice();
    render();
  });
  add('prior', 'a grid the prior invented', () => {
    st.source = 'prior';
    st.seedP = (st.seedP || 7717) + 613;
    st.codes = prior.sample(st.seedP);
    render();
  });
  add('uniform', 'four codes at random', () => {
    st.source = 'uniform';
    st.seedU = (st.seedU || 31) + 977;
    st.codes = uniformCodes(vq.k, vq.cells, st.seedU);
    render();
  });

  const cellButtons = [];
  for (let c = 0; c < vq.cells; c++) {
    const b = document.createElement('button');
    b.className = 'cell-btn';
    b.addEventListener('click', () => { st.cell = c; render(); });
    grid.appendChild(b);
    cellButtons.push(b);
  }

  function paintOne() {
    const px = vq.decode(st.codes);
    const g = new Float64Array(px.length);
    for (let i = 0; i < px.length; i++) g[i] = 1 - px[i];
    paintGray(cv.ctx, g, side, side, { zoom: 13 });
    return px;
  }

  /* every entry of the alphabet, in the chosen cell, with the rest held */
  function paintAtlas() {
    const rows = vq.k / ATLAS_COLS;
    const W = side * ATLAS_COLS;
    const big = new Float64Array(W * side * rows);
    for (let j = 0; j < vq.k; j++) {
      const codes = st.codes.slice();
      codes[st.cell] = j;
      const px = vq.decode(codes);
      const r = Math.floor(j / ATLAS_COLS);
      const c = j % ATLAS_COLS;
      for (let a = 0; a < side; a++) {
        for (let b = 0; b < side; b++) {
          big[(r * side + a) * W + c * side + b] = 1 - px[a * side + b];
        }
      }
    }
    paintGray(atlas.ctx, big, W, side * rows, { zoom: 2 });
  }

  /* the atlas is the picker: clicking an entry puts it in the chosen cell,
     which is what the caption has been promising */
  atlas.canvas.style.cursor = 'pointer';
  atlas.canvas.addEventListener('click', (ev) => {
    const box = atlas.canvas.getBoundingClientRect();
    const col = Math.floor(((ev.clientX - box.left) / box.width) * ATLAS_COLS);
    const rows = vq.k / ATLAS_COLS;
    const row = Math.floor(((ev.clientY - box.top) / box.height) * rows);
    const j = row * ATLAS_COLS + col;
    if (j < 0 || j >= vq.k) return;
    st.codes = st.codes.slice();
    st.codes[st.cell] = j;
    st.source = 'edited';
    render();
  });

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.source));
    cellButtons.forEach((b, c) => {
      b.classList.toggle('is-picked', c === st.cell);
      b.innerHTML = `cell ${c + 1}<br /><span class="value">${st.codes[c]}</span> of ${vq.k}`;
    });
    paintOne();
    paintAtlas();

    caption.textContent = `pick a cell here, then click any entry in the atlas on the right to `
      + `put it in that cell: ${vq.cells} cells times ${Math.round(Math.log2(vq.k))} bits is `
      + `${N.arch.bits} bits for the whole picture`;
    canvasCaption.textContent = st.source === 'digit'
      ? `held out digit ${shots[st.shot].label}, rebuilt from its ${vq.cells} codes`
      : st.source === 'prior'
        ? 'decoded from a grid the prior over codes invented'
        : st.source === 'edited'
          ? 'decoded from a code you edited by hand'
          : 'decoded from four entries picked uniformly at random';
    atlasCaption.textContent = `all ${vq.k} entries in cell ${st.cell + 1}, with the other `
      + `${vq.cells - 1} held where they are`;

    const usage = N.usage;
    const total = usage.reduce((a, b) => a + b, 0);
    const share = ((usage[st.codes[st.cell]] / total) * 100).toFixed(2);
    const q = data.net.quadrant_share[st.cell];
    const strongest = Math.max(...q);
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>the code</th><th>bits</th><th>this entry's share</th>
          <th>strongest quadrant of this cell</th><th>rebuilding</th>
        </tr>
        <tr>
          <td><span class="value">${st.codes.join(', ')}</span></td>
          <td><span class="value">${N.arch.bits}</span> for the whole picture</td>
          <td>${share}% of ${total.toLocaleString('en-US')}</td>
          <td><span class="value">${(strongest * 100).toFixed(1)}%</span>, 25% owns nothing</td>
          <td>${N.quant.mse_quantised}</td>
        </tr>
      </table>
      <div class="gen-note">
        ${st.source === 'edited'
    ? `You just changed one symbol of four. The other three cells are exactly where they were,
       and the whole picture moved: that is what it means for the code to have no spatial layout.
       At ${N.arch.bits} bits, this model rebuilds a held out digit at ${N.quant.mse_quantised}.`
    : st.source === 'uniform'
    ? `Four entries drawn uniformly. Over ${data.prior.samples} such grids the shared judge is
       ${data.prior.samplers.uniform.confidence.mean} confident, against
       ${data.prior.judge_real_confidence} on real held out digits, and the pictures spread over
       ${data.prior.samplers.uniform.class_entropy.mean} nats of class entropy where the ten
       classes flat out would be ${Math.log(10).toFixed(3)}. A codebook is not a distribution:
       nothing in the training objective ever said which combinations of symbols are legal.`
    : st.source === 'prior'
      ? `This grid came from a second model, trained on nothing but the ${data.meta.digits.train}
         code grids the encoder produced. It spends ${data.prior.ar_bits.mean} bits per cell
         against the ${data.prior.uniform_bits} a uniform prior costs, so the codes really do
         carry structure, and the judge gives its samples
         ${data.prior.samplers.prior.confidence.mean}.`
      : `The four numbers under the picture are the entire code: change one and the whole digit
         moves, which is the answer to the obvious question about whether a cell owns a corner
         of the image. It does not: the strongest quadrant of a cell takes
         ${(Math.max(...data.net.quadrant_share.flat()) * 100).toFixed(1)}% of the change it
         causes at most, where a cell that owned a quarter of the picture would take 100%.`}
      </div>`;
  }

  render();
  return st;
}
