/* The noise inputs, live.
 *
 * Same style vector, different per-pixel noise: whatever changes is what the
 * network chose to leave to chance, and whatever does not is what the style
 * decides. The right hand panel is the standard deviation over realisations,
 * amplified so it is visible at all, which is the picture the StyleGAN paper
 * uses to argue that the noise buys stochastic detail rather than content.
 * Both halves are measured here rather than asserted: the factor readout under
 * the panels is the same measurement the article's table used.
 */
import { drawTiles } from './panels.js';
import { measure, hueDist } from './stylekit.js';
import { makeCanvas } from '../../assets/js/imagery.js';

export function initNoiseWidget(data, gen) {
  const M = data.meta;
  const nb = M.blocks.length;
  const state = { z: Float32Array.from(data.check.z[2]), on: true, seed: 11 };

  function noiseFor(seed) {
    let s = seed >>> 0;
    const rng = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
    return M.blocks.map((b, i) => {
      const res = [4, 8, 8, 16, 32][i];
      const out = new Float32Array(res * res);
      for (let k = 0; k < res * res; k++) {
        const u1 = Math.max(rng(), 1e-12);
        const u2 = rng();
        out[k] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      return out;
    });
  }

  function render() {
    const w = gen.map(state.z);
    const ws = new Array(nb).fill(w);
    const n = 6;
    const imgs = [];
    for (let i = 0; i < n; i++) {
      imgs.push(gen.synth(ws, state.on ? noiseFor(state.seed + i * 7919) : null));
    }
    drawTiles(document.querySelector('#noise-tiles'), imgs, M.res, n, { zoom: 3, gap: 6 });

    /* the standard deviation over the realisations, per pixel */
    const sd = new Float32Array(M.res * M.res);
    for (let p = 0; p < M.res * M.res; p++) {
      let mu = 0;
      for (let i = 0; i < n; i++) {
        mu += (imgs[i][p * 3] + imgs[i][p * 3 + 1] + imgs[i][p * 3 + 2]) / 3;
      }
      mu /= n;
      let v = 0;
      for (let i = 0; i < n; i++) {
        const g = (imgs[i][p * 3] + imgs[i][p * 3 + 1] + imgs[i][p * 3 + 2]) / 3;
        v += (g - mu) ** 2;
      }
      sd[p] = Math.sqrt(v / Math.max(n - 1, 1));
    }
    const peak = Math.max(...sd, 1e-6);
    const host = document.querySelector('#noise-map');
    host.innerHTML = '';
    const { ctx } = makeCanvas(host, M.res * 4, M.res * 4);
    const img = ctx.createImageData(M.res, M.res);
    for (let p = 0; p < M.res * M.res; p++) {
      const v = sd[p] / peak;
      /* white where nothing moved, the article's accent where the most did */
      img.data[p * 4] = Math.round(255 - 124 * v);
      img.data[p * 4 + 1] = Math.round(255 - 231 * v);
      img.data[p * 4 + 2] = Math.round(255 - 188 * v);
      img.data[p * 4 + 3] = 255;
    }
    const off = document.createElement('canvas');
    off.width = M.res;
    off.height = M.res;
    off.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, M.res * 4, M.res * 4);

    const ms = imgs.map((im) => measure(im, M.res));
    const spread = (k) => {
      const vals = ms.map((m) => m[k]).filter((v) => Number.isFinite(v));
      if (!vals.length) return 0;
      const mu = vals.reduce((s, v) => s + v, 0) / vals.length;
      if (k === 'hue') {
        return vals.reduce((s, v) => s + hueDist(v, mu), 0) / vals.length;
      }
      return Math.sqrt(vals.reduce((s, v) => s + (v - mu) ** 2, 0) / vals.length);
    };
    const N = data.noise;
    document.querySelector('#noise-readout').innerHTML =
      `<p>${n} images from one style vector, `
      + (state.on ? 'each with its own draw of the noise inputs'
        : 'with every noise input set to zero, so all six are the same picture')
      + `. Across these six the position moves by ${spread('cx').toFixed(3)} pixels, the size by `
      + `${spread('size').toFixed(3)} and the hue by ${spread('hue').toFixed(4)}. Across style `
      + `vectors, the same three move by ${N.across_w.cx.toFixed(2)}, ${N.across_w.size.toFixed(2)} `
      + `and ${N.across_w.hue.toFixed(3)}. The style decides what the sprite is; on this dataset `
      + `the noise decides essentially nothing, which is what a network should do when the data `
      + `it is copying has nothing left to chance.</p>`
      + `<p>The right hand panel is the per-pixel standard deviation over these realisations, `
      + `scaled so its largest value is full strength, which is the only reason there is anything `
      + `to see in it: its peak is <span class="bold">${peak.toFixed(4)}</span> of a unit of `
      + `brightness, and the sprites in this dataset carry no stochastic detail at all, so the `
      + `network learned to leave nothing to chance.</p>`;
  }

  const row = document.querySelector('#noise-buttons');
  const mk = (label, fn) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = label;
    b.addEventListener('click', fn);
    row.appendChild(b);
    return b;
  };
  mk('another sprite', () => {
    state.z = Float32Array.from(data.check.z[(state.seed + 1) % data.check.z.length]);
    state.seed += 13;
    render();
  });
  mk('new noise', () => { state.seed += 101; render(); });
  const toggle = mk('noise on', () => {
    state.on = !state.on;
    toggle.textContent = state.on ? 'noise on' : 'noise off';
    toggle.className = state.on ? 'atlas-btn ghost' : 'atlas-btn';
    render();
  });
  render();
  return { state, render };
}
