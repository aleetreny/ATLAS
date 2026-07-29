/* Three samplers and one decoder, run in the page.
 *
 * They all have the same shape on purpose, because the article is a comparison
 * and a comparison between two different architectures measures the
 * architecture. What differs is only what they were trained to say:
 *
 *   flow    the velocity along a straight line from noise to data
 *   reflow  the same thing, retrained on pairs the flow itself produced
 *   student the ENDPOINT, from anywhere on a teacher trajectory, which is why
 *           it needs no chain at all and is evaluated once
 *
 * The first two are walked with Euler from t=0 to t=1 in however many steps the
 * reader asks for, and the whole question of the article is what happens to the
 * answer as that number falls.
 */
import { byName, decodeWeights, dense, lcg } from '../../assets/js/halfweights.js';

export function makeFlow(data) {
  const N = data.net;
  const W = decodeWeights(N.b64, N.count);
  const L = byName(N.spec);
  const K = N.k;

  const silu = (v) => v.map((t) => t / (1 + Math.exp(-t)));
  const tanh = (v) => v.map(Math.tanh);
  const sigmoid = (v) => v.map((t) => 1 / (1 + Math.exp(-t)));

  /* read off the spec rather than assumed, so a generator that changes the
     depth of these networks does not silently break the page */
  const pick = (re) => N.spec.filter((s) => re.test(s.name)).map((s) => s.name);
  const NETS = { flow: pick(/^f\d+$/), reflow: pick(/^r\d+$/), student: pick(/^s\d+$/) };
  const decLayers = pick(/^dec\d+$/);

  /* the same embedding the generator used, and t arrives in [0, 1] for all
     three networks, so no rescaling hides between the arms */
  function embed(t) {
    const out = new Float64Array(32);
    for (let i = 0; i < 16; i++) {
      const f = Math.exp((Math.log(200) * i) / 15);
      const a = t * f * Math.PI;
      out[i] = Math.sin(a);
      out[i + 16] = Math.cos(a);
    }
    return out;
  }

  /** What network `tag` says at point `z` and time `t`. */
  function field(tag, z, t) {
    const e = embed(t);
    const inp = new Float64Array(K + 32);
    for (let i = 0; i < K; i++) inp[i] = z[i];
    for (let i = 0; i < 32; i++) inp[K + i] = e[i];
    let h = inp;
    const layers = NETS[tag];
    layers.forEach((name, i) => {
      h = dense(W, L[name], h);
      if (i < layers.length - 1) h = silu(h);
    });
    return h;
  }

  /**
   * Euler from noise to data. With one step this is z0 + v(z0, 0): the straight
   * line guess, taken from the very start of the trajectory, which is exactly
   * the guess a curved path punishes.
   */
  function walk(tag, z0, steps, keepPath = false) {
    let z = Float64Array.from(z0);
    const path = keepPath ? [Float64Array.from(z)] : null;
    const dt = 1 / steps;
    for (let i = 0; i < steps; i++) {
      const v = field(tag, z, i * dt);
      const nz = new Float64Array(K);
      for (let d = 0; d < K; d++) nz[d] = z[d] + dt * v[d];
      z = nz;
      if (keepPath) path.push(Float64Array.from(z));
    }
    return { z, path };
  }

  /** The student does not walk. It is asked once, at t=0, and answers with the
   *  place the teacher would have arrived at. */
  function jump(z0) {
    return field('student', z0, 0);
  }

  /** Whichever of the three the reader picked, at whatever budget. */
  function run(tag, z0, steps, keepPath = false) {
    if (tag === 'student') return { z: jump(z0), path: keepPath ? [Float64Array.from(z0)] : null };
    return walk(tag, z0, steps, keepPath);
  }

  function decode(z) {
    const un = new Float64Array(K);
    for (let i = 0; i < K; i++) un[i] = z[i] * N.sd[i] + N.mu[i];
    let h = un;
    decLayers.forEach((name, i) => {
      h = dense(W, L[name], h);
      h = i < decLayers.length - 1 ? tanh(h) : sigmoid(h);
    });
    return h;
  }

  /* the eight numbers drawn on a flat page, in the frame the generator chose,
     so a stored path and a live one land on top of each other */
  const P = data.proj;
  function project(z) {
    let a = 0;
    let b = 0;
    for (let i = 0; i < K; i++) {
      a += z[i] * P[i][0];
      b += z[i] * P[i][1];
    }
    return [a, b];
  }

  function noise(n, seed) {
    const rnd = lcg(seed);
    const out = [];
    for (let i = 0; i < n; i++) {
      const z = new Float64Array(K);
      for (let j = 0; j < K; j++) z[j] = rnd.normal();
      out.push(z);
    }
    return out;
  }

  return { field, walk, jump, run, decode, project, noise, K, names: NETS };
}
