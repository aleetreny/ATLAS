/* The arithmetic of a stack of convolutions, which is all VGG's argument is.
 *
 * Nothing here needs data: the receptive field and the parameter count of a
 * stack follow from the kernel size and the number of layers, and writing
 * them out is the point. The measured accuracies live in depth.json.
 */

/* A stack of L convolutions of size k, all stride 1, sees a window of
 * 1 + L(k-1) pixels: each layer adds (k-1) on top of what the one below it
 * already saw. */
export const receptiveField = (k, layers) => 1 + layers * (k - 1);

/* Weights in one convolution layer with C channels in and C out. */
export const layerParams = (k, C) => k * k * C * C;

/* The single kernel that would see the same window as a stack. */
export const equivalentKernel = (k, layers) => receptiveField(k, layers);

/* How many 3x3 layers are needed to match a k x k window, and what that
 * costs: the trade VGG made in 2014. */
export function vggTrade(k, C) {
  const layers = (k - 1) / 2;
  const one = layerParams(k, C);
  const stacked = layers * layerParams(3, C);
  return { layers, one, stacked, saving: one - stacked, ratio: one / stacked };
}

/* The number of layers a signal passes through, which is what the depth
 * article is really counting: two convolutions per block, plus the stem and
 * the classifier. depth = 6n + 2 for n blocks per stage, three stages. */
export const blocksPerStage = (depth) => (depth - 2) / 6;
export const totalBlocks = (depth) => 3 * blocksPerStage(depth);

/* Nice, evenly spaced sample points for drawing a curve through measured
 * depths without implying we measured in between. */
export function series(ladder, family, field) {
  return ladder.map((r) => ({ depth: r.depth, v: r[family][field] }));
}
