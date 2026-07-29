/* The trained network, run again here.
 *
 * Two inputs, one hidden layer of rectified units, four outputs. Written out
 * because it is four lines and because a page that draws what a network
 * believes should be running the network rather than reading a picture of it:
 * the guard checks these outputs against the file for all forty eight squares.
 */
export function forward(weights, x) {
  const { W1, b1, W2, b2 } = weights;
  const hidden = b1.map((bias, j) => {
    let s = bias;
    for (let i = 0; i < x.length; i++) s += x[i] * W1[i][j];
    return Math.max(s, 0);
  });
  return b2.map((bias, k) => {
    let s = bias;
    for (let j = 0; j < hidden.length; j++) s += hidden[j] * W2[j][k];
    return s;
  });
}

/* The only thing the network is ever shown: where it is, as two numbers between
 * zero and one. No entry per square, which is the whole point and the whole
 * risk. */
export const observe = (rc, rows, cols) => [rc[1] / (cols - 1), rc[0] / (rows - 1)];
