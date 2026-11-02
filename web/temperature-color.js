/* Shared, fixed physical scale: 0 K is near white; -14 K is dark blue.
   The mapping changes display color only, never the underlying field values. */
window.temperatureColor = function temperatureColor(value) {
  const magnitude = Math.max(0, Math.min(1, -value / 14));
  const t = Math.pow(magnitude, 0.7);
  const base = [250, 250, 247], cold = [25, 76, 135];
  return base.map((v, i) => Math.round(v + (cold[i] - v) * t));
};
