#!/usr/bin/env node
/**
 * Contrast under the glass, at the worst point of the ambient wash.
 *
 * The panel and the chrome are translucent, so their effective background is not
 * `--color-surface` — it is the surface plus whatever the wash contributes where
 * it is brightest. Every ink contrast measured for the flat palette is therefore
 * an upper bound, and the number that matters is this one.
 *
 * Run: node scripts/check_glass_contrast.mjs
 * Exits non-zero if any ink falls under 4.5:1 on any combination.
 */
const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4 }
const L = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
const ratio = (a, b) => { const [x, y] = [L(a), L(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05) }
let failed = false
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16))
const over = (fg, bg, a) => fg.map((c, i) => a * c + (1 - a) * bg[i])

// Worst case: the brightest point of the ambient wash, directly behind the panel.
for (const [mode, surface, inks, washes, panelAlpha, chromeAlpha, layerOpacity] of [
  ['dark', '#0a0b14', { ink: '#f4f2ec', 'ink-2': '#b9b7c6', 'ink-3': '#9a98a8' },
   { gold: ['#efc373', 0.62], coral: ['#e05c42', 0.52], blue: ['#3e8fdb', 0.46] }, 0.82, 0.72, 0.62],
  ['light', '#fbfaf6', { ink: '#14131a', 'ink-2': '#4a4756', 'ink-3': '#6b6878' },
   { gold: ['#8a5a12', 0.62], coral: ['#c9492f', 0.52], blue: ['#2a6fd6', 0.46] }, 0.78, 0.70, 0.30],
]) {
  const base = hex(surface)
  // In light mode the panel glass is white, not the surface colour.
  const glassTint = mode === 'light' ? [255, 255, 255] : base
  console.log(`\n${mode} — worst effective background under the glass, and the ink on it`)
  for (const [name, [c, mix]] of Object.entries(washes)) {
    const alpha = mix * layerOpacity
    const backdrop = over(hex(c), base, alpha)
    for (const [layer, a] of [['panel', panelAlpha], ['chrome', chromeAlpha]]) {
      const eff = over(glassTint, backdrop, a)
      const worst = Math.min(...Object.values(inks).map((k) => ratio(hex(k), eff)))
      const px = eff.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')
      if (worst < 4.5) failed = true
      console.log(`  ${name.padEnd(6)} under ${layer.padEnd(6)} #${px}  weakest ink ${worst.toFixed(2)}:1  ${worst >= 4.5 ? 'ok' : 'FAIL'}`)
    }
  }
}

if (failed) {
  console.error('\nAt least one ink falls below 4.5:1 under the glass.')
  process.exit(1)
}
console.log('\nAll inks clear 4.5:1 under the glass, in both schemes.')
