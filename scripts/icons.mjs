#!/usr/bin/env node
/**
 * Render the PWA icons from the mark, with no new dependency.
 *
 * Playwright is already here for the smoke test, and a browser is a perfectly
 * good SVG rasteriser — better than most, in fact, since it is the same renderer
 * that draws the favicon. Adding `sharp` or `resvg` to turn one 300-byte vector
 * into three PNGs would be the larger cost.
 *
 * Two shapes are produced, and they are not the same picture scaled:
 *
 *  - `icon-192` / `icon-512`: the mark as it appears everywhere else, rounded
 *    corners included.
 *  - `icon-maskable-512`: full-bleed background with the bars shrunk into the
 *    central 60%. Android crops a maskable icon to whatever shape the launcher
 *    uses — circle, squircle, teardrop — and anything outside the safe zone can
 *    be cut. Shipping the rounded version as maskable is the classic mistake: it
 *    gets rounded a second time and loses its corners.
 *
 * Usage: node scripts/icons.mjs
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const OUT = join(ROOT, 'public', 'icons')
const CHROMIUM = process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium'

const SURFACE = '#0a0b14'
const GOLD = '#efc373'

/** The three bars, in a 32×32 frame. Shared by both shapes. */
const BARS = `<g fill="${GOLD}">
  <rect x="7"  y="17" width="4" height="8"  rx="2"/>
  <rect x="14" y="10" width="4" height="15" rx="2"/>
  <rect x="21" y="14" width="4" height="11" rx="2"/>
</g>`

const rounded = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="7" fill="${SURFACE}"/>
  ${BARS}
</svg>`

// 0.6 scale about the centre: the mark occupies the inner 60%, comfortably
// inside the 80% circle a maskable icon is guaranteed to keep.
const maskable = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" fill="${SURFACE}"/>
  <g transform="translate(16 16) scale(0.6) translate(-16 -16)">${BARS}</g>
</svg>`

const TARGETS = [
  { name: 'icon-192.png', size: 192, svg: rounded },
  { name: 'icon-512.png', size: 512, svg: rounded },
  { name: 'icon-maskable-512.png', size: 512, svg: maskable },
  // iOS ignores the manifest icons for "add to home screen" and reads
  // <link rel="apple-touch-icon">, which must be opaque and square — it applies
  // its own rounding on top.
  { name: 'apple-touch-icon.png', size: 180, svg: maskable },
]

const browser = await chromium.launch({ executablePath: CHROMIUM })
mkdirSync(OUT, { recursive: true })

for (const { name, size, svg } of TARGETS) {
  const page = await browser.newPage({
    viewport: { width: size, height: size },
    deviceScaleFactor: 1,
  })
  await page.setContent(
    `<style>html,body{margin:0;padding:0;background:transparent}svg{display:block;width:${size}px;height:${size}px}</style>${svg}`,
  )
  writeFileSync(join(OUT, name), await page.screenshot({ omitBackground: true }))
  await page.close()
  console.log(`  ${name}  ${size}×${size}`)
}

await browser.close()
console.log(`\nWrote ${TARGETS.length} icons to public/icons/`)
