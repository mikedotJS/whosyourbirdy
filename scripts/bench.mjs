#!/usr/bin/env node
/**
 * Measure per-window inference time in a real browser.
 *
 * Reports median and p95 over every window of the fixture, plus the realtime
 * factor (how many seconds of audio are analysed per second of wall clock).
 * That last number is the one that decides whether P3's live microphone mode is
 * feasible at all: it has to stay comfortably above 1.
 *
 * Runs against dist/, single-threaded WASM with SIMD, which is what the site
 * actually ships.
 *
 * Usage: pnpm bench [--runs=3]
 */
import { spawnSync } from 'node:child_process'
import { createServer } from 'node:http'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

// This environment pre-installs Chromium at a build number that may not match the
// one our Playwright pins; launching it by path avoids a download we cannot do
// (and must not do: the browser is already here).
const CHROMIUM = process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium'
const launchOptions = existsSync(CHROMIUM) ? { executablePath: CHROMIUM } : {}

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const WORK = join(ROOT, '.parity')
const FIXTURE = join(ROOT, '.cache', 'birdnet', 'soundscape.wav')

const runs = Number(process.argv.find((a) => a.startsWith('--runs='))?.slice(7) ?? 3)

function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
}

function serveStatic() {
  const roots = [join(ROOT, 'dist'), WORK]
  const types = {
    '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
    '.wasm': 'application/wasm', '.html': 'text/html', '.css': 'text/css',
    '.onnx': 'application/octet-stream', '.txt': 'text/plain', '.wav': 'audio/wav',
  }
  const server = createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0])
    for (const root of roots) {
      const path = join(root, url)
      if (path.startsWith(root) && existsSync(path) && !path.endsWith('/')) {
        res.writeHead(200, { 'content-type': types[path.slice(path.lastIndexOf('.'))] ?? 'application/octet-stream' })
        res.end(readFileSync(path))
        return
      }
    }
    res.writeHead(404).end('not found')
  })
  return new Promise((resolve) => server.listen(0, () => resolve(server)))
}

async function main() {
  if (!existsSync(FIXTURE)) throw new Error('run `python3 scripts/fetch_artifacts.py` first')
  if (!existsSync(join(ROOT, 'public', 'models', 'manifest.json'))) {
    throw new Error('run `pnpm model:build` first')
  }

  const build = spawnSync('pnpm', ['build'], { cwd: ROOT, encoding: 'utf8' })
  if (build.status !== 0) throw new Error(`pnpm build failed:\n${build.stderr}`)

  mkdirSync(WORK, { recursive: true })
  writeFileSync(join(WORK, 'soundscape.wav'), readFileSync(FIXTURE))

  const server = await serveStatic()
  const port = server.address().port
  const browser = await chromium.launch(launchOptions)

  try {
    const page = await browser.newPage()
    await page.goto(`http://127.0.0.1:${port}/parity-harness.html`)

    const all = []
    let meta = null
    for (let run = 0; run < runs; run++) {
      const result = await page.evaluate(
        async ([url]) => {
          const out = await window.benchmark(url)
          return out
        },
        ['/soundscape.wav'],
        { timeout: 900_000 },
      )
      all.push(...result.timings)
      meta = result
      process.stdout.write(
        `  run ${run + 1}/${runs}: median ${result.median.toFixed(0)} ms/window over ${result.timings.length} windows\n`,
      )
    }

    const sorted = all.sort((a, b) => a - b)
    const median = quantile(sorted, 0.5)
    const p95 = quantile(sorted, 0.95)
    const realtime = 3000 / median

    const report = {
      platform: `${process.platform} ${process.arch}`,
      browser: `Chromium ${browser.version()}`,
      backend: 'onnxruntime-web wasm, SIMD, 1 thread',
      windows: sorted.length,
      medianMs: Number(median.toFixed(1)),
      p95Ms: Number(p95.toFixed(1)),
      minMs: Number(sorted[0].toFixed(1)),
      maxMs: Number(sorted[sorted.length - 1].toFixed(1)),
      realtimeFactor: Number(realtime.toFixed(1)),
      decodeMs: meta ? Number(meta.decodeMs.toFixed(0)) : null,
      modelLoadMs: meta ? Number(meta.modelLoadMs.toFixed(0)) : null,
    }

    console.log('\n' + JSON.stringify(report, null, 2))
    console.log(
      `\n  ${report.medianMs} ms per 3 s window -> ${report.realtimeFactor}x realtime ` +
        `(a 2-minute file takes about ${((sorted.length / runs) * median / 1000).toFixed(0)} s)`,
    )
    writeFileSync(join(ROOT, 'docs', 'bench-latest.json'), JSON.stringify(report, null, 2) + '\n')
    console.log('  written to docs/bench-latest.json')
  } finally {
    await browser.close()
    server.close()
  }
}

main().catch((error) => {
  console.error(error.message)
  process.exit(1)
})
