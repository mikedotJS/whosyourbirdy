#!/usr/bin/env node
/**
 * End-to-end smoke test of the actual UI, in a real browser.
 *
 * The parity harness proves the numbers are right; this proves a user can get to
 * them. It drives the production build in Chromium: drop a file, watch the
 * analysis run, read the detections, drag the threshold, click a row to play its
 * segment. Anything that throws, 404s, or logs an error fails the run.
 *
 * Note the MIME map below. ORT loads its runtime as a dynamic ES module import
 * from `/ort/*.mjs`, and a browser refuses a module served as
 * application/octet-stream — the first version of this script omitted `.mjs` and
 * the app died with "no available backend found". Any static host serving this
 * site must send a JavaScript content type for .mjs.
 *
 * Usage: pnpm smoke [--headed] [--shots=DIR]
 */
import { spawnSync } from 'node:child_process'
import { createServer } from 'node:http'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const FIXTURE = join(ROOT, '.cache', 'birdnet', 'soundscape.wav')
const CHROMIUM = process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium'

const shotsDir = process.argv.find((a) => a.startsWith('--shots='))?.slice(8) ?? join(ROOT, '.smoke')

const MIME = {
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.html': 'text/html',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.txt': 'text/plain',
  '.wav': 'audio/wav',
}

function serve() {
  const root = join(ROOT, 'dist')
  const server = createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0])
    if (url === '/favicon.ico') return void res.writeHead(204).end()
    const path = join(root, url === '/' ? '/index.html' : url)
    if (path.startsWith(root) && existsSync(path) && !path.endsWith('/')) {
      res.writeHead(200, { 'content-type': MIME[path.slice(path.lastIndexOf('.'))] ?? 'application/octet-stream' })
      return void res.end(readFileSync(path))
    }
    res.writeHead(404).end('not found')
  })
  return new Promise((resolve) => server.listen(0, () => resolve(server)))
}

const checks = []
function check(name, ok, detail = '') {
  checks.push({ name, ok, detail })
  console.log(`  ${ok ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m'}  ${name}${detail ? ` — ${detail}` : ''}`)
}

async function main() {
  if (!existsSync(FIXTURE)) throw new Error('run `python3 scripts/fetch_artifacts.py` first')
  if (!existsSync(join(ROOT, 'public', 'models', 'manifest.json'))) {
    throw new Error('run `pnpm model:build` first')
  }

  const build = spawnSync('pnpm', ['build'], { cwd: ROOT, encoding: 'utf8' })
  if (build.status !== 0) throw new Error(`pnpm build failed:\n${build.stdout}\n${build.stderr}`)

  mkdirSync(shotsDir, { recursive: true })
  const server = await serve()
  const port = server.address().port
  const browser = await chromium.launch({
    executablePath: existsSync(CHROMIUM) ? CHROMIUM : undefined,
    headless: !process.argv.includes('--headed'),
  })

  const problems = []
  try {
    const page = await browser.newPage({ viewport: { width: 900, height: 1100 } })
    page.on('console', (m) => m.type() === 'error' && problems.push(m.text()))
    page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`))
    page.on('response', (r) => r.status() >= 400 && problems.push(`HTTP ${r.status()} ${r.url()}`))

    console.log('\n\x1b[1mUI smoke test\x1b[0m\n')
    await page.goto(`http://127.0.0.1:${port}/`)
    await page.screenshot({ path: join(shotsDir, '1-idle.png') })
    check('page renders', (await page.locator('h1').textContent()) === 'whosyourbirdy')
    check(
      'attribution visible without interaction',
      (await page.locator('footer').innerText()).includes('Powered by BirdNET'),
    )

    await page.setInputFiles('input[type=file]', FIXTURE)
    await page.waitForTimeout(2500)
    await page.screenshot({ path: join(shotsDir, '2-progress.png') })
    const progressText = await page.locator('main').innerText()
    check(
      'progress is reported while analysing',
      /Téléchargement|Analyse|Décodage|Préparation/.test(progressText),
      progressText.split('\n').find((l) => /Téléchargement|Analyse|Décodage/.test(l)) ?? '',
    )

    await page.waitForSelector('input[type=range]', { timeout: 600_000 })
    await page.waitForTimeout(400)
    await page.screenshot({ path: join(shotsDir, '3-results.png'), fullPage: true })

    const rows = await page.locator('ul > li').count()
    check('detections listed at the default threshold', rows > 0, `${rows} rows`)
    const first = await page.locator('ul > li').first().innerText()
    check(
      'each row carries species, timecode and score',
      /\d:\d\d–\d:\d\d/.test(first) && /0\.\d\d/.test(first),
      first.replace(/\n/g, ' / '),
    )

    // The slider must filter in memory, not re-run the model.
    const t0 = Date.now()
    await page.locator('input[type=range]').fill('0.7')
    await page.waitForTimeout(120)
    const high = await page.locator('ul > li').count()
    const elapsed = Date.now() - t0
    check('threshold filters instantly', elapsed < 1000 && high <= rows, `${rows} → ${high} rows in ${elapsed} ms`)

    await page.locator('input[type=range]').fill('0.05')
    await page.waitForTimeout(120)
    const low = await page.locator('ul > li').count()
    check('lowering the threshold reveals more', low >= rows, `${low} rows at 0.05`)

    await page.locator('input[type=range]').fill('0.25')
    await page.waitForTimeout(120)

    // Assert on what a user perceives — the row reporting itself as playing —
    // rather than on the <audio> element, which is created via `new Audio()` and
    // never attached to the document.
    const firstRow = page.locator('ul > li button').first()
    await firstRow.click()
    await page.waitForTimeout(700)
    check(
      'clicking a detection starts playback',
      (await firstRow.getAttribute('aria-pressed')) === 'true',
    )
    await page.screenshot({ path: join(shotsDir, '4-playing.png') })

    // The segment must stop at the end of its 3 s window rather than running on.
    await page.waitForTimeout(3000)
    check(
      'playback stops at the end of the 3 s window',
      (await firstRow.getAttribute('aria-pressed')) === 'false',
    )

    check('no console errors, page errors or 404s', problems.length === 0, problems.slice(0, 3).join(' | '))
  } finally {
    await browser.close()
    server.close()
  }

  const failed = checks.filter((c) => !c.ok)
  console.log(
    failed.length
      ? `\n\x1b[31m  ${failed.length}/${checks.length} checks failed\x1b[0m\n`
      : `\n\x1b[32m  all ${checks.length} checks passed\x1b[0m  (screenshots in ${shotsDir.replace(ROOT + '/', '')})\n`,
  )
  process.exit(failed.length ? 1 : 0)
}

main().catch((error) => {
  console.error(`\n\x1b[31m  ${error.message}\x1b[0m\n`)
  process.exit(1)
})
