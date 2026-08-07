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
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const FIXTURE = join(ROOT, '.cache', 'birdnet', 'soundscape.wav')
const CHROMIUM = process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium'

const shotsDir = process.argv.find((a) => a.startsWith('--shots='))?.slice(8) ?? join(ROOT, '.smoke')

/**
 * Golden values for the reference fixture, taken from the parity ground truth.
 * Exact counts and an exact top row are what make this test able to fail: a
 * dead slider, a shuffled list or a mis-mapped score all pass `rows > 0`.
 */
const GOLDEN = {
  detections025: 24,
  detections070: 2,
  detections005: 63,
  species025: 8,
  top: { common: 'Mésange à tête noire', scientific: 'Poecile atricapillus', count: '2×', score: 0.81 },
}

const ROWS = 'main > ul > li'

async function readRows(page) {
  return page.locator('main > ul > li').evaluateAll((items) =>
    items.map((li) => {
      const text = li.innerText.split('\n').map((t) => t.trim()).filter(Boolean)
      return {
        common: text[0],
        scientific: text[1],
        count: text.find((t) => /^\d+×$/.test(t)) ?? '',
        score: Number(text.find((t) => /^0\.\d\d$/.test(t))),
      }
    }),
  )
}

/** Read the detection count out of the summary line. */
async function countDetections(page) {
  const text = await page.locator('main').innerText()
  const match = text.match(/(\d+)\s+détection/)
  return match ? Number(match[1]) : -1
}

/** Count pixels close to the focus blue, in either theme's step. */
async function countFocusColour(page) {
  return page.evaluate(() => {
    const canvas = document.querySelector('canvas')
    const ctx = canvas.getContext('2d')
    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height)
    let n = 0
    for (let i = 0; i < data.length; i += 4) {
      const [r, g, b] = [data[i], data[i + 1], data[i + 2]]
      // Blue-dominant and not grey: the focus colour, at any alpha.
      if (b > 90 && b - r > 40 && b - g > 20) n++
    }
    return n
  })
}

function toSeconds(clock) {
  const [m, s] = clock.split(':').map(Number)
  return m * 60 + s
}

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

    // `canvas.count() === 1` passed with a blank canvas, a wrong FFT or an
    // inverted axis. Read the pixels instead.
    const canvasStats = await page.evaluate(() => {
      const canvas = document.querySelector('canvas')
      const ctx = canvas.getContext('2d')
      const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height)
      let min = 255, max = 0, sum = 0, n = 0
      // Sample the plot area only, skipping the axis gutter and the band lane.
      for (let y = 10; y < height * 0.8; y += 4) {
        for (let x = Math.floor(width * 0.1); x < width; x += 4) {
          const v = data[(y * width + x) * 4]
          if (v < min) min = v
          if (v > max) max = v
          sum += v
          n++
        }
      }
      return { min, max, mean: sum / n }
    })
    check(
      'the spectrogram has real content, not a blank canvas',
      canvasStats.max - canvasStats.min > 60,
      `luminance ${canvasStats.min}–${canvasStats.max}, mean ${canvasStats.mean.toFixed(0)}`,
    )

    // Golden values from the parity ground truth. `rows > 0` would pass with the
    // species, scores and counts all shuffled; these would not.
    const rows = await page.locator(ROWS).count()
    check('exactly the expected species at 0.25', rows === GOLDEN.species025, `${rows} species`)

    const summary = await page.locator('main').innerText()
    check(
      'the summary states the species and detection counts',
      summary.includes(`${GOLDEN.species025} espèces`) &&
        summary.includes(`${GOLDEN.detections025} détections`) &&
        summary.includes('2:00'),
      summary.split('\n').find((l) => /espèces/.test(l)) ?? '',
    )

    const parsed = await readRows(page)
    check(
      'top species matches the reference exactly',
      parsed[0].common === GOLDEN.top.common &&
        parsed[0].scientific === GOLDEN.top.scientific &&
        parsed[0].count === GOLDEN.top.count &&
        parsed[0].score === GOLDEN.top.score,
      JSON.stringify(parsed[0]),
    )

    const scores = parsed.map((r) => r.score)
    check(
      'species are sorted by descending best score',
      scores.every((v, i) => i === 0 || v <= scores[i - 1]),
      `${scores[0]} … ${scores[scores.length - 1]}`,
    )

    // Focusing a species reveals its occurrences, every one on the 3 s grid.
    await page.locator(`${ROWS} button`).first().click()
    await page.waitForTimeout(300)
    const chips = await page.locator(`${ROWS}`).first().locator('ul button').allInnerTexts()
    check('focusing a species reveals its occurrences', chips.length === 2, chips.join(' '))
    // innerText runs the chip's timecode and score together ("0:000.81"), so
    // pull the clock out by shape rather than by splitting on whitespace.
    const offGrid = chips.filter((c) => {
      const match = c.match(/^(\d+:\d\d)/)
      return !match || toSeconds(match[1]) % 3 !== 0
    })
    check('every occurrence sits on the 3 s analysis grid', offGrid.length === 0, offGrid.join(', '))

    // Focusing must actually change the picture, not just the list.
    const blueWhenFocused = await countFocusColour(page)
    await page.locator(`${ROWS} button`).first().click() // unpin
    // The pointer is still resting on the row after the click, and hover alone
    // keeps a species focused — by design. Move away before measuring, or this
    // compares the focused state with itself.
    await page.mouse.move(5, 5)
    await page.waitForTimeout(500)
    const blueWhenNot = await countFocusColour(page)
    check(
      'focusing a species repaints its bands in the picture',
      blueWhenFocused > blueWhenNot,
      `${blueWhenNot} → ${blueWhenFocused} focus-coloured pixels`,
    )

    // The timeline must scrub by dragging, not only by clicking.
    const box = await page.locator('canvas').boundingBox()
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.4)
    await page.mouse.down()
    for (let i = 1; i <= 8; i++) {
      await page.mouse.move(box.x + box.width * (0.2 + 0.06 * i), box.y + box.height * 0.4)
    }
    await page.mouse.up()
    await page.waitForTimeout(200)
    const scrubbed = Number(await page.locator('[role=slider]').getAttribute('aria-valuenow'))
    check('dragging the timeline scrubs', scrubbed > 60, `position ${scrubbed}s after drag to ~68%`)

    // A parked playhead must not keep the canvas animating forever. Count real
    // repaints by instrumenting the call every draw makes, rather than trusting
    // that rAF is idle.
    const draws = await page.evaluate(
      () =>
        new Promise((resolve) => {
          const proto = CanvasRenderingContext2D.prototype
          const original = proto.clearRect
          let n = 0
          proto.clearRect = function (...args) {
            n++
            return original.apply(this, args)
          }
          setTimeout(() => {
            proto.clearRect = original
            resolve(n)
          }, 1200)
        }),
    )
    check(
      'the canvas stops repainting once nothing is moving',
      draws < 5,
      `${draws} repaints in 1.2 s with a parked playhead`,
    )

    // Keyboard shortcuts. Driven from outside the spectrogram on purpose: the
    // canvas has its own arrow handler, so pressing keys while it holds focus
    // would exercise that path and never the global one. Placed after the
    // repaint count, because focusing a species starts a fade and would spend
    // that budget.
    await page.keyboard.press('/')
    const listFocus = await page.evaluate(() => {
      const el = document.activeElement
      return { tag: el?.tagName ?? '', inList: el?.closest('#species-list') !== null }
    })
    check(
      '`/` moves focus into the species list',
      listFocus.tag === 'BUTTON' && listFocus.inList,
      `${listFocus.tag}, in list: ${listFocus.inList}`,
    )

    // Arrows must move the *painted* playhead, not just the audio element.
    // Reading aria-valuenow catches a seek that writes to a ref nobody re-reads,
    // which is exactly what happened before `positionVersion` existed: the sound
    // moved, the picture did not, and the slider kept reporting the old second.
    const readPosition = async () =>
      Number(await page.locator('[role=slider]').getAttribute('aria-valuenow'))
    const before = await readPosition()
    await page.keyboard.press('ArrowLeft')
    await page.waitForTimeout(120)
    const oneWindowBack = await readPosition()
    await page.keyboard.press('Shift+ArrowLeft')
    await page.waitForTimeout(120)
    const tenBack = await readPosition()
    check(
      'arrows scrub by one window, Shift by ten seconds',
      before - oneWindowBack === 3 && oneWindowBack - tenBack === 10,
      `${before} → ${oneWindowBack} → ${tenBack}`,
    )

    // Space must still activate whatever holds focus — it is the native key for
    // a button, and a global shortcut that swallows it makes the list unusable
    // from the keyboard. Focus is on a species row here, so this press has to
    // reach the row and toggle it, not start playback.
    await page.keyboard.press('Space')
    await page.waitForTimeout(200)
    // Then move focus away. Hover-focus is released by the blur, so the
    // occurrences only stay open if that press really reached the row and
    // pinned it. If the shortcut had swallowed it, they would close here.
    await page.locator('[role=slider]').focus()
    await page.waitForTimeout(200)
    const stillOpen = await page.locator(`${ROWS} ul button`).count()
    check(
      'space still activates the focused control instead of being swallowed',
      stillOpen === 2,
      `${stillOpen} occurrence chips after moving focus off the row`,
    )

    // With focus on the timeline instead, the same key is the play shortcut.
    await page.keyboard.press('Space')
    await page.waitForTimeout(700)
    const playingChips = await page.locator('[aria-pressed=true]').count()
    check('space starts playback when no control has focus', playingChips === 1, `${playingChips} chip playing`)
    await page.keyboard.press('Space')
    await page.waitForTimeout(300)
    check(
      'space stops it again',
      (await page.locator('[aria-pressed=true]').count()) === 0,
    )

    // Escape releases the selection, which also puts the page back in the
    // nothing-focused state the playback checks below start from.
    await page.keyboard.press('Escape')
    await page.waitForTimeout(250)
    check(
      'escape releases the selected species',
      (await page.locator(`${ROWS} ul button`).count()) === 0,
    )

    // The slider must filter in memory, not re-run the model.
    const t0 = Date.now()
    await page.locator('input[type=range]').fill('0.7')
    await page.waitForTimeout(120)
    const high = await countDetections(page)
    const elapsed = Date.now() - t0
    // Strict: a `>=` comparison is satisfied by an onChange that does nothing.
    check(
      'raising the threshold hides detections, instantly',
      high === GOLDEN.detections070 && elapsed < 1000,
      `${GOLDEN.detections025} → ${high} detections in ${elapsed} ms`,
    )

    await page.locator('input[type=range]').fill('0.05')
    await page.waitForTimeout(150)
    const low = await countDetections(page)
    check('lowering the threshold reveals more', low === GOLDEN.detections005, `${low} at 0.05`)

    await page.locator('input[type=range]').fill('0.25')
    await page.waitForTimeout(120)

    // Assert on what a user perceives — the row reporting itself as playing —
    // rather than on the <audio> element, which is created via `new Audio()` and
    // never attached to the document.
    // Re-pin a species: the focus test above deliberately left nothing focused,
    // so there are no occurrence chips to click until we open one again.
    await page.locator(`${ROWS} button`).first().click()
    await page.waitForTimeout(300)

    const firstRow = page.locator(`${ROWS} ul button`).first()
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

    // ---- regression: switching files mid-analysis -------------------------
    // A cancelled run used to finish anyway and write its results under the new
    // file's name — 24 bird detections displayed for a 29-byte text file. The
    // worker now yields so `cancel` is actually delivered, and the hook drops
    // state writes from superseded runs.
    const decoy = join(shotsDir, 'not-audio.wav')
    writeFileSync(decoy, 'this is not audio, it is a text file pretending to be one')

    await page.reload()
    await page.setInputFiles('input[type=file]', FIXTURE)
    await page.waitForSelector('text=/fenêtre \\d+ \\/ 40/', { timeout: 600_000 })
    await page.click('text=Autre fichier')
    await page.setInputFiles('input[type=file]', decoy)

    // Long enough that the abandoned run would have finished if it were still live.
    await page.waitForTimeout(20_000)
    const after = await page.locator('main').innerText()
    await page.screenshot({ path: join(shotsDir, '5-switched.png'), fullPage: true })

    check(
      'a superseded analysis cannot report results under the new file',
      !/espèces? ·/.test(after) && (await page.locator(ROWS).count()) === 0,
      after.split('\n').slice(0, 3).join(' | '),
    )
    check(
      'the undecodable file reports its own error',
      /pas pu être décodé/.test(after),
      after.split('\n').find((l) => /décod/.test(l)) ?? after.slice(0, 80),
    )

    // ---- regression: reset mid-analysis leaves no ghost list ---------------
    await page.reload()
    await page.setInputFiles('input[type=file]', FIXTURE)
    await page.waitForSelector('text=/fenêtre \\d+ \\/ 40/', { timeout: 600_000 })
    await page.click('text=Autre fichier')
    await page.waitForTimeout(20_000)
    const afterReset = await page.locator('main').innerText()
    check(
      'reset mid-analysis leaves the drop zone and nothing else',
      /Déposez un enregistrement/.test(afterReset) && (await page.locator(ROWS).count()) === 0,
      afterReset.split('\n').slice(0, 2).join(' | '),
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
