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

/**
 * The geo filter's expected effect on the reference fixture.
 *
 * `soundscape.wav` is a North American recording, and BirdNET reports a
 * Chestnut-winged Cuckoo (*Clamator coromandus*) in it at 0.32 — an Asian
 * species that cannot possibly be there. Filtering at New York in week 20 must
 * remove exactly that kind of thing and leave the chickadee, which belongs.
 */
const GEO = {
  newYork: [40.71, -74.01],
  week: 20,
  removed: 'Coucou à collier',
  kept: 'Mésange à tête noire',
  masked: 3,
}

const ROWS = '#species-list > li'
/** Occurrences live in the bottom sheet, which is only open with a selection. */
const CHIPS = 'dialog.sheet[open] ul button'
/** Header, scrolling content and action bar. The status line is in the bar. */
const PANEL = '.app-panel'

async function readRows(page) {
  return page.locator(ROWS).evaluateAll((items) =>
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

/** Read the detection count out of the summary line, which lives in the bar. */
async function countDetections(page) {
  const text = await page.locator(PANEL).innerText()
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
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webmanifest': 'application/manifest+json',
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
    // A fake capture device fed from the reference fixture, so live listening
    // can be driven end to end. `--use-fake-ui-for-media-stream` accepts the
    // permission prompt; nothing else in the suite touches getUserMedia.
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${FIXTURE}`,
    ],
  })

  const problems = []
  try {
    const page = await browser.newPage({ viewport: { width: 900, height: 1100 } })
    page.on('console', (m) => m.type() === 'error' && problems.push(m.text()))
    page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`))
    page.on('response', (r) => r.status() >= 400 && problems.push(`HTTP ${r.status()} ${r.url()}`))

    console.log('\n\x1b[1mUI smoke test\x1b[0m\n')
    // 127.0.0.1 is a secure context in Chromium, which is what lets the service
    // worker register over plain HTTP.
    var base = `http://127.0.0.1:${port}`
    await page.goto(`${base}/`)
    await page.screenshot({ path: join(shotsDir, '1-idle.png') })
    check('page renders', (await page.locator('h1').textContent()) === 'whosyourbirdy')
    check(
      'attribution visible without interaction',
      (await page.locator('footer').innerText()).includes('Powered by BirdNET'),
    )

    await page.setInputFiles('input[type=file]', FIXTURE)
    await page.waitForTimeout(2500)
    await page.screenshot({ path: join(shotsDir, '2-progress.png') })
    const progressText = await page.locator(PANEL).innerText()
    check(
      'progress is reported while analysing',
      /Téléchargement|Analyse|Décodage|Préparation/.test(progressText),
      progressText.split('\n').find((l) => /Téléchargement|Analyse|Décodage/.test(l)) ?? '',
    )

    await page.waitForSelector('#threshold', { timeout: 600_000 })
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

    const summary = await page.locator(PANEL).innerText()
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

    // Selecting a species opens its occurrences in a modal sheet, every one on
    // the 3 s grid.
    await page.locator(`${ROWS} button`).first().click()
    await page.waitForTimeout(400)
    check(
      'selecting a species opens a modal sheet',
      (await page.locator('dialog.sheet[open]').count()) === 1,
    )
    const chips = await page.locator(CHIPS).allInnerTexts()
    check('the sheet lists the species occurrences', chips.length === 2, chips.join(' '))
    // innerText runs the chip's timecode and score together ("0:000.81"), so
    // pull the clock out by shape rather than by splitting on whitespace.
    const offGrid = chips.filter((c) => {
      const match = c.match(/^(\d+:\d\d)/)
      return !match || toSeconds(match[1]) % 3 !== 0
    })
    check('every occurrence sits on the 3 s analysis grid', offGrid.length === 0, offGrid.join(', '))

    // Focusing must actually change the picture, not just the list.
    const blueWhenFocused = await countFocusColour(page)

    // Escape closes the sheet, and `<dialog>` hands focus back to whatever
    // opened it. That restoration is the reason this is a real dialog and not a
    // positioned div, so it is worth asserting rather than assuming.
    await page.keyboard.press('Escape')
    await page.waitForTimeout(300)
    const returned = await page.evaluate(() => {
      const el = document.activeElement
      return el?.closest('#species-list > li') === document.querySelector('#species-list > li')
    })
    check('escape closes the sheet and returns focus to the row that opened it', returned)
    // A closed `<dialog>` is hidden by the browser's own `display: none`, which
    // a `display: flex` on the element outranks — that left an empty strip of
    // sheet parked over the action bar, invisible to every other assertion here.
    // Every sheet, not just the one we opened: there is more than one now, and
    // the bug this catches — a `display` rule outranking the browser's own
    // `display:none` for a closed dialog — would apply to all of them.
    const ghosts = await page.evaluate(
      () =>
        [...document.querySelectorAll('dialog.sheet')].filter(
          (d) => !d.open && getComputedStyle(d).display !== 'none',
        ).length,
    )
    check('a closed sheet takes no space', ghosts === 0, `${ghosts} closed sheets still displayed`)

    // Focus alone keeps a species highlighted — by design, for keyboard users —
    // so blur before measuring, or this compares the focused state with itself.
    await page.evaluate(() => document.activeElement?.blur())
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
    await page.waitForTimeout(400)
    check(
      'space still activates the focused control instead of being swallowed',
      (await page.locator('dialog.sheet[open]').count()) === 1,
      'the focused row opened its sheet',
    )

    // Back out, and take focus off the list so the next press has no control to
    // activate — that is when space becomes the play shortcut.
    await page.keyboard.press('Escape')
    await page.waitForTimeout(300)
    await page.locator('[role=slider]').focus()
    await page.keyboard.press('Space')
    await page.waitForTimeout(800)
    check(
      'space starts playback when no control has focus',
      (await page.locator('#species-list [data-playing=true]').count()) === 1,
      'one species reports itself as sounding',
    )
    await page.keyboard.press('Space')
    await page.waitForTimeout(300)
    check(
      'space stops it again',
      (await page.locator('#species-list [data-playing=true]').count()) === 0,
    )

    // The slider must filter in memory, not re-run the model.
    const t0 = Date.now()
    await page.locator('#threshold').fill('0.7')
    await page.waitForTimeout(120)
    const high = await countDetections(page)
    const elapsed = Date.now() - t0
    // Strict: a `>=` comparison is satisfied by an onChange that does nothing.
    check(
      'raising the threshold hides detections, instantly',
      high === GOLDEN.detections070 && elapsed < 1000,
      `${GOLDEN.detections025} → ${high} detections in ${elapsed} ms`,
    )

    await page.locator('#threshold').fill('0.05')
    await page.waitForTimeout(150)
    const low = await countDetections(page)
    check('lowering the threshold reveals more', low === GOLDEN.detections005, `${low} at 0.05`)

    await page.locator('#threshold').fill('0.25')
    await page.waitForTimeout(120)

    // Assert on what a user perceives — the chip reporting itself as playing —
    // rather than on the <audio> element, which is created via `new Audio()` and
    // never attached to the document. Re-open a species: the checks above closed
    // the sheet, so there are no occurrence chips to click until one is open.
    await page.locator(`${ROWS} button`).first().click()
    await page.waitForTimeout(400)

    const firstRow = page.locator(CHIPS).first()
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

    // ---- the shell, at phone size -----------------------------------------
    // Resizing the existing page rather than opening a mobile context: what is
    // under test here is layout, and a second context would pay for the 52 MB
    // model and a second full analysis to reach the same screen.
    await page.keyboard.press('Escape')
    await page.waitForTimeout(300)
    await page.setViewportSize({ width: 390, height: 844 }) // iPhone 14
    await page.waitForTimeout(600)
    await page.screenshot({ path: join(shotsDir, '6-phone.png') })

    const overflow = await page.evaluate(() => {
      const panel = document.querySelector('.app-panel')
      return {
        doc: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        panel: panel.scrollWidth - panel.clientWidth,
      }
    })
    check(
      'nothing overflows horizontally at 390 px',
      overflow.doc <= 0 && overflow.panel <= 0,
      `document +${overflow.doc}px, panel +${overflow.panel}px`,
    )

    // WCAG 2.5.8 exempts links inline in a block of text, which is what the
    // attribution is, so the audit covers controls: buttons and the slider.
    const small = await page.evaluate(() => {
      const out = []
      for (const el of document.querySelectorAll('button, input[type=range], select')) {
        const box = el.getBoundingClientRect()
        if (box.width === 0 && box.height === 0) continue // not rendered
        if (box.height >= 44 && box.width >= 44) continue
        const name = (el.getAttribute('aria-label') ?? el.textContent ?? '').trim().slice(0, 22)
        out.push(`${name || el.tagName.toLowerCase()} ${Math.round(box.width)}×${Math.round(box.height)}`)
      }
      return out
    })
    check('every tap target clears 44 px', small.length === 0, small.slice(0, 4).join(' | '))

    // The header and the action bar are the app's fixed furniture. If either
    // moves when the content scrolls, this is a page again.
    const barBefore = await page.locator('.app-bar').boundingBox()
    const headerBefore = await page.locator('.app-header').boundingBox()
    const scrolledBy = await page.locator('.app-scroll').evaluate((el) => {
      el.scrollTo(0, el.scrollHeight)
      return el.scrollTop
    })
    await page.waitForTimeout(300)
    const barAfter = await page.locator('.app-bar').boundingBox()
    const headerAfter = await page.locator('.app-header').boundingBox()
    check(
      'the header and action bar stay put while the content scrolls',
      // 2px, not 0: the header is sticky and sits one pixel below the scroll
      // sentinel at rest, so it gives that pixel back when it pins. A bar that
      // actually scrolled away would move by hundreds.
      scrolledBy > 50 &&
        Math.abs(barBefore.y - barAfter.y) <= 2 &&
        Math.abs(headerBefore.y - headerAfter.y) <= 2,
      `content scrolled ${Math.round(scrolledBy)}px, bar moved ${Math.abs(barBefore.y - barAfter.y).toFixed(1)}px`,
    )
    await page.screenshot({ path: join(shotsDir, '7-phone-scrolled.png') })
    await page.setViewportSize({ width: 900, height: 1100 })

    // ---- the geo-temporal filter -------------------------------------------
    // The fixture is a North American soundscape, and BirdNET finds a Chestnut-
    // winged Cuckoo in it at 0.32 — an Asian species that cannot be there. That
    // false positive is the reason this feature exists, so it is what the test
    // asserts on, rather than "some number went down".
    await page.click('text=Lieu et saison')
    await page.waitForSelector('dialog.sheet[open] #geo-latitude')

    // The week is pinned rather than left at today's, so the expected species
    // set does not depend on the day the suite runs.
    await page.fill('#geo-latitude', String(GEO.newYork[0]))
    await page.fill('#geo-longitude', String(GEO.newYork[1]))
    await page.locator('#geo-week').fill(String(GEO.week))
    await page.locator('dialog.sheet[open] input[type=checkbox]').first().check()

    // The switch lives in the sheet's header, which is also its drag handle.
    // Capturing the pointer there once swallowed this click entirely.
    check(
      'the switch in the sheet header is clickable, not eaten by the drag handle',
      await page.locator('dialog.sheet[open] input[type=checkbox]').first().isChecked(),
    )

    // 29 MB on first use, and it is not precached — give it room.
    await page.waitForFunction(
      () => !/calcul en cours|Téléchargement du modèle géographique/.test(document.body.innerText),
      undefined,
      { timeout: 300_000 },
    )
    await page.waitForTimeout(600)
    await page.screenshot({ path: join(shotsDir, '9-geo.png') })
    await page.keyboard.press('Escape')
    await page.waitForTimeout(400)

    const remaining = (await page.locator(ROWS).allInnerTexts()).map((t) => t.split('\n')[0].trim())
    check(
      'the geo filter removes the exotic false positive',
      !remaining.includes(GEO.removed) && remaining.includes(GEO.kept),
      `${remaining.length} species left; ${GEO.removed} gone: ${!remaining.includes(GEO.removed)}`,
    )

    // Removed is not the same as hidden: the species must still be reachable,
    // with the score that removed it. A filter that makes a detection vanish
    // without saying so is what this whole disclosure exists to prevent.
    const maskedSummary = await page.locator('details summary').innerText()
    await page.locator('details summary').click()
    await page.waitForTimeout(200)
    const maskedRows = await page.locator('details ul li').allInnerTexts()
    check(
      'the masked species stay reachable, with the geo score that removed them',
      maskedRows.length === GEO.masked &&
        maskedRows.some((r) => r.includes(GEO.removed) && /lieu 0\.\d{3}/.test(r)),
      `${maskedSummary.trim()} — ${maskedRows.length} listed`,
    )

    // The week defaults to BirdNET's own convention: four per month, 1..48.
    // An ISO week would be off by up to a month, worst exactly during migration.
    await page.click('text=Lieu et saison')
    await page.waitForSelector('dialog.sheet[open] #geo-week')
    await page.locator('dialog.sheet[open] input[type=checkbox]').first().uncheck()
    await page.keyboard.press('Escape')
    await page.waitForTimeout(400)
    check(
      'disabling the filter restores every species',
      (await page.locator(ROWS).count()) === GOLDEN.species025,
      `${await page.locator(ROWS).count()} species`,
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
    await page.setInputFiles('input[type=file]', decoy)

    // Long enough that the abandoned run would have finished if it were still live.
    await page.waitForTimeout(20_000)
    const after = await page.locator(PANEL).innerText()
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
    await page.click('[aria-label="Recommencer avec un autre fichier"]')
    await page.waitForTimeout(20_000)
    const afterReset = await page.locator(PANEL).innerText()
    check(
      'reset mid-analysis leaves the drop zone and nothing else',
      /Choisir un enregistrement/.test(afterReset) && (await page.locator(ROWS).count()) === 0,
      afterReset.split('\n').slice(0, 2).join(' | '),
    )

    // ---- live listening ----------------------------------------------------
    // Driven by a fake device playing the same fixture, so the species that come
    // out are checkable rather than "some rows appeared".
    await page.click('text=Micro')
    await page.waitForTimeout(300)

    // Instrument the real thing that must happen on stop. Counting calls to
    // `MediaStreamTrack.stop` proves the device is released, which closing the
    // AudioContext alone would not do — the recording indicator would stay lit.
    await page.evaluate(() => {
      const proto = MediaStreamTrack.prototype
      const original = proto.stop
      window.__trackStops = 0
      proto.stop = function (...args) {
        window.__trackStops++
        return original.apply(this, args)
      }
    })

    await page.click('[aria-label="Écouter le micro"]')
    await page.waitForSelector('[aria-label="Arrêter l\'écoute"]', { timeout: 300_000 })
    await page.waitForTimeout(1500)

    const liveDraws = await page.evaluate(
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
      // Low bar on purpose: the failure this catches is a loop that never runs
      // at all, and a headless browser under load does not hold 60 fps.
      'the rings animate while listening',
      liveDraws > 10,
      `${liveDraws} repaints in 1.2 s`,
    )

    await page.waitForTimeout(30_000)
    await page.screenshot({ path: join(shotsDir, '10-live.png'), fullPage: true })

    const liveRows = (await page.locator(ROWS).allInnerTexts()).map((t) => t.split('\n')[0].trim())
    const windowsText = await page.locator('.app-bar').innerText()
    const windowCount = Number(windowsText.match(/(\d+)\s+fenêtre/)?.[1] ?? 0)
    check(
      'live listening streams detections from the fixture',
      liveRows.length >= 3 && liveRows.includes(GOLDEN.top.common),
      `${liveRows.length} species, incl. ${GOLDEN.top.common}: ${liveRows.includes(GOLDEN.top.common)}`,
    )
    // 3 s windows hopping every second: about one window per second of audio.
    check(
      'windows advance at the live hop, not the file hop',
      windowCount >= 25 && windowCount <= 45,
      `${windowCount} windows in ~32 s of listening`,
    )

    await page.click('[aria-label="Arrêter l\'écoute"]')
    await page.waitForTimeout(500)
    const stoppedDraws = await page.evaluate(
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
      'the animation loop stops when capture stops',
      stoppedDraws < 5,
      `${stoppedDraws} repaints in 1.2 s after stopping`,
    )
    check(
      'stopping releases the microphone, not just the loop',
      (await page.evaluate(() => window.__trackStops)) > 0,
      `${await page.evaluate(() => window.__trackStops)} track.stop() calls`,
    )

    // ---- installable, and usable with the network off ----------------------
    // A fresh context, because a service worker registration and its caches are
    // exactly the state the rest of this run must not inherit.
    const pwa = await browser.newContext({ viewport: { width: 390, height: 844 } })
    const pwaPage = await pwa.newPage()
    pwaPage.on('console', (m) => m.type() === 'error' && problems.push(`pwa: ${m.text()}`))
    pwaPage.on('pageerror', (e) => problems.push(`pwa pageerror: ${e.message}`))

    const manifestResponse = await pwaPage.goto(`${base}/manifest.webmanifest`)
    const manifest = JSON.parse(await manifestResponse.text())
    check(
      'the manifest is served and describes a standalone app',
      manifestResponse.status() === 200 &&
        manifest.display === 'standalone' &&
        manifest.start_url === '/',
      `${manifestResponse.status()}, display ${manifest.display}`,
    )
    // A rounded icon shipped as maskable is the classic mistake: Android crops
    // it a second time and eats the corners.
    check(
      'it ships a 512 px icon and a separate maskable one',
      manifest.icons.some((i) => i.sizes === '512x512' && i.purpose === 'any') &&
        manifest.icons.some((i) => i.purpose === 'maskable'),
      manifest.icons.map((i) => `${i.sizes} ${i.purpose}`).join(', '),
    )

    await pwaPage.goto(`${base}/`)
    // Bounded on purpose. `serviceWorker.ready` never rejects — if registration
    // silently does not happen, it simply waits forever, which turns a failed
    // check into a hung suite. It cost fifteen minutes to learn that once.
    const swState = await pwaPage.evaluate(() =>
      Promise.race([
        // `ready` resolves as soon as there *is* an active worker, which
        // includes `activating` — ours spends that window pruning old caches and
        // claiming clients. Wait for the state it actually has to reach.
        navigator.serviceWorker.ready.then(
          (r) =>
            new Promise((resolve) => {
              const worker = r.active
              if (!worker) return resolve('none')
              if (worker.state === 'activated') return resolve('activated')
              worker.addEventListener('statechange', () => {
                if (worker.state === 'activated') resolve('activated')
              })
            }),
        ),
        new Promise((resolve) => setTimeout(() => resolve('never registered'), 20_000)),
      ]),
    )
    check('the service worker reaches activated', swState === 'activated', swState)

    // The precache must not quietly include the 52 MB model or the 24 MB
    // runtime: both are fetched on first use, with progress, and precaching them
    // would turn a first visit into a 76 MB download nobody asked for.
    const precached = await pwaPage.evaluate(async () => {
      const names = await caches.keys()
      const shell = names.find((n) => n.startsWith('shell-'))
      if (!shell) return null
      const keys = await (await caches.open(shell)).keys()
      return keys.map((r) => new URL(r.url).pathname)
    })
    check(
      'the precache holds the shell only, not the model or the runtime',
      precached !== null &&
        precached.length > 8 &&
        !precached.some((p) => p.endsWith('.onnx') || p.startsWith('/ort/') || p.includes('parity')),
      `${precached?.length ?? 0} entries`,
    )

    // The real question is not "is a worker registered" but "does the app come
    // back with the network off".
    await pwa.setOffline(true)
    await pwaPage.reload()
    await pwaPage.waitForSelector('h1', { timeout: 15_000 })
    check(
      'the app loads with the network off',
      (await pwaPage.locator('h1').textContent()) === 'whosyourbirdy' &&
        (await pwaPage.locator('.app-bar').count()) === 1,
    )
    await pwaPage.screenshot({ path: join(shotsDir, '8-offline.png') })
    await pwa.setOffline(false)
    await pwa.close()

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
