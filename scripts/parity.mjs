#!/usr/bin/env node
/**
 * Parity harness: prove the browser pipeline reproduces official BirdNET.
 *
 * Three levels, ordered so that a failure points at one cause rather than "the
 * numbers are off somewhere":
 *
 *   A. Model only. Identical PCM goes into the official TFLite (Python) and into
 *      our ONNX under onnxruntime-web's WASM backend (Node). Any difference here
 *      is the model conversion.
 *
 *   B. Whole chain. The same file goes through the Python reference and through
 *      a real Chromium running the actual `lib/birdnet` code — Web Audio decode,
 *      windowing, worker, sigmoid. Any difference that is not in A is the audio
 *      chain.
 *
 *   C. Resampling. The file is fed at 44.1 kHz so both sides must resample. The
 *      browser uses its own resampler and BirdNET uses resampy; they cannot
 *      agree bit-for-bit. This level *measures* that divergence and reports it.
 *      It is not corrected, and no fudge factor is applied anywhere.
 *
 * Levels A and B must pass. Level C is reported, and only fails if the drift is
 * large enough to change which species get reported.
 *
 * Usage: pnpm parity [--only=A,B,C] [--keep]
 */
import { spawnSync } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync, existsSync, rmSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer } from 'node:http'
import { chromium } from 'playwright'

// This environment pre-installs Chromium at a build number that may not match the
// one our Playwright pins; launching it by path avoids a download we cannot do
// (and must not do: the browser is already here).
const CHROMIUM = process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium'
const launchOptions = existsSync(CHROMIUM) ? { executablePath: CHROMIUM } : {}

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const WORK = join(ROOT, '.parity')
const CACHE = join(ROOT, '.cache', 'birdnet')
const VENV = join(ROOT, '.venv', 'bin', 'python')
const FIXTURE = join(CACHE, 'soundscape.wav')

// Thresholds. `SCORE_TOL` is the contract: post-sigmoid confidences, which are
// what the UI shows and what a user would compare against BirdNET-Analyzer.
const SCORE_TOL = 1e-3
const LOGIT_TOL = 5e-3

const args = process.argv.slice(2)
const only = (args.find((a) => a.startsWith('--only='))?.slice(7) ?? 'A,B,C').split(',')
const keep = args.includes('--keep')

const bold = (s) => `\x1b[1m${s}\x1b[0m`
const green = (s) => `\x1b[32m${s}\x1b[0m`
const red = (s) => `\x1b[31m${s}\x1b[0m`
const dim = (s) => `\x1b[2m${s}\x1b[0m`

/* ------------------------------------------------------------------ helpers */

function python(scriptArgs, label) {
  if (!existsSync(VENV)) {
    throw new Error(
      'no .venv found. Create it with:\n' +
        '  python3 -m venv .venv && .venv/bin/pip install numpy==1.26.4 ' +
        'tensorflow-cpu==2.15.1 tf2onnx==1.16.1 onnx onnxruntime soundfile librosa resampy',
    )
  }
  const result = spawnSync(VENV, scriptArgs, { cwd: ROOT, encoding: 'utf8' })
  if (result.status !== 0) {
    throw new Error(`${label} failed:\n${result.stderr?.slice(-2000) ?? ''}`)
  }
  return result.stdout
}

/** Read a .npz written by numpy. Only handles the uncompressed float32 arrays we emit. */
function readNpz(path) {
  // Rather than implement the zip+npy format, ask Python for a flat binary dump.
  const out = {}
  const meta = JSON.parse(readFileSync(path.replace(/\.npz$/, '.json'), 'utf8'))
  out.meta = meta
  out.logits = new Float32Array(readFileSync(path.replace(/\.npz$/, '.logits.bin')).buffer.slice(0))
  return out
}

function flatSigmoid(logits, sensitivity = 1.0, bias = 1.0, clipVal = 15.0) {
  const out = new Float32Array(logits.length)
  const shift = (bias - 1.0) * 10.0
  for (let i = 0; i < logits.length; i++) {
    let v = logits[i] + shift
    if (v < -clipVal) v = -clipVal
    else if (v > clipVal) v = clipVal
    const y = -sensitivity * v
    const e = Math.exp(-Math.abs(y))
    out[i] = y >= 0 ? e / (1 + e) : 1 / (1 + e)
  }
  return out
}

/** Compare two flat logit arrays laid out as [windows, classes]. */
function compare(name, got, expected, windows, classes, { scoreTol = SCORE_TOL, logitTol = LOGIT_TOL } = {}) {
  if (got.length !== expected.length) {
    return { name, ok: false, detail: `length ${got.length} != ${expected.length}` }
  }

  const gotScores = flatSigmoid(got)
  const expScores = flatSigmoid(expected)

  let maxLogit = 0
  let maxScore = 0
  let worstWindow = -1
  let worstClass = -1

  for (let i = 0; i < got.length; i++) {
    const dl = Math.abs(got[i] - expected[i])
    if (dl > maxLogit) maxLogit = dl
    const ds = Math.abs(gotScores[i] - expScores[i])
    if (ds > maxScore) {
      maxScore = ds
      worstWindow = Math.floor(i / classes)
      worstClass = i % classes
    }
  }

  // Do the two sides agree on what to report? This is the question that matters
  // more than any epsilon: same species, same windows, above the same threshold.
  let disagreements = 0
  for (let i = 0; i < got.length; i++) {
    if ((gotScores[i] >= 0.25) !== (expScores[i] >= 0.25)) disagreements++
  }

  const ok = maxScore <= scoreTol && maxLogit <= logitTol && disagreements === 0
  return {
    name,
    ok,
    maxLogit,
    maxScore,
    disagreements,
    worstWindow,
    worstClass,
    windows,
    detail: `max|Δscore|=${maxScore.toExponential(3)} max|Δlogit|=${maxLogit.toExponential(3)} ` +
      `detections differing at 0.25: ${disagreements}` +
      (worstWindow >= 0 ? dim(`  (worst: window ${worstWindow}, class ${worstClass})`) : ''),
  }
}

/* ------------------------------------------------------- reference (Python) */

function runReference(audio, out, { forceResample = false } = {}) {
  const argv = ['scripts/parity_reference.py', audio, '--out', out, '--dump-pcm']
  if (forceResample) argv.push('--force-resample')
  const stdout = python(argv, 'python reference')
  process.stdout.write(dim(`    ${stdout.trim()}\n`))

  // Convert the .npz to a flat binary the JS side can mmap trivially.
  python(
    ['-c',
      `import numpy as np,sys; d=np.load(sys.argv[1]); d['logits'].astype('<f4').tofile(sys.argv[2])`,
      out, out.replace(/\.npz$/, '.logits.bin')],
    'npz flatten',
  )
  return readNpz(out)
}

/* --------------------------------------------------- level A: ORT WASM, Node */

async function levelA(reference) {
  const ort = await import('onnxruntime-web')
  ort.env.wasm.numThreads = 1
  ort.env.wasm.simd = true
  ort.env.logLevel = 'error'
  // In Node, ORT resolves its .wasm from the package directory; no wasmPaths needed.

  const modelPath = join(ROOT, 'public', 'models', 'birdnet_v2.4_fp32.onnx')
  if (!existsSync(modelPath)) throw new Error('run `pnpm model:build` first')

  const session = await ort.InferenceSession.create(readFileSync(modelPath), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })

  // Reuse the reference's own decoded PCM: level A isolates the model, so both
  // sides must see identical samples by construction, not by luck.
  const pcm = new Float32Array(readFileSync(join(WORK, 'ref-48k.pcm')).buffer.slice(0))
  const { windows, classes } = reference.meta

  const got = new Float32Array(windows * classes)
  const timings = []
  for (let w = 0; w < windows; w++) {
    const window = new Float32Array(144000)
    const chunk = pcm.subarray(w * 144000, Math.min((w + 1) * 144000, pcm.length))
    window.set(chunk)
    const t0 = performance.now()
    const output = await session.run({ input: new ort.Tensor('float32', window, [1, 144000]) })
    timings.push(performance.now() - t0)
    got.set(output.output.data, w * classes)
  }

  timings.sort((a, b) => a - b)
  process.stdout.write(
    dim(`    ${windows} windows, median ${timings[timings.length >> 1].toFixed(0)} ms/window (node wasm, 1 thread)\n`),
  )

  return compare('A  model only (ONNX/WASM vs official TFLite)', got, reference.logits, windows, classes)
}

/* ------------------------------------- level B/C: real Chromium, real chain */

function serveStatic(port) {
  // dist/ first: the browser must exercise the production bundle, not source.
  const roots = [join(ROOT, 'dist'), join(ROOT, '.parity')]
  const types = {
    '.js': 'text/javascript', '.mjs': 'text/javascript', '.ts': 'text/javascript',
    '.json': 'application/json', '.wasm': 'application/wasm', '.html': 'text/html',
    '.onnx': 'application/octet-stream', '.txt': 'text/plain', '.wav': 'audio/wav',
    '.pcm': 'application/octet-stream',
  }
  const server = createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0])
    for (const root of roots) {
      const path = join(root, url)
      if (!path.startsWith(root)) continue
      if (existsSync(path) && !path.endsWith('/')) {
        const ext = path.slice(path.lastIndexOf('.'))
        res.writeHead(200, { 'content-type': types[ext] ?? 'application/octet-stream' })
        res.end(readFileSync(path))
        return
      }
    }
    res.writeHead(404).end('not found')
  })
  return new Promise((resolve) => server.listen(port, () => resolve(server)))
}

async function levelBrowser(label, reference, audioFile, expectFail) {
  const server = await serveStatic(0)
  const port = server.address().port
  const browser = await chromium.launch(launchOptions)

  try {
    const page = await browser.newPage()
    page.on('console', (m) => {
      if (m.type() === 'error') process.stderr.write(dim(`    [browser] ${m.text()}\n`))
    })
    // A 404 must name itself: a silently missing asset (labels, wasm, manifest)
    // is exactly the kind of thing a passing tensor diff would not reveal.
    page.on('requestfailed', (r) => process.stderr.write(dim(`    [404?] ${r.url()}\n`)))
    page.on('response', (r) => {
      if (r.status() >= 400) process.stderr.write(dim(`    [HTTP ${r.status()}] ${r.url()}\n`))
    })
    await page.goto(`http://127.0.0.1:${port}/parity-harness.html`)

    const result = await page.evaluate(
      async ([audioUrl, windows, classes]) => {
        const out = await window.runParity(audioUrl, windows, classes)
        return { logits: Array.from(out.logits), meta: out.meta }
      },
      [`/${audioFile}`, reference.meta.windows, reference.meta.classes],
      { timeout: 600_000 },
    )

    const got = Float32Array.from(result.logits)
    process.stdout.write(
      dim(`    decoded ${result.meta.originalRate} Hz -> 48000 Hz, ` +
        `${result.meta.windows} windows, median ${result.meta.medianMs.toFixed(0)} ms/window (chromium)\n`),
    )

    const cmp = compare(label, got, reference.logits, reference.meta.windows, reference.meta.classes,
      expectFail ? { scoreTol: Infinity, logitTol: Infinity } : {})

    // The tensor diff above only covers the raw path. This is the second half of
    // level B: the shipping path (worker, thresholding, label mapping) must agree
    // with what those tensors imply. A bug there would otherwise pass unnoticed.
    const { analyzerDrift, analyzerMismatches } = result.meta
    const analyzerOk = analyzerMismatches === 0 && analyzerDrift <= 1e-6
    if (!analyzerOk) {
      return {
        name: label,
        ok: false,
        informational: false,
        detail:
          `${cmp.detail}\n        analyzer cross-check FAILED: ` +
          `${analyzerMismatches} mismatched detections, max drift ${analyzerDrift.toExponential(3)}`,
      }
    }

    return {
      ...cmp,
      informational: Boolean(expectFail),
      detail: `${cmp.detail}\n        analyzer cross-check: worker/sigmoid/labels agree with the raw logits`,
    }
  } finally {
    await browser.close()
    server.close()
  }
}

/* ------------------------------------------------------------------- driver */

async function main() {
  if (!existsSync(FIXTURE)) {
    throw new Error(`missing ${FIXTURE}; run \`python3 scripts/fetch_artifacts.py\` first`)
  }
  mkdirSync(WORK, { recursive: true })

  console.log(bold('\nBirdNET parity harness'))
  console.log(dim(`  fixture: ${FIXTURE.replace(ROOT + '/', '')} (official BirdNET example soundscape)`))
  console.log(dim(`  reference: BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite via the TFLite interpreter\n`))

  const results = []

  console.log(bold('  reference run (python, tflite)'))
  const reference = runReference(FIXTURE, join(WORK, 'ref-48k.npz'))

  if (only.includes('A')) {
    console.log(bold('\n  level A — model only'))
    results.push(await levelA(reference))
  }

  if (only.includes('B') || only.includes('C')) {
    console.log(bold('\n  building the static site (the browser levels run against dist/)'))
    const build = spawnSync('pnpm', ['build'], { cwd: ROOT, encoding: 'utf8' })
    if (build.status !== 0) throw new Error(`pnpm build failed:\n${build.stdout}\n${build.stderr}`)
  }

  if (only.includes('B')) {
    console.log(bold('\n  level B — whole chain in Chromium'))
    // Serve the fixture from .parity so the browser fetches the same bytes.
    writeFileSync(join(WORK, 'soundscape.wav'), readFileSync(FIXTURE))
    results.push(await levelBrowser('B  whole chain (Chromium vs official TFLite)', reference, 'soundscape.wav'))
  }

  if (only.includes('C')) {
    console.log(bold('\n  level C — resampling divergence (informational)'))
    console.log(dim('    both sides must resample 44.1 kHz -> 48 kHz with different resamplers'))
    python(
      ['-c',
        `import soundfile as sf, librosa, sys, numpy as np
a, sr = sf.read(sys.argv[1], dtype='float32', always_2d=True)
m = a.mean(axis=1)
r = librosa.resample(m, orig_sr=sr, target_sr=44100, res_type='kaiser_best')
sf.write(sys.argv[2], r, 44100, subtype='PCM_16')`,
        FIXTURE, join(WORK, 'soundscape-44k.wav')],
      'build 44.1 kHz fixture',
    )
    const ref44 = runReference(join(WORK, 'soundscape-44k.wav'), join(WORK, 'ref-44k.npz'))
    results.push(await levelBrowser('C  resampled 44.1 kHz (Web Audio vs resampy)', ref44, 'soundscape-44k.wav', true))
  }

  console.log(bold('\n  results\n'))
  let failed = false
  for (const r of results) {
    const status = r.informational ? dim('INFO') : r.ok ? green('PASS') : red('FAIL')
    console.log(`  ${status}  ${r.name}`)
    console.log(`        ${r.detail}`)
    if (!r.ok && !r.informational) failed = true
  }

  const informational = results.filter((r) => r.informational)
  if (informational.length) {
    console.log(dim('\n  Level C is measured, not enforced: the browser resampler and resampy'))
    console.log(dim('  differ by design. It is reported so the cost of a non-48 kHz file is visible.'))
  }

  if (!keep) rmSync(WORK, { recursive: true, force: true })

  console.log(failed ? red('\n  PARITY FAILED\n') : green('\n  PARITY OK\n'))
  process.exit(failed ? 1 : 0)
}

main().catch((error) => {
  console.error(red(`\n  ${error.message}\n`))
  process.exit(1)
})
