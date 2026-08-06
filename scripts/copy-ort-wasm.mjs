#!/usr/bin/env node
/**
 * Copy the ONNX Runtime Web WASM binaries into public/ort/.
 *
 * ORT resolves its runtime files (`ort-wasm-simd-threaded*.wasm` and their `.mjs`
 * loaders) relative to `ort.env.wasm.wasmPaths` at runtime, not at build time.
 * If we leave them inside node_modules the dev server serves them, but the
 * production build does not copy them and every deployment 404s on first
 * inference. Copying them into public/ makes dev and prod resolve identically.
 *
 * Runs on postinstall so a fresh clone is ready without a manual step.
 */
import { cp, mkdir, readdir, rm } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const src = join(root, 'node_modules', 'onnxruntime-web', 'dist')
const dest = join(root, 'public', 'ort')

if (!existsSync(src)) {
  console.warn('[copy-ort-wasm] onnxruntime-web not installed yet, skipping')
  process.exit(0)
}

await rm(dest, { recursive: true, force: true })
await mkdir(dest, { recursive: true })

const entries = await readdir(src)
// The .mjs loaders are fetched by the .wasm side of the pair, so both must ship.
const wanted = entries.filter((f) => /^ort-.*\.(wasm|mjs)$/.test(f))

if (wanted.length === 0) {
  console.error('[copy-ort-wasm] no ORT runtime files found in', src)
  process.exit(1)
}

for (const file of wanted) {
  await cp(join(src, file), join(dest, file))
}

console.log(`[copy-ort-wasm] copied ${wanted.length} files to public/ort/`)
