import { spawn } from 'node:child_process'
import { appendFileSync } from 'node:fs'
import { chromium } from 'playwright'
const log = (m) => appendFileSync('/tmp/devcheck.log', m + '\n')

const dev = spawn('pnpm', ['dev', '--port', '5199', '--strictPort'], { cwd: process.cwd() })
let ready = false
dev.stdout.on('data', d => { if (/Local:/.test(d.toString())) ready = true })
for (let i = 0; i < 120 && !ready; i++) await new Promise(r => setTimeout(r, 500))
if (!ready) { dev.kill(); log('dev server never started'); process.exit(1) }
log('dev server up')

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' })
const page = await browser.newPage()
const problems = []
page.on('console', m => m.type() === 'error' && problems.push(m.text()))
page.on('response', r => r.status() >= 400 && problems.push(`HTTP ${r.status()} ${r.url()}`))
await page.goto('http://127.0.0.1:5199/')
log('page loaded')
await page.setInputFiles('input[type=file]', '.cache/birdnet/soundscape.wav')
try {
  await page.waitForSelector('ul > li', { timeout: 500000 })
  const rows = await page.locator('ul > li').count()
  const first = (await page.locator('ul > li').first().innerText()).replace(/\n/g, ' / ')
  log(`DEV OK — ${rows} detections; first: ${first}`)
} catch {
  log('DEV FAILED: ' + (await page.locator('main').innerText()).slice(0, 250).replace(/\n/g, ' | '))
}
log('problems: ' + (problems.length ? JSON.stringify(problems.slice(0,3)) : 'none'))
await browser.close(); dev.kill('SIGTERM')
