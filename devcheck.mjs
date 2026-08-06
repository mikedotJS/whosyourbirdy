// Finding #3: `pnpm dev` could not analyse anything (Vite refused to serve
// public/ files reached by an import). Verify the plugin fix on the DEV server.
import { spawn } from 'node:child_process'
import { chromium } from 'playwright'

const dev = spawn('pnpm', ['dev', '--port', '5199', '--strictPort'], { cwd: '/home/user/whosyourbirdy' })
let ready = false
dev.stdout.on('data', d => { if (/Local:/.test(d.toString())) ready = true })
dev.stderr.on('data', d => process.stderr.write(d))
for (let i = 0; i < 60 && !ready; i++) await new Promise(r => setTimeout(r, 500))
if (!ready) { dev.kill(); throw new Error('dev server never started') }

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' })
const page = await browser.newPage()
const problems = []
page.on('console', m => m.type() === 'error' && problems.push(m.text()))
page.on('response', r => r.status() >= 400 && problems.push(`HTTP ${r.status()} ${r.url()}`))
await page.goto('http://127.0.0.1:5199/')
await page.setInputFiles('input[type=file]', '/home/user/whosyourbirdy/.cache/birdnet/soundscape.wav')
try {
  await page.waitForSelector('ul > li', { timeout: 420000 })
  const rows = await page.locator('ul > li').count()
  const first = (await page.locator('ul > li').first().innerText()).replace(/\n/g, ' / ')
  console.log(`DEV OK — ${rows} detections; first: ${first}`)
} catch (e) {
  console.log('DEV FAILED:', (await page.locator('main').innerText()).slice(0, 300))
}
console.log('problems:', problems.length ? problems.slice(0,4) : 'none')
await browser.close(); dev.kill('SIGTERM')
