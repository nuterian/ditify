// Accuracy and speed on seeded synthetic datasets: `npm run bench`.
import { Ditify, Forest } from '../ditify.js'

function rng(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), a | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const pick = (r, n) => Math.floor(r() * n)

// MONK's problem 1 style: class = (a1 == a2) OR (a5 == 1), 5% label noise.
function monk(n, seed, missing = 0) {
  const r = rng(seed)
  const sizes = [3, 3, 2, 3, 4, 2]
  const rows = []
  for (let i = 0; i < n; i++) {
    const a = sizes.map((s) => pick(r, s) + 1)
    let y = a[0] === a[1] || a[4] === 1
    if (r() < 0.05) y = !y
    const row = a.map((v, j) => (r() < missing ? null : `v${v}`))
    rows.push([...row, y ? 'yes' : 'no'])
  }
  return { attributes: ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'class'], rows }
}

// Two informative numeric features (inside a circle?) plus three noise features.
function circle(n, seed) {
  const r = rng(seed)
  const rows = []
  for (let i = 0; i < n; i++) {
    const x = r() * 2 - 1
    const y = r() * 2 - 1
    let inside = x * x + y * y < 0.5
    if (r() < 0.05) inside = !inside
    rows.push([x, y, r(), r(), r(), inside ? 'in' : 'out'])
  }
  return { attributes: ['x', 'y', 'n1', 'n2', 'n3', 'label'], rows }
}

const datasets = {
  'monk (categorical)': (s) => monk(300, s),
  'monk, 20% missing': (s) => monk(300, s, 0.2),
  'circle (numeric)': (s) => circle(300, s),
}

const models = {
  'ID3 (entropy, raw leaves)': (attributes) => new Ditify({ attributes, criterion: 'entropy', shrinkage: 0 }),
  'C4.5 + shrinkage (default)': (attributes) => new Ditify({ attributes }),
  'Forest, 100 trees': (attributes) => new Forest({ attributes }),
}

const pad = (s, n) => String(s).padEnd(n)
console.log(pad('dataset', 22), pad('model', 30), 'accuracy  log loss')
for (const [name, make] of Object.entries(datasets)) {
  const test = make(999)
  for (const [label, build] of Object.entries(models)) {
    let acc = 0
    let ll = 0
    const runs = 5
    for (let s = 1; s <= runs; s++) {
      const train = make(s)
      const r = build(train.attributes).train(train.rows).evaluate(test.rows.slice(0, 2000))
      acc += r.accuracy / runs
      ll += r.logLoss / runs
    }
    console.log(pad(name, 22), pad(label, 30), pad(acc.toFixed(3), 9), ll.toFixed(3))
  }
}

console.log('\nspeed (10 000 rows, mixed numeric + categorical):')
const big = circle(10000, 7)
big.rows.forEach((row, i) => row.splice(5, 0, `c${i % 7}`))
big.attributes.splice(5, 0, 'cat')
for (const [label, build] of [
  ['tree', (a) => new Ditify({ attributes: a })],
  ['forest, 50 trees', (a) => new Forest({ attributes: a, trees: 50 })],
]) {
  const model = build(big.attributes).train(big.rows)
  let t = performance.now()
  model.classify(big.rows[0].slice(0, -1))
  const fit = performance.now() - t
  t = performance.now()
  for (const row of big.rows) model.classify(row.slice(0, -1))
  const per = ((performance.now() - t) / big.rows.length) * 1000
  console.log(`  ${pad(label, 18)} fit ${fit.toFixed(0)} ms, predict ${per.toFixed(1)} µs/row`)
}
