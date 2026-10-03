import { test } from 'node:test'
import assert from 'node:assert/strict'
import Ditify, { Forest } from '../ditify.js'

// Quinlan's "play tennis" data; ID3's textbook tree splits on outlook first.
const TENNIS = `sunny hot high weak no|sunny hot high strong no|overcast hot high weak yes|rain mild high weak yes|rain cool normal weak yes|rain cool normal strong no|overcast cool normal strong yes|sunny mild high weak no|sunny cool normal weak yes|rain mild normal weak yes|sunny mild normal strong yes|overcast mild high strong yes|overcast hot normal weak yes|rain mild high strong no`
  .split('|')
  .map((r) => r.split(' '))
const TENNIS_ATTRS = ['outlook', 'temp', 'humidity', 'wind', 'play']

const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0)

test('learns the textbook ID3 tree regardless of column order', () => {
  for (const criterion of ['entropy', 'gainRatio', 'gini']) {
    const t = new Ditify({ attributes: TENNIS_ATTRS, criterion }).train(TENNIS)
    assert.equal(t.tree().attribute, 'outlook', criterion)
    assert.equal(t.evaluate(TENNIS).accuracy, 1)
  }
  const order = [3, 2, 1, 0, 4]
  const t = new Ditify({ attributes: order.map((i) => TENNIS_ATTRS[i]) }).train(TENNIS.map((r) => order.map((i) => r[i])))
  assert.equal(t.tree().attribute, 'outlook')
  assert.deepEqual(
    new Ditify({ attributes: TENNIS_ATTRS, shrinkage: 0 }).train(TENNIS).rules(),
    [
      'IF outlook = sunny AND humidity = high THEN play = no (100%, n=3)',
      'IF outlook = sunny AND humidity = normal THEN play = yes (100%, n=2)',
      'IF outlook = overcast THEN play = yes (100%, n=4)',
      'IF outlook = rain AND wind = weak THEN play = yes (100%, n=3)',
      'IF outlook = rain AND wind = strong THEN play = no (100%, n=2)',
    ]
  )
})

test('v1 README example: blank value marks the attribute to predict', () => {
  const c = new Ditify({ attribs: ['meal', 'weather', 'speed', 'restaurant'] })
  c.train(['breakfast', 'hot', 'quick', 'subway'])
  c.train(['lunch', 'hot', 'medium', "moumon's"])
  c.train(['lunch', 'rainy', 'leisurely', "percy's"])
  c.train(['dinner', 'rainy', 'medium', 'dominos'])
  c.train(['breakfast', 'cold', 'quick', 'subway'])
  c.train(['lunch', 'cold', 'leisurely', "megan's"])
  c.train(['dinner', 'cold', 'medium', 'dominos'])

  const r = c.classify(['lunch', 'rainy', 'medium', ''])
  assert.equal(r.target, 'restaurant')
  assert.equal(r.label, "percy's")
  assert.ok(r.chance > 0.5 && r.chance < 1)
  assert.ok(Math.abs(sum(r.probabilities) - 1) < 1e-9)

  // Any column can be the target.
  assert.equal(c.classify(['', 'cold', 'medium', 'dominos']).label, 'dinner')
  assert.equal(c.classify(['', 'cold', 'medium', 'dominos']).target, 'meal')
  assert.equal(c.classify({ weather: 'hot', speed: 'quick', restaurant: 'subway' }, { target: 'meal' }).label, 'breakfast')
})

test('object rows infer attributes; explicit target', () => {
  const c = new Ditify({ target: 'play' })
  c.train(TENNIS.map((r) => Object.fromEntries(TENNIS_ATTRS.map((a, j) => [a, r[j]]))))
  assert.deepEqual(c.attributes, TENNIS_ATTRS)
  assert.equal(c.classify({ outlook: 'overcast', temp: 'cool', humidity: 'high', wind: 'strong' }).label, 'yes')
  const e = c.explain({ outlook: 'sunny', humidity: 'high' })
  assert.equal(e.label, 'no')
  assert.equal(e.because, 'outlook = sunny AND humidity = high')
})

test('numeric attributes get threshold splits', () => {
  const rows = []
  for (let i = 0; i < 40; i++) rows.push([i, i % 3 === 0 ? 'a' : 'b', i < 17 ? 'low' : 'high'])
  const c = new Ditify({ attributes: ['x', 'noise', 'level'] }).train(rows)
  const root = c.tree()
  assert.equal(root.attribute, 'x')
  assert.equal(root.children[0].threshold, 16.5)
  assert.equal(c.classify([3, 'a']).label, 'low')
  assert.equal(c.classify([30, 'a']).label, 'high')
  assert.equal(c.classify([16.9, 'b']).label, 'high')

  // Numeric strings (e.g. from CSV) can be declared numeric.
  const s = new Ditify({ attributes: ['x', 'noise', 'level'], numeric: ['x'] }).train(rows.map((r) => [String(r[0]), r[1], r[2]]))
  assert.equal(s.tree().children[0].threshold, 16.5)
  assert.equal(s.classify(['30', 'a']).label, 'high')
})

test('missing and unseen values are averaged over branches', () => {
  const rows = TENNIS.map((r, i) => (i % 4 === 0 ? [r[0], null, r[2], '', r[4]] : r))
  const c = new Ditify({ attributes: TENNIS_ATTRS }).train(rows)
  for (const q of [['snow', 'hot', 'high', 'weak'], [null, 'hot', 'high', 'weak'], ['sunny', 'hot', undefined, 'weak']]) {
    const r = c.classify(q)
    assert.ok(['yes', 'no'].includes(r.label))
    assert.ok(Math.abs(sum(r.probabilities) - 1) < 1e-9)
  }
  // Unknown outlook, but humidity=normal + wind=weak is "yes" on every branch.
  assert.equal(c.classify(['snow', 'cool', 'normal', 'weak']).label, 'yes')
  assert.match(c.explain(['snow', 'cool', 'normal', 'weak']).because, /outlook unknown/)
})

test('learns XOR through a zero-gain first split', () => {
  const rows = []
  for (const a of ['0', '1']) for (const b of ['0', '1']) for (let k = 0; k < 3; k++) rows.push([a, b, a === b ? 'same' : 'diff'])
  const c = new Ditify({ attributes: ['a', 'b', 'y'] }).train(rows)
  assert.equal(c.evaluate(rows).accuracy, 1)
})

test('shrinkage keeps probabilities a valid distribution and tempers small leaves', () => {
  const raw = new Ditify({ attributes: TENNIS_ATTRS, shrinkage: 0 }).train(TENNIS)
  const hs = new Ditify({ attributes: TENNIS_ATTRS, shrinkage: 5 }).train(TENNIS)
  const q = ['sunny', 'cool', 'high', 'strong']
  assert.equal(raw.classify(q).chance, 1)
  const r = hs.classify(q)
  assert.equal(r.label, 'no')
  assert.ok(r.chance < 1 && r.chance > 0.5)
  for (const p of Object.values(r.probabilities)) assert.ok(p >= 0)
})

test('pre-pruning options limit the tree', () => {
  const stump = new Ditify({ attributes: TENNIS_ATTRS, maxDepth: 1 }).train(TENNIS)
  assert.ok(stump.tree().children.every((c) => !c.node.children))
  const leaf = new Ditify({ attributes: TENNIS_ATTRS, minSamplesLeaf: 8 }).train(TENNIS)
  assert.equal(leaf.tree().children, undefined)
})

test('awkward values cannot pollute prototypes', () => {
  const c = new Ditify({ attributes: ['k', 'y'] }).train([
    ['__proto__', 'a'],
    ['constructor', 'b'],
    ['toString', 'a'],
  ])
  assert.equal(c.classify(['constructor']).label, 'b')
  assert.equal({}.a, undefined)
})

test('toJSON / fromJSON round trip', () => {
  const c = new Ditify({ attributes: TENNIS_ATTRS, maxDepth: 2 }).train(TENNIS)
  const copy = Ditify.fromJSON(JSON.stringify(c))
  assert.deepEqual(copy.tree(), c.tree())
  const f = new Forest({ attributes: TENNIS_ATTRS, trees: 10 }).train(TENNIS)
  const g = Forest.fromJSON(JSON.parse(JSON.stringify(f)))
  const q = ['rain', 'mild', 'high', 'strong']
  assert.deepEqual(g.classify(q), f.classify(q))
})

test('forest is reproducible and accurate', () => {
  const a = new Forest({ attributes: TENNIS_ATTRS, trees: 50, seed: 7 }).train(TENNIS)
  const b = new Forest({ attributes: TENNIS_ATTRS, trees: 50, seed: 7 }).train(TENNIS)
  const q = ['sunny', 'mild', 'high', 'weak']
  assert.deepEqual(a.classify(q), b.classify(q))
  assert.equal(a.classify(q).label, 'no')
  assert.ok(a.evaluate(TENNIS).accuracy >= 0.9)
  const imp = a.importance()
  assert.ok(Math.abs(sum(imp) - 1) < 1e-9)
  assert.equal(Object.entries(imp).sort((x, y) => y[1] - x[1])[0][0], 'outlook')
})

test('training is lazy and incremental', () => {
  const c = new Ditify({ attributes: ['x', 'y'] })
  c.train(['a', 'p'])
  assert.equal(c.classify(['a']).label, 'p')
  c.train([['a', 'q'], ['a', 'q']])
  assert.equal(c.classify(['a']).label, 'q')
})

test('input errors', () => {
  const c = new Ditify({ attributes: ['x', 'y'] })
  assert.throws(() => c.train(['a', 'b', 'c']), /3 values but there are 2/)
  assert.throws(() => c.train('a'), TypeError)
  assert.throws(() => c.classify(['a']), /no training rows/)
  c.train(['a', 'b'])
  assert.throws(() => c.classify(['a'], { target: 'z' }), /unknown attribute/)
  assert.throws(() => new Ditify().train(['a']), /set `attributes`/)
})
