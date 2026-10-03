// ditify: a small, dependency-free decision tree and random forest classifier.
// Runs in Node (>= 18) and modern browsers as an ES module.
//
// Algorithm: C4.5-style trees (gain ratio, numeric thresholds, fractional
// handling of missing values) with hierarchical shrinkage of leaf probabilities
// (Agarwal et al., ICML 2022), plus an optional bagged random forest.

const DEFAULTS = {
  criterion: 'gainRatio', // 'gainRatio' (C4.5) | 'entropy' (ID3 information gain) | 'gini' (CART)
  maxDepth: Infinity,
  minSamplesLeaf: 1,
  minGain: 0,
  shrinkage: 1, // 0 disables hierarchical shrinkage (raw leaf frequencies)
  numeric: [], // attributes to treat as numbers even when given as strings
}

const FOREST_DEFAULTS = {
  criterion: 'gini',
  trees: 100,
  maxFeatures: 'sqrt', // 'sqrt' | 'log2' | 'all' | count | fraction in (0, 1)
  seed: 1,
}

// Row copies created by missing-value fan-out lighter than this are dropped.
const MIN_WEIGHT = 1e-6

const isMissing = (v) =>
  v === undefined || v === null || v === '' || (typeof v === 'number' && Number.isNaN(v))

const keyOf = (v) => String(v)

function entropy(counts, total) {
  if (total <= 0) return 0
  let h = 0
  for (let k = 0; k < counts.length; k++) {
    if (counts[k] > 0) {
      const p = counts[k] / total
      h -= p * Math.log2(p)
    }
  }
  return h
}

function gini(counts, total) {
  if (total <= 0) return 0
  let s = 0
  for (let k = 0; k < counts.length; k++) {
    const p = counts[k] / total
    s += p * p
  }
  return 1 - s
}

// Small seeded PRNG (mulberry32) so forests are reproducible.
function rng(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ---------------------------------------------------------------------------
// Encoding: training rows -> per-attribute typed columns

function encodeColumn(rows, j, forceNumeric) {
  let numeric = forceNumeric
  if (!numeric) {
    let seen = false
    numeric = true
    for (const r of rows) {
      const v = r[j]
      if (isMissing(v)) continue
      seen = true
      if (typeof v !== 'number') {
        numeric = false
        break
      }
    }
    numeric = numeric && seen
  }
  if (numeric) {
    const values = new Float64Array(rows.length)
    for (let i = 0; i < rows.length; i++) {
      const v = rows[i][j]
      values[i] = isMissing(v) ? NaN : Number(v)
    }
    return { index: j, numeric: true, values }
  }
  const codes = new Int32Array(rows.length)
  const cats = []
  const lookup = new Map()
  for (let i = 0; i < rows.length; i++) {
    const v = rows[i][j]
    if (isMissing(v)) {
      codes[i] = -1
      continue
    }
    const key = keyOf(v)
    let c = lookup.get(key)
    if (c === undefined) {
      c = cats.length
      lookup.set(key, c)
      cats.push(v)
    }
    codes[i] = c
  }
  return { index: j, numeric: false, codes, cats, lookup }
}

// Encode one query value; categorical: -1 missing, -2 never seen in training.
function encodeValue(f, v) {
  if (f.numeric) return isMissing(v) ? NaN : Number(v)
  if (isMissing(v)) return -1
  const c = f.lookup.get(keyOf(v))
  return c === undefined ? -2 : c
}

function buildProblem(rows, attributes, t, numeric) {
  const features = attributes.map((name, j) =>
    j === t ? null : encodeColumn(rows, j, numeric.includes(name))
  )
  const y = encodeColumn(rows, t, false)
  return {
    attributes,
    target: t,
    features,
    candidates: features.filter(Boolean),
    y: y.codes,
    classes: y.cats,
    K: y.cats.length,
  }
}

// ---------------------------------------------------------------------------
// Tree growing

function evalCategorical(f, idx, w, ctx, W) {
  const { y, K, impurity, minLeaf } = ctx
  const V = f.cats.length
  const m = new Float64Array(V * K)
  const vw = new Float64Array(V)
  let missW = 0
  for (let i = 0; i < idx.length; i++) {
    const c = f.codes[idx[i]]
    if (c < 0) {
      missW += w[i]
      continue
    }
    m[c * K + y[idx[i]]] += w[i]
    vw[c] += w[i]
  }
  const known = W - missW
  const knownCounts = new Float64Array(K)
  const sizes = [missW]
  let branches = 0
  let big = 0
  let childImp = 0
  for (let v = 0; v < V; v++) {
    if (vw[v] <= 0) continue
    branches++
    if (vw[v] >= minLeaf) big++
    const row = m.subarray(v * K, (v + 1) * K)
    childImp += vw[v] * impurity(row, vw[v])
    for (let k = 0; k < K; k++) knownCounts[k] += row[k]
    sizes.push(vw[v])
  }
  if (branches < 2 || big < 2) return null
  const gain = (known / W) * (impurity(knownCounts, known) - childImp / known)
  return { f, gain, splitInfo: entropy(sizes, W), vw, known }
}

function evalNumeric(f, idx, w, ctx, W) {
  const { y, K, impurity, minLeaf, criterion } = ctx
  const order = []
  let missW = 0
  for (let i = 0; i < idx.length; i++) {
    if (Number.isNaN(f.values[idx[i]])) missW += w[i]
    else order.push(i)
  }
  if (order.length < 2) return null
  order.sort((a, b) => f.values[idx[a]] - f.values[idx[b]])

  const known = W - missW
  const total = new Float64Array(K)
  for (const i of order) total[y[idx[i]]] += w[i]
  const left = new Float64Array(K)
  const right = new Float64Array(K)
  let leftW = 0
  let distinct = 1
  let bestImp = Infinity
  let threshold = NaN
  let bestLeftW = 0
  for (let p = 0; p < order.length - 1; p++) {
    const i = order[p]
    left[y[idx[i]]] += w[i]
    leftW += w[i]
    const v = f.values[idx[i]]
    const next = f.values[idx[order[p + 1]]]
    if (v === next) continue
    distinct++
    const rightW = known - leftW
    if (leftW < minLeaf || rightW < minLeaf) continue
    for (let k = 0; k < K; k++) right[k] = total[k] - left[k]
    const imp = leftW * impurity(left, leftW) + rightW * impurity(right, rightW)
    if (imp < bestImp) {
      bestImp = imp
      threshold = (v + next) / 2
      bestLeftW = leftW
    }
  }
  if (Number.isNaN(threshold)) return null
  let gain = (known / W) * (impurity(total, known) - bestImp / known)
  // C4.5 release 8: penalise picking the best of many candidate thresholds.
  if (criterion !== 'gini') gain -= Math.log2(distinct - 1) / W
  const splitInfo = entropy([bestLeftW, known - bestLeftW, missW], W)
  return { f, gain, splitInfo, threshold, leftW: bestLeftW, known }
}

function chooseSplit(candidates, ctx) {
  if (candidates.length === 0) return null
  if (ctx.criterion !== 'gainRatio') {
    return candidates.reduce((a, b) => (b.gain > a.gain ? b : a))
  }
  // C4.5: best gain ratio among splits with at least average gain.
  const avg = candidates.reduce((s, c) => s + c.gain, 0) / candidates.length
  let best = null
  let bestRatio = -Infinity
  for (const c of candidates) {
    if (c.gain < avg - 1e-12) continue
    const ratio = c.splitInfo > 0 ? c.gain / c.splitInfo : 0
    if (ratio > bestRatio) {
      bestRatio = ratio
      best = c
    }
  }
  return best
}

function sampleFeatures(ctx) {
  const all = ctx.problem.candidates
  if (ctx.maxFeatures >= all.length) return all
  const pool = all.slice()
  for (let i = 0; i < ctx.maxFeatures; i++) {
    const j = i + Math.floor(ctx.random() * (pool.length - i))
    ;[pool[i], pool[j]] = [pool[j], pool[i]]
  }
  return pool.slice(0, ctx.maxFeatures)
}

function grow(ctx, idx, w, depth) {
  const { y, K } = ctx
  const counts = new Float64Array(K)
  let W = 0
  for (let i = 0; i < idx.length; i++) {
    counts[y[idx[i]]] += w[i]
    W += w[i]
  }
  const node = { n: W, raw: counts.map((c) => c / W) }
  const pure = counts.some((c) => c >= W - 1e-9)
  if (pure || depth >= ctx.maxDepth || W < 2 * ctx.minLeaf) return node

  const candidates = []
  for (const f of sampleFeatures(ctx)) {
    const c = f.numeric ? evalNumeric(f, idx, w, ctx, W) : evalCategorical(f, idx, w, ctx, W)
    if (c) candidates.push(c)
  }
  const best = chooseSplit(candidates, ctx)
  // Like ID3, an impure node may take a zero-gain categorical split (this is what
  // lets a tree learn XOR-style interactions); numeric splits must gain something.
  if (!best || best.gain < ctx.minGain || (best.f.numeric && best.gain <= 0)) return node

  ctx.importance[best.f.index] += Math.max(best.gain, 0) * W
  node.feature = best.f.index
  node.branches = []
  if (best.f.numeric) {
    node.threshold = best.threshold
    const frac = best.leftW / best.known
    const goesLeft = (v) => (Number.isNaN(v) ? null : v <= best.threshold)
    node.branches.push({ frac, node: growChild(ctx, idx, w, depth, (i) => goesLeft(best.f.values[i]) === true, frac) })
    node.branches.push({ frac: 1 - frac, node: growChild(ctx, idx, w, depth, (i) => goesLeft(best.f.values[i]) === false, 1 - frac) })
  } else {
    node.lookup = new Map()
    for (let v = 0; v < best.vw.length; v++) {
      if (best.vw[v] <= 0) continue
      const frac = best.vw[v] / best.known
      const branch = { code: v, frac, node: growChild(ctx, idx, w, depth, (i) => best.f.codes[i] === v, frac) }
      node.branches.push(branch)
      node.lookup.set(v, branch)
    }
  }
  return node

  function isMissingAt(i) {
    return best.f.numeric ? Number.isNaN(best.f.values[i]) : best.f.codes[i] < 0
  }

  // Rows whose split value is missing go down every branch with a fractional
  // weight proportional to that branch's share of the known rows (C4.5).
  function growChild(ctx, idx, w, depth, matches, frac) {
    const cIdx = []
    const cW = []
    for (let i = 0; i < idx.length; i++) {
      if (matches(idx[i])) {
        cIdx.push(idx[i])
        cW.push(w[i])
      } else if (isMissingAt(idx[i]) && w[i] * frac > MIN_WEIGHT) {
        cIdx.push(idx[i])
        cW.push(w[i] * frac)
      }
    }
    return grow(ctx, cIdx, cW, depth + 1)
  }
}

// Hierarchical shrinkage: each node's distribution is its parent's plus the
// change in raw frequency, damped by 1 / (1 + lambda / parentSamples). The
// result is a convex mix of the class frequencies along the root-to-leaf path,
// so leaves backed by few samples lean on their ancestors.
function shrink(node, parent, lambda) {
  if (!parent) node.p = node.raw
  else {
    const s = 1 / (1 + lambda / parent.n)
    node.p = parent.p.map((pp, k) => pp + (node.raw[k] - parent.raw[k]) * s)
  }
  if (node.branches) for (const b of node.branches) shrink(b.node, node, lambda)
}

function fitTree(problem, idx, w, opts, random) {
  const ctx = {
    problem,
    y: problem.y,
    K: problem.K,
    criterion: opts.criterion,
    impurity: opts.criterion === 'gini' ? gini : entropy,
    minLeaf: opts.minSamplesLeaf,
    minGain: opts.minGain,
    maxDepth: opts.maxDepth,
    maxFeatures: opts.maxFeaturesCount ?? Infinity,
    random,
    importance: new Float64Array(problem.attributes.length),
  }
  const root = grow(ctx, idx, w, 0)
  shrink(root, null, opts.shrinkage)
  return { root, importance: ctx.importance }
}

function predictDist(node, x, K) {
  if (!node.branches) return node.p
  const v = x[node.feature]
  if (node.threshold !== undefined) {
    if (!Number.isNaN(v)) return predictDist(node.branches[v <= node.threshold ? 0 : 1].node, x, K)
  } else if (v >= 0) {
    const b = node.lookup.get(v)
    if (b) return predictDist(b.node, x, K)
  }
  // Missing or never-seen value: blend every branch by its training share.
  const out = new Float64Array(K)
  for (const b of node.branches) {
    const d = predictDist(b.node, x, K)
    for (let k = 0; k < K; k++) out[k] += b.frac * d[k]
  }
  return out
}

// ---------------------------------------------------------------------------
// Public API

class Classifier {
  constructor(options = {}, defaults = DEFAULTS) {
    this.options = { ...DEFAULTS, ...defaults, ...stripUndefined(options) }
    this.attributes = [...(options.attributes ?? options.attribs ?? [])]
    this._target = options.target ?? options.label
    this.rows = []
    this._models = new Map()
  }

  /** Default attribute to predict: `options.target`, else the last attribute. */
  get target() {
    return this._target ?? this.attributes[this.attributes.length - 1]
  }

  /**
   * Add one example or many. A row is an array ordered like `attributes`, or an
   * object keyed by attribute name. Pass an array of rows to add several.
   */
  train(input) {
    if (Array.isArray(input) && input.length === 0) return this
    const many = Array.isArray(input) && input.length > 0 && input.every((r) => r !== null && typeof r === 'object')
    for (const row of many ? input : [input]) this.rows.push(this._toArray(row, true))
    this._models.clear()
    return this
  }

  /**
   * Predict the value of one attribute. The attribute predicted is
   * `opts.target` if given, else the default target if it is blank in `row`,
   * else the first blank attribute. Other blank values are treated as unknown.
   */
  classify(row, opts = {}) {
    const { x, t } = this._resolve(row, opts.target)
    const model = this._model(t)
    return this._format(t, model.problem, this._predict(model, x))
  }

  /** Normalised per-attribute importance (total impurity decrease). */
  importance(target) {
    const model = this._model(this._targetIndex(target ?? this.target))
    const total = model.importance.reduce((a, b) => a + b, 0) || 1
    const out = {}
    this.attributes.forEach((a, j) => {
      if (j !== model.problem.target) out[a] = model.importance[j] / total
    })
    return out
  }

  /** Accuracy and log loss on labelled rows (for `opts.target` or the default target). */
  evaluate(rows, opts = {}) {
    const target = opts.target ?? this.target
    const t = this._targetIndex(target)
    let correct = 0
    let logLoss = 0
    let n = 0
    for (const row of rows) {
      const arr = this._toArray(row)
      const truth = arr[t]
      if (isMissing(truth)) continue
      arr[t] = undefined
      const r = this.classify(arr, { target })
      n++
      if (keyOf(r.label) === keyOf(truth)) correct++
      logLoss -= Math.log(Math.max(r.probabilities[keyOf(truth)] ?? 0, 1e-15))
    }
    return { n, accuracy: n ? correct / n : NaN, logLoss: n ? logLoss / n : NaN }
  }

  toJSON() {
    const options = { ...this.options, attributes: this.attributes, target: this._target }
    delete options.attribs
    delete options.label
    return { ditify: 2, type: this.constructor.name, options, rows: this.rows }
  }

  static fromJSON(json) {
    const data = typeof json === 'string' ? JSON.parse(json) : json
    const model = new this(data.options)
    model.rows = data.rows.map((r) => r.slice())
    return model
  }

  _toArray(row, training = false) {
    if (Array.isArray(row)) {
      if (training && this.attributes.length === 0) throw new Error('ditify: set `attributes` to train with array rows')
      if (row.length > this.attributes.length) {
        throw new Error(`ditify: row has ${row.length} values but there are ${this.attributes.length} attributes`)
      }
      const out = row.slice()
      out.length = this.attributes.length
      return out
    }
    if (row !== null && typeof row === 'object') {
      if (training && this.attributes.length === 0) this.attributes = Object.keys(row)
      return this.attributes.map((a) => (Object.hasOwn(row, a) ? row[a] : undefined))
    }
    throw new TypeError('ditify: a row must be an array or an object')
  }

  _targetIndex(name) {
    const t = this.attributes.indexOf(name)
    if (t < 0) throw new Error(`ditify: unknown attribute "${name}"`)
    return t
  }

  _resolve(row, target) {
    const arr = this._toArray(row)
    if (target === undefined) {
      const blank = this.attributes.filter((_, j) => isMissing(arr[j]))
      target = blank.length === 0 || blank.includes(this.target) ? this.target : blank[0]
    }
    const t = this._targetIndex(target)
    const model = this._model(t)
    const x = model.problem.features.map((f, j) => (f ? encodeValue(f, arr[j]) : null))
    return { x, t }
  }

  _model(t) {
    let model = this._models.get(t)
    if (!model) {
      const problem = buildProblem(this.rows, this.attributes, t, this.options.numeric)
      const idx = []
      for (let i = 0; i < this.rows.length; i++) if (problem.y[i] >= 0) idx.push(i)
      if (idx.length === 0) throw new Error(`ditify: no training rows have a value for "${this.attributes[t]}"`)
      model = this._fit(problem, idx)
      model.problem = problem
      this._models.set(t, model)
    }
    return model
  }

  _format(t, problem, dist) {
    let best = 0
    for (let k = 1; k < problem.K; k++) if (dist[k] > dist[best]) best = k
    const probabilities = {}
    problem.classes.forEach((c, k) => (probabilities[keyOf(c)] = dist[k]))
    return { target: this.attributes[t], label: problem.classes[best], chance: dist[best], probabilities }
  }
}

/** A single, interpretable decision tree. */
export class Ditify extends Classifier {
  _fit(problem, idx) {
    return fitTree(problem, idx, idx.map(() => 1), this.options, Math.random)
  }

  _predict(model, x) {
    return predictDist(model.root, x, model.problem.K)
  }

  /** Like `classify`, plus the path of tests that led to the prediction. */
  explain(row, opts = {}) {
    const { x, t } = this._resolve(row, opts.target)
    const model = this._model(t)
    const { problem } = model
    const path = []
    let node = model.root
    while (node.branches) {
      const f = problem.features[node.feature]
      const attribute = this.attributes[node.feature]
      const v = x[node.feature]
      let next
      if (f.numeric && !Number.isNaN(v)) {
        const left = v <= node.threshold
        path.push({ attribute, test: `${left ? '<=' : '>'} ${round(node.threshold)}` })
        next = node.branches[left ? 0 : 1]
      } else if (!f.numeric && v >= 0) {
        next = node.lookup.get(v)
        if (next) path.push({ attribute, test: `= ${keyOf(f.cats[v])}` })
      }
      if (!next) {
        path.push({ attribute, test: 'unknown (averaged over all branches)' })
        break
      }
      node = next.node
    }
    const result = this._format(t, problem, predictDist(model.root, x, problem.K))
    const because = path.map((s) => `${s.attribute} ${s.test}`).join(' AND ')
    return { ...result, path, because }
  }

  /** The fitted tree as plain JSON, e.g. for visualisation. */
  tree(target) {
    const model = this._model(this._targetIndex(target ?? this.target))
    const { problem } = model
    const toPlain = (node) => {
      const { label, chance, probabilities } = this._format(problem.target, problem, node.p)
      const out = { samples: node.n, label, chance, probabilities }
      if (!node.branches) return out
      const f = problem.features[node.feature]
      out.attribute = this.attributes[node.feature]
      out.children = node.branches.map((b, i) => ({
        test: f.numeric ? `${i === 0 ? '<=' : '>'} ${round(node.threshold)}` : `= ${keyOf(f.cats[b.code])}`,
        ...(f.numeric ? { threshold: node.threshold } : { value: f.cats[b.code] }),
        node: toPlain(b.node),
      }))
      return out
    }
    return toPlain(model.root)
  }

  /** Every root-to-leaf path as a readable IF ... THEN ... rule. */
  rules(target) {
    const root = this.tree(target)
    const name = target ?? this.target
    const out = []
    const walk = (node, conds) => {
      if (!node.children) {
        const when = conds.length ? conds.join(' AND ') : 'always'
        out.push(`IF ${when} THEN ${name} = ${node.label} (${Math.round(node.chance * 100)}%, n=${round(node.samples)})`)
        return
      }
      for (const c of node.children) walk(c.node, [...conds, `${node.attribute} ${c.test}`])
    }
    walk(root, [])
    return out
  }
}

/** Bagged ensemble of randomised trees: usually more accurate, less readable. */
export class Forest extends Classifier {
  constructor(options = {}) {
    super(options, FOREST_DEFAULTS)
  }

  _fit(problem, idx) {
    const opts = { ...this.options, maxFeaturesCount: featureCount(this.options.maxFeatures, problem.candidates.length) }
    const random = rng(this.options.seed)
    const trees = []
    const importance = new Float64Array(problem.attributes.length)
    for (let t = 0; t < this.options.trees; t++) {
      // Bootstrap sample expressed as integer weights.
      const counts = new Map()
      for (let i = 0; i < idx.length; i++) {
        const r = idx[Math.floor(random() * idx.length)]
        counts.set(r, (counts.get(r) ?? 0) + 1)
      }
      const tree = fitTree(problem, [...counts.keys()], [...counts.values()], opts, random)
      trees.push(tree.root)
      tree.importance.forEach((v, j) => (importance[j] += v))
    }
    return { trees, importance }
  }

  _predict(model, x) {
    const K = model.problem.K
    const out = new Float64Array(K)
    for (const root of model.trees) {
      const d = predictDist(root, x, K)
      for (let k = 0; k < K; k++) out[k] += d[k] / model.trees.length
    }
    return out
  }
}

function featureCount(spec, d) {
  if (spec === 'sqrt') return Math.max(1, Math.round(Math.sqrt(d)))
  if (spec === 'log2') return Math.max(1, Math.round(Math.log2(d)))
  if (typeof spec === 'number' && spec > 0 && spec < 1) return Math.max(1, Math.round(spec * d))
  if (typeof spec === 'number' && spec >= 1) return Math.min(d, Math.floor(spec))
  return d
}

function stripUndefined(obj) {
  return Object.fromEntries(Object.entries(obj).filter(([, v]) => v !== undefined && v !== null))
}

const round = (v) => Math.round(v * 1000) / 1000

export default Ditify
