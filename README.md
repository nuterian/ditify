# ditify

A tiny, dependency-free decision tree classifier for JavaScript. Feed it rows,
leave a value blank, and it predicts that value, with a probability and an
explanation. A single ~600-line ES module that runs in Node and in the browser.

```js
import Ditify from 'ditify'

const classifier = new Ditify({ attributes: ['meal', 'weather', 'speed', 'restaurant'] })

classifier.train([
  ['breakfast', 'hot',   'quick',     'subway'],
  ['lunch',     'hot',   'medium',    "moumon's"],
  ['lunch',     'rainy', 'leisurely', "percy's"],
  ['dinner',    'rainy', 'medium',    'dominos'],
  ['breakfast', 'cold',  'quick',     'subway'],
  ['lunch',     'cold',  'leisurely', "megan's"],
  ['dinner',    'cold',  'medium',    'dominos'],
])

classifier.classify(['lunch', 'rainy', 'medium', ''])
// { target: 'restaurant', label: "percy's", chance: 0.81, probabilities: { "percy's": 0.81, ... } }

classifier.explain(['lunch', 'rainy', 'medium', '']).because
// "meal = lunch AND weather = rainy"

// Any column can be predicted: just leave it blank.
classifier.classify(['', 'cold', 'medium', 'dominos']).label // 'dinner'
```

## What it does

- **C4.5 trees.** Splits are chosen by gain ratio (or `entropy` for ID3, or
  `gini` for CART), so the order of the columns doesn't matter.
- **Numbers and categories.** Columns whose values are all numbers get binary
  threshold splits (`age <= 31.5`). Everything else is categorical.
- **Missing values.** `null`, `undefined`, `''` and `NaN` count as unknown, both
  in training and in queries. So does a category never seen in training. Unknown
  values are spread across branches in proportion to the training data, as C4.5
  does.
- **Calibrated probabilities.** Leaves with few samples are pulled toward their
  ancestors' class frequencies (*hierarchical shrinkage*, Agarwal et al.,
  ICML 2022). Without it, a leaf that has seen one example claims 100%.
- **Explanations.** `explain()` returns the decision path, `rules()` prints the
  whole tree as IF/THEN rules, `tree()` returns plain JSON for visualisation, and
  `importance()` ranks attributes.
- **Random forest.** `Forest` has the same API and is usually more accurate, at
  the cost of readability.
- **Fast enough to not think about.** On 10,000 rows × 6 attributes, a tree
  trains in about 0.2 s and predicts in a couple of microseconds per row.

## API

```js
import Ditify, { Forest } from 'ditify'
```

### `new Ditify(options)` / `new Forest(options)`

| option | default | |
|---|---|---|
| `attributes` | inferred from the first object row | Column names. `attribs` is accepted for v1 compatibility. |
| `target` | last attribute | Default column to predict. `label` is accepted for v1 compatibility. |
| `criterion` | `'gainRatio'` (tree), `'gini'` (forest) | `'gainRatio'`, `'entropy'` or `'gini'`. |
| `maxDepth` | `Infinity` | Stop growing below this depth. |
| `minSamplesLeaf` | `1` | Minimum samples in each branch of a split. |
| `minGain` | `0` | Minimum gain needed to split. |
| `shrinkage` | `1` | Hierarchical shrinkage strength λ. `0` gives raw leaf frequencies. Try 0–50. |
| `numeric` | `[]` | Attributes to parse as numbers even if they arrive as strings (e.g. from CSV). |
| `trees` | `100` | *Forest only.* Number of trees. |
| `maxFeatures` | `'sqrt'` | *Forest only.* Attributes considered per split: `'sqrt'`, `'log2'`, `'all'`, a count, or a fraction. |
| `seed` | `1` | *Forest only.* Random seed. The same seed gives the same forest. |

### Methods

- `train(row)` / `train(rows)`: add one example, or an array of them. A row is
  an array ordered like `attributes`, or an object keyed by attribute name.
  Training is lazy: the tree is (re)built on the next prediction.
- `classify(row, { target }?)`: returns `{ target, label, chance, probabilities }`.
  The attribute predicted is `target` if given. Otherwise it is the default
  target if that is blank in `row`, or else the first blank attribute.
- `explain(row, { target }?)` *(tree only)*: like `classify`, plus `path` and a
  readable `because`.
- `rules(target?)` *(tree only)*: an array of `IF … THEN …` strings.
- `tree(target?)` *(tree only)*: the fitted tree as nested plain objects
  (`attribute`, `children[].test`, `samples`, `label`, `chance`, `probabilities`).
- `importance(target?)`: `{ attribute: share }` of the total impurity decrease.
- `evaluate(rows, { target }?)`: `{ n, accuracy, logLoss }` on labelled rows.
- `toJSON()` / `Ditify.fromJSON(json)` / `Forest.fromJSON(json)`: save and
  restore a model, including its training data.

## Development

```sh
npm test        # node --test
npm run bench   # accuracy and speed on seeded synthetic data
```

Sample `npm run bench` output (5 seeds, 300 training rows, 2,000 test rows):

```
dataset                model                          accuracy  log loss
monk (categorical)     ID3 (entropy, raw leaves)      0.847     4.683
monk (categorical)     C4.5 + shrinkage (default)     0.850     0.481
monk (categorical)     Forest, 100 trees              0.898     0.402
circle (numeric)       ID3 (entropy, raw leaves)      0.881     1.200
circle (numeric)       C4.5 + shrinkage (default)     0.881     0.385
circle (numeric)       Forest, 100 trees              0.899     0.306
```

## Upgrading from v1

v1 was a browser global (`new ditify({ attribs })`). v2 is an ES module, but the
`attribs` and `label` options, `train(row)`, and the "leave one value blank"
`classify` all work as before. The result still has `label` and `chance`, plus
`target` and `probabilities`. Errors are now thrown as `Error` objects instead of
strings.

v1 also had a typo in its entropy sum that made every attribute score zero, so
it always split on the first column. It also crashed on object rows, returned
`undefined` for unseen values, and leaked globals. v2 fixes all of these.

## Roadmap

Ideas that would keep ditify small and local:

- **Gradient-boosted trees.** The strongest general-purpose method for tabular
  data (the XGBoost / LightGBM / CatBoost family). For categorical columns, use
  CatBoost-style ordered target statistics.
- **Hoeffding trees** (VFDT / Extremely Fast Decision Tree). These update with
  each `train()` call instead of rebuilding from scratch, which suits streaming
  use.
- **Optimal sparse trees** (GOSDT, MurTree). Provably optimal shallow trees for
  small categorical datasets, for when readability matters most.
- **Faster forest training.** Presort numeric columns once, and use histogram
  binning as LightGBM does.
- **TreeSHAP** for per-prediction attribution in forests.
