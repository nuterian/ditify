# ditify

**Leave a blank. Get an answer, a confidence, and a reason.**

ditify is a tiny decision-tree classifier for JavaScript. It's one file with no
dependencies, and it runs in Node or the browser.

```js
import Ditify from 'ditify'

const lunch = new Ditify({ attributes: ['meal', 'weather', 'speed', 'restaurant'] })

lunch.train([
  ['breakfast', 'hot',   'quick',     'subway'],
  ['lunch',     'hot',   'medium',    "moumon's"],
  ['lunch',     'rainy', 'leisurely', "percy's"],
  ['dinner',    'rainy', 'medium',    'dominos'],
  ['breakfast', 'cold',  'quick',     'subway'],
  ['lunch',     'cold',  'leisurely', "megan's"],
  ['dinner',    'cold',  'medium',    'dominos'],
])

lunch.classify(['lunch', 'rainy', 'medium', ''])
// → { label: "percy's", chance: 0.81, probabilities: {...} }

lunch.explain(['lunch', 'rainy', 'medium', '']).because
// → "meal = lunch AND weather = rainy"

lunch.classify(['', 'cold', 'medium', 'dominos']).label
// → 'dinner'   (any column can be the question)
```

## Why it's nice

- 🕳️ **Blank = question.** Whichever value you leave out is the one it predicts.
- 🧾 **It shows its work.** `explain()` gives the path, `rules()` prints the
  tree as IF/THEN rules, `tree()` returns JSON you can draw, and `importance()`
  ranks the columns.
- 🎯 **Honest confidence.** A leaf trained on one example doesn't claim 100%.
  Small leaves lean on their ancestors (hierarchical shrinkage, ICML 2022).
- 🔢 **Numbers and gaps are fine.** Numeric columns split on thresholds
  (`age <= 31.5`). Missing or never-seen values are averaged across branches
  rather than crashing. ID-like columns (every value different) are ignored.
- 🌲 **Swap in a forest** when you care more about accuracy than reading the
  tree: `new Forest(...)` has the same API.
- ⚡ **Quick.** On 10,000 rows, a tree trains in about 0.2 s and answers in
  microseconds.

## API

```js
import Ditify, { Forest } from 'ditify'

new Ditify({ attributes, target, criterion, maxDepth, minSamplesLeaf, minGain, shrinkage, numeric })
new Forest({ ...same, trees: 100, maxFeatures: 'sqrt', seed: 1 })
```

| method | returns |
|---|---|
| `train(row \| rows)` | adds examples. Rows can be arrays or `{ name: value }` objects. |
| `classify(row, { target }?)` | `{ target, label, chance, probabilities }` |
| `explain(row)` | the same, plus `path` and `because` *(tree only)* |
| `rules()` / `tree()` | IF/THEN strings / nested JSON *(tree only)* |
| `importance()` | `{ column: share }` |
| `evaluate(rows)` | `{ n, accuracy, logLoss }` |
| `toJSON()` / `Ditify.fromJSON()` | save and restore |

<details><summary>Options</summary>

| option | default | |
|---|---|---|
| `attributes` | keys of the first object row | Column names. |
| `target` | last column | What to predict when the row doesn't make it obvious. |
| `criterion` | `gainRatio` (tree), `gini` (forest) | Also `entropy` (classic ID3). |
| `maxDepth`, `minSamplesLeaf`, `minGain` | `∞`, `1`, `0` | Pre-pruning. |
| `shrinkage` | `1` | `0` gives raw leaf frequencies. Higher values give more cautious probabilities. |
| `numeric` | `[]` | Columns to parse as numbers (useful for CSV strings). |
| `trees`, `maxFeatures`, `seed` | `100`, `'sqrt'`, `1` | Forest only. |

</details>

## How good is it?

`npm run bench` runs 5 seeds, training on 300 rows and testing on 2,000:

| | categorical (MONK-1) | numeric (circle) |
|---|---|---|
| plain ID3 | 84.7%, log loss 4.68 | 88.1%, log loss 1.20 |
| **ditify tree** | **85.0%, log loss 0.48** | **88.1%, log loss 0.39** |
| **ditify forest** | **89.8%, log loss 0.40** | **89.9%, log loss 0.31** |

Shrinkage leaves accuracy about the same and cuts log loss roughly tenfold. In
other words, the probabilities become believable.

## See it

**[Try the playground →](http://nuterian.github.io/DecisionTree)**
It has 1,000-row datasets, a live tree you can click, knobs that retrain as you
drag, and a 100-tree forest trained in your browser.
([source](https://github.com/nuterian/DecisionTree))

## What's next

- **Gradient boosting** (as in XGBoost and LightGBM): the most accurate approach
  for tabular data.
- **Hoeffding trees:** learn from each `train()` call instead of rebuilding.
- **Optimal sparse trees** (GOSDT, MurTree): the smallest tree that's still
  right.
- **Faster forest training:** presorting and histogram bins.

## Development

```sh
npm test       # node --test
npm run bench  # accuracy and speed
```

MIT © Jugal Manjeshwar
