ditify
======

`ditify` is a simple decision tree (ID3) based classification library written in JavaScript.

```javascript
var classifier = new ditify({
  attribs: [ "meal", "weather", "speed", "restaurant" ],
});

classifier.train([ "breakfast",  "hot",    "quick",      "subway" ]);
classifier.train([ "lunch",      "hot",    "medium",     "moumon's" ]);
classifier.train([ "lunch",      "rainy",  "leisurely",  "percy's" ]);
classifier.train([ "dinner",     "rainy",  "medium",     "dominos" ]);
classifier.train([ "breakfast",  "cold",   "quick",      "subway" ]);
classifier.train([ "lunch",      "cold",   "leisurely",  "megan's" ]);
classifier.train([ "dinner",     "cold",   "medium",     "dominos" ]);

classifier.classify([ "lunch", "rainy", "medium", "" ]); // {label: percy's, chance: 1}
```

##options
The constructor `ditify()` accepts an options hash whose properties can be defined as follows:

###attribs
The attribs property takes an array of strings containing all possible attribute names in the training data.

###label
The label property takes a string denotes the name of the attribute used for classification. If nothing is specified, the last string in the `attribs` array is set as the class attribute.
