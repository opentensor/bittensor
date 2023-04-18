var test = require('tap').test;

test("asyncListener preserves function length", function (t) {
  t.plan(2);

  var fsLengthsPre = computeValueLengths(require('fs'));
  var httpLengthsPre = computeValueLengths(require('http'));

  if (!process.addAsyncListener) require('../index.js');

  var fsLengthsPost = computeValueLengths(require('fs'));
  var httpLengthsPost = computeValueLengths(require('http'));

  t.same(fsLengthsPre, fsLengthsPost);
  t.same(httpLengthsPre, httpLengthsPost);
});

function computeValueLengths(o) {
  var lengths = [];
  for (var k in o) {
    lengths.push(o[k].length);
  }
  return lengths;
}
