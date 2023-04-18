var test = require('tap').test;

if (!global.setImmediate) global.setImmediate = setTimeout;

test("asyncListeners work as expected with process.nextTick", function (t) {
  t.plan(1);

  if (!process.addAsyncListener) require('../index.js');

  console.log('+');
  // comment out this line to get the expected result:
  setImmediate(function () { console.log('!'); });

  var counter = 1;
  var current;
  process.addAsyncListener(
    {
      create : function listener() { return counter++; },
      before : function (_, domain) { current = domain; },
      after  : function () { current = null; }
    }
  );

  setImmediate(function () { t.equal(current, 1); });
  // uncomment this line to get the expected result:
  // process.removeAsyncListener(id);
});

