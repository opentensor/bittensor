var test = require('tap').test;

if (!global.setImmediate) global.setImmediate = setTimeout;

test("after handler not run on throw", function (t) {
  t.plan(2);

  if (!process.addAsyncListener) require('../index.js');

  var key = process.createAsyncListener(
    {
      create : function () { return {}; },
      after  : function asyncAfter() { t.fail("after was called"); },
      error  : function asyncError(domain) { t.ok(domain, "got error"); }
    }
  );

  process.addAsyncListener(key);

  setImmediate(function () {
    throw new Error('whoops');
  });

  function handler(err) {
    process.removeAsyncListener(key);
    process.removeListener('uncaughtException', handler);
    t.ok(err, "error was propagated");
  }

  process.on('uncaughtException', handler);
});
