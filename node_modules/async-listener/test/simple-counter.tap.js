var test = require('tap').test;

test("asyncListeners work as expected with process.nextTick", function (t) {
  t.plan(4);

  if (!process.addAsyncListener) require('../index.js');

  var active
    , cntr   = 0
    ;

  process.addAsyncListener(
    {
      create : function () { return { val : ++cntr }; },
      before : function (context, data) { active = data.val; },
      after  : function () { active = null; }
    }
  );

  process.nextTick(function () {
    t.equal(active, 1);
    process.nextTick(function () { t.equal(active, 3); });
  });

  process.nextTick(function () {
    t.equal(active, 2);
    process.nextTick(function () { t.equal(active, 4); });
  });
});
