'use strict';

var test = require('tap').test;

test("async listener lifecycle", function (t) {
  t.plan(8);

  if (!process.addAsyncListener) require('../index.js');

  t.ok(process.createAsyncListener, "can create async listeners");
  var counted = 0;
  var listener = process.createAsyncListener(
    {
      create : function () { counted++; },
      before : function () {},
      after  : function () {},
      error  : function () {}
    },
    Object.create(null)
  );

  t.ok(process.addAsyncListener, "can add async listeners");
  t.doesNotThrow(function () {
    listener = process.addAsyncListener(listener);
  }, "adding does not throw");

  t.ok(listener, "have a listener we can later remove");

  t.ok(process.removeAsyncListener, "can remove async listeners");
  t.doesNotThrow(function () {
    process.removeAsyncListener(listener);
  }, "removing does not throw");

  t.doesNotThrow(function () {
    process.removeAsyncListener(listener);
  }, "failing remove does not throw");

  t.equal(counted, 0, "didn't hit any async functions");
});
