'use strict';

var tap             = require('tap')
  , test            = tap.test
  , createNamespace = require('../context.js').createNamespace
  ;

test("continuation-local state with timers", function (t) {
  t.plan(4);

  var namespace = createNamespace('namespace');
  namespace.run(function () {
    namespace.set('test', 0xabad1dea);

    t.test("process.nextTick", function (t) {
      namespace.run(function () {
        namespace.set('test', 31337);
        t.equal(namespace.get('test'), 31337, "state has been mutated");

        process.nextTick(function () {
          t.equal(namespace.get('test'), 31337,
                  "mutated state has persisted to process.nextTick's callback");

          t.end();
        });
      });
    });

    t.test("setImmediate", function (t) {
      // setImmediate only in Node > 0.9.x
      if (!global.setImmediate) return t.end();

      namespace.run(function () {
        namespace.set('test', 999);
        t.equal(namespace.get('test'), 999, "state has been mutated");

        setImmediate(function () {
          t.equal(namespace.get('test'), 999,
                  "mutated state has persisted to setImmediate's callback");

          t.end();
        });
      });
    });

    t.test("setTimeout", function (t) {
      namespace.run(function () {
        namespace.set('test', 54321);
        t.equal(namespace.get('test'), 54321, "state has been mutated");

        setTimeout(function () {
          t.equal(namespace.get('test'), 54321,
                  "mutated state has persisted to setTimeout's callback");

          t.end();
        });
      });
    });

    t.test("setInterval", function (t) {
      namespace.run(function () {
        namespace.set('test', 10101);
        t.equal(namespace.get('test'), 10101,
                "continuation-local state has been mutated");

        var ref = setInterval(function () {
          t.equal(namespace.get('test'), 10101,
                  "mutated state has persisted to setInterval's callback");

          clearInterval(ref);
          t.end();
        }, 20);
      });
    });
  });
});
