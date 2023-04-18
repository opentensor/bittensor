'use strict';

var tap             = require('tap')
  , test            = tap.test
  , createNamespace = require('../context.js').createNamespace
  ;

test("continuation-local state with promises", function (t) {
  t.plan(4);

  var namespace = createNamespace('namespace');
  namespace.run(function () {
    namespace.set('test', 0xabad1dea);

    t.test("chained promises", function (t) {
      if (!global.Promise) return t.end();

      namespace.run(function () {
        namespace.set('test', 31337);
        t.equal(namespace.get('test'), 31337, "state has been mutated");

        Promise.resolve()
          .then(function () {
            t.equal(namespace.get('test'), 31337,
                    "mutated state has persisted to first continuation");
          })
          .then(function () {
            t.equal(namespace.get('test'), 31337,
                    "mutated state has persisted to second continuation");
          })
          .then(function () {
            t.equal(namespace.get('test'), 31337,
                    "mutated state has persisted to third continuation");
            t.end();
          });
      });
    });

    t.test("chained unwrapped promises", function (t) {
      if (!global.Promise) return t.end();

      namespace.run(function () {
        namespace.set('test', 999);
        t.equal(namespace.get('test'), 999, "state has been mutated");

        Promise.resolve()
          .then(function () {
            t.equal(namespace.get('test'), 999,
                    "mutated state has persisted to first continuation");
            return Promise.resolve();
          })
          .then(function () {
            t.equal(namespace.get('test'), 999,
                    "mutated state has persisted to second continuation");
            return Promise.resolve();
          })
          .then(function () {
            t.equal(namespace.get('test'), 999,
                    "mutated state has persisted to third continuation");
            t.end();
          });
      });
    });

    t.test("nested promises", function (t) {
      if (!global.Promise) return t.end();

      namespace.run(function () {
        namespace.set('test', 54321);
        t.equal(namespace.get('test'), 54321, "state has been mutated");

        Promise.resolve()
          .then(function () {
            t.equal(namespace.get('test'), 54321,
              "mutated state has persisted to first continuation");

            Promise.resolve()
              .then(function () {
                t.equal(namespace.get('test'), 54321,
                  "mutated state has persisted to second continuation");

                Promise.resolve()
                  .then(function () {
                    t.equal(namespace.get('test'), 54321,
                      "mutated state has persisted to third continuation");
                    t.end();
                  });
              });
          });
      });
    });

    t.test("forked continuations", function (t) {
      if (!global.Promise) return t.end();

      namespace.run(function () {
        namespace.set('test', 10101);
        t.equal(namespace.get('test'), 10101, "state has been mutated");

        var promise = Promise.resolve();

        promise
          .then(function () {
            t.equal(namespace.get('test'), 10101,
              "mutated state has persisted to first continuation");
          });
        promise
          .then(function () {
            t.equal(namespace.get('test'), 10101,
              "mutated state has persisted to second continuation");
          });
        promise
          .then(function () {
            t.equal(namespace.get('test'), 10101,
              "mutated state has persisted to third continuation");
            t.end();
          });
      });
    });
  });
});
