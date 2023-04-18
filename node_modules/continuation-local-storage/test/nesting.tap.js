'use strict';

var tap = require('tap');
var test = tap.test;

var cls = require('../context.js');

test("nested contexts on a single namespace", function (t) {
  t.plan(7);

  var namespace = cls.createNamespace("namespace");
  namespace.run(function () {
    namespace.set("value", 1);

    t.equal(namespace.get("value"), 1,
            "namespaces have associated data even without contexts.");

    namespace.run(function () {
      t.equal(namespace.get("value"), 1, "lookup will check enclosing context");
      namespace.set("value", 2);
      t.equal(namespace.get("value"), 2, "setting works on top-level context");

      namespace.run(function () {
        t.equal(namespace.get("value"), 2, "lookup will check enclosing context");
        namespace.set("value", 3);
        t.equal(namespace.get("value"), 3, "setting works on nested context");
      });

      t.equal(namespace.get("value"), 2,
              "should revert to value set in top-level context");
    });

    t.equal(namespace.get("value"), 1, "namespace retains its outermost value.");
  });
});

test("the example from the docs", function (t) {
  var writer = cls.createNamespace('writer');
  writer.run(function () {
    writer.set('value', 0);

    t.equal(writer.get('value'), 0, "outer hasn't been entered yet");
    function requestHandler() {
      writer.run(function (outer) {
        t.equal(writer.active, outer, "writer.active == outer");

        writer.set('value', 1);
        t.equal(writer.get('value'), 1, "writer.active == outer");
        t.equal(outer.value, 1, "outer is active");

        process.nextTick(function () {
          t.equal(writer.active, outer, "writer.active == outer");
          t.equal(writer.get('value'), 1, "inner has been entered");
          writer.run(function (inner) {
            t.equal(writer.active, inner, "writer.active == inner");

            writer.set('value', 2);
            t.equal(outer.value, 1, "outer is unchanged");
            t.equal(inner.value, 2, "inner is active");
            t.equal(writer.get('value'), 2, "writer.active == inner");
          });
        });
      });

      setTimeout(function () {
        t.equal(writer.get('value'), 0, "writer.active == global");
        t.end();
      }, 100);
    }

    requestHandler();
  });
});
