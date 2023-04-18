'use strict';

var tap = require('tap');
var test = tap.test;

var context = require('../context.js');

test("namespace management", function (t) {
  t.plan(8);

  t.throws(function () { context.createNamespace(); }, "name is required");

  var namespace = context.createNamespace('test');
  t.ok(namespace, "namespace is returned upon creation");

  t.equal(context.getNamespace('test'), namespace, "namespace lookup works");

  t.doesNotThrow(function () { context.reset(); }, "allows resetting namespaces");

  t.equal(Object.keys(process.namespaces).length, 0, "namespaces have been reset");

  namespace = context.createNamespace('another');
  t.ok(process.namespaces.another, "namespace is available from global");

  t.doesNotThrow(function () { context.destroyNamespace('another'); },
                 "destroying works");

  t.notOk(process.namespaces.another, "namespace has been removed");
});
