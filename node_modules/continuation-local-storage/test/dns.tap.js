'use strict';

var dns             = require('dns')
  , tap             = require('tap')
  , test            = tap.test
  , createNamespace = require('../context.js').createNamespace
  ;

test("continuation-local state with MakeCallback and DNS module", function (t) {
  t.plan(11);

  var namespace = createNamespace('dns');
  namespace.run(function () {
    namespace.set('test', 0xabad1dea);

    t.test("dns.lookup", function (t) {
      namespace.run(function () {
        namespace.set('test', 808);
        t.equal(namespace.get('test'), 808, "state has been mutated");

        dns.lookup('www.newrelic.com', 4, function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 808,
                  "mutated state has persisted to dns.lookup's callback");

          t.end();
        });
      });
    });

    t.test("dns.resolve", function (t) {
      namespace.run(function () {
        namespace.set('test', 909);
        t.equal(namespace.get('test'), 909, "state has been mutated");

        dns.resolve('newrelic.com', 'NS', function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 909,
                  "mutated state has persisted to dns.resolve's callback");

          t.end();
        });
      });
    });

    t.test("dns.resolve4", function (t) {
      namespace.run(function () {
        namespace.set('test', 303);
        t.equal(namespace.get('test'), 303, "state has been mutated");

        dns.resolve4('www.newrelic.com', function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 303,
                  "mutated state has persisted to dns.resolve4's callback");

          t.end();
        });
      });
    });

    t.test("dns.resolve6", function (t) {
      namespace.run(function () {
        namespace.set('test', 101);
        t.equal(namespace.get('test'), 101, "state has been mutated");

        dns.resolve6('google.com', function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 101,
                  "mutated state has persisted to dns.resolve6's callback");

          t.end();
        });
      });
    });

    t.test("dns.resolveCname", function (t) {
      namespace.run(function () {
        namespace.set('test', 212);
        t.equal(namespace.get('test'), 212, "state has been mutated");

        dns.resolveCname('mail.newrelic.com', function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 212,
                  "mutated state has persisted to dns.resolveCname's callback");

          t.end();
        });
      });
    });

    t.test("dns.resolveMx", function (t) {
      namespace.run(function () {
        namespace.set('test', 707);
        t.equal(namespace.get('test'), 707, "state has been mutated");

        dns.resolveMx('newrelic.com', function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 707,
                  "mutated state has persisted to dns.resolveMx's callback");

          t.end();
        });
      });
    });

    t.test("dns.resolveNs", function (t) {
      namespace.run(function () {
        namespace.set('test', 717);
        t.equal(namespace.get('test'), 717, "state has been mutated");

        dns.resolveNs('newrelic.com', function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 717,
                  "mutated state has persisted to dns.resolveNs's callback");

          t.end();
        });
      });
    });

    t.test("dns.resolveTxt", function (t) {
      namespace.run(function () {
        namespace.set('test', 2020);
        t.equal(namespace.get('test'), 2020, "state has been mutated");

        dns.resolveTxt('newrelic.com', function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 2020,
                  "mutated state has persisted to dns.resolveTxt's callback");

          t.end();
        });
      });
    });

    t.test("dns.resolveSrv", function (t) {
      namespace.run(function () {
        namespace.set('test', 9000);
        t.equal(namespace.get('test'), 9000, "state has been mutated");

        dns.resolveSrv('_xmpp-server._tcp.google.com', function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 9000,
                  "mutated state has persisted to dns.resolveSrv's callback");

          t.end();
        });
      });
    });

    t.test("dns.resolveNaptr", function (t) {
      // dns.resolveNaptr only in Node > 0.9.x
      if (!dns.resolveNaptr) return t.end();

      namespace.run(function () {
        namespace.set('test', 'Polysix');
        t.equal(namespace.get('test'), 'Polysix', "state has been mutated");

        dns.resolveNaptr('columbia.edu', function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 'Polysix',
                  "mutated state has persisted to dns.resolveNaptr's callback");

          t.end();
        });
      });
    });

    t.test("dns.reverse", function (t) {
      namespace.run(function () {
        namespace.set('test', 1000);
        t.equal(namespace.get('test'), 1000, "state has been mutated");

        dns.reverse('204.93.223.144', function (err, addresses) {
          t.notOk(err, "lookup succeeded");
          t.ok(addresses.length > 0, "some results were found");

          t.equal(namespace.get('test'), 1000,
                  "mutated state has persisted to dns.reverse's callback");

          t.end();
        });
      });
    });
  });
});
