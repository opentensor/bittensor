'use strict';

var tap             = require('tap')
  , semver          = require('semver')
  , test            = tap.test
  , createNamespace = require('../context.js').createNamespace
  ;

var crypto;
try { crypto = require('crypto'); }
catch (err) {}

if (crypto) {
  test("continuation-local state with crypto.randomBytes", function (t) {
    t.plan(1);

    var namespace = createNamespace('namespace');
    namespace.run(function () {
      namespace.set('test', 0xabad1dea);

      t.test("randomBytes", function (t) {
        namespace.run(function () {
          namespace.set('test', 42);
          crypto.randomBytes(100, function (err) {
            if (err) throw err;
            t.equal(namespace.get('test'), 42, "mutated state was preserved");
            t.end();
          });
        });
      });
    });
  });

  test("continuation-local state with crypto.pseudoRandomBytes", function (t) {
    t.plan(1);

    var namespace = createNamespace('namespace');
    namespace.run(function () {
      namespace.set('test', 0xabad1dea);

      t.test("pseudoRandomBytes", function (t) {
        namespace.run(function () {
          namespace.set('test', 42);
          crypto.pseudoRandomBytes(100, function (err) {
            if (err) throw err;
            t.equal(namespace.get('test'), 42, "mutated state was preserved");
            t.end();
          });
        });
      });
    });
  });

  test("continuation-local state with crypto.pbkdf2", function (t) {
    t.plan(1);

    var namespace = createNamespace('namespace');
    namespace.run(function () {
      namespace.set('test', 0xabad1dea);

      t.test("pbkdf2", function (t) {
        namespace.run(function () {
          namespace.set('test', 42);
          // this API changed after 0.10.0, and errors if digest is missing after v6
          if (semver.gte(process.version, "0.12.0")) {
            crypto.pbkdf2("s3cr3tz", "451243", 10, 40, "sha512", function (err) {
              if (err) throw err;
              t.equal(namespace.get('test'), 42, "mutated state was preserved");
              t.end();
            });
          } else {
            crypto.pbkdf2("s3cr3tz", "451243", 10, 40, function (err) {
              if (err) throw err;
              t.equal(namespace.get('test'), 42, "mutated state was preserved");
              t.end();
            });
          }
        });
      });
    });
  });
}
