'use strict';

var EventEmitter = require('events').EventEmitter
  , assert       = require('assert')
  , test         = require('tap').test
  , cls          = require('../context.js')
  ;

var nextID = 1;
function fresh(name) {
  assert.ok(!cls.getNamespace(name), "namespace " + name + " already exists");
  return cls.createNamespace(name);
}

function destroy(name) {
  return function destroyer(t) {
    cls.destroyNamespace(name);
    assert.ok(!cls.getNamespace(name), "namespace '" + name + "' should no longer exist");
    t.end();
  };
}

function runInTransaction(name, fn) {
  var namespace = cls.getNamespace(name);
  assert(namespace, "namespaces " + name + " doesn't exist");

  var context = namespace.createContext();
  context.transaction = ++nextID;
  process.nextTick(namespace.bind(fn, context));
}

test("asynchronous state propagation", function (t) {
  t.plan(24);

  t.test("a. async transaction with setTimeout", function (t) {
    t.plan(2);

    var namespace = fresh('a', this);

    function handler() {
      t.ok(namespace.get('transaction'), "transaction should be visible");
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('a', function () { setTimeout(handler, 100); });
  });

  t.test("a. cleanup", destroy('a'));

  t.test("b. async transaction with setInterval", function (t) {
    t.plan(4);

    var namespace = fresh('b', this)
      , count     = 0
      , handle
      ;

    function handler() {
      count += 1;
      if (count > 2) clearInterval(handle);
      t.ok(namespace.get('transaction'), "transaction should be visible");
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('b', function () { handle = setInterval(handler, 50); });
  });

  t.test("b. cleanup", destroy('b'));

  t.test("c. async transaction with process.nextTick", function (t) {
    t.plan(2);

    var namespace = fresh('c', this);

    function handler() {
      t.ok(namespace.get('transaction'), "transaction should be visible");
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('c', function () { process.nextTick(handler); });
  });

  t.test("c. cleanup", destroy('c'));

  t.test("d. async transaction with EventEmitter.emit", function (t) {
    t.plan(2);

    var namespace = fresh('d', this)
      , ee        = new EventEmitter()
      ;

    function handler() {
      t.ok(namespace.get('transaction'), "transaction should be visible");
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('d', function () {
      ee.on('transaction', handler);
      ee.emit('transaction');
    });
  });

  t.test("d. cleanup", destroy('d'));

  t.test("e. two overlapping async transactions with setTimeout", function (t) {
    t.plan(6);

    var namespace = fresh('e', this)
      , first
      , second
      ;

    function handler(id) {
      t.ok(namespace.get('transaction'), "transaction should be visible");
      t.equal(namespace.get('transaction'), id, "transaction matches");
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('e', function () {
      first = namespace.get('transaction');
      setTimeout(handler.bind(null, first), 100);
    });

    setTimeout(function () {
      runInTransaction('e', function () {
        second = namespace.get('transaction');
        t.notEqual(first, second, "different transaction IDs");
        setTimeout(handler.bind(null, second), 100);
      });
    }, 25);
  });

  t.test("e. cleanup", destroy('e'));

  t.test("f. two overlapping async transactions with setInterval", function (t) {
    t.plan(15);

    var namespace = fresh('f', this);

    function runInterval() {
      var count = 0
        , handle
        , id
        ;

      function handler() {
        count += 1;
        if (count > 2) clearInterval(handle);
        t.ok(namespace.get('transaction'), "transaction should be visible");
        t.equal(id, namespace.get('transaction'), "transaction ID should be immutable");
      }

      function run() {
        t.ok(namespace.get('transaction'), "transaction should have been created");
        id = namespace.get('transaction');
        handle = setInterval(handler, 50);
      }

      runInTransaction('f', run);
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInterval(); runInterval();
  });

  t.test("f. cleanup", destroy('f'));

  t.test("g. two overlapping async transactions with process.nextTick", function (t) {
    t.plan(6);

    var namespace = fresh('g', this)
      , first
      , second
      ;

    function handler(id) {
      var transaction = namespace.get('transaction');
      t.ok(transaction, "transaction should be visible");
      t.equal(transaction, id, "transaction matches");
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('g', function () {
      first = namespace.get('transaction');
      process.nextTick(handler.bind(null, first));
    });

    process.nextTick(function () {
      runInTransaction('g', function () {
        second = namespace.get('transaction');
        t.notEqual(first, second, "different transaction IDs");
        process.nextTick(handler.bind(null, second));
      });
    });
  });

  t.test("g. cleanup", destroy('g'));

  t.test("h. two overlapping async runs with EventEmitter.prototype.emit", function (t) {
    t.plan(3);

    var namespace = fresh('h', this)
      , ee        = new EventEmitter()
      ;

    function handler() {
      t.ok(namespace.get('transaction'), "transaction should be visible");
    }

    function lifecycle() {
      ee.once('transaction', process.nextTick.bind(process, handler));
      ee.emit('transaction');
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('h', lifecycle);
    runInTransaction('h', lifecycle);
  });

  t.test("h. cleanup", destroy('h'));

  t.test("i. async transaction with an async sub-call with setTimeout", function (t) {
    t.plan(5);

    var namespace = fresh('i', this);

    function inner(callback) {
      setTimeout(function () {
        t.ok(namespace.get('transaction'), "transaction should (yep) still be visible");
        callback();
      }, 50);
    }

    function outer() {
      t.ok(namespace.get('transaction'), "transaction should be visible");
      setTimeout(function () {
        t.ok(namespace.get('transaction'), "transaction should still be visible");
        inner(function () {
          t.ok(namespace.get('transaction'), "transaction should even still be visible");
        });
      }, 50);
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('i', setTimeout.bind(null, outer, 50));
  });

  t.test("i. cleanup", destroy('i'));

  t.test("j. async transaction with an async sub-call with setInterval", function (t) {
    t.plan(5);

    var namespace = fresh('j', this)
      , outerHandle
      , innerHandle
      ;

    function inner(callback) {
      innerHandle = setInterval(function () {
        clearInterval(innerHandle);
        t.ok(namespace.get('transaction'), "transaction should (yep) still be visible");
        callback();
      }, 50);
    }

    function outer() {
      t.ok(namespace.get('transaction'), "transaction should be visible");
      outerHandle = setInterval(function () {
        clearInterval(outerHandle);
        t.ok(namespace.get('transaction'), "transaction should still be visible");
        inner(function () {
          t.ok(namespace.get('transaction'), "transaction should even still be visible");
        });
      }, 50);
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('j', outer);
  });

  t.test("j. cleanup", destroy('j'));

  t.test("k. async transaction with an async call with process.nextTick", function (t) {
    t.plan(5);

    var namespace = fresh('k', this);

    function inner(callback) {
      process.nextTick(function () {
        t.ok(namespace.get('transaction'), "transaction should (yep) still be visible");
        callback();
      });
    }

    function outer() {
      t.ok(namespace.get('transaction'), "transaction should be visible");
      process.nextTick(function () {
        t.ok(namespace.get('transaction'), "transaction should still be visible");
        inner(function () {
          t.ok(namespace.get('transaction'), "transaction should even still be visible");
        });
      });
    }

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('k', function () { process.nextTick(outer); });
  });

  t.test("k. cleanup", destroy('k'));

  t.test("l. async transaction with an async call with EventEmitter.emit", function (t) {
    t.plan(4);

    var namespace = fresh('l', this)
      , outer     = new EventEmitter()
      , inner     = new EventEmitter()
      ;

    inner.on('pong', function (callback) {
      t.ok(namespace.get('transaction'), "transaction should still be visible");
      callback();
    });

    function outerCallback() {
      t.ok(namespace.get('transaction'), "transaction should even still be visible");
    }

    outer.on('ping', function () {
      t.ok(namespace.get('transaction'), "transaction should be visible");
      inner.emit('pong', outerCallback);
    });

    t.notOk(namespace.get('transaction'), "transaction should not yet be visible");
    runInTransaction('l', outer.emit.bind(outer, 'ping'));
  });

  t.test("l. cleanup", destroy('l'));
});
