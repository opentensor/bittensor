'use strict';

var test   = require('tap').test
  , cls    = require('../context.js')
  , domain = require('domain')
  ;

test("continuation-local storage glue with a throw in the continuation chain",
     function (t) {
  var namespace = cls.createNamespace('test');
  namespace.run(function () {
    var d = domain.create();
    namespace.set('outer', true);
    namespace.bindEmitter(d);

    d.on('error', function (blerg) {
      t.equal(blerg.message, "explicitly nonlocal exit", "got the expected exception");
      t.ok(namespace.get('outer'), "outer context is still active");
      t.notOk(namespace.get('inner'), "inner context should have been exited by throw");
      t.equal(namespace._set.length, 1, "should be back to outer state");

      cls.destroyNamespace('test');
      t.end();
    });

    // tap is only trying to help
    process.nextTick(d.bind(function () {
      t.ok(namespace.get('outer'), "outer mutation worked");
      t.notOk(namespace.get('inner'), "inner mutation hasn't happened yet");

      namespace.run(function () {
        namespace.set('inner', true);
        throw new Error("explicitly nonlocal exit");
      });
    }));
  });
});

test("synchronous throw attaches the context", function (t) {
  t.plan(3);

  var namespace = cls.createNamespace('cls@synchronous');
  namespace.run(function () {
    namespace.set('value', 'transaction clear');
    try {
      namespace.run(function () {
        namespace.set('value', 'transaction set');
        throw new Error('cls@synchronous explosion');
      });
    }
    catch (e) {
      t.ok(namespace.fromException(e), "context was attached to error");
      t.equal(namespace.fromException(e)['value'], 'transaction set',
              "found the inner value");
    }

    t.equal(namespace.get('value'), 'transaction clear', "everything was reset");
  });

  cls.destroyNamespace('cls@synchronous');
});

test("synchronous throw checks if error exists", function (t) {
  t.plan(2);

  var namespace = cls.createNamespace('cls@synchronous-null-error');
  namespace.run(function () {
    namespace.set('value', 'transaction clear');
    try {
      namespace.run(function () {
        namespace.set('value', 'transaction set');
        throw null;
      });
    }
    catch (e) {
      // as we had a null error, cls couldn't set the new inner value
      t.equal(namespace.get('value'), 'transaction clear', 'from outer value');
    }

    t.equal(namespace.get('value'), 'transaction clear', "everything was reset");
  });

  cls.destroyNamespace('cls@synchronous-null-error');
});

test("throw in process.nextTick attaches the context", function (t) {
  t.plan(3);

  var namespace = cls.createNamespace('cls@nexttick');
  var d = domain.create();
  namespace.bindEmitter(d);

  d.on('error', function (e) {
    t.ok(namespace.fromException(e), "context was attached to error");
    t.equal(namespace.fromException(e)['value'], 'transaction set',
            "found the inner value");

    cls.destroyNamespace('cls@nexttick');
  });

  namespace.run(function () {
    namespace.set('value', 'transaction clear');

    // tap is only trying to help
    process.nextTick(d.bind(function () {
      namespace.run(function () {
        namespace.set('value', 'transaction set');
        throw new Error("cls@nexttick explosion");
      });
    }));

    t.equal(namespace.get('value'), 'transaction clear', "everything was reset");
  });
});

test("throw in setTimeout attaches the context", function (t) {
  t.plan(3);

  var namespace = cls.createNamespace('cls@setTimeout');
  var d = domain.create();
  namespace.bindEmitter(d);

  d.on('error', function (e) {
    t.ok(namespace.fromException(e), "context was attached to error");
    t.equal(namespace.fromException(e)['value'], 'transaction set',
            "found the inner value");

    cls.destroyNamespace('cls@setTimeout');
  });

  namespace.run(function () {
    namespace.set('value', 'transaction clear');

    // tap is only trying to help
    setTimeout(d.bind(function () {
      namespace.run(function () {
        namespace.set('value', 'transaction set');
        throw new Error("cls@setTimeout explosion");
      });
    }));

    t.equal(namespace.get('value'), 'transaction clear', "everything was reset");
  });
});
