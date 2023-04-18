var test   = require('tap').test
  , assert = require('assert')
  ;

if (!global.setImmediate) global.setImmediate = setTimeout;

/**
 *
 *
 *
 *
 * SETUP AND BOILERPLATE
 *
 *
 *
 */
if (!process.addAsyncListener) require('../index.js');

/*
 * CLS code
 */
function Namespace () {
  this.active = Object.create(null);
  this._stack = [];
  this.id     = null;
}

Namespace.prototype.set = function (key, value) {
  this.active[key] = value;
  return value;
};

Namespace.prototype.get = function (key) {
  return this.active[key];
};

Namespace.prototype.enter = function (context) {
  assert.ok(context, "context must be provided for entering");

  this._stack.push(this.active);
  this.active = context;
};

Namespace.prototype.exit = function (context) {
  assert.ok(context, "context must be provided for exiting");

  if (this.active === context) {
    assert.ok(this._stack.length, "can't remove top context");
    this.active = this._stack.pop();
    return;
  }

  var index = this._stack.lastIndexOf(context);

  assert.ok(index >= 0, "context not currently entered; can't exit");
  assert.ok(index,      "can't remove top context");

  this.active = this._stack[index - 1];
  this._stack.length = index - 1;
};

Namespace.prototype.createContext = function () {
  return Object.create(this.active);
};

Namespace.prototype.run = function (fn) {
  var context = this.createContext();
  this.enter(context);
  try {
    fn(context);
    return context;
  }
  finally {
    this.exit(context);
  }
};

Namespace.prototype.bind = function (fn, context) {
  if (!context) context = this.active;
  var self = this;
  return function () {
    self.enter(context);
    try {
      return fn.apply(this, arguments);
    }
    finally {
      self.exit(context);
    }
  };
};

function create(name) {
  assert.ok(name, "namespace must be given a name!");

  var namespace = new Namespace(name);
  namespace.id = process.addAsyncListener(
    {
      create : function () { return namespace.active; },
      before : function (context, domain) { namespace.enter(domain); },
      after  : function (context, domain) { namespace.exit(domain); }
    }
  );

  return namespace;
}

/*
 * Transaction code
 */
var id = 1337;
function Transaction() { this.id = id++; }

/*
 * Tracer code
 */
var namespace = create("__NR_tracer");
function getTransaction() {
  var state = namespace.get('state');
  if (state) return state.transaction;
}

function transactionProxy(handler) {
  return function wrapTransactionInvocation() {
    var state   = {transaction : new Transaction()};

    var context = namespace.createContext();
    context.state = state;

    return namespace.bind(handler, context).apply(this, arguments);
  };
}


/**
 *
 *
 *
 *
 * TESTS
 *
 *
 *
 */

test("overlapping requests", function (t) {
  t.plan(2);

  t.test("simple overlap", function (t) {
    t.plan(3);

    setImmediate(function () { console.log('!'); });

    var n = create("test2");
    t.ok(!n.get('state'), "state should not yet be visible");

    n.run(function () {
      n.set('state', true);
      t.ok(n.get('state'), "state should be visible");

      setImmediate(function () { t.ok(n.get('state'), "state should be visible"); });
    });
  });

  t.test("two process.nextTicks", function (t) {
    t.plan(6);

    function handler(id) {
      var transaction = getTransaction();
      t.ok(transaction, "transaction should be visible");
      t.equal((transaction || {}).id, id, "transaction matches");
    }

    t.ok(!getTransaction(), "transaction should not yet be visible");

    var first;
    var proxied = transactionProxy(function () {
      t.ok(getTransaction(), "transaction should be visible");

      first = getTransaction().id;
      process.nextTick(function () { handler(first); }, 42);
    });
    proxied();

    process.nextTick(transactionProxy(function () {
      t.ok(getTransaction(), "transaction should be visible");

      var second = getTransaction().id;
      t.notEqual(first, second, "different transaction IDs");
      process.nextTick(function () { handler(second); }, 42);
    }), 42);
  });
});
