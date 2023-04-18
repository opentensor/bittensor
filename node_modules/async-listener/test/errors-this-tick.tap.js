'use strict';

var assert = require('assert');

if (!process.addAsyncListener) require('../index.js');

function MiniCLS() {
  this.active = Object.create(null);
  this._stack = [];
}

MiniCLS.prototype.enter = function (context) {
  assert.ok(context, "context must be provided for entering");

  this._stack.push(this.active);
  this.active = context;
};

MiniCLS.prototype.exit = function (context) {
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

MiniCLS.prototype.run = function (fn) {
  var context = Object.create(this.active);
  this.enter(context);
  try {
    fn(context);
    return context;
  }
  finally {
    this.exit(context);
  }
};

var cls = new MiniCLS();
process.addAsyncListener(
  {
    create : function () { return cls.active; },
    before : function (context, domain) { cls.enter(domain); },
    after  : function (context, domain) { cls.exit(domain); },
    error  : function (domain) { if (domain) cls.exit(domain); }
  }
);

process.on('uncaughtException', function (err) {
  if (err.message === 'oops') {
    console.log('ok got expected error: %s', err.message);
  }
  else {
    console.log('not ok got expected error: %s', err.stack);
  }
});

cls.run(function () {
  throw new Error('oops');
});
