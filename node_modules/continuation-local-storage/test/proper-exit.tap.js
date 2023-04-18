'use strict';

// I love when a tap.plan() comes together
console.log('1..1');

process.on('uncaughtException', function (err) {
  if (err.message === 'oops') {
    console.log("ok got expected message: %s", err.message);
  }
  else {
    throw err;
  }
});

var cls = require('../context.js');
var ns = cls.createNamespace('x');
ns.run(function () { throw new Error('oops'); });
