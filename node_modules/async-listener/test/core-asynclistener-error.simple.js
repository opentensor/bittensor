// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

var PORT = 12346;

if (!process.addAsyncListener) require('../index.js');
if (!global.setImmediate) global.setImmediate = setTimeout;

var assert = require('assert');
var dns = require('dns');
var fs = require('fs');
var net = require('net');
var addListener = process.addAsyncListener;
var removeListener = process.removeAsyncListener;

var caught = 0;
var expectCaught = 0;

function asyncL() { }

var callbacksObj = {
  error: function(domain, er) {
    caught++;

    switch (er.message) {
      case 'sync throw':
      case 'setTimeout - simple':
      case 'setImmediate - simple':
      case 'setInterval - simple':
      case 'process.nextTick - simple':
      case 'setTimeout - nested':
      case 'process.nextTick - nested':
      case 'setImmediate - nested':
      case 'setTimeout2 - nested':
      case 'setInterval - nested':
      case 'fs - file does not exist':
      case 'fs - nested file does not exist':
      case 'fs - exists':
      case 'fs - realpath':
      case 'net - connection listener':
      case 'net - server listening':
      case 'net - client connect':
      case 'dns - lookup':
        return true;

      default:
        return false;
    }
  }
};

process.on('exit', function() {
  console.log('caught:', caught);
  console.log('expected:', expectCaught);
  assert.equal(caught, expectCaught, 'caught all expected errors');
  console.log('ok');
});

var listener = process.createAsyncListener(asyncL, callbacksObj);


// Catch synchronous throws
process.nextTick(function() {
  addListener(listener);

  expectCaught++;
  throw new Error('sync throw');

  removeListener(listener);
});


// Simple cases
process.nextTick(function() {
  addListener(listener);

  setTimeout(function() {
    throw new Error('setTimeout - simple');
  });
  expectCaught++;

  setImmediate(function() {
    throw new Error('setImmediate - simple');
  });
  expectCaught++;

  var b = setInterval(function() {
    clearInterval(b);
    throw new Error('setInterval - simple');
  });
  expectCaught++;

  process.nextTick(function() {
    throw new Error('process.nextTick - simple');
  });
  expectCaught++;

  removeListener(listener);
});


// Deeply nested
process.nextTick(function() {
  addListener(listener);

  setTimeout(function() {
    process.nextTick(function() {
      setImmediate(function() {
        var b = setInterval(function() {
          clearInterval(b);
          throw new Error('setInterval - nested');
        });
        expectCaught++;
        throw new Error('setImmediate - nested');
      });
      expectCaught++;
      throw new Error('process.nextTick - nested');
    });
    expectCaught++;
    setTimeout(function() {
      throw new Error('setTimeout2 - nested');
    });
    expectCaught++;
    throw new Error('setTimeout - nested');
  });
  expectCaught++;

  removeListener(listener);
});


// FS
process.nextTick(function() {
  addListener(listener);

  fs.stat('does not exist', function() {
    throw new Error('fs - file does not exist');
  });
  expectCaught++;

  fs.exists('hi all', function() {
    throw new Error('fs - exists');
  });
  expectCaught++;

  fs.realpath('/some/path', function() {
    throw new Error('fs - realpath');
  });
  expectCaught++;

  removeListener(listener);
});


// Nested FS
process.nextTick(function() {
  addListener(listener);

  setTimeout(function() {
    setImmediate(function() {
      var b = setInterval(function() {
        clearInterval(b);
        process.nextTick(function() {
          fs.stat('does not exist', function() {
            throw new Error('fs - nested file does not exist');
          });
          expectCaught++;
        });
      });
    });
  });

  removeListener(listener);
});


// Net
process.nextTick(function() {
  addListener(listener);

  var server = net.createServer(function() {
    server.close();
    throw new Error('net - connection listener');
  });
  expectCaught++;

  server.listen(PORT, function() {
    var client = net.connect(PORT, function() {
      client.end();
      throw new Error('net - client connect');
    });
    expectCaught++;
    throw new Error('net - server listening');
  });
  expectCaught++;

  removeListener(listener);
});


// DNS
process.nextTick(function() {
  addListener(listener);

  dns.lookup('localhost', function() {
    throw new Error('dns - lookup');
  });
  expectCaught++;

  removeListener(listener);
});
