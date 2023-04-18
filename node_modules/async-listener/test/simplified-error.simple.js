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

if (!process.addAsyncListener) require('../index.js');

var assert = require('assert');
var fs = require('fs');
var addListener = process.addAsyncListener;
var removeListener = process.removeAsyncListener;

var caught = 0;
var expectCaught = 0;

var callbacksObj = {
  error: function(domain, er) {
    caught++;

    switch (er.message) {
      case 'fs - nested file does not exist':
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

var listener = process.createAsyncListener(callbacksObj);

// Nested FS
process.nextTick(function() {
  addListener(listener);

  setTimeout(function() {
    fs.stat('does not exist', function() {
      throw new Error('fs - nested file does not exist');
    });
    expectCaught++;
  });

  removeListener(listener);
});
