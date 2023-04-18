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
if (!global.setImmediate) global.setImmediate = setTimeout;

var assert = require('assert');

var once = 0;
function onAsync0() {}

var handlers = {
  after: function () {
    throw new Error('erk');
  },
  error: function () {
    // error handler must be called exactly *once*
    once++;

    return true;
  }
};

var key = process.addAsyncListener(onAsync0, handlers);

process.on('uncaughtException', function () {
  // process should propagate error regardless of
  // error handlers return value
  assert.equal(once, 1);
  console.log('ok');
});

setImmediate(function () {
  return 1;
});

process.removeAsyncListener(key);
