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


if (!process.addAsyncListener) require('../../index.js');

var assert = require('assert');

function onAsync0() {}
function onAsync1() {}

var once = 0;
var handlers0 = {
  before: function (stor, err) {
    // should catch the error *once*
    once++;
  }
}

var handlers1 = {
  before: function (stor, err) {
    // handler was added in flight, and should not be called
    throw "Should Never Be Called";
  }
}

var key0 = process.addAsyncListener(onAsync0, handlers0);

process.on('exit', function (err) {
  // handlers0 before handler must be called once only
  assert.equal(once, 1);
  console.log('ok');
});

setImmediate(function () {
  1;
});

process.addAsyncListener(onAsync1, handlers1);
process.removeAsyncListener(key0);
