if (!process.addAsyncListener) require('../index.js');

var test = require('tap').test;
var net = require('net');

test('test process.nextTick', function (t) {
  test_helper(t, function (listener, done) {
    listener.currentName = 'process.nextTick';
    process.nextTick(done);
  }, {
    name: 'root',
    children: [
      {
        name: 'process.nextTick',
        children: [],
        before: 1,
        after: 1,
        error: 0
      }
    ],
    before: 0,
    after: 0,
    error: 0
  });
});

test('test setTimeout', function (t) {
  test_helper(t, function (listener, done) {
    listener.currentName = 'setTimeout';
    setTimeout(done, 1);
  }, {
    name: 'root',
    children: [
      {
        name: 'setTimeout',
        children: [],
        before: 1,
        after: 1,
        error: 0
      }
    ],
    before: 0,
    after: 0,
    error: 0
  });
});

test('test setImmediate', function (t) {
  test_helper(t, function (listener, done) {
    listener.currentName = 'setImmediate';
    setImmediate(done);
  }, {
    name: 'root',
    children: [
      {
        name: 'setImmediate',
        children: [],
        before: 1,
        after: 1,
        error: 0
      }
    ],
    before: 0,
    after: 0,
    error: 0
  });
});

test('test setInterval', function (t) {
  test_helper(t, function (listener, done) {
    listener.currentName = 'setInterval';
    var count = 0;
    var interval = setInterval(function () {
      if (++count === 2) {
        clearInterval(interval);
        done();
      }
    });
  }, {
    name: 'root',
    children: [
      {
        name: 'setInterval',
        children: [],
        before: 2,
        after: 2,
        error: 0
      }
    ],
    before: 0,
    after: 0,
    error: 0
  });
});

function test_helper (t, run, expect) {
  // Trigger callback out-of-band from async listener
  var done;
  var interval = setInterval(function () {
    if (done) {
      clearInterval(interval);
      t.deepEqual(listener.root, expect);
      t.end();
    }
  }, 5);

  var listener = addListner();
  run(listener, function () { done = true; });
  process.removeAsyncListener(listener.listener);
}

function addListner() {
  var listener = process.addAsyncListener({
    create: create,
    before: before,
    after: after,
    error: error
  });


  var state = {
    listener: listener,
    currentName: 'root'
  };

  state.root = create();
  state.current = state.root;

  return state;

  function create () {
    var node = {
      name: state.currentName,
      children: [],
      before: 0,
      after: 0,
      error: 0
    };

    if(state.current) state.current.children.push(node);
    return node;
  }

  function before(ctx, node) {
    state.current = node;
    state.current.before++;
  }

  function after(ctx, node) {
    node.after++;
    state.current = null;
  }

  function error(ctx, node) {
    node.error++;
    state.current = null;
    return false;
  }
}
