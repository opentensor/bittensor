if (!process.addAsyncListener) require('../index.js');

var test = require('tap').test;
var zlib = require('zlib');

// Convert semver string to number set
// TODO: This is *very* naive structure to check versions with,
// but it works well enough for now...
var nodeVersion = process.version.slice(1).split('.').map(Number)

var compressors = ['deflate', 'deflateRaw', 'gzip'];
var decompressors = ['inflate', 'inflateRaw', 'gunzip'];

compressors.forEach(function (method) {
  var name = 'zlib.' + method;
  var el = {
    name: name,
    children: [],
    before: 1,
    after: 1,
    error: 0
  };
  var list = [ el ];
  if (nodeVersion[0] >= 6) {
    list.push(el);
  }

  var children = [
    {
      name: name,
      // Compressors use streams internally,
      // so there's a bunch of nested stuff.
      children: [
        {
          name: name,
          children: [
            {
              name: name,
              children: list,
              before: 1,
              after: 1,
              error: 0
            }
          ],
          before: 1,
          after: 1,
          error: 0
        }
      ],
      before: 1,
      after: 1,
      error: 0
    }
  ]
  if (nodeVersion[0] >= 9) {
    children.unshift({
      name: name,
      children: [],
      before: 1,
      after: 1,
      error: 0
    })
  }

  test('test ' + name, function (t) {
    test_helper(t, function (listener, done) {
      listener.currentName = name;
      zlib[method](new Buffer('Goodbye World'), done);
    }, {
      name: 'root',
      children: children,
      before: 0,
      after: 0,
      error: 0
    });
  });
});

decompressors.forEach(function (method, i) {
  var preMethod = compressors[i];
  var name = 'zlib.' + method;
  var el = {
    name: name,
    children: [],
    before: 1,
    after: 1,
    error: 0
  };
  var list = [ el ];
  if (nodeVersion[0] >= 6) {
    list.push(el);
  }

  var children = [
    {
      name: name,
      // Compressors use streams internally,
      // so there's a bunch of nested stuff.
      children: [
        {
          name: name,
          children: [
            {
              name: name,
              children: list,
              before: 1,
              after: 1,
              error: 0
            }
          ],
          before: 1,
          after: 1,
          error: 0
        }
      ],
      before: 1,
      after: 1,
      error: 0
    }
  ]
  if (nodeVersion[0] >= 9) {
    children.unshift({
      name: name,
      children: [],
      before: 1,
      after: 1,
      error: 0
    })
  }

  test('test ' + name, function (t) {
    zlib[preMethod](new Buffer('Goodbye World'), function (err, buf) {
      t.ok(!err, 'should not have errored in preparation')
      test_helper(t, function (listener, done) {
        listener.currentName = name;
        zlib[method](buf, done);
      }, {
        name: 'root',
        children: children,
        before: 0,
        after: 0,
        error: 0
      });
    });
  });
});

function test_helper (t, run, expect) {
  // Trigger callback out-of-band from async listener
  var args;
  var interval = setInterval(function () {
    if (args) {
      clearInterval(interval);
      t.ok(!args[0], 'should not have errored');
      t.deepEqual(listener.root, expect);
      t.end();
    }
  }, 5);

  var listener = addListner();
  run(listener, function () {
    args = Array.prototype.slice.call(arguments);
  });
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
