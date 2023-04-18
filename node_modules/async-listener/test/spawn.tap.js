var test   = require('tap').test
  , assert = require('assert')
  ;

if (!global.setImmediate) global.setImmediate = setTimeout;

if (!process.addAsyncListener) require('../index.js');

var childProcess = require('child_process')
  , exec         = childProcess.exec
  , execFile     = childProcess.execFile
  , spawn        = childProcess.spawn
  ;

test('ChildProcess', function (t) {
  t.plan(3);

  t.test('exec', function (t) {
    t.plan(3);

    var active
      , cntr   = 0
      ;

    process.addAsyncListener(
      {
        create : function () { return { val : ++cntr }; },
        before : function (context, data) { active = data.val; },
        after  : function () { active = null; }
      }
    );

    t.equal(active, undefined,
      'starts in initial context');
    process.nextTick(function () {
      t.equal(active, 1,
        'after tick: 1st context');
      var child = exec('node --version');
      child.on('exit', function (code) {
        t.ok(active >= 2,
          'after exec#exit: entered additional contexts');
      })
    });
  });

  t.test('execFile', function (t) {
    t.plan(3);

    var active
      , cntr   = 0
      ;

    process.addAsyncListener(
      {
        create : function () { return { val : ++cntr }; },
        before : function (context, data) { active = data.val; },
        after  : function () { active = null; }
      }
    );

    t.equal(active, undefined,
      'starts in initial context');
    process.nextTick(function () {
      t.equal(active, 1,
        'after nextTick: 1st context');
      execFile('node', ['--version'], function (err, code) {
        t.ok(active >= 2,
          'after execFile: entered additional contexts');
      });
    });
  });

  t.test('spawn', function (t) {
    t.plan(3);

    var active
      , cntr   = 0
      ;

    process.addAsyncListener(
      {
        create : function () { return { val : ++cntr }; },
        before : function (context, data) { active = data.val; },
        after  : function () { active = null; }
      }
    );

    t.equal(active, undefined,
      'starts in initial context');
    process.nextTick(function () {
      t.equal(active, 1,
        'after tick: 1st context');
      var child = spawn('node', ['--version']);
      child.on('exit', function (code) {
        t.ok(active >= 2,
          'after spawn#exit: entered additional contexts');
      })
    });
  });
});
