'use strict';

var fork = require('child_process').fork;
var test = require('tap').test;

var server

test("parent listener", function (t) {
  server = require('net').createServer();

  server.listen(8585, function () {
    t.ok(server, "parent listening on port 8585");

    var listener = fork(__dirname + '/fork-listener.js');
    t.ok(listener, "child process started");

    listener.on('message', function (message) {
      if (message === 'shutdown') {
        t.ok(message, "child handled error properly");
        listener.send('shutdown');
      }
      else {
        t.fail("parent got unexpected message " + message);
      }
      t.end();
    });
  });
});

test("tearDown", function (t) {
  server.close();
  t.end();
})
