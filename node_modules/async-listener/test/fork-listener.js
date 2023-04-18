'use strict';

var domain = require('domain');

if (!process.addAsyncListener) require('../index.js');

var d = domain.create();
d.on('error', function (error) {
  process.send(error.message);
});

process.on('message', function (message) {
  if (message === 'shutdown') {
    process.exit();
  }
  else {
    process.send("child got unexpected message " + message);
  }
});

d.run(function () {
  var server = require('net').createServer();

  server.on('error', function () {
    process.send('shutdown');
  });

  server.listen(8585, function () {
    process.send("child shouldn't be able to listen on port 8585");
  });
});
