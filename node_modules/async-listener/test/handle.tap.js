if (!process.addAsyncListener) require('../index.js');

var test = require('tap').test;
var net = require('net');

test('synchronous errors during connect return a null _handle', function(t){
  t.plan(3);

  // listening server
  var server = net.createServer().listen(8000);

  // client
  var client = net.connect({port: 8000});

  client.on('connect', function(){
    t.ok(true, 'connect');
    // kill connection
    client.end();
  });

  client.on('error', function(){
    server.close();
    t.ok(true, 'done test');
  });

  client.on('end', function() {
    setTimeout(function(){
      // try to reconnect, but this has an error
      // rather than throw the right error, we're going to get an async-listener error
      t.ok(true, 'end');
      client.connect(8001);
    }, 100);
  });
});
