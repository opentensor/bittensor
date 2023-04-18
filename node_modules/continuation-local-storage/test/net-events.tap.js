'use strict';

var net             = require('net')
  , tap             = require('tap')
  , test            = tap.test
  , createNamespace = require('../context').createNamespace
  ;

test("continuation-local state with net connection", function (t) {
  t.plan(4);

  var namespace = createNamespace('net');
  namespace.run(function () {
    namespace.set('test', 0xabad1dea);

    var server;
    namespace.run(function () {
      namespace.set('test', 0x1337);

      server = net.createServer(function (socket) {
        t.equal(namespace.get('test'), 0x1337, "state has been mutated");
        socket.on("data", function () {
          t.equal(namespace.get('test'), 0x1337, "state is still preserved");
          server.close();
          socket.end("GoodBye");
        });
      });
      server.listen(function () {
        var address = server.address();
        namespace.run(function () {
          namespace.set("test", "MONKEY");
          var client = net.connect(address.port, function () {
            t.equal(namespace.get("test"), "MONKEY",
                    "state preserved for client connection");
            client.write("Hello");
            client.on("data", function () {
              t.equal(namespace.get("test"), "MONKEY", "state preserved for client data");
              t.end();
            });
          });
        });
      });
    });
  });
});
