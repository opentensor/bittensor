if (!process.addAsyncListener) require('../index.js');

var extend = require('util')._extend;
var test = require('tap').test;
var http = require('http');

// Convert semver string to number set
// TODO: This is *very* naive structure to check versions with,
// but it works well enough for now...
var nodeVersion = process.version.slice(1).split('.').map(Number)

test('http.Agent socket reuse works', function(t){
  function main (done) {
    var listener = addListner();
    var times = 2;

    var agent = new http.Agent({
      keepAlive: true,
      maxFreeSockets: 1,
      maxSockets: 1
    });

    function after(rand, i) {
      if (--times === 0) {
        t.deepEqual(
          listener.root,
          expected,
          'should have equal state structures'
        );
        if (agent.destroy) {
          agent.destroy();
        }
        done();
      }
    }

    function ping(i) {
      listener.currentName = 'ping #' + i + ' request';
      var addr = server.address();
      var req = http.request({
        agent: agent,
        port: addr.port,
        host: addr.host,
        path: '/sub'
      }, function (res) {
        // The second request is a logical continuation of
        // the first request, due to the http.Agent pooling
        if (i === 0) {
          t.equal(
            listener.current.name,
            'ping #' + i + ' request',
            'should be ping #' + i + ' request'
          );
        } else {
          t.equal(
            listener.current.name,
            'setImmediate to after #' + (i - 1),
            'should be setImmediate to after #' + (i - 1)
          );
        }

        listener.currentName = 'res.resume ping #' + i;
        const bufs = [];
        res.on('data', function (chunk) {
          bufs.push(chunk);
        });
        res.on('end', function () {
          const body = Buffer.concat(bufs).toString();
          t.equal('hello', body, 'should have body of "hello"')
          t.equal(
            listener.current.name,
            'res.resume ping #' + i,
            'should be res.resume ping #' + i
          );
          listener.currentName = 'setImmediate to after #' + i;
          setImmediate(after, i);
        });
      });
      listener.currentName = 'req.end ping #' + i;
      req.end();
    }

    for (var i = 0; i < times; i++) {
      listener.currentName = 'setImmediate #' + i;
      setImmediate(ping, i);
    }

    process.removeAsyncListener(listener.listener);

    //
    // NOTE: This expected structure building stuff is really complicated
    // because the interactions in node internals changed so much from 0.10
    // until now. It could be a lot simpler if we only cared about testing
    // the current stable, but this really needs to be tested back to 0.10.
    //
    // I'm sorry. :'(
    //
    function make (name, override) {
      return extend({
        name: name,
        children: [],
        before: 1,
        after: 1,
        error: 0
      }, override || {})
    }

    //
    // First ping branch
    //
    var innerResumeChildren = [];
    if (nodeVersion[0] < 8) {
      innerResumeChildren.push(make('res.resume ping #0'));
    }
    innerResumeChildren.push(make('setImmediate to after #0'));

    var innerResumeChildrenWrapped = [
      make('res.resume ping #0', {
        children: innerResumeChildren
      }),
      make('res.resume ping #0'),
      make('res.resume ping #0')
    ];
    var innerPingChildren = [];
    if (nodeVersion[0] == 0 && nodeVersion[1] < 12) {
      innerPingChildren.push(make('res.resume ping #0'));
    }
    innerPingChildren.push(make('res.resume ping #0', {
      children: nodeVersion[0] == 0 && nodeVersion[1] < 12
        ? innerResumeChildren
        : innerResumeChildrenWrapped
    }));
    if (nodeVersion[0] > 0 || nodeVersion[1] > 10) {
      if (nodeVersion[0] < 6 && nodeVersion[0] !== 4) {
        innerPingChildren.push(make('res.resume ping #0'));
      }
      innerPingChildren.push(
        make('res.resume ping #0', {
          children: [make('res.resume ping #0')]
        }),
        make('res.resume ping #0')
      );
    }

    var firstImmediateChildren = [
      make('ping #0 request', {
        children: [
          make('ping #0 request', {
            children: innerPingChildren
          }),
          make('ping #0 request', {
            children: nodeVersion[0] > 0 || nodeVersion[1] > 10
              ? [make('req.end ping #1')]
              : []
          })
        ]
      })
    ];

    if (nodeVersion[0] > 4) {
      firstImmediateChildren.push(make('ping #0 request'));
    };
    
    firstImmediateChildren.push(
      make('ping #0 request'),
      make('ping #0 request', {
        before: 0,
        after: 0
      })
    );

    var firstImmediate = make('setImmediate #0', {
      children: firstImmediateChildren
    });

    //
    // Second ping branch
    //
    var innerPingChildren = [];
    if (nodeVersion[0] < 8) {
      innerPingChildren.push(make('res.resume ping #1'));
    }

    innerPingChildren.push(make('setImmediate to after #1', {
      after: 0
    }));

    var innerPingChildrenWrapped = [
      make('res.resume ping #1', {
        children: innerPingChildren
      }),
      make('res.resume ping #1'),
      make('res.resume ping #1')
    ];
    var innerImmediateChildren = [];
    if (nodeVersion[0] == 0 && nodeVersion[1] < 12) {
      innerImmediateChildren.push(make('res.resume ping #1'));
    }
    innerImmediateChildren.push(make('res.resume ping #1', {
      children: nodeVersion[0] == 0 && nodeVersion[1] < 12
        ? innerPingChildren
        : innerPingChildrenWrapped
    }));
    if (nodeVersion[0] > 0 || nodeVersion[1] > 10) {
      if (nodeVersion[0] < 6 && nodeVersion[0] !== 4) {
        innerImmediateChildren.push(make('res.resume ping #1'));
      }
      innerImmediateChildren.push(
        make('res.resume ping #1', {
          children: [make('res.resume ping #1')]
        }),
        make('res.resume ping #1')
      );
    }

    var secondImmediate = make('setImmediate #1', {
      children: [
        make('ping #1 request', {
          children: [
            make('setImmediate to after #0', {
              children: innerImmediateChildren
            }),
            make('setImmediate to after #0', {
              children: [make('setImmediate to after #0')]
            })
          ]
        })
      ]
    });

    //
    // Make expected object
    //
    var expected = make('root', {
      children: [
        firstImmediate,
        secondImmediate
      ],
      before: 0,
      after: 0
    });
  }

  var server = http.createServer(function (req, res) {
    res.end('hello');
  });

  //
  // Test client
  //
  server.listen(function () {
    main(function () {
      server.close();
      server.on('close', function () {
        t.end();
      });
    });
  });
});

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
