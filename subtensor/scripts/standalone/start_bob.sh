../../target/release/node-subtensor \
--base-path /tmp/bob \
--chain ../../config/chain_spec_raw.json \
--bob \
--port 30334 \
--ws-port 9946 \
--rpc-port 9934 \
--node-key 0000000000000000000000000000000000000000000000000000000000000002 \
--bootnodes /ip4/127.0.0.1/tcp/30333/p2p/$1 \
--telemetry-url 'wss://telemetry.polkadot.io/submit/ 0' \
--rpc-methods=Unsafe \

