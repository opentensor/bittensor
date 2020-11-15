../../target/release/node-subtensor \
--base-path /tmp/charlie \
--chain ../../config/chain_spec_raw.json \
--charlie \
--port 30335 \
--ws-port 9947 \
--rpc-port 9935 \
--node-key 0000000000000000000000000000000000000000000000000000000000000003 \
--bootnodes /ip4/127.0.0.1/tcp/30333/p2p/$1 \
--telemetry-url 'wss://telemetry.polkadot.io/submit/ 0' \
--rpc-methods=Unsafe \

