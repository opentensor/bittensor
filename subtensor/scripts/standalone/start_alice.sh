../../target/release/node-subtensor \
--base-path /tmp/alice \
--chain ../../config/chain_spec_raw.json \
--alice \
--port 30333 \
--ws-port 9945 \
--rpc-port 9933 \
--node-key 0000000000000000000000000000000000000000000000000000000000000001 \
--telemetry-url 'wss://telemetry.polkadot.io/submit/ 0' \
--rpc-methods=Unsafe \

