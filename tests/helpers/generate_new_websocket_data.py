"""
This is a script whose purpose is to generate new stub data for given commands for use in the integration tests.

The integration tests rely on actual websocket sends/responses, which have been more-or-less manually entered.

The Async Substrate Interface package includes a raw websocket logger [logging.getLogger("raw_websocket")] which will
be used to gather this data. It is imperative that this script only uses the SubstrateInterface class, as sorting the
sent/received data from an asynchronous manner will be significantly more difficult to parse (though it's doable by
checking IDs if we ever would absolutely need to.

I'm writing the following parts before adding any code, and is mostly just my train of thought, and should be removed
before this makes it into the codebase:

 - received websocket responses begin with `WEBSOCKET_RECEIVE> `
 - sent websocket begin with `WEBSOCKET_SEND> `
 - both are stringified JSON
 - logging level is DEBUG
 - metadata and metadataV15 (metadata at version) must be discarded, or rather just dumped to their respective txt files
 - metadata/metadataV15 txt files are just the "result" portion of the response:
    e.g. `{"jsonrpc": "2.0", "id": _id, "result": METADATA}`
 - This script should NOT overwrite "retry_archive", as that uses a specific cycling mechanism to cycle between possible
    runtime responses
 - the data is structured as follows:
    seed_name: {
      rpc_method: {
        params: {
          response
        }
      }
    }

 - seed_name is basically just the name of the test that is being run. Specifying the seed name tells the FakeWebsocket
    which set of data to use
 - some workflows may specify the same parameters for a given RPC call, and expect different results. This is alleviated
    typically by specifying block hashes/numbers, though that does not always fit well into the tests.
 - the tests should be short and concise, testing just a single "thing" because of the possibility of conflicting rpc
    requests with the same params
 - all requests include {"json_rpc": "2.0"} in them — this can be removed for the sake of saving space, but is not
    imperative to do so
 - the websocket logger will include the id of the request: these should be stripped away, as they are dynamically
    created in SubstrateInterface, and then attached to the next response by FakeWebsocket
"""