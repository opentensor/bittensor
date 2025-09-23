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
 - metadata and metadatav15 (metadata at version) must be discarded, or rather just dumped to their respective txt files
 - metadata/metadatav15 txt files are just the "result" portion of the response:
    e.g. `{"jsonrpc": "2.0", "id": _id, "result": METADATA}`
"""