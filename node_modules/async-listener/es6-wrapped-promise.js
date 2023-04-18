'use strict';

module.exports = (Promise, ensureAslWrapper) => {
  // Updates to this class should also be applied to the the ES3 version
  // in index.js.
  return class WrappedPromise extends Promise {
    constructor(executor) {
      var context, args;
      super(wrappedExecutor);
      var promise = this;

      try {
        executor.apply(context, args);
      } catch (err) {
        args[1](err);
      }

      return promise;
      function wrappedExecutor(resolve, reject) {
        context = this;
        args = [wrappedResolve, wrappedReject];

        // These wrappers create a function that can be passed a function and an argument to
        // call as a continuation from the resolve or reject.
        function wrappedResolve(val) {
          ensureAslWrapper(promise, false);
          return resolve(val);
        }

        function wrappedReject(val) {
          ensureAslWrapper(promise, false);
          return reject(val);
        }
      }
    }
  }
};
