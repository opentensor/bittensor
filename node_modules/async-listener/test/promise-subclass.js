'use strict';

module.exports = (tap, customSpecies) => {
  if (customSpecies) {
    return class SubclassedPromise extends Promise {
      static get [Symbol.species]() { return Promise; }
      then(onSuccess, onReject) {
        tap.type(onSuccess, 'function');
        tap.type(onReject, 'undefined');
        return super.then(onSuccess, onReject);
      }
    };
  } else {
    return class SubclassedPromise extends Promise {
      then(onSuccess, onReject) {
        tap.type(onSuccess, 'function');
        tap.type(onReject, 'undefined');
        return super.then(onSuccess, onReject);
      }
    };
  }
};
