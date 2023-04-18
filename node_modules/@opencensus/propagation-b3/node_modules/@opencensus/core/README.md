# OpenCensus Core Node.js
[![Gitter chat][gitter-image]][gitter-url] ![Node Version][node-img] [![NPM Published Version][npm-img]][npm-url] ![dependencies Status][dependencies-status] ![devDependencies Status][devdependencies-status] ![Apache License][license-image]

OpenCensus for Node.js is an implementation of OpenCensus, a toolkit for collecting application performance and behavior monitoring data. It currently includes 3 apis: stats, tracing and tags.

The library is in alpha stage and the API is subject to change.

## Installation

Install the opencensus-core package with NPM:
```bash
npm install @opencensus/core
```

## Usage

#### Set up a new Stats manager instance.

To enable metrics, we’ll import a few items from OpenCensus Core package.

```javascript
const { Stats, MeasureUnit, AggregationType } = require('@opencensus/core');

// Create the Stats manager
const stats = new Stats();

// The latency in milliseconds
const mLatencyMs = stats.createMeasureDouble(
  "repl/latency",
  MeasureUnit.MS,
  "The latency in milliseconds"
);
```

#### Create Views and Tags:

We now determine how our metrics will be organized by creating ```Views```. We will also create the variable needed to add extra text meta-data to our metrics – ```methodTagKey```, ```statusTagKey```, and ```errorTagKey```.

```javascript
const methodTagKey = "method";
const statusTagKey = "status";
const errorTagKey = "error";

const latencyView = stats.createView(
  "demo/latency",
  mLatencyMs,
  AggregationType.DISTRIBUTION,
  [methodTagKey, statusTagKey, errorTagKey],
  "The distribution of the latencies",
  // Bucket Boundaries:
  // [>=0ms, >=25ms, >=50ms, >=75ms, >=100ms, >=200ms, >=400ms, >=600ms, >=800ms, >=1s, >=2s, >=4s, >=6s]
  [0, 25, 50, 75, 100, 200, 400, 600, 800, 1000, 2000, 4000, 6000]
);
```

#### Recording Metrics:

Now we will record the desired metrics. To do so, we will use ```stats.record()``` and pass in measurements.

```javascript
const [_, startNanoseconds] = process.hrtime();
const tags = {method: "repl", status: "OK"};

stats.record({
  measure: mLatencyMs,
  tags,
  value: sinceInMilliseconds(startNanoseconds)
});


function sinceInMilliseconds(startNanoseconds) {
  const [_, endNanoseconds] = process.hrtime();
  return (endNanoseconds - startNanoseconds) / 1e6;
}
```

See [Quickstart/Metrics](https://opencensus.io/quickstart/nodejs/metrics/) for a full example of registering and collecting metrics.

## Useful links
- For more information on OpenCensus, visit: <https://opencensus.io/>
- To checkout the OpenCensus for Node.js, visit: <https://github.com/census-instrumentation/opencensus-node>
- For help or feedback on this project, join us on [gitter](https://gitter.im/census-instrumentation/Lobby)

[gitter-image]: https://badges.gitter.im/census-instrumentation/lobby.svg
[gitter-url]: https://gitter.im/census-instrumentation/lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
[npm-url]: https://www.npmjs.com/package/@opencensus/core
[npm-img]: https://badge.fury.io/js/%40opencensus%2Fcore.svg
[node-img]: https://img.shields.io/node/v/@opencensus/core.svg
[license-image]: https://img.shields.io/badge/license-Apache_2.0-green.svg?style=flat
[dependencies-status]: https://david-dm.org/census-instrumentation/opencensus-node/status.svg?path=packages/opencensus-core
[devdependencies-status]:
https://david-dm.org/census-instrumentation/opencensus-node/dev-status.svg?path=packages/opencensus-core

## LICENSE

Apache License 2.0
