# OpenCensus B3 Format Propagation for Node.js
[![Gitter chat][gitter-image]][gitter-url]

OpenCensus B3 Format Propagation sends a span context on the wire in an HTTP request, allowing other services to create spans with the right context.

This project is still at an early stage of development. It's subject to change.

## Installation

Install OpenCensus B3 Propagation with:
```bash
npm install @opencensus/propagation-b3
```

## Usage

To propagate span context arround services with B3 Propagation, pass an instance of B3 Propagation to your tracing instance. For Javascript:

```javascript
const tracing = require('@opencensus/nodejs');
const propagation = require('@opencensus/propagation-b3');

const b3 = new propagation.B3Format();

tracing.start({propagation: b3});
```

Similarly for Typescript:

```typescript
import * as tracing from '@opencensus/nodejs';
import { B3Format } from '@opencensus/propagation-b3';

const b3 = new B3Format();

tracing.start({propagation: b3});
```

## Useful links
- To know more about B3 Format propagation, visit: <https://github.com/openzipkin/b3-propagation>
- For more information on OpenCensus, visit: <https://opencensus.io/>
- To checkout the OpenCensus for Node.js, visit: <https://github.com/census-instrumentation/opencensus-node>
- For help or feedback on this project, join us on [gitter](https://gitter.im/census-instrumentation/Lobby)

[gitter-image]: https://badges.gitter.im/census-instrumentation/lobby.svg
[gitter-url]: https://gitter.im/census-instrumentation/lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
