# Changelog

## 5.0.0 / 2023-05-17

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v4.0.1...v5.0.0


## 4.0.1 / 2023-04-21

* Fix btcli my_delegates bug by @camfairchild in ef32a4da0d0827ab5977af1454d66ffe97cbc572
* Fix endpoint protocol check bug by @camfairchild and @Eugene-hu in https://github.com/opentensor/bittensor/pull/1296
* Fix changelog script and perms by @camfairchild in f5e7f1e9e9717d229fdec6875fdb9a3051c4bd6b and 1aed09a162ef0fe4d9def2faf261b15dc4c1fa8d

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v4.0.0...v4.0.1


## 4.0.0 / 2023-04-20

## What's Changed
* add mnrv-ai to delegates.json by @SFuller4 in https://github.com/opentensor/bittensor/pull/1226
* Update delegates list by @adriansmares in https://github.com/opentensor/bittensor/pull/1225
* Update delegates.json by @whiterhinoTAO in https://github.com/opentensor/bittensor/pull/1230
* Hotfix - Cli unstake fix by @Eugene-hu in https://github.com/opentensor/bittensor/pull/1233
* Fix permissions for release github script by @eduardogr in https://github.com/opentensor/bittensor/pull/1224
* Staging into Release branch by @camfairchild in https://github.com/opentensor/bittensor/pull/1275
* Remove codecov by @camfairchild in https://github.com/opentensor/bittensor/pull/1282
* Use alt new preseal by @camfairchild in https://github.com/opentensor/bittensor/pull/1269

## New Contributors
* @SFuller4 made their first contribution in https://github.com/opentensor/bittensor/pull/1226
* @whiterhinoTAO made their first contribution in https://github.com/opentensor/bittensor/pull/1230

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.7.0...v4.0.0


## 3.6.3 / 2023-01-21

## What's Changed
* [hotfix][3.6.3] Fixing no version checking by @eduardogr in https://github.com/opentensor/bittensor/pull/1063


**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.6.2...v3.6.3


## 3.6.2 / 2023-01-19

## What's Changed
* Hotfix/3.6.2/validator logit parameters by @Eugene-hu in https://github.com/opentensor/bittensor/pull/1057


**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.6.1...v3.6.2


## 3.6.1 / 2022-12-21

## What's Changed
* V3.6.0 nobunaga merge by @Eugene-hu in https://github.com/opentensor/bittensor/pull/1028
* Integration dendrite test fixes by @Eugene-hu in https://github.com/opentensor/bittensor/pull/1029
* Adding 3.6.0 release notes to CHANGELOG by @eduardogr in https://github.com/opentensor/bittensor/pull/1032
* [BIT-612] Validator robustness improvements by @opentaco in https://github.com/opentensor/bittensor/pull/1034
* [Hotfix 3.6.1] Validator robustness by @opentaco in https://github.com/opentensor/bittensor/pull/1035


**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.6.0...v3.6.1


## 3.6.0 / 2022-12-13

## What's Changed
* Removal of dendrite multiprocessing by @Eugene-hu in https://github.com/opentensor/bittensor/pull/1017
* Merging back 3.5.1 fix to nobunaga by @eduardogr in https://github.com/opentensor/bittensor/pull/1018
* Release/3.5.0 post release by @eduardogr in https://github.com/opentensor/bittensor/pull/1010
* Fixes issue with --neuron.no_set_weights by @camfairchild in https://github.com/opentensor/bittensor/pull/1020
* Removing GitHub workflow push docker by @eduardogr in https://github.com/opentensor/bittensor/pull/1011
* [Fix] fix max stake for single by @camfairchild in https://github.com/opentensor/bittensor/pull/996
* [Feature] mention balance if not no prompt by @camfairchild in https://github.com/opentensor/bittensor/pull/995
* Add signature v2 format by @adriansmares in https://github.com/opentensor/bittensor/pull/983
* Improving the way we manage requirements by @eduardogr in https://github.com/opentensor/bittensor/pull/1003
* [BIT-601] Scaling law on EMA loss by @opentaco in https://github.com/opentensor/bittensor/pull/1022
* [BIT-602] Update scaling power from subtensor by @opentaco in https://github.com/opentensor/bittensor/pull/1027
* Release 3.6.0 by @eduardogr in https://github.com/opentensor/bittensor/pull/1023

## New Contributors
* @adriansmares made their first contribution in https://github.com/opentensor/bittensor/pull/976

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.5.1...v3.6.0


## 3.5.1 / 2022-11-24

## What's Changed
* [hotfix] pin scalecodec lower by @camfairchild in https://github.com/opentensor/bittensor/pull/1013


**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.5.0...v3.5.1

## 3.5.0 / 2022-11-24

## What's Changed

- [Fix] allow synapse all (https://github.com/opentensor/bittensor/pull/988)
  - allow set synapse All using flag
  - add test
  - use dot get
  
- [Feature] Mark registration threads as daemons (https://github.com/opentensor/bittensor/pull/998)
  - make solver processes daemons

- [Feature] Validator debug response table (https://github.com/opentensor/bittensor/pull/999)
  - Add response table to validator debugging

- [Feature] Validator weight setting improvements (https://github.com/opentensor/bittensor/pull/1000)
  - Remove responsive prioritization from validator weight calculation
  - Move metagraph_sync just before weight setting
  - Add metagraph register to validator
  - Update validator epoch conditions
  - Log epoch while condition details
  - Consume validator nucleus UID queue fully
  - Increase synergy table display precision
  - Round before casting to int in phrase_cross_entropy
- small fix for changelog and version by @Eugene-hu in https://github.com/opentensor/bittensor/pull/993
- release/3.5.0 by @eduardogr in https://github.com/opentensor/bittensor/pull/1006

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.4.3...v3.5.0


## 3.4.3 / 2022-11-15

## What's Changed
* [Hotfix] Synapse security update by @opentaco in https://github.com/opentensor/bittensor/pull/991


**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.4.2...v3.4.3

## 3.4.2 / 2022-11-09

## What's Changed
* Adding 3.4.0 changelog to CHANGELOG.md by @eduardogr in https://github.com/opentensor/bittensor/pull/953
* Release 3.4.2 by @unconst in https://github.com/opentensor/bittensor/pull/970


**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.4.1...v3.4.2

## 3.4.1 / 2022-10-13

## What's Changed
* [Hotfix] Fix CUDA Reg update block by @camfairchild in https://github.com/opentensor/bittensor/pull/954


**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.4.0...v3.4.1

## 3.4.0 / 2022-10-13

## What's Changed
* Parameters update by @Eugene-hu  #936 
* Bittensor Generate by @unconst  #941 
* Prometheus by @unconst  #928 
* [Tooling][Release] Adding release script by @eduardogr in https://github.com/opentensor/bittensor/pull/948


**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.3.4...v3.4.0


## 3.3.4 / 2022-10-03

### What's Changed
* [hot-fix] fix indent again. add test by @camfairchild in https://github.com/opentensor/bittensor/pull/907
* Delete old gitbooks by @quac88 in https://github.com/opentensor/bittensor/pull/924
* Release/3.3.4 by @Eugene-hu in https://github.com/opentensor/bittensor/pull/927

### New Contributors
* @quac88 made their first contribution in https://github.com/opentensor/bittensor/pull/924

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.3.3...v3.3.4


## 3.3.3 / 2022-09-06

### What's Changed
* [feature] cpu register faster by @camfairchild in https://github.com/opentensor/bittensor/pull/854
* [hotfix] fix flags for multiproc register limit by @camfairchild in https://github.com/opentensor/bittensor/pull/876
* Fix/diff unpack bit shift by @camfairchild in https://github.com/opentensor/bittensor/pull/878
* [Feature] [cubit] CUDA registration solver by @camfairchild in https://github.com/opentensor/bittensor/pull/868
* Fix/move overview args to cli by @camfairchild in https://github.com/opentensor/bittensor/pull/867
* Add/address CUDA reg changes by @camfairchild in https://github.com/opentensor/bittensor/pull/879
* [Fix] --help command by @camfairchild in https://github.com/opentensor/bittensor/pull/884
* Validator hotfix min allowed weights by @Eugene-hu in https://github.com/opentensor/bittensor/pull/885
* [BIT-552] Validator improvements (nucleus permute, synergy avg) by @opentaco in https://github.com/opentensor/bittensor/pull/889
* Bit 553 bug fixes by @isabella618033 in https://github.com/opentensor/bittensor/pull/886
* add check to add ws:// if needed by @camfairchild in https://github.com/opentensor/bittensor/pull/896
* [BIT-572] Exclude lowest quantile from weight setting by @opentaco in https://github.com/opentensor/bittensor/pull/895
* [BIT-573] Improve validator epoch and responsives handling by @opentaco in https://github.com/opentensor/bittensor/pull/901
* Nobunaga Release V3.3.3 by @Eugene-hu in https://github.com/opentensor/bittensor/pull/899


**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.3.2...v3.3.3

## 3.3.2 / 2022-08-18

### SynapseType fix in dendrite
### What's Changed
* SynapseType fix in dendrite by @robertalanm in https://github.com/opentensor/bittensor/pull/874

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.3.1...v3.3.2

## 3.3.1 / 2022-08-17

### What's Changed
* [hotfix] Fix GPU reg bug. bad indent by @camfairchild in https://github.com/opentensor/bittensor/pull/883

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.3.0...v3.3.1

## 3.3.0 / 2022-08-16

### CUDA registration
This release adds the ability to complete the registration using a CUDA-capable device.   
See https://github.com/opentensor/cubit/releases/tag/v1.0.5 for the required `cubit` v1.0.5 release

Also a few bug fixes for the CLI

### What's Changed
* [hotfix] fix flags for run command, fix hotkeys flag for overview, and [feature] CUDA reg by @camfairchild in https://github.com/opentensor/bittensor/pull/877

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.2.0...v3.3.0

## 3.2.0 / 2022-08-12

### Validator saving and responsive-priority weight-setting

### What's Changed
* [BIT-540] Choose responsive UIDs for setting weights in validator + validator save/load by @opentaco in https://github.com/opentensor/bittensor/pull/872

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.1.0...v3.2.0

## 3.1.0 / 2022-08-11

### Optimizing multi-processed CPU registration
This release refactors the registration code for CPU registration to improve solving performance.

### What's Changed
* [feature] cpu register faster (#854) by @camfairchild in https://github.com/opentensor/bittensor/pull/875

**Full Changelog**: https://github.com/opentensor/bittensor/compare/v3.0.0...v3.1.0

## 3.0.0 / 2022-08-08

### Synapse update

## 

