# screepsMachineLearning

Repo that focuses on using machine learning on data that comes out of screeps plus

Data is constructed using a modified version of screepsplus [node-agent](https://github.com/ScreepsPlus/node-agent) and the [screeps](https://github.com/LispEngineer/screeps) stats data collector

#### Requirements
Node.js 6+

### Setup

#### Prerequisites

1. Token from [ScreepsPl.us](https://screepspl.us/agent)
2. Screeps Login info
3. Node + NPM
4. Unmodified [Screeps](https://github.com/LispEngineer/screeps) stats data collector installed in your screeps account

#### Instructions

1. Download
2. Configure with config.js in `node-agent`
3. `cd node-agent`
4. `npm install`
5. `node app.js`
6. Collect some data and run various machine learning scripts

#### Data collections
- Data is stored in in a folder called `screepsData`

### Machine Learning Scripts
| Plugin | README |
| ------ | ------ |
| kmeans.py | creates one cluster and displays the 5 most anomalous rows |

#### Observations
- kmeans.py
-- most anomalous rows are usually very recent data
-- even though center changes the average distance to the center always increases as new data points arrive
