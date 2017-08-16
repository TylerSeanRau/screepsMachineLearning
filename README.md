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
4. [Screeps](https://github.com/LispEngineer/screeps) stats data collector installed in your screeps account

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
| kmeans.py | creates one cluster and displays the 5 most anomalous rows based on distance to center of a single cluster |
| dbscan.py | noise is considered anomalous, epsilon = 500 and minimum points = 10 |

#### Observations
* kmeans.py
    * most anomalous rows are usually very recent data when a low number of data points are fitted on
    * most anomalous rows are very close to the first points collected when a high amount of data points are fitted on
    * even though center changes the average distance to the center always increases as new data points arrive
    * changes made because of these observations
        * removed controller_progress and replaced it with controller_progress_delta
        * removed a ton of noise features
            * controller_level controller_needed controller_safemode controller_safemode_avail energy_cap num_sources num_extractors storage_energy storage_minerals terminal_energy terminal_minerals num_containers container_energy num_links link_energy num_spawns num_source_containers
 * dbscan.py
    * most anomalous points are 3 point clusters where somewhere among those 3 points there's a source regen (meaning available source jumps up by 3000)
    * even though most of the anomalous points are source regens it does not detect all the source regens within a given set of data, unsure what percentage it does find
