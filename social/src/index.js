// src/index.js
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { loadConfig } = require('./parseConfig');
const { fetchTeamsData } = require('./fetchData');
const { createChart } = require('./createChart');

(async () => {
  try {
    // Parse CLI arguments
    const argv = yargs(hideBin(process.argv))
      .option('config', {
        type: 'string',
        describe: 'Name of the config file (without .yaml)',
        demandOption: true
      })
      .option('teams', {
        type: 'string',
        describe: 'Comma-separated list of team IDs (e.g. 149,93)',
        demandOption: true
      })
      .option('competitionID', {
        type: 'number',
        describe: 'Competition ID (e.g., 9660)',
        demandOption: true
      })
      .help()
      .argv;

    // Load YAML config
    const yamlConfig = loadConfig(argv.config);

    // Parse teams: e.g. "149,93" -> [149, 93]
    const teamIDs = argv.teams.split(',').map(id => Number(id.trim()));

    // Fetch data
    const teamsData = await fetchTeamsData({
      teamIDs,
      competitionID: argv.competitionID,
      datapoints: yamlConfig.datapoints,
      collection: yamlConfig.dataSource?.collection || 'teams'
    });

    if (!teamsData || teamsData.length === 0) {
      console.error("No data retrieved for the specified teams/competition.");
      return;
    }

    // Create chart
    await createChart(teamsData, yamlConfig);

    console.log("Done generating chart.");
  } catch (error) {
    console.error("Error in main script:", error);
  }
})();
