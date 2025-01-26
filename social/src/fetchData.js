// src/fetchData.js
const { MongoClient } = require('mongodb');

const uri = "mongodb://localhost:27017";
const databaseName = "footballDB";

async function fetchTeamsData({ teamIDs, competitionID, datapoints, collection }) {
  const client = new MongoClient(uri);
  try {
    await client.connect();
    const db = client.db(databaseName);
    const col = db.collection(collection);

    // Build projection from YAML datapoints
    const projection = {};
    datapoints.forEach(dp => {
      projection[dp.field] = 1;
    });

    // Filter by "id" for teams, and "competition_id"
    const filter = {
      id: { $in: teamIDs },
      competition_id: competitionID
    };

    const teams = await col.find(filter, { projection }).toArray();
    return teams; // likely an array of two docs (for two teams)
  } catch (err) {
    console.error("Error fetching teams data", err);
  } finally {
    await client.close();
  }
}

module.exports = {
  fetchTeamsData
};
