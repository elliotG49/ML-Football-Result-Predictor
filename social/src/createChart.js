// src/createChart.js
const fs = require('fs');
const { ChartJSNodeCanvas } = require('chartjs-node-canvas');

const width = 800;
const height = 800;
const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height });

async function createChart(teamsDocs, yamlConfig) {
  // `teamsDocs` should be an array of documents, e.g. [ { _id: 149, corners_per_match: ... }, { _id: 93, ... } ]
  // If you expect exactly 2 teams, you might do some checks:
  if (teamsDocs.length < 2) {
    console.warn("Expected data for 2 teams, found: ", teamsDocs.length);
  }

  // The fields/labels come from the YAML "datapoints"
  const { datapoints } = yamlConfig;

  // Build the X-axis (radar) labels from the datapoints
  const labels = datapoints.map(dp => dp.label);

  // Create a dataset for each team doc
  // For color, you can define a simple array or do something more dynamic
  const datasetColors = [
    { bg: 'rgba(54, 162, 235, 0.2)', border: 'rgba(54, 162, 235, 1)' },
    { bg: 'rgba(255, 99, 132, 0.2)', border: 'rgba(255, 99, 132, 1)' },
    // add more if you might compare >2 teams
  ];

  // Transform each team doc into a dataset
  const datasets = teamsDocs.map((teamDoc, index) => {
    const colorSet = datasetColors[index] || datasetColors[0];
    const dataValues = datapoints.map(dp => teamDoc[dp.field] || 0);

    return {
      label: `Team #${teamDoc._id}`,   // or any label you prefer
      data: dataValues,
      backgroundColor: colorSet.bg,
      borderColor: colorSet.border,
      borderWidth: 2
    };
  });

  // If you want a dynamic max, get the largest value from all data
  const allValues = [];
  datasets.forEach(ds => { allValues.push(...ds.data); });
  const suggestedMax = Math.max(...allValues) + 2;

  // Chart type from YAML
  const chartType = yamlConfig.type || 'radar';

  // Build final chart config
  const chartConfig = {
    type: chartType,
    data: {
      labels,
      datasets
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: yamlConfig.chartOptions?.title || 'Team Comparison'
        },
        subtitle: {
          display: !!yamlConfig.chartOptions?.subtitle,
          text: yamlConfig.chartOptions?.subtitle
        },
        legend: {
          position: 'top'
        }
      },
      scales: {
        r: {
          suggestedMin: 0,
          suggestedMax
        }
      }
    }
  };

  // Render and save
  const imageBuffer = await chartJSNodeCanvas.renderToBuffer(chartConfig);
  fs.writeFileSync('./radar-chart.png', imageBuffer);
  console.log("Radar chart saved as radar-chart.png");
}

module.exports = { createChart };
