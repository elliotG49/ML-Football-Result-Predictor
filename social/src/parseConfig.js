// src/parseConfig.js
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

function loadConfig(configName) {
  // If user didn't include .yaml, we add it
  const configFileName = configName.endsWith('.yaml')
    ? configName
    : `${configName}.yaml`;

  // This will correctly handle subdirectories (like "team/defensive-strength.yaml")
  const filePath = path.join(__dirname, '../configs', configFileName);

  const fileContents = fs.readFileSync(filePath, 'utf8');
  const config = yaml.load(fileContents);
  return config;
}

module.exports = {
  loadConfig
};
