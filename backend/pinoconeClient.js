// backend/pineconeClient.js
import { exec } from 'child_process';
import path from 'path';

// Function to run topic modeling Python script
export function runTopicModeling(documentId, documentText) {
  const scriptPath = path.join(process.cwd(), 'python-scripts', 'topic_modeling.py');
  
  exec(`python3 ${scriptPath} ${documentId} "${documentText}"`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing topic modeling: ${error}`);
      return;
    }
    console.log(`Topic Modeling Output:\n${stdout}`);
  });
}
