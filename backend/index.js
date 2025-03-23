import express from 'express';
import multer from 'multer';
import bodyParser from 'body-parser';
import { runTopicModeling } from './pinoconeClient.js'

const app = express();
const upload = multer();
const PORT = process.env.PORT || 3000;

// Middleware to parse JSON
app.use(bodyParser.json());

// Endpoint for document upload
app.post('/upload', upload.single('document'), (req, res) => {
    if (!req.file) {
        return res.status(400).send('No document provided.');
    }

    const documentId = req.body.documentId; // Assuming documentId is sent in the request body
    const documentText = req.file.buffer.toString('utf-8'); // Convert buffer to string

    // Here we can implement document processing logic (like extracting text from pdf)

    // Run the topic modeling Py script after document ingestion
    runTopicModeling(documentId, documentText);

    res.status(200).send('Document uploaded and processing started.');
});

// Endpoint for question answering
app.post('/qa', async (req, res) => {
    const question = req.body.question;
    const documentId = req.body.documentId; // Assuming we want to specify which document to query

    if (!question || !documentId) {
        return res.status(400).send('Question and document ID are required.');
    }

  
    try {
        // Dummy response for illustration
        const answer = `This is a dummy answer to the question: ${question}`;
        
        res.status(200).json({ answer });
    } catch (error) {
        console.error(error);
        res.status(500).send('Error processing the question.');
    }
});


app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
