import pdfParse from 'pdf-parse';
import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import dotenv from 'dotenv';

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENV,
});
const index = pinecone.index('document-qa');


async function extractText(file) {
  const data = await pdfParse(file);
  return data.text.split('\n').filter(Boolean).join(' ');
}

function chunkText(text, chunkSize = 500) {
  const chunks = [];
  for (let i = 0; i < text.length; i += chunkSize) {
    chunks.push(text.slice(i, i + chunkSize));
  }
  return chunks;
}

/**
 * Placeholder function for Named Entity Recognition (NER)
 */
async function performNER(text) {
  // Placeholder: In a production app, integrate a model or service for NER
  return [];
}

/**
 * Generate vector embeddings for each text chunk using OpenAI API.
 */
async function generateEmbeddings(textChunks) {
  const embeddings = await Promise.all(
    textChunks.map(async (chunk) => {
      const response = await openai.embeddings.create({
        input: chunk,
        model: 'text-embedding-ada-002',
      });
      return response.data[0].embedding;
    })
  );
  return embeddings;
}

/**
 * Upsert (add or update) vectors to Pinecone index.
 */
async function upsertVectors(embeddings, chunks) {
  const vectors = embeddings.map((embedding, i) => ({
    id: `chunk-${i}`,
    values: embedding,
    metadata: { text: chunks[i] },
  }));
  await index.upsert({ vectors });
}

/**
 * Process a document: extract text, chunk, perform NER, generate embeddings, and upsert to Pinecone.
 */
async function processDocument(file) {
  const text = await extractText(file.path);
  const textChunks = chunkText(text);
  const entities = await performNER(text);
  const embeddings = await generateEmbeddings(textChunks);
  await upsertVectors(embeddings, textChunks);

  return { message: 'Document ingested successfully', entities };
}

/**
 * Generate a response to a question by finding relevant chunks and synthesizing an answer.
 */
async function answerQuestion(question) {
  const questionEmbedding = await generateEmbeddings([question]);
  const queryResponse = await index.query({
    vector: questionEmbedding[0],
    topK: 5,
    includeMetadata: true,
  });
  const relevantChunks = queryResponse.matches.map((match) => match.metadata.text);

  const answer = await openai.chat.completions.create({
    messages: [{ role: 'user', content: `Answer the following question based on the information: ${relevantChunks.join(' ')}\nQuestion: ${question}` }],
    model: 'gpt-4o-mini',
    max_tokens: 150,
  });

  return { answer: answer.choices[0].message.content.trim() };
}

export { processDocument, answerQuestion };
