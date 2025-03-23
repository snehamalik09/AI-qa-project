import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Load environment variables from .env file
load_dotenv()

# Download NLTK resources (only if not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize Pinecone with gRPC
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key)
index_name = "dms-index"

# Check if the index exists, and create if it doesn’t
if index_name not in pc.list_indexes():
    try:
        pc.create_index(
            name=index_name,
            dimension=1536,  # Update based on embedding model
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    except Exception as e:
        print(f"Error creating index: {e}")

# Connect to the index
index = pc.index(index_name)

# Define stopwords and preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Topic Modeling with LDA
def get_topics(texts, num_topics=3):
    processed_texts = [preprocess_text(text) for text in texts]
    dictionary = Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    
    topics = []
    for idx, topic in lda_model.print_topics(num_words=3):
        topics.append(" + ".join([word.split("*")[1].replace('"', '') for word in topic.split(" + ")]))
    return topics

# Update Pinecone metadata
def update_pinecone_metadata(document_id, topics):
    metadata = {"topics": topics}
    index.update(id=document_id, set_metadata=metadata)
    print(f"Updated metadata for document {document_id} with topics: {topics}")

# Main function to perform topic modeling and update metadata
def process_document(document_id, document_text):
    topics = get_topics([document_text])
    update_pinecone_metadata(document_id, topics)
    print("Document processed successfully.")

if __name__ == "__main__":
    document_id = "doc_123"
    document_text = "This is an example document text that needs to be processed."
    
    # Process the document
    process_document(document_id, document_text)
