from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from pypdf import PdfReader
import uuid
import httpx

transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
db = PersistentClient(path='./chromadb')

def train(file: str):
    pdf = PdfReader(file)
    for p in pdf.pages:
        text = p.extract_text()
        for c in chunks(text):
            db.get_or_create_collection('aeroport').add(
                ids=str(uuid.uuid4()),
                embeddings=[transformer.encode(c)],
                metadatas={'original_text': c}
            )

def chunks(text: str, size: int = 500, overlap: int = 100):
    for i in range(0, len(text) + 1, size - overlap):
        yield text[i:i + size]

# train('aeroport.pdf')

def query(text: str):
    result = db.get_collection('aeroport').query(
        query_embeddings=[transformer.encode(text)],
        n_results=5
    )
    return [t['original_text'] for t in result['metadatas'][0]]

def question(text: str):
    systemMessage = 'Tu es un assistant qui répond uniquement à des questions concernant un aéroport. Tu ne dois répondre qu\'avec les données du contexte que l\'on te fournira, n\'invente surtout pas et si tu ne sais pas. Réponds simplement que tu ne sais pas.'
    userMessage= f'''
        question: {text}
        contexte: {query(text)}
    '''

    result = httpx.post('http://localhost:11434/api/chat', headers={
        'ContentType': 'application/json'
    }, json={
        'model': 'mistral',
        'messages': [
            { 'role': 'system', 'content': systemMessage },
            { 'role': 'user', 'content': userMessage },
        ],
        'stream': False
    }, timeout=None)
    print(userMessage)
    print(result.json())
    
question('Combien y a t\'il de piste ?')

