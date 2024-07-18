from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import spacy
import pickle
from model import Encoder, Decoder, Seq2Seq
from attention import Attention
from firebase_admin import credentials, firestore, initialize_app
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Firestore DB
cred = credentials.Certificate('trek-ai-firebase-adminsdk-m8l32-db5180792f.json')
initialize_app(cred)
db = firestore.client()

spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def load_vocab(filepath):
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def load_vocab_itos(vocab):
    return {v: k for k, v in vocab.items()}

query_vocab = load_vocab('query_vocab.pkl')
response_vocab = load_vocab('response_vocab.pkl')
query_vocab_itos = load_vocab_itos(query_vocab)
response_vocab_itos = load_vocab_itos(response_vocab)

INPUT_DIM = len(query_vocab)
OUTPUT_DIM = len(response_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('seq2seq_model.pt', map_location=device))
model.eval()

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    tokens = [token.lower() for token in tokenize_en(sentence)]
    tokens = [src_vocab['<sos>']] + [src_vocab.get(token, src_vocab['<unk>']) for token in tokens] + [src_vocab['<eos>']]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
    trg_indices = [trg_vocab['<sos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        if pred_token == trg_vocab['<eos>']:
            break
    trg_tokens = [response_vocab_itos[i] for i in trg_indices]
    return trg_tokens[1:-1]

def extract_coordinates(text):
    coordinates = re.findall(r'([+-]?\d+\.\d+)', text)
    if len(coordinates) >= 2:
        return f"<{coordinates[0]}:{coordinates[1]}>"
    return None

def print_destination_ids():
    destinations_ref = db.collection('destinations')
    docs = destinations_ref.stream()
    for doc in docs:
        print(f'Destination ID: {doc.id}')

def check_destination_ids(tags):
    des_tags = []
    destinations_ref = db.collection('destinations')
    docs = destinations_ref.stream()
    for doc in docs:
        for tag in tags:
            if tag.strip().lower() == doc.id.lower():
                des_tags.append(doc.id)
    return des_tags

def get_all_ids(tags):
    ids = []
    destinations_ref = db.collection('destinations')
    docs = destinations_ref.stream()
    for doc in docs:
        doc_tags = doc.get('tags') or []
        if any(tag.lower() in [t.lower() for t in doc_tags] for tag in tags):
            ids.append(doc.id)

    return ids  

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    query = data.get('query')
    response_tokens = translate_sentence(query, query_vocab, response_vocab, model, device)
    response_text = ' '.join(response_tokens)
    
    # Extract tags from the response text
    tags = re.findall(r'<(.+?)>', response_text)
    cleaned_response_text = re.sub(r'<.+?>', '', response_text).strip()

    # Extract coordinates from the response text
    coordinates = extract_coordinates(cleaned_response_text)
    
    print("Response Text:", repr(response_text))
    tags_2 = re.findall(r'<\s*tags\s*:\s*(.+?)\s*>', response_text)
    #print(tags_2)
    tags_2 = [tag.strip() for tag in tags_2]
    
    response = {
        'response': cleaned_response_text,
        'tags': check_destination_ids(tags),
        'coordinates': coordinates  # Include coordinates in the response
    }

    if tags_2:
        print(get_all_ids(tags_2))
        response['tags'] = get_all_ids(tags_2)

    return jsonify(response)



    #return jsonify(response)

if __name__ == '__main__':
    print_destination_ids()
    app.run(debug=True)