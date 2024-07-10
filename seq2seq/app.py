from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import spacy
import pickle
from model import Encoder, Decoder, Seq2Seq
from attention import Attention

app = Flask(__name__)
app.config.from_object('config.Config')
CORS(app, resources={r"/*": {"origins": "*"}})

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

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    query = data.get('query')
    response = translate_sentence(query, query_vocab, response_vocab, model, device)
    return jsonify({'response': ' '.join(response)})

if __name__ == '__main__':
    app.run(debug=True)
