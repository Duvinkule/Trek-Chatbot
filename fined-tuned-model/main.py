from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
from ChatData import ChatData
def train(chatData, model, optim, epochs=60):
    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for X, a in chatData:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            outputs = model(X, attention_mask=a, labels=X)
            loss = outputs.loss
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(chatData)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"model_state_epoch_{epoch + 1}.pt")
        print(infer("Recommend a place for camping", model))

def infer(inp, model):
    inp = "<startofstring> " + inp + " <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(X, attention_mask=a, max_length=50)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "pad_token": "<pad>",
    "bos_token": "<startofstring>",
    "eos_token": "<endofstring>"
})
tokenizer.add_tokens(["<bot>:"])

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

chatData = ChatData("./query_response_data.json", tokenizer)
chatData = DataLoader(chatData, batch_size=64, shuffle=True)

optim = Adam(model.parameters(), lr=1e-3)

print("training ....")
train(chatData, model, optim)

print("infer from model:")
while True:
    inp = input()
    print(infer(inp, model))
