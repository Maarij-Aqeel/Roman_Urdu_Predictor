import torch
import re
import torch.nn as nn
import random





class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)
        # Embedding
        embedded = self.embedding(x)
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        # Final linear layer
        output = self.fc(dropped)  # (batch_size, vocab_size)

        return output



class Predictive_Keyboard:
    def __init__(self,device) -> None:
        self.device=device
    
    def load_model(self, filepath):
        """Load a trained model and vocabulary"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.word_to_idx = checkpoint['word_to_idx']
        self.idx_to_word = checkpoint['idx_to_word']
        self.vocab_size = checkpoint['vocab_size']
        self.max_sequence_length = checkpoint['max_sequence_length']
        self.embedding_dim = checkpoint['embedding_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint.get('dropout', 0.3)

        self.model = LSTMModel(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.num_layers,
            self.dropout
        )

        # Clean state_dict: remove 'module.' prefix if present
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {
            k.replace('module.', ''): v for k, v in state_dict.items()
        }

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {filepath} successfully.")

        

    def preprocess_text(self,text,keep=1):
            """preprocess the text"""

            text=text.lower()                      
            text=re.sub(r'(.)\1{2,}', r'\1' * keep, text)
            text = re.sub(r'\.{2,}', ' ', text)                          # replace 2+ dots with space
            text = re.sub(r'[^a-zA-Z0-9.,? ]|(?<= ) {2,}', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()                     # normalize spaces

            return text.strip()


    def predict_next_word(self,input_text,top_k=10):
        """Predict the next word given input text"""
        self.model.eval()
        with torch.no_grad():
            # Preprocess input
            processed_text = self.preprocess_text(input_text)
            words = processed_text.split()

            # Take last max_sequence_length-1 words
            if len(words) >= self.max_sequence_length:
                words = words[-(self.max_sequence_length-1):]

            # Pad if necessary
            if len(words) < self.max_sequence_length - 1:
                words = ['<PAD>'] * (self.max_sequence_length - 1 - len(words)) + words

            # Convert to indices and move to GPU
            input_seq = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(self.device)

            # Get predictions
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs[0], dim=0)

            # Get top k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)

            predictions = []
            for i in range(top_k):
                word = self.idx_to_word[top_indices[i].item()]
                prob = top_probs[i].item()
                if word not in ['<PAD>', '<UNK>']:
                    predictions.append((word, prob))

            return predictions



if __name__=='__main__':
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    Keyboard = Predictive_Keyboard(device)
    Keyboard.load_model('./Roman_urdu_predictor.pth')

    text = input('Enter sentence or word: ')
    predictions = Keyboard.predict_next_word(text)

    print("\nTop predictions:")
    for word, prob in predictions:
        print(f"{word}: {prob:.4f}")
    
