import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import os
import nltk
import json
import argparse
from collections import Counter 
from torch.serialization import safe_globals 

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.idx = 4
    def build_vocabulary(self, sentence_list): 
        frequencies = Counter()
        for sentence in sentence_list: tokens = nltk.tokenize.word_tokenize(sentence.lower()); frequencies.update(tokens)
        filtered_freq = {word: freq for word, freq in frequencies.items() if freq >= self.freq_threshold}
        for word in filtered_freq:
            if word not in self.word2idx: self.word2idx[word] = self.idx; self.idx2word[self.idx] = word; self.idx += 1
    def numericalize(self, text):
        tokens = nltk.tokenize.word_tokenize(text.lower())
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
    def __len__(self): return self.idx

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout_p=0.5, fine_tune=True):
        super(EncoderCNN, self).__init__()
        try:
            resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        except TypeError: 
             resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters(): param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(dropout_p) 
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.squeeze(3).squeeze(2)
        features = self.fc(features)
        features = self.bn(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout_p=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed_dropout = nn.Dropout(dropout_p) 
        lstm_dropout = dropout_p if num_layers > 1 else 0
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout_p) 
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers 

    def init_hidden_state(self, features):
        h0 = self.init_h(features).unsqueeze(0)
        c0 = self.init_c(features).unsqueeze(0)
        if self.num_layers > 1:
             h0 = h0.repeat(self.num_layers, 1, 1)
             c0 = c0.repeat(self.num_layers, 1, 1)
        return (h0, c0)

    def forward_step(self, embedded_input, hidden_state):
        lstm_out, hidden_state = self.lstm(embedded_input, hidden_state)
        outputs = self.linear(lstm_out.squeeze(1))
        return outputs, hidden_state


CHECKPOINT_PATH = 'best_model_improved.pth' # Path to the saved model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 25  # Max length for generated captions


def load_image(image_path, transform=None):
    try:
        image = Image.open(image_path).convert("RGB")
        if transform:
            image = transform(image)
        image = image.unsqueeze(0)
        return image
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def tokens_to_sentence(tokens, vocab):
    words = [vocab.idx2word.get(token, "<unk>") for token in tokens]
    # Filter out special tokens for final output (optional)
    words = [word for word in words if word not in ["<start>", "<end>", "<pad>"]]
    return " ".join(words)

def predict_caption(image_path, encoder, decoder, vocab, transform, device, max_len=MAX_LEN):

    image_tensor = load_image(image_path, transform)
    if image_tensor is None:
        return None

    image_tensor = image_tensor.to(device)

    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()

    generated_indices = []
    with torch.no_grad(): 
        features = encoder(image_tensor) # Shape: (1, embed_size)

        hidden_state = decoder.init_hidden_state(features) # Tuple (h, c)

        # Start token
        start_token_idx = vocab.word2idx["<start>"]
        inputs = torch.tensor([[start_token_idx]], dtype=torch.long).to(device) # Shape: (1, 1)

        # Generate token by token
        for _ in range(max_len):
            embedded = decoder.embed(inputs) # Shape: (1, 1, embed_size)
            outputs, hidden_state = decoder.forward_step(embedded, hidden_state) # outputs shape: (1, vocab_size)
            predicted_idx = outputs.argmax(1)

            predicted_word_idx = predicted_idx.item()
            generated_indices.append(predicted_word_idx)

            # Stop if <end> token is generated
            if predicted_word_idx == vocab.word2idx["<end>"]:
                break

            # Prepare input for the next step
            inputs = predicted_idx.unsqueeze(1) # Shape: (1, 1)

    # Convert indices to sentence
    caption = tokens_to_sentence(generated_indices, vocab)
    return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate caption for an image using a pre-trained model.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help='Path to the model checkpoint file.')
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Loading checkpoint: {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        exit()

    try:
        with safe_globals([Vocabulary, Counter]): 
             checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying with weights_only=False (less secure)...")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        except Exception as e2:
            print(f"Failed to load checkpoint even with weights_only=False: {e2}")
            exit()


    try:
        vocab = checkpoint['vocab']
        embed_size = checkpoint.get('embed_size', 256)
        hidden_size = checkpoint.get('hidden_size', 512)
        num_layers = checkpoint.get('num_layers', 1)
        dropout_prob = checkpoint.get('dropout_prob', 0.5)
        fine_tune_encoder = checkpoint.get('fine_tune_encoder', True)
        vocab_size = len(vocab)
        print(f"Vocabulary loaded (size: {vocab_size}). Hyperparameters extracted.")
    except KeyError as e:
        print(f"Error: Missing key {e} in checkpoint. Cannot configure model.")
        exit()
    except Exception as e:
        print(f"Error processing checkpoint data: {e}")
        exit()


    # Initialize models
    try:
        encoder = EncoderCNN(embed_size, dropout_p=dropout_prob, fine_tune=fine_tune_encoder).to(DEVICE)
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, dropout_p=dropout_prob).to(DEVICE)

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print("Models initialized and weights loaded.")
    except KeyError as e:
        print(f"Error: Missing model state_dict key {e} in checkpoint.")
        exit()
    except RuntimeError as e:
         print(f"Error loading state_dict (likely architecture mismatch): {e}")
         exit()
    except Exception as e:
         print(f"Error initializing or loading models: {e}")
         exit()


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Generate the caption
    print(f"\nGenerating caption for: {args.image_path}")
    generated_caption = predict_caption(args.image_path, encoder, decoder, vocab, transform, DEVICE)

    if generated_caption:
        print("\nGenerated Caption:")
        print(generated_caption)
    else:
        print("\nFailed to generate caption.")