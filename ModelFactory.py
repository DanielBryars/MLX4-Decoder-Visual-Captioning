import torch
from CaptionTransformerDecoder import CaptionTransformerDecoder

class ModelFactory:
    def CreateModelFromHyperparameters(self, hyperparameters, vocab_size):
        D_txt = 512
        
        return CaptionTransformerDecoder(
            embed_dim=D_txt,
            vocab_size=vocab_size,
            num_layers=hyperparameters['num_layers'], 
            num_heads=hyperparameters['num_heads'], 
            dropout=hyperparameters['dropout']
        )
    
    def CreateFromSnapshot(self, snapshot_path, vocab_size):
        checkpoint = torch.load(snapshot_path, weights_only=True)
        
        hyperparameters = checkpoint['hyperparameters']
        
        model = self.CreateModelFromHyperparameters(hyperparameters, vocab_size)        
        model.load_state_dict(checkpoint['model'])
        
        return model