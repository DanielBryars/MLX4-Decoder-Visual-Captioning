

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