from transformers import ViTForImageClassification, ViTConfig
import torch

if __name__ == '__main__':
    config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
    config.num_labels = 17355
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                           config=config,
                                                           ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

    print(model)
    print()
