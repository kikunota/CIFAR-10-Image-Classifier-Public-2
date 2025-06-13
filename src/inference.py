import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import json

sys.path.append(os.path.dirname(__file__))

from src.models.models import create_model
from src.data.transforms import ImageTransforms
from src.utils.utils import load_checkpoint

class ImageClassifierInference:
    def __init__(self, checkpoint_path, model_name='resnet18'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = load_checkpoint(checkpoint_path)
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        num_classes = len(self.class_to_idx)
        self.model = create_model(model_name, num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = ImageTransforms.get_test_transforms()
    
    def predict(self, image_path, top_k=5):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
        
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        predictions = []
        for i in range(top_k):
            class_name = self.idx_to_class[top_indices[i]]
            confidence = top_probs[i]
            predictions.append({
                'class': class_name,
                'confidence': float(confidence)
            })
        
        return predictions
    
    def predict_batch(self, image_paths, top_k=5):
        results = {}
        for image_path in image_paths:
            try:
                predictions = self.predict(image_path, top_k)
                results[image_path] = predictions
            except Exception as e:
                results[image_path] = {'error': str(e)}
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Image Classification Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file path')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image-dir', type=str, help='Directory containing images')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--top-k', type=int, default=5, help='Top K predictions')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified")
    
    classifier = ImageClassifierInference(args.checkpoint, args.model)
    
    if args.image:
        predictions = classifier.predict(args.image, args.top_k)
        print(f"Predictions for {args.image}:")
        for pred in predictions:
            print(f"  {pred['class']}: {pred['confidence']:.4f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({args.image: predictions}, f, indent=2)
    
    elif args.image_dir:
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_paths.extend([os.path.join(args.image_dir, f) 
                              for f in os.listdir(args.image_dir) 
                              if f.lower().endswith(ext)])
        
        results = classifier.predict_batch(image_paths, args.top_k)
        
        for image_path, predictions in results.items():
            print(f"\nPredictions for {os.path.basename(image_path)}:")
            if 'error' in predictions:
                print(f"  Error: {predictions['error']}")
            else:
                for pred in predictions:
                    print(f"  {pred['class']}: {pred['confidence']:.4f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()