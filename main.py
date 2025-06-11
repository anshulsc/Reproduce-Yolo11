"""
- To validate a pre-trained model:
   python main.py --val --model yolo11n.pt --data coco.yaml

- To train a model from a YAML configuration:
    python main.py --train --model yolo11.yaml --data coco.yaml --epochs 600 --batch 128
"""

import argparse
import sys

from pathlib import Path
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(Path.cwd())

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='Run the model in training mode.')
    parser.add_argument('--val', action='store_true', help='Run the model in validation mode.')

    parser.add_argument('--model', type=str, required=True, help='Path to the model file (.pt for eval, .yaml for training). E.g., "yolov8n.pt"')
    parser.add_argument('--data', type=str, default='coco.yaml', help='Path to the dataset YAML file.')
    
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=128, help='Batch size for training. Use -1 for auto-batch.')
    parser.add_argument('--copy_paste', type=float, default=0.1, help='Copy-paste augmentation factor for training.')

    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training and validation.')
    parser.add_argument('--device', type=str, default='0', help='Device to run on, e.g., "cpu", "0" for GPU 0, or "mps" for Apple Silicon.')
    parser.add_argument('--project', type=str, default='runs/detect', help='Directory to save results.')
    parser.add_argument('--name', type=str, default='exp', help='Name of the experiment run.')

    return parser.parse_args()

def main():

    args = parse_args()


    if not args.train and not args.val:
        print("Error: Please specify an execution mode: --train or --val")
        sys.exit(1)


    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model file '{args.model}' exists and is valid.")
        sys.exit(1)

    if args.train:
        print(f"Starting training for {args.epochs} epochs...")
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            copy_paste=args.copy_paste,
            project=args.project,
            name=args.name + "_train"
        )
        print("Training complete.")

    elif args.val:
        print("Starting validation...")
        results = model.val(
            data=args.data,
            imgsz=args.imgsz,
            device=args.device,
            plots=True,          
            save_json=True,      
            project=args.project,
            name=args.name + "_val"
        )
        print("Validation complete. Results saved.")
        

if __name__ == '__main__':
    main()