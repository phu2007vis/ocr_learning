import argparse

from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='see example at ',default=r'C:\Users\9999\phuoc\ocr\vietocr\config\vgg_transformer_phuoc.yml')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint',default=None)

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)
    
    trainer = Trainer(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    trainer.train()
    

if __name__ == '__main__':
    main()
