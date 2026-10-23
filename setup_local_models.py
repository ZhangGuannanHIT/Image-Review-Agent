"""Prepare local OCR assets once; run --check for an offline presence check."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'local_models'
REQUIRED = {
    'glm_ocr': ['config.json', 'model.safetensors', 'preprocessor_config.json',
                'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'],
    'rapidocr': ['PP-OCRv6_det_small.onnx', 'PP-OCRv6_rec_small.onnx',
                 'ch_ppocr_mobile_v2.0_cls_mobile.onnx'],
    'easyocr': ['craft_mlt_25k.pth', 'english_g2.pth'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.check:
        from huggingface_hub import snapshot_download
        from rapidocr import RapidOCR
        import easyocr
        snapshot_download('zai-org/GLM-OCR',
                          revision='2e85a62840ccac27daa451df36c736c4636b8628',
                          local_dir=ROOT / 'glm_ocr',
                          allow_patterns=['*.json', '*.jinja', '*.safetensors'])
        RapidOCR(params={'Global.model_root_dir': str(ROOT / 'rapidocr')})
        easyocr.Reader(['en'], gpu=False, model_storage_directory=str(ROOT / 'easyocr'),
                       user_network_directory=str(ROOT / 'easyocr'), download_enabled=True)
    missing = [str(ROOT / folder / name) for folder, names in REQUIRED.items()
               for name in names if not (ROOT / folder / name).is_file()]
    if missing:
        raise SystemExit('Missing local OCR files:\n' + '\n'.join(missing))
    print('Local OCR assets ready.')


if __name__ == '__main__':
    main()
