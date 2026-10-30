# Anime style router

Lightweight 768→128→1 binary router over frozen FG-CLIP2 visual embeddings.

Training data:

- 120 labeled anime/game images;
- 100 solid real images;
- 60 quality real images.

Production threshold: `0.90`.

The router only chooses anime versus real policy. It must not be used as a
violation classifier. Low-confidence and missing-router cases fall back to the
existing real-image policy.

Retrain:

```bash
python -B -m src.train_anime_style_router \
  --anime-root /path/to/anime/images \
  --anime-annotations data/anime_safe/human_annotations_2026-10-27.csv \
  --solid-root /path/to/solid \
  --quality-root /path/to/quality \
  --output-dir checkpoints/anime_style_router
```
