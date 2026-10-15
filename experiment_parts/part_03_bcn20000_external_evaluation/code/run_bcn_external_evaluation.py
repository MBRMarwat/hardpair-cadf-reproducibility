"""Command-line entry point for the frozen BCN20000 external evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hardpair.external_bcn import (
    audit_ham_overlap,
    build_bcn_manifest,
    cache_external_predictions,
    evaluate_external_predictions,
    load_external_config,
    preflight_external_cohort,
    stratified_smoke_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bcn_external_evaluation/external_evaluation_config.yaml")
    parser.add_argument("--output-dir", default="bcn_external_evaluation/full")
    parser.add_argument("--stage", choices=("preflight", "overlap", "inference", "evaluate", "all"), default="all")
    parser.add_argument("--smoke-images", type=int, default=0)
    parser.add_argument("--skip-image-verification", action="store_true")
    args = parser.parse_args()

    config = load_external_config(args.config)
    output_dir = Path(args.output_dir).resolve()
    if args.stage in ("preflight", "all"):
        manifest = preflight_external_cohort(config, output_dir, verify_images=not args.skip_image_verification)
    else:
        manifest = build_bcn_manifest(config, verify_images=False)
    if args.smoke_images:
        manifest = stratified_smoke_manifest(manifest, args.smoke_images, int(config["protocol"]["seed"]))
        manifest.to_csv(output_dir / "smoke_manifest.csv", index=False)
    if args.stage in ("overlap", "all"):
        audit_ham_overlap(manifest, config, output_dir)
    if args.stage in ("inference", "all"):
        cache = cache_external_predictions(manifest, config, output_dir)
    elif args.stage == "evaluate":
        cache_path = output_dir / "external_prediction_cache.pt"
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu")
        else:
            cache = cache_external_predictions(manifest, config, output_dir)
    else:
        cache = None
    if args.stage in ("evaluate", "all"):
        assert cache is not None
        metrics, bootstrap = evaluate_external_predictions(manifest, cache, config, output_dir)
        print(metrics.to_string(index=False))
        print("\nPaired lesion-level bootstrap:\n", bootstrap.to_string(index=False))
    else:
        print(f"Completed {args.stage}: {output_dir}")


if __name__ == "__main__":
    main()

