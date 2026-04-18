"""Train TabTransformer + baselines using drevalpy's experiment pipeline.

Usage:
    # Quick test with toy dataset:
    python train_pharmaai.py --toy

    # Full training on GDSC2 (takes hours):
    python train_pharmaai.py --dataset GDSC2

    # Cross-study: train on GDSC2, test on CCLE:
    python train_pharmaai.py --dataset GDSC2 --cross-study CCLE
"""

import argparse
import sys

# Register TabTransformer before importing experiment
import register_model  # noqa: F401

from drevalpy.experiment import drug_response_experiment
from drevalpy.datasets.loader import load_dataset
from drevalpy.models import MODEL_FACTORY


def main():
    parser = argparse.ArgumentParser(description="PharmaAI Predictor Training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="GDSC2",
        choices=["GDSC1", "GDSC2", "CCLE", "TOYv1", "TOYv2", "CTRPv1", "CTRPv2"],
        help="Primary dataset for training/evaluation (default: GDSC2)",
    )
    parser.add_argument(
        "--cross-study",
        type=str,
        default=None,
        choices=["GDSC1", "GDSC2", "CCLE", "CTRPv1", "CTRPv2"],
        help="Optional cross-study dataset for external validation",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Use TOYv2 dataset for quick testing",
    )
    parser.add_argument(
        "--test-mode",
        type=str,
        default="LPO",
        choices=["LPO", "LCO", "LDO", "LTO"],
        help="Test mode: LPO=Leave-Pair-Out, LCO=Leave-Cell-Line-Out, LDO=Leave-Drug-Out, LTO=Leave-Tissue-Out",
    )
    parser.add_argument("--n-cv-splits", type=int, default=5, help="Number of CV splits (default: 5)")
    parser.add_argument("--measure", type=str, default="LN_IC50", help="Response measure (default: LN_IC50)")
    parser.add_argument("--no-baselines", action="store_true", help="Skip baseline models")
    parser.add_argument("--no-hpam-tuning", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--path-data", type=str, default="data", help="Path to data directory")
    parser.add_argument("--path-out", type=str, default="results", help="Path to output directory")
    parser.add_argument("--run-id", type=str, default="PharmaAI_Transformer_2025", help="Run identifier")

    args = parser.parse_args()

    if args.toy:
        args.dataset = "TOYv2"
        args.n_cv_splits = 2
        args.run_id = "PharmaAI_TOY_test"
        print("Running in TOY mode for quick testing...")

    # Load primary dataset
    print(f"Loading dataset: {args.dataset} (measure: {args.measure})")
    response_data = load_dataset(
        dataset_name=args.dataset,
        path_data=args.path_data,
        measure=args.measure,
    )

    # Load cross-study dataset if specified
    cross_study_datasets = None
    if args.cross_study:
        print(f"Loading cross-study dataset: {args.cross_study}")
        cross_study_data = load_dataset(
            dataset_name=args.cross_study,
            path_data=args.path_data,
            measure=args.measure,
        )
        cross_study_datasets = [cross_study_data]

    # Select models
    models = [MODEL_FACTORY["TabTransformer"]]

    baselines = None
    if not args.no_baselines:
        baseline_names = ["ElasticNet", "SimpleNeuralNetwork", "RandomForest"]
        baselines = [MODEL_FACTORY[name] for name in baseline_names if name in MODEL_FACTORY]
        print(f"Baselines: {[b.get_model_name() for b in baselines]}")

    print(f"\nStarting experiment:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Test mode: {args.test_mode}")
    print(f"  CV splits: {args.n_cv_splits}")
    print(f"  Hyperparameter tuning: {not args.no_hpam_tuning}")
    print(f"  Cross-study: {args.cross_study or 'None'}")
    print(f"  Output: {args.path_out}/{args.run_id}")
    print()

    drug_response_experiment(
        models=models,
        baselines=baselines,
        response_data=response_data,
        n_cv_splits=args.n_cv_splits,
        test_mode=args.test_mode,
        run_id=args.run_id,
        path_data=args.path_data,
        path_out=args.path_out,
        hyperparameter_tuning=not args.no_hpam_tuning,
        cross_study_datasets=cross_study_datasets,
    )

    print(f"\nExperiment complete! Results saved to: {args.path_out}/{args.run_id}")


if __name__ == "__main__":
    main()
