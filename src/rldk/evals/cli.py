"""CLI for running evaluation suites."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import typer

from .suites import QUICK_SUITE, COMPREHENSIVE_SUITE, SAFETY_SUITE
from .metrics import evaluate_throughput, evaluate_toxicity, evaluate_bias

app = typer.Typer(help="RL Debug Kit Evaluation CLI")

logger = logging.getLogger(__name__)


def load_jsonl_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        data = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
        
        if not data:
            raise ValueError("No valid JSON records found in file")
        
        return pd.DataFrame(data)
    
    except Exception as e:
        logger.error(f"Failed to load JSONL file: {e}")
        raise


def run_evaluation_suite(
    data: pd.DataFrame,
    suite_name: str,
    output_column: str = "output",
    events_column: str = "events",
    **kwargs
) -> Dict[str, Any]:
    """
    Run evaluation suite on data.
    
    Args:
        data: Input data DataFrame
        suite_name: Name of evaluation suite
        output_column: Column containing model outputs
        events_column: Column containing event logs
        **kwargs: Additional evaluation parameters
        
    Returns:
        Dictionary with evaluation results
    """
    if suite_name == "quick":
        suite = QUICK_SUITE
    elif suite_name == "comprehensive":
        suite = COMPREHENSIVE_SUITE
    elif suite_name == "safety":
        suite = SAFETY_SUITE
    else:
        raise ValueError(f"Unknown suite: {suite_name}")
    
    results = {
        "suite_name": suite_name,
        "suite_description": suite["description"],
        "evaluations": {},
        "summary": {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "errors": []
        }
    }
    
    # Add output column to data if not present
    if output_column not in data.columns:
        data[output_column] = "No output data available"
    
    # Add events column to data if not present
    if events_column not in data.columns:
        data[events_column] = "[]"
    
    for eval_name, eval_func in suite["evaluations"].items():
        try:
            logger.info(f"Running evaluation: {eval_name}")
            
            # Handle different evaluation types
            if eval_name == "throughput":
                result = evaluate_throughput(data, log_column=events_column, **kwargs)
            elif eval_name == "toxicity":
                result = evaluate_toxicity(data, output_column=output_column, **kwargs)
            elif eval_name == "bias":
                result = evaluate_bias(data, output_column=output_column, **kwargs)
            else:
                # For other evaluations, try with default parameters
                result = eval_func(data, **kwargs)
            
            results["evaluations"][eval_name] = result
            
            # Check if the evaluation actually succeeded (no error in result)
            if "error" in result and result["error"]:
                logger.warning(f"Evaluation {eval_name} completed but with errors: {result['error']}")
                results["summary"]["failed_evaluations"] += 1
            else:
                results["summary"]["successful_evaluations"] += 1
            
        except Exception as e:
            logger.error(f"Evaluation {eval_name} failed: {e}")
            results["evaluations"][eval_name] = {
                "score": 0.0,
                "details": f"Evaluation failed: {str(e)}",
                "error": str(e)
            }
            results["summary"]["errors"].append({
                "evaluation": eval_name,
                "error": str(e)
            })
            results["summary"]["failed_evaluations"] += 1
        
        results["summary"]["total_evaluations"] += 1
    
    # Note: failed_evaluations is now correctly maintained throughout the loop
    
    # Calculate overall score
    successful_scores = [
        eval_result["score"] 
        for eval_result in results["evaluations"].values()
        if "score" in eval_result and "error" not in eval_result
    ]
    
    if successful_scores:
        results["summary"]["overall_score"] = sum(successful_scores) / len(successful_scores)
    else:
        results["summary"]["overall_score"] = 0.0
    
    return results


@app.command()
def evaluate(
    input_file: Path = typer.Argument(..., help="Path to JSONL input file"),
    suite: str = typer.Option("quick", "--suite", "-s", help="Evaluation suite to run (quick/comprehensive/safety)"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to output JSON file"),
    output_column: str = typer.Option("output", "--output-column", help="Column name containing model outputs"),
    events_column: str = typer.Option("events", "--events-column", help="Column name containing event logs"),
    min_samples: int = typer.Option(10, "--min-samples", help="Minimum samples required for evaluation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """
    Run evaluation suite on JSONL data.
    
    Example:
        rldk-evals evaluate data.jsonl --suite comprehensive --output results.json
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Load data
        logger.info(f"Loading data from {input_file}")
        data = load_jsonl_data(input_file)
        logger.info(f"Loaded {len(data)} records")
        
        # Run evaluation
        logger.info(f"Running {suite} evaluation suite")
        results = run_evaluation_suite(
            data=data,
            suite_name=suite,
            output_column=output_column,
            events_column=events_column,
            min_samples=min_samples
        )
        
        # Output results
        if output_file:
            logger.info(f"Writing results to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            # Print to stdout
            print(json.dumps(results, indent=2))
        
        # Print summary
        summary = results["summary"]
        logger.info(f"Evaluation complete: {summary['successful_evaluations']}/{summary['total_evaluations']} successful")
        logger.info(f"Overall score: {summary['overall_score']:.3f}")
        
        if summary["errors"]:
            logger.warning(f"{summary['failed_evaluations']} evaluations failed")
            for error in summary["errors"]:
                logger.warning(f"  {error['evaluation']}: {error['error']}")
        
        # Exit with error code if any evaluations failed
        if summary["failed_evaluations"] > 0:
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


@app.command()
def list_suites():
    """List available evaluation suites."""
    suites = {
        "quick": QUICK_SUITE,
        "comprehensive": COMPREHENSIVE_SUITE,
        "safety": SAFETY_SUITE
    }
    
    print("Available evaluation suites:")
    print()
    
    for name, suite in suites.items():
        print(f"  {name}:")
        print(f"    Description: {suite['description']}")
        print(f"    Default sample size: {suite['default_sample_size']}")
        print(f"    Estimated runtime: {suite['estimated_runtime']}")
        print(f"    Evaluations: {', '.join(suite['evaluations'].keys())}")
        print()


@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="Path to JSONL input file to validate"),
    output_column: str = typer.Option("output", "--output-column", help="Column name containing model outputs"),
    events_column: str = typer.Option("events", "--events-column", help="Column name containing event logs")
):
    """
    Validate JSONL file structure and data.
    
    Example:
        rldk-evals validate data.jsonl
    """
    try:
        logger.info(f"Validating {input_file}")
        data = load_jsonl_data(input_file)
        
        print(f"File validation results:")
        print(f"  Total records: {len(data)}")
        print(f"  Columns: {list(data.columns)}")
        
        # Check required columns
        if output_column in data.columns:
            output_count = data[output_column].notna().sum()
            print(f"  Output column '{output_column}': {output_count} non-null values")
        else:
            print(f"  Output column '{output_column}': NOT FOUND")
        
        if events_column in data.columns:
            events_count = data[events_column].notna().sum()
            print(f"  Events column '{events_column}': {events_count} non-null values")
        else:
            print(f"  Events column '{events_column}': NOT FOUND")
        
        # Check data quality
        print(f"  Missing values: {data.isnull().sum().sum()}")
        
        logger.info("Validation complete")
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    app()