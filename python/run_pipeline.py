"""
Data Analytics Pipeline Orchestrator
======================================

Executes the complete data analytics pipeline in the correct order:
1. Data Validation (00_data_validation.py) - Checks data integrity
2. EDA (01_eda.py) - Generates daily and monthly KPIs
3. RFM Segmentation (02_rfm_segmentation.py) - Customer segmentation
4. A/B Testing (03_ab_testing.py) - Statistical hypothesis tests
5. Anomaly Detection (04_anomaly_detection.py) - Identifies unusual patterns
6. Advanced Analysis (05_data_profiling.py, etc.)

Features:
- Runs all scripts in correct dependency order
- Handles errors and stops on critical failures
- Generates execution logs and timestamps
- Provides detailed progress reporting
- Supports dry-run mode to validate configuration

Usage:
    python python/run_pipeline.py [--dry-run] [--skip-validation] [--stop-on-error]

Exit codes:
    0 = All scripts succeeded
    1 = Warnings (some scripts had issues but completed)
    2 = Failure (critical script failed)
"""

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json

# Configuration
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the execution of the analytics pipeline."""

    def __init__(self, dry_run: bool = False, skip_validation: bool = False, stop_on_error: bool = True):
        """Initialize orchestrator."""
        self.dry_run = dry_run
        self.skip_validation = skip_validation
        self.stop_on_error = stop_on_error
        self.results = []
        self.start_time = datetime.now()

    def run_script(self, script_path: str, description: str) -> bool:
        """Execute a Python script and log results."""
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING: {description}")
        logger.info(f"Script: {script_path}")
        logger.info(f"{'='*80}")

        if self.dry_run:
            logger.info("[DRY RUN] Skipping actual execution")
            return True

        try:
            result = subprocess.run(
                ["python", script_path],
                cwd=ROOT,
                capture_output=False,
                timeout=300,  # 5 minute timeout
            )

            success = result.returncode == 0
            
            # Data validation warnings (exit code 1) are acceptable
            if script_path.endswith("00_data_validation.py") and result.returncode == 1:
                success = True
            
            status = "SUCCESS" if success else f"FAILED (exit code {result.returncode})"

            self.results.append({
                "script": script_path,
                "description": description,
                "status": status,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            })

            logger.info(f"Result: {status}")

            if not success and self.stop_on_error:
                logger.error(f"Pipeline stopped due to failure in {description}")
                return False

            return success

        except subprocess.TimeoutExpired:
            logger.error(f"Script {script_path} timed out")
            self.results.append({
                "script": script_path,
                "description": description,
                "status": "TIMEOUT",
                "success": False,
                "timestamp": datetime.now().isoformat(),
            })
            if self.stop_on_error:
                return False
            return False

        except Exception as e:
            logger.error(f"Error running {script_path}: {e}")
            self.results.append({
                "script": script_path,
                "description": description,
                "status": f"ERROR: {str(e)}",
                "success": False,
                "timestamp": datetime.now().isoformat(),
            })
            if self.stop_on_error:
                return False
            return False

    def run_pipeline(self) -> int:
        """Execute the complete pipeline."""
        logger.info("=" * 80)
        logger.info("DATA ANALYTICS PIPELINE ORCHESTRATOR")
        logger.info(f"Start Time: {self.start_time.isoformat()}")
        logger.info(f"Dry Run: {self.dry_run}")
        logger.info(f"Skip Validation: {self.skip_validation}")
        logger.info(f"Stop on Error: {self.stop_on_error}")
        logger.info("=" * 80)

        pipeline_stages = []

        if not self.skip_validation:
            pipeline_stages.append(
                ("python/00_data_validation.py", "Data Validation & Profiling")
            )

        pipeline_stages.extend([
            ("python/01_eda.py", "Exploratory Data Analysis (EDA)"),
            ("python/02_rfm_segmentation.py", "RFM Customer Segmentation"),
            ("python/03_ab_testing.py", "A/B Testing Analysis"),
            ("python/04_anomaly_detection.py", "Anomaly Detection"),
        ])

        # Run pipeline
        all_success = True
        for script, description in pipeline_stages:
            success = self.run_script(str(ROOT / script), description)
            all_success = all_success and success

        # Print summary
        self.print_summary()

        # Save results
        self.save_results()

        # Determine exit code
        if all_success:
            logger.info("Pipeline execution COMPLETED successfully")
            return 0
        else:
            logger.warning("Pipeline execution COMPLETED with failures")
            return 2

    def print_summary(self):
        """Print execution summary."""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)

        for result in self.results:
            status_symbol = "[PASS]" if result["success"] else "[FAIL]"
            print(f"{status_symbol} {result['description']:45s} {result['status']}")

        elapsed = datetime.now() - self.start_time
        print(f"\nTotal Elapsed Time: {elapsed}")

        success_count = sum(1 for r in self.results if r["success"])
        print(f"Success Rate: {success_count}/{len(self.results)} scripts")

        print("=" * 80 + "\n")

    def save_results(self):
        """Save execution results to JSON."""
        results_file = LOG_DIR / "pipeline_results.json"
        try:
            with open(results_file, "w") as f:
                json.dump({
                    "timestamp": self.start_time.isoformat(),
                    "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                    "dry_run": self.dry_run,
                    "results": self.results,
                }, f, indent=2)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Data analytics pipeline orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without executing scripts")
    parser.add_argument("--skip-validation", action="store_true", help="Skip initial data validation")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue pipeline even if a script fails")

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        dry_run=args.dry_run,
        skip_validation=args.skip_validation,
        stop_on_error=not args.continue_on_error,
    )

    exit_code = orchestrator.run_pipeline()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
