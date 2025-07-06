# run_pipeline.py (enhanced with error handling and logging)
import subprocess
import datetime

log_file = "pipeline_log.txt"

def log_message(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def run_step(step_name, command):
    try:
        log_message(f"Starting {step_name}...")
        subprocess.run(command, check=True)
        log_message(f"Completed {step_name} successfully.")
    except subprocess.CalledProcessError as e:
        log_message(f"ERROR in {step_name}: {e}")
        exit(1)

if __name__ == "__main__":
    log_message("Pipeline execution started.")

    # Run Screener
    run_step("Screener", ["python", "screener.py"])

    # Run Backtest
    run_step("Backtest", ["python", "backtest.py"])

    # Run Metrics Calculation and Ranking
    run_step("Metrics Calculation", ["python", "metrics.py"])

    log_message("Pipeline execution complete. Top candidates available in 'top_candidates.csv'.")