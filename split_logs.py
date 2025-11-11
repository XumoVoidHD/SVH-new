import os
import re

def split_logs_by_strategy(log_file, start, end):
    # Base directory = where log file exists
    base_dir = os.path.dirname(os.path.abspath(log_file))
    
    # Output directory inside "log" folder
    output_dir = os.path.join(base_dir, "detailed_logs")
    os.makedirs(output_dir, exist_ok=True)

    # Regex pattern to extract strategy number
    strategy_pattern = re.compile(r"\[Strategy-(\d+)-")

    # Dictionary to hold open file handles
    file_handles = {}

    try:
        with open(log_file, "r") as infile:
            for line in infile:
                match = strategy_pattern.search(line)
                if match:
                    strategy_num = int(match.group(1))
                    if start <= strategy_num <= end:
                        if strategy_num not in file_handles:
                            # Create file for this strategy number
                            file_path = os.path.join(output_dir, f"Strategy-{strategy_num}.log")
                            file_handles[strategy_num] = open(file_path, "w")
                        # Write log line into strategy file
                        file_handles[strategy_num].write(line)
    finally:
        # Close all opened files
        for f in file_handles.values():
            f.close()

    print(f"✅ Logs split completed! Files saved inside: {output_dir}")


# Example usage
if __name__ == "__main__":
    # Example: if your log file is "log/system.log"
    log_file_path = "C:/Users/vedan/Desktop/projects/SVH new/logs/2025-10-31_13-47-29.log"  # <-- change this to your log file name
    split_logs_by_strategy(log_file_path, start=0, end=50)
