#!/bin/bash

# Usage: ./run_backbone_extraction.sh /path/to/guroback /some/path/to/dataset '*.opb'

GUROBACK_EXEC="$1"
DATASET_PATH="$2"
FILES_PATTERN="$3"

# Check if the dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path '$DATASET_PATH' does not exist."
    exit 1
fi

INSTANCE_DIR="$DATASET_PATH/instance"
BACKBONE_DIR="$DATASET_PATH/backbone"
LOG_DIR="$DATASET_PATH/backbone_extraction_log"

mkdir -p "$BACKBONE_DIR"
mkdir -p "$LOG_DIR"

file_list=$(find "$INSTANCE_DIR" -type f \( -name "$FILES_PATTERN" \))

COUNTER_FILE="/tmp/parallel_counter.$$"  # Unique counter file per run
TOTAL_FILES=$( echo "$file_list" | wc -l )
# Initialize counter
echo 0 > "$COUNTER_FILE"

# Export the script and variables to use in parallel subshells
export INSTANCE_DIR BACKBONE_DIR LOG_DIR COUNTER_FILE TOTAL_FILES


# Counter function (thread-safe using flock)
next_counter() {
  ( flock -x 200
    read n < "$COUNTER_FILE"
    n=$((n + 1))
    echo "$n" > "$COUNTER_FILE"
    echo "$n"
  ) 200<>"$COUNTER_FILE"
}
export -f next_counter

# Function to process a single file
process_file() {
    local file="$1"
    local filename
    filename=$(basename "$file")
    local output="$BACKBONE_DIR/${filename}.backbone"
    local logfile="$LOG_DIR/${filename}.log"
    count=$(next_counter)
    if [ ! -f "$logfile" ]; then
        echo "[$count/$TOTAL_FILES] Processing $filename"
        # Replace this line with your actual processing command
        ulimit -v $((4 * 1024 * 1024)) # Set memory limit (in KB)
        "$GUROBACK_EXEC" Threads=1 FeasibilityTol=1e-9 OptimalityTol=1e-9 IntFeasTol=1e-9 "$file" "$output" > "$logfile" 2>&1
    else
        echo "[$count/$TOTAL_FILES] Skipping $filename (log file exists)"
    fi
    
}

export -f process_file

# Find matching files and run in parallel (max 8 jobs)
# Process files in parallel
#find "$INSTANCE_DIR" -type f \( -name 'valid_easy_*' \) | \
echo "$file_list" | \
    xargs -P 8 -I{} bash -c 'process_file "$@"' _ {}
rm -f "$COUNTER_FILE"