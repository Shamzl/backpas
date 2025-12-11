#!/bin/bash

# Usage: ./run_gurobi.sh /some/path/to/dataset

FILE_LIST_PATH="$1"

file_list=$(cat "$FILE_LIST_PATH")

COUNTER_FILE="/tmp/parallel_counter.$$"  # Unique counter file per run
TOTAL_FILES=$( echo "$file_list" | wc -l )
# Initialize counter
echo 0 > "$COUNTER_FILE"

# Export the script and variables to use in parallel subshells
export COUNTER_FILE TOTAL_FILES


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
    file_parent_folder=$(dirname "$file")
    log_folder="${file_parent_folder}_log"
    mkdir -p "$log_folder"
    local logfile="${log_folder}/${filename}.log"
    count=$(next_counter)
    if [ ! -f "$logfile" ]; then
        echo "[$count/$TOTAL_FILES] Processing $filename : $logfile"
        # Replace this line with your actual processing command
        ulimit -v $((4 * 1024 * 1024)) # Set memory limit (in KB)
        gurobi_cl threads=1 MIPGap=0 IncumbentLog=1 timelimit=1000 logFile="$logfile" "$file"
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