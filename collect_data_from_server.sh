#!/usr/bin/env bash
# Run on server A:  bash collect_data_from_server.sh   -> creates ~/release_data.tar.gz
# Collects the full trace sets and the baseline/NetLLM replay logs from ~/ESM.
set -e
cd ~/ESM
for d in training/cooked_traces evaluation/test_sim_traces evaluation/rep_6/test_results; do
  [ -d "$d" ] || { echo "missing $d"; exit 1; }
  echo "$d: $(find "$d" -type f | wc -l) files"
done
tar czf ~/release_data.tar.gz training/cooked_traces evaluation/test_sim_traces evaluation/rep_6/test_results
ls -lh ~/release_data.tar.gz
