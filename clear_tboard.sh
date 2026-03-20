#!/bin/bash
# Clear TensorBoard logs for a fresh training graph

TBOARD_DIR="gpt2_openwebtext_pretrain/tboard_log"

if [ -d "$TBOARD_DIR" ]; then
    rm -f "$TBOARD_DIR"/events.out.tfevents.*
    echo "TensorBoard logs cleared from $TBOARD_DIR"
else
    echo "No TensorBoard log directory found at $TBOARD_DIR"
fi
