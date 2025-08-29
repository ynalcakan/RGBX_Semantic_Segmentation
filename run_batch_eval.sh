#!/bin/bash

# Batch Evaluation Runner Script for RGBX Semantic Segmentation
# This script activates the conda environment and runs comprehensive evaluation

echo "========================================================"
echo "           RGBX Semantic Segmentation"
echo "           Batch Evaluation Runner"
echo "========================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install conda first."
    exit 1
fi

# Activate the edge environment as specified in memory
echo "🔄 Activating conda environment 'edge'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate edge

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment 'edge'"
    echo "Please make sure the 'edge' environment exists and is properly configured."
    exit 1
fi

echo "✅ Conda environment 'edge' activated successfully"

# Check if we're in the right directory
if [ ! -f "config.py" ] || [ ! -f "batch_eval.py" ]; then
    echo "❌ Error: Please run this script from the RGBX_Semantic_Segmentation root directory"
    exit 1
fi

# Set CUDA device visibility (default to GPU 0 if not set)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "🔧 Set CUDA_VISIBLE_DEVICES to 0"
fi

echo "🚀 Starting batch evaluation..."
echo "📍 Working directory: $(pwd)"
echo "🎯 Target epochs: 50, 100, 150, 200, 250, 300, 350, 400, 450, 500"

# Run the batch evaluation
python batch_eval.py "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo "✅ Batch evaluation completed successfully!"
    echo "📊 Check the 'batch_evaluation_results' directory for:"
    echo "   - evaluation_report.txt (comprehensive text report)"
    echo "   - summary_metrics.csv (metrics table)"
    echo "   - per_class_iou.csv (per-class IoU data)"
    echo "   - *.png (visualization plots)"
    echo "   - detailed_results.json (raw data for further analysis)"
    echo "========================================================"
else
    echo ""
    echo "❌ Batch evaluation failed. Check the logs above for errors."
    echo "💡 Common issues:"
    echo "   - Missing checkpoint files in logs/[dataset]/checkpoint/"
    echo "   - CUDA memory issues (try smaller batch size or single GPU)"
    echo "   - Missing dependencies in conda environment"
    exit 1
fi
