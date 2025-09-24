#!/bin/bash

set -e

echo "🚀 RLDK GRPO Training Demonstration"
echo "=================================="

echo "📋 Checking RLDK installation..."
if ! python -c "import rldk; print('RLDK imported successfully')" 2>/dev/null; then
    echo "❌ RLDK import failed - installing in editable mode..."
    pip install -e .
fi

echo "🔍 Checking compute environment..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "📦 Installing dependencies..."
pip install datasets transformers torch

echo "🎯 Running enhanced GRPO demonstration..."
python enhanced_grpo_demo.py --model distilgpt2 --steps 50 --output-dir enhanced_grpo_demo_results

echo ""
echo "📊 Demonstration Results:"
echo "========================"

if [ -f "enhanced_grpo_demo_results/comparison/demonstration_report.md" ]; then
    echo "✅ Demonstration completed successfully!"
    echo "📄 View results: enhanced_grpo_demo_results/comparison/demonstration_report.md"
    echo ""
    echo "Key metrics comparison:"
    echo "----------------------"
    
    if [ -f "enhanced_grpo_demo_results/comparison/comparison_report.json" ]; then
        python -c "
import json
with open('enhanced_grpo_demo_results/comparison/comparison_report.json') as f:
    data = json.load(f)
    
baseline = data['sessions']['baseline']
monitored = data['sessions']['monitored']

print(f'Baseline  - KL: {baseline[\"final_kl\"]:.4f}, Entropy: {baseline[\"final_entropy\"]:.4f}, Alerts: {baseline[\"alerts_triggered\"]}')
print(f'Monitored - KL: {monitored[\"final_kl\"]:.4f}, Entropy: {monitored[\"final_entropy\"]:.4f}, Alerts: {monitored[\"alerts_triggered\"]}')
print(f'Detection difference: {monitored[\"alerts_triggered\"] - baseline[\"alerts_triggered\"]} alerts')
"
    fi
else
    echo "❌ Demonstration failed - check logs above"
    exit 1
fi

echo ""
echo "🎉 GRPO demonstration complete!"
echo "📁 All files saved to: enhanced_grpo_demo_results/"
