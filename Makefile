# Makefile for RL Debug Kit Reference Suite

.PHONY: help reference\:cpu_smoke reference\:bisect_demo clean profile profile-check profile-train profile-dashboard profile-clean

help:
	@echo "RL Debug Kit Reference Suite"
	@echo ""
	@echo "Available targets:"
	@echo "  reference:cpu_smoke    - Run CPU smoke tests and generate cards"
	@echo "  reference:bisect_demo  - Run bisect demonstration"
	@echo "  profile                - Run profiler test"
	@echo "  profile-check          - Check profiler artifacts"
	@echo "  profile-train          - Run training with profiler"
	@echo "  profile-dashboard      - Start profiler dashboard"
	@echo "  profile-clean          - Clean profiler artifacts"
	@echo "  clean                  - Clean generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make reference:cpu_smoke"
	@echo "  make reference:bisect_demo"
	@echo "  make profile"
	@echo "  make profile-train"

# CPU Smoke Test Target
reference\:cpu_smoke:
	@echo "Running CPU smoke tests..."
	@echo "1. Materializing dataset manifests..."
	python3 reference/scripts/materialize.py --output-dir reference/datasets --max-samples 50
	
	@echo "2. Running good summarization training..."
	python3 reference/tasks/summarization/train.py \
		--manifest reference/datasets/ag_news_manifest.jsonl \
		--output-dir reference/runs/summarization/good \
		--max-steps 50 \
		--seed 42 \
		--pad-direction right \
		--truncate-at 512
	
	@echo "3. Running doctored summarization training (tokenizer changed)..."
	python3 reference/tasks/summarization/train.py \
		--manifest reference/datasets/ag_news_manifest.jsonl \
		--output-dir reference/runs/summarization/tokenizer_changed \
		--max-steps 50 \
		--seed 42 \
		--pad-direction left \
		--truncate-at 256
	
	@echo "4. Running safety evaluations..."
	python3 reference/tasks/safety_evals/run.py \
		--manifest reference/datasets/imdb_manifest.jsonl \
		--output-dir reference/runs/safety_evals/good \
		--seed 42
	
	@echo "5. Running code fix evaluations..."
	python3 reference/tasks/code_fix/run.py \
		--manifest reference/datasets/ag_news_code_manifest.jsonl \
		--output-dir reference/runs/code_fix/good \
		--seed 42
	
	@echo "6. Checking determinism on good run..."
	rldk check-determinism \
		--cmd "python3 reference/tasks/summarization/train.py --manifest reference/datasets/ag_news_manifest.jsonl --output-dir reference/runs/summarization/determinism_test --max-steps 10 --seed 42" \
		--compare "loss,reward_scalar" \
		--replicas 3 \
		--output-dir reference/expected/determinism_analysis
	
	@echo "7. Finding first divergence between good and doctored runs..."
	rldk diff \
		--a reference/runs/summarization/good \
		--b reference/runs/summarization/tokenizer_changed \
		--signals "sample_id,input_ids_sha256,attention_mask_sha256,outputs.text,reward_scalar,loss" \
		--output-dir reference/expected/drift_analysis
	
	@echo "8. Computing reward health on good run..."
	rldk reward-health \
		--run reference/runs/summarization/good \
		--output-dir reference/expected/reward_analysis \
		--reward-col reward_scalar \
		--step-col global_step
	
	@echo "9. Copying determinism card to expected directory..."
	cp reference/expected/determinism_analysis/determinism_card.md reference/expected/determinism_card.json
	cp reference/expected/determinism_analysis/determinism_card.png reference/expected/
	
	@echo "10. Copying drift card to expected directory..."
	cp reference/expected/drift_analysis/drift_card.md reference/expected/drift_card.json
	
	@echo "11. Copying reward health card to expected directory..."
	cp reference/expected/reward_analysis/reward_health_card.md reference/expected/reward_card.json
	
	@echo "CPU smoke tests complete!"
	@echo "Generated files in reference/expected/:"
	@ls -la reference/expected/

# Bisect Demo Target
reference\:bisect_demo:
	@echo "Running bisect demonstration..."
	@echo "1. Creating good and bad commits..."
	
	# Get current good commit
	@echo "Current commit (good): $(shell git rev-parse HEAD)"
	
	# Create bad commit
	@echo "Creating bad commit..."
	@bash reference/scripts/make_bad_commit.sh
	
	# Get bad commit hash
	@echo "Bad commit created: $(shell git rev-parse HEAD)"
	
	# Run bisect
	@echo "2. Running git bisect..."
	@git bisect start
	@git bisect bad
	@git bisect good HEAD~1
	
	@echo "3. Running rldk bisect..."
	rldk bisect \
		--good HEAD~1 \
		--bad HEAD \
		--cmd "python reference/tasks/summarization/train.py --manifest reference/datasets/ag_news_manifest.jsonl --output-dir reference/runs/summarization/bisect_test --max-steps 10 --seed 42" \
		--metric reward_scalar \
		--cmp "> 0.1" \
		--window 10
	
	@echo "4. Writing bisect results..."
	@echo '{"bisect_complete": true, "culprit_commit": "$(shell git rev-parse HEAD)", "timestamp": "$(shell date -Iseconds)"}' > reference/expected/bisect.json
	
	@echo "5. Resetting bisect..."
	@git bisect reset
	
	@echo "6. Reverting bad commit..."
	@git revert HEAD --no-edit
	
	@echo "Bisect demonstration complete!"
	@echo "Results saved to reference/expected/bisect.json"

# Clean target
clean:
	@echo "Cleaning generated files..."
	rm -rf reference/datasets/
	rm -rf reference/runs/
	rm -rf reference/expected/
	rm -f reference/tasks/summarization/train.py.backup
	@echo "Clean complete!"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -e .
	@echo "Installation complete!"

# Test target
test:
	@echo "Running tests..."
	pytest tests/ -v
	@echo "Tests complete!"

# Profiler targets
profile:
	@echo "Running profiler test..."
	python3 tools/run_profile.py --output-dir runs/profiler_test --steps 20
	@echo "Profiler test complete!"

profile-check:
	@echo "Checking profiler artifacts..."
	python3 tools/check_profile.py runs/profiler_test/context --analysis --report
	@echo "Profiler check complete!"

profile-train:
	@echo "Running training with profiler enabled..."
	python3 train.py --profiler on --epochs 3 --steps-per-epoch 10
	@echo "Training with profiler complete!"

profile-dashboard:
	@echo "Starting profiler dashboard..."
	@echo "Dashboard will be available at http://localhost:8501"
	streamlit run monitor/app.py

profile-clean:
	@echo "Cleaning profiler artifacts..."
	rm -rf runs/profiler_test
	rm -rf runs/run_*
	@echo "Profiler cleanup complete!"