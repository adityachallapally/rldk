# Makefile for RL Debug Kit Reference Suite

.PHONY: help init lint test cli-smoke docs-serve reference\:cpu_smoke reference\:bisect_demo reference\:setup clean profile profile-check profile-train profile-dashboard profile-clean test-trl test-trl-unit test-trl-integration test-trl-slow golden-master-test monitor-demo monitor-grpo bench-stability

help:
	@echo "RL Debug Kit Development Commands"
	@echo ""
	@echo "Development targets:"
	@echo "  init                   - Initialize development environment"
	@echo "  lint                   - Run linting and formatting checks"
	@echo "  test                   - Run all tests"
	@echo "  cli-smoke              - Run CLI smoke tests"
	@echo "  docs-serve             - Serve documentation locally"
	@echo "  monitor-demo           - Run the live monitoring demo with auto-stop"
	@echo "  monitor-grpo           - Stream the GRPO loop with grpo_safe rules"
	@echo "  bench-stability        - Run the stability micro-benchmark end-to-end"
	@echo ""
	@echo "Reference Suite targets:"
	@echo "  reference:setup        - Setup reference runs for testing"
	@echo "  reference:cpu_smoke    - Run CPU smoke tests and generate cards"
	@echo "  reference:bisect_demo  - Run bisect demonstration"
	@echo "  golden-master-test     - Run golden master test (capture + replay)"
	@echo "  profile                - Run profiler test"
	@echo "  profile-check          - Check profiler artifacts"
	@echo "  profile-train          - Run training with profiler"
	@echo "  profile-dashboard      - Start profiler dashboard"
	@echo "  profile-clean          - Clean profiler artifacts"
	@echo "  test-trl               - Run all TRL tests (unit + integration without downloads)"
	@echo "  test-trl-unit          - Run TRL unit tests only"
	@echo "  test-trl-integration   - Run TRL integration tests (without model downloads)"
	@echo "  test-trl-slow          - Run TRL tests with real model downloads (slow)"
	@echo "  clean                  - Clean generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make init"
	@echo "  make lint"
	@echo "  make test"
	@echo "  make cli-smoke"
	@echo "  make docs-serve"
	@echo "  make reference:setup"
	@echo "  make reference:cpu_smoke"
	@echo "  make reference:bisect_demo"
	@echo "  make golden-master-test"
	@echo "  make profile"
	@echo "  make profile-train"
	@echo "  make bench-stability"

# Development targets
init:
	@echo "Initializing development environment..."
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[dev]"
	@echo "✅ Development environment initialized!"

lint:
	@echo "Running linting and formatting checks..."
	ruff check .
	ruff format --check .
	mypy src --ignore-missing-imports --no-strict-optional
	@echo "✅ Linting complete!"

test:
	@echo "Running all tests..."
	pytest tests/ -v --cov=src/rldk --cov-report=xml --cov-report=html
	@echo "✅ Tests complete!"

fixtures-grpo:
	@echo "Generating GRPO fixtures..."
	python tests/_make_fixtures.py --make grpo
	@echo "✅ GRPO fixtures ready!"

detector-metrics:
	@mkdir -p test_artifacts/logs_doctored_kl_spike
	@touch test_artifacts/logs_doctored_kl_spike/alerts.jsonl
	@for seed in 1 2 3; do mkdir -p test_artifacts/logs_grpo/seed_$${seed}; touch test_artifacts/logs_grpo/seed_$${seed}/alerts.jsonl; done
	@echo "Computing detector metrics for doctored PPO run..."
	python scripts/compute_detector_metrics.py \
	        --alerts test_artifacts/logs_doctored_kl_spike/alerts.jsonl \
	        --window-start 800 \
	        --window-end 804 \
	        --rule-id ppo_high_kl_guard \
	        --output-json docs/assets/blog_catch_failures/metrics_ppo.json \
	        --output-markdown docs/assets/blog_catch_failures/metrics_ppo.md \
	        --label "PPO doctored"
	@echo "Computing detector metrics for doctored GRPO seeds..."
	python scripts/compute_detector_metrics.py \
	        --alerts "test_artifacts/logs_grpo/seed_*/alerts.jsonl" \
	        --window-start 800 \
	        --window-end 804 \
	        --rule-id ppo_high_kl_guard \
	        --output-json docs/assets/blog_catch_failures/metrics_grpo.json \
	        --output-markdown docs/assets/blog_catch_failures/metrics_grpo.md \
	        --label "GRPO doctored"
	@echo "✅ Detector metrics written to docs/assets/blog_catch_failures/"

cli-smoke:
	@echo "Running CLI smoke tests..."
	rldk --help
	rldk version
	rldk seed --help
	rldk forensics --help
	rldk reward --help
	rldk evals --help
	rldk track --help
	@echo "✅ CLI smoke tests complete!"

docs-serve:
	@echo "Starting documentation server..."
	@echo "Documentation will be available at http://localhost:8000"
	@python scripts/stamp_methods_box.py
	RLDK_COMMIT_SHORT=$$(git rev-parse --short HEAD) mkdocs serve

docs-build:
	@echo "Building documentation..."
	@python scripts/stamp_methods_box.py
	RLDK_COMMIT_SHORT=$$(git rev-parse --short HEAD) mkdocs build

monitor-demo:
	@echo "Running live monitoring demo..."
	@set -eu; \
	        export PYTHONPATH="src$${PYTHONPATH:+:$${PYTHONPATH}}"; \
	        artifacts_dir=artifacts; \
		mkdir -p $$artifacts_dir; \
		rm -f $$artifacts_dir/run.jsonl $$artifacts_dir/alerts.jsonl $$artifacts_dir/report.json $$artifacts_dir/demo_loop.log $$artifacts_dir/monitor.log; \
		python examples/minimal_streaming_loop.py > $$artifacts_dir/demo_loop.log 2>&1 & \
		loop_pid=$$!; \
		echo $$loop_pid > $$artifacts_dir/demo_loop.pid; \
		trap "kill -TERM $$loop_pid >/dev/null 2>&1 || true; wait $$loop_pid >/dev/null 2>&1 || true; rm -f $$artifacts_dir/demo_loop.pid" EXIT; \
		sleep 1; \
		echo "Monitor attaching to PID $$loop_pid"; \
		if command -v rldk >/dev/null 2>&1; then \
			timeout 15s rldk monitor --stream $$artifacts_dir/run.jsonl --rules rules.yaml --pid $$loop_pid --alerts $$artifacts_dir/alerts.jsonl --report $$artifacts_dir/report.json > $$artifacts_dir/monitor.log 2>&1 || true; \
		else \
			timeout 15s python -m rldk.cli monitor --stream $$artifacts_dir/run.jsonl --rules rules.yaml --pid $$loop_pid --alerts $$artifacts_dir/alerts.jsonl --report $$artifacts_dir/report.json > $$artifacts_dir/monitor.log 2>&1 || true; \
		fi; \
		kill -TERM $$loop_pid >/dev/null 2>&1 || true; \
		wait $$loop_pid >/dev/null 2>&1 || true; \
		echo "Monitor output:"; \
		cat $$artifacts_dir/monitor.log; \
		echo "Alerts written to $$artifacts_dir/alerts.jsonl"; \
		tail -n 5 $$artifacts_dir/alerts.jsonl 2>/dev/null || echo "No alerts recorded yet."; \
	        echo "Report available at $$artifacts_dir/report.json"; \
	        echo "Trainer stdout captured in $$artifacts_dir/demo_loop.log"; \
	        echo "✅ Monitor demo complete!"

monitor-grpo:
	@echo "Running GRPO monitoring demo..."
	@set -eu; \
	        export PYTHONPATH="src$${PYTHONPATH:+:$${PYTHONPATH}}"; \
	        artifacts_dir=artifacts; \
	        mkdir -p $$artifacts_dir; \
	        grpo_stream=$$artifacts_dir/grpo_run.jsonl; \
	        alerts_path=$$artifacts_dir/grpo_alerts.jsonl; \
	        report_path=$$artifacts_dir/grpo_report.json; \
	        loop_log=$$artifacts_dir/grpo_loop.log; \
	        monitor_log=$$artifacts_dir/grpo_monitor.log; \
	        rm -f $$grpo_stream $$alerts_path $$report_path $$loop_log $$monitor_log $$artifacts_dir/grpo_loop.pid; \
	        python examples/grpo_minimal_loop.py > $$loop_log 2>&1 & \
	        loop_pid=$$!; \
	        echo $$loop_pid > $$artifacts_dir/grpo_loop.pid; \
	        trap "kill -TERM $$loop_pid >/dev/null 2>&1 || true; wait $$loop_pid >/dev/null 2>&1 || true; rm -f $$artifacts_dir/grpo_loop.pid" EXIT; \
	        sleep 1; \
	        echo "Monitor attaching to PID $$loop_pid"; \
	        if command -v rldk >/dev/null 2>&1; then \
	                timeout 20s rldk monitor --stream $$grpo_stream --rules grpo_safe --preset grpo --pid $$loop_pid --alerts $$alerts_path --report $$report_path > $$monitor_log 2>&1 || true; \
	        else \
	                timeout 20s python -m rldk.cli monitor --stream $$grpo_stream --rules grpo_safe --preset grpo --pid $$loop_pid --alerts $$alerts_path --report $$report_path > $$monitor_log 2>&1 || true; \
	        fi; \
	        kill -TERM $$loop_pid >/dev/null 2>&1 || true; \
	        wait $$loop_pid >/dev/null 2>&1 || true; \
	        echo "Monitor output:"; \
	        cat $$monitor_log; \
	        echo "Alerts written to $$alerts_path"; \
	        tail -n 5 $$alerts_path 2>/dev/null || echo "No alerts recorded yet."; \
	        echo "Report available at $$report_path"; \
	        echo "Trainer stdout captured in $$loop_log"; \
	        echo "✅ GRPO monitor demo complete!"

bench-stability:
	@echo "Running stability micro-benchmark..."
	bash benchmarks/stability_micro/run_all.sh

# Setup Reference Runs Target
reference\:setup:
	@echo "Setting up reference runs for testing..."
	python3 scripts/setup_reference_runs.py
	@echo "✅ Reference runs setup complete!"

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

# Golden Master Test Target
golden-master-test:
	@echo "Running golden master test..."
	@echo "This will create a fresh virtual environment and run capture + replay tests"
	@echo "Expected duration: 2-5 minutes"
	@echo ""
	./scripts/run_golden_master_test.sh
	@echo "✅ Golden master test completed!"

# Clean target
clean:
	@echo "Cleaning generated files..."
	rm -rf reference/datasets/
	rm -rf reference/runs/
	rm -rf reference/expected/
	rm -f reference/tasks/summarization/train.py.backup
	rm -rf golden_master_output/
	rm -rf replay_output/
	rm -f golden_master.zip
	rm -f golden_master_test_report.md
	@echo "Clean complete!"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -e .
	@echo "Installation complete!"


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

# TRL Test Targets
test-trl:
	@echo "Running all TRL tests (unit + integration without downloads)..."
	@echo "1. Running unit tests..."
	python3 -m pytest tests/integration/test_trl_unit.py -v --tb=short
	@echo "2. Running integration tests (without model downloads)..."
	SKIP_MODEL_LOADING=true python3 tests/integration/test_trl_integration.py
	@echo "✅ All TRL tests completed!"

test-trl-unit:
	@echo "Running TRL unit tests..."
	python3 -m pytest tests/integration/test_trl_unit.py -v --tb=short
	@echo "✅ TRL unit tests completed!"

test-trl-integration:
	@echo "Running TRL integration tests (without model downloads)..."
	SKIP_MODEL_LOADING=true python3 tests/integration/test_trl_integration.py
	@echo "✅ TRL integration tests completed!"

test-trl-slow:
	@echo "Running TRL tests with real model downloads (slow)..."
	@echo "⚠️  This will download models and may take several minutes"
	python3 -m pytest tests/integration/test_trl_integration_optional.py -m integration -v --tb=short
	@echo "✅ TRL slow tests completed!"
