.PHONY: lint fix test

lint:
	@cargo fmt --all -- --check
	@cargo clippy --all-targets --all-features

fix:
	@cargo fmt --all
	@cargo clippy --fix --workspace --tests

test:
	@cargo test --workspace --all-features -- --nocapture
