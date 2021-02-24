all: test nits

.PHONY: test
test:
	cargo build
	cargo test

.PHONY: nits
nits:
	cargo clippy --tests
	cargo fmt -- --check
