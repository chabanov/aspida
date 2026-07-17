# Contributing to Aspida

Thank you for your interest in contributing to Aspida! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be considerate of others and follow standard open-source community guidelines.

## How to Contribute

### Reporting Bugs

Before submitting a bug report, please:
1. Check if the issue has already been reported
2. Test with the latest version
3. Collect debug information (logs, stack traces, environment details)

When submitting a bug report, include:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, GNAT version, CUDA version if applicable)
- Relevant logs or screenshots

### Suggesting Features

Feature suggestions are welcome! Please:
1. Check if the feature has already been suggested
2. Provide a clear use case
3. Explain why it would benefit the project

### Pull Requests

1. **Fork and Branch**
   ```bash
   git clone https://github.com/YOUR_USERNAME/aspida.git
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   make test
   make prove  # SPARK verification
   ```

4. **Commit Messages**
   - Use clear, descriptive commit messages
   - Follow conventional commits format:
     - `feat:` for new features
     - `fix:` for bug fixes
     - `docs:` for documentation
     - `test:` for tests
     - `refactor:` for refactoring
     - `perf:` for performance improvements

5. **Submit PR**
   - Reference any related issues
   - Describe your changes
   - Ensure CI passes

## Coding Standards

### Ada/SPARK

- Use consistent indentation (3 spaces)
- Follow Ada naming conventions:
  - `PascalCase` for types, packages
  - `snake_case` for variables, procedures
- Add SPARK contracts where applicable
- Document public APIs with comments

### Example

```ada
--  Compute matrix-vector product for quantized weight
function QMatVec
  (W : Weight;
   X : Tensor;
   Thread_Pool : Pool_Access := null) return Tensor
with
  Pre  => X.Dim = 1 and then X.Shape (1) = W.Cols,
  Post => QMatVec'Result.Dim = 1 and then QMatVec'Result.Shape (1) = W.Rows;
```

### CUDA

- Follow CUDA best practices
- Use meaningful variable names
- Add comments for kernel logic
- Test with compute capability 7.0+

## Project Structure

```
src/
├── llm/      — Inference engine
├── crypto/   — Cryptographic primitives
├── secure/   — Secure channel
├── server/   — HTTP/WebSocket servers
└── session/  — Session management

tests/        — Test suites
gpu/          — CUDA kernels
docs/         — Documentation
```

## Testing

### Unit Tests

```bash
make test          # All tests
make test-crypto   # Crypto tests
make test-llm      # LLM tests
```

### SPARK Verification

```bash
make prove         # Full verification
make prove-flow    # Flow analysis only
```

### Model-Dependent Tests

Some tests require a local GGUF file:
```bash
QWEN_MODEL_PATH=/path/to/model.gguf make test-weights-real
```

## Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for architecture changes
- Add inline comments for complex logic
- Update API documentation for new endpoints

## Questions?

- Open an issue for questions
- Check existing documentation in `docs/`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Aspida! 🚀
