# Contributing to Whetstone

Thank you for your interest in contributing to Whetstone! This guide will help you get set up locally and explain how the project's extensibility architecture works.

---

## Developer Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nandansai08/whetstone.git
   cd whetstone
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   ```

3. **Install the package in editable mode with development tools**:
   ```bash
   pip install -e ".[dev]"
   ```
   *Note: For optional model/embedding provider dependencies (e.g., Anthropic, Voyage, sentence-transformers), please refer to the [Optional Provider Dependencies](README.md#optional-provider-dependencies) section of the README.*

4. **Configure your API keys**:
   Create a `.env` file in the project root containing your API keys (e.g., `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). See the [Quick Start](README.md#quick-start) section for details.

5. **Verify the installation**:
   Run the test suite to ensure everything is set up correctly:
   ```bash
   pytest
   ```

---

## Coding Style & Guidelines

We target **Python 3.11** or newer and enforce style rules via **Ruff** (configured in `pyproject.toml`).

* **Linting and Formatting**:
  Run Ruff to verify code compliance before submitting changes:
  ```bash
  ruff check builder_agent/
  ```
* **Line Length**: The project configures a maximum line length of **88 characters**.
* **Linter Selection**: Ruff is configured to check rules `["E", "F", "I", "W"]` (Pycodestyle, Pyflakes, Isort, Warnings).
* **Documentation**: Ensure all public modules, functions, classes, and methods use Google-style docstrings.

---

## Extensibility

### 1. Adding a Custom LLM Provider
All LLM SDK queries must remain encapsulated inside `builder_agent/llm.py` to preserve the provider-agnostic abstraction. Whetstone exposes explicit hooks in `llm.py` to register custom providers:

* **Text Generation**:
  ```python
  from builder_agent.llm import register_provider

  def my_provider(prompt: str, *, model: ModelConfig, system: str = "", max_tokens: int = 4096) -> str:
      # Call custom provider logic here
      return response_text

  register_provider("my_provider_name", my_provider)
  ```
* **Streaming Generation**:
  ```python
  from builder_agent.llm import register_stream_provider

  def my_stream_provider(prompt: str, *, model: ModelConfig, system: str = "", max_tokens: int = 4096) -> Generator[str, None, None]:
      # Yield streaming text chunks here
      yield chunk

  register_stream_provider("my_provider_name", my_stream_provider)
  ```
* **Embeddings**:
  ```python
  from builder_agent.llm import register_embed_provider

  def my_embed_provider(text: str, *, model: ModelConfig) -> list[float]:
      # Return vector embedding list of floats
      return embedding_vector

  register_embed_provider("my_provider_name", my_embed_provider)
  ```

---

### 2. Adding a Custom Embedder
Embedders are pluggable components configured via `[memory]` settings. They are defined in [`builder_agent/embedders.py`](builder_agent/embedders.py):

1. **Implement the Embedder Protocol**:
   Your class must conform to the `Embedder` protocol:
   ```python
   class Embedder(Protocol):
       def embed(self, text: str) -> list[float]: ...
   ```
2. **Register in the Factory**:
   Add your new embedder registration to the factory function `get_embedder(name)` in `builder_agent/embedders.py`:
   ```python
   def get_embedder(name: str = "sentence_transformer") -> Embedder:
       # ...
       if name == "my_new_embedder":
           return MyNewEmbedder()
       raise ValueError(f"Unknown embedder: {name}")
   ```

---

### 3. Adding a Custom Plugin
Whetstone supports a pluggable architecture to extend or customize **generation**, **verification**, and **post-processing** behavior.

#### Plugin Directory and Configuration
By default, Whetstone discovers plugins from:
1. Python entry points registered under the `whetstone.plugins` group.
2. A local `plugins/` directory in the current working directory. Any `.py` file placed inside this directory containing a class conforming to one or more of the plugin protocols is loaded automatically.

To disable specific plugins, edit your `.whetstone.toml` file:
```toml
[plugins]
disabled = ["BadPlugin"] # List class names of plugins to disable
dir = "custom_plugins_directory" # Override default plugins directory
```

> **Thread safety:** Whetstone may run subtasks concurrently. Plugin instances can be called from multiple threads, so plugins should be stateless or protect shared mutable state appropriately.

#### Plugin Interfaces
All plugin classes must define one or more of the following interface methods:

* **Generator Plugins**: Intercept or override the default LLM code generation. If `generate()` returns `None`, Whetstone falls back to the default LLM generator.
  ```python
  from builder_agent.plugin_system import (
      register_generator,
      GenerationContext,
      PluginContext,
  )

  @register_generator
  class MyGenerator:
      def generate(
          self,
          gen_context: GenerationContext,
          context: PluginContext,
          on_chunk = None,
      ) -> str | None:
          # Return generated code string, or None to fall back to the built-in generator
          return "print('Hello from custom generator')"
  ```

* **Verifier Plugins**: Run additive verification steps (e.g. running code quality tools). Return a `PluginVerificationResult` indicating pass/fail status.
  ```python
  from builder_agent.plugin_system import (
      register_verifier,
      PluginContext,
      PluginVerificationResult,
  )

  @register_verifier
  class MyVerifier:
      def verify(
          self,
          subtask,
          code: str | dict[str, str],
          context: PluginContext,
      ) -> PluginVerificationResult | None:
          # Perform checks and return verification result
          return PluginVerificationResult(
              passed=True,
              issues=[],
              exec_output="My custom check passed",
              blocking=True,
          )
  ```

* **Post-Processor Plugins**: Clean up, format, or transform generated code before verification and final output.
  ```python
  from builder_agent.plugin_system import (
      register_post_processor,
      PluginContext,
  )

  @register_post_processor
  class MyFormatter:
      def post_process_subtask(self, subtask, code: str, context: PluginContext) -> str:
          # Modify subtask generated code
          return code + "\n# formatted"

      def post_process_artifact(self, spec, code: str | dict[str, str], context: PluginContext) -> str | dict[str, str]:
          # Modify final integrated artifact
          return code
  ```

---
## Testing Guidelines

* **Unit Testing**: All code modifications and features must have corresponding unit tests in the `builder_agent/tests/` directory.
* **Mocking API Calls**: The test suite must run offline. **Never make live API calls inside tests**. Mock all LLM and embedding completions using pytest mocks or the helper response mock utilities in [`tests/test_llm.py`](builder_agent/tests/test_llm.py).
* **Running Tests**:
  ```bash
  pytest
  ```

---

## Pull Request Guidelines

1. Create a clean branch from `main`:
   ```bash
   git checkout -b my-feature-branch
   ```
2. Write your code, add test coverage, and ensure Ruff runs clean.
3. Commit with descriptive, logical messages.
4. Open a pull request against the upstream repository using our [PR Template](.github/pull_request_template.md).
