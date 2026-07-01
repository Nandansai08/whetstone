# Getting Started

Follow this quick guide to configure and run Whetstone.

## Installation

Install Whetstone from source in development mode with all required dependencies:

```bash
git clone https://github.com/Nandansai08/whetstone.git
cd whetstone
pip install -e ".[all]"
```

## Configuration

Set up your model provider credentials in a `.env` file in your project root directory:

```bash
# Copy template env
cp .env.example .env

# Edit with your credentials
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

Launch the Whetstone interactive REPL:

```bash
whetstone
```

Or run a one-shot build direct from the command line:

```bash
whetstone "build a calculator with add and subtract functions"
```
