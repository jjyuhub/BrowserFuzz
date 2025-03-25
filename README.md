# BrowserFuzz: AI-Guided Browser IPC Fuzzing Framework

BrowserFuzz is an advanced fuzzing framework that uses reinforcement learning to discover potential vulnerabilities in browser sandboxing mechanisms, with a focus on inter-process communication (IPC) channels.

![GitHub Actions Workflow Status](https://github.com/jjyuhub/BrowserFuzz/actions/workflows/browserfuzz-testing.yml/badge.svg)

## Features

- **AI-Guided Fuzzing**: Uses reinforcement learning to intelligently generate and adapt test cases
- **IPC-Focused Testing**: Specifically targets browser inter-process communication channels
- **Automated Detection**: Identifies and categorizes potential sandbox escapes
- **Cross-Browser Support**: Works with Chrome, Firefox, Edge, and Safari (with varying degrees of support)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/browserfuzz.git
cd browserfuzz

# Install dependencies
pip install -r requirements.txt

# Install a browser if you don't have one already
# For Ubuntu:
sudo apt-get install google-chrome-stable
```

## Usage

Basic usage:

```bash
python browserfuzz.py --browser chrome --iterations 1000
```

Advanced usage with reinforcement learning:

```bash
python browserfuzz.py --browser chrome --iterations 10000 --use-rl --model-path model.keras
```

Available options:

```
--browser        Browser to target (chrome, firefox, edge, safari)
--timeout        Timeout for each test in seconds (default: 30)
--iterations     Number of fuzzing iterations (default: 10000)
--save-dir       Directory to save findings (default: findings)
--use-rl         Use reinforcement learning to guide fuzzing
--model-path     Path to save/load RL model
--mutation-rate  Rate of mutation vs random generation (default: 0.3)
```

## GitHub Actions Integration

This repository includes a GitHub Actions workflow that automatically tests the BrowserFuzz tool against different browsers. You can trigger it manually from the Actions tab with custom parameters.

## How It Works

1. **Input Generation**: Creates diverse IPC messages to test browser communication channels
2. **Reinforcement Learning**: Uses rewards from previous tests to refine input generation
3. **Browser Interaction**: Monitors browser behavior for crashes and anomalies
4. **Result Analysis**: Identifies and categorizes potential security issues


