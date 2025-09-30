# DreamPRM MinIF2F Dataset Visualizer

An interactive web application to explore and analyze the MinIF2F dataset with chain-of-thought reasoning, step-wise rewards, and key metrics.

## Features

- 📊 **Interactive Dashboard**: Overview of dataset statistics and visualizations
- 🔍 **Example Browser**: Browse through examples with pagination and search
- 📝 **Step-by-step Analysis**: Detailed view of chain-of-thought reasoning
- 🎯 **Reward Visualization**: Step-wise reward analysis with color coding
- 📈 **Statistics**: Comprehensive dataset analysis and metrics
- 🔎 **Search Functionality**: Find examples by content
- ✅ **Quality Filtering**: Automatically excludes poor quality examples (all rewards = 0)
- 🎛️ **Flexible Filtering**: Filter by step count, reward count, and quality thresholds

## Quick Start

### Option 1: Auto-install dependencies and run
```bash
python launch_visualizer.py --data_file prm_minif2f_valid_train.json
```

### Option 2: Manual installation
```bash
# Install dependencies
pip install -r visualizer_requirements.txt

# Run the visualizer (all examples)
python visualizer.py --data_file prm_minif2f_valid_train.json

# Run with step filtering (e.g., only 5-step examples)
python visualizer.py --data_file prm_minif2f_valid_train.json --filter_steps 5

# Run with reward filtering (e.g., at least 4 rewards)
python visualizer.py --data_file prm_minif2f_valid_train.json --min_rewards 4

# Run with both filters (5 steps AND at least 4 rewards)
python visualizer.py --data_file prm_minif2f_valid_train.json --filter_steps 5 --min_rewards 4
```

## Usage

1. **Start the visualizer** with your dataset file:
   ```bash
   python launch_visualizer.py --data_file your_dataset.json
   ```

2. **Open your browser** and navigate to: `http://localhost:5000`

3. **Explore the dashboard**:
   - View overall statistics
   - Browse reward and step count distributions
   - Search and filter examples
   - Click on examples for detailed analysis

## Command Line Options

- `--data_file`: Path to your dataset file (JSON or JSONL format) **[Required]**
- `--filter_steps`: Filter to show only examples with exactly N steps (optional)
- `--min_rewards`: Filter to show only examples with at least N rewards (optional)
- `--include_zero_rewards`: Include examples where all rewards are 0 (by default skipped for quality)
- `--host`: Host to run the server on (default: localhost)
- `--port`: Port to run the server on (default: 5000)
- `--debug`: Run in debug mode for development

Examples:
```bash
# View all examples (excluding zero-reward examples by default)
python visualizer.py --data_file data.json

# View examples with 5 steps AND at least 4 rewards (high quality examples)
python visualizer.py --data_file data.json --filter_steps 5 --min_rewards 4

# Include even poor quality examples (all rewards = 0)
python visualizer.py --data_file data.json --include_zero_rewards

# Comprehensive quality filtering (5 steps, 4+ rewards, exclude zero rewards)
python visualizer.py --data_file data.json --filter_steps 5 --min_rewards 4

# Debug mode with quality filtering
python visualizer.py --data_file data.json --filter_steps 5 --min_rewards 4 --debug
```

## Dataset Format

The visualizer expects a JSON or JSONL file with examples containing:

- `informal_prefix`: Human-readable problem description
- `formal_statement`: Formal mathematical statement
- `cot_response`: Chain-of-thought reasoning text
- `cot_steps`: List of reasoning steps with reward annotations
- `meta_label`: Problem classification label

## Dashboard Features

### Main Dashboard
- **Statistics Cards**: Total examples, average steps, rewards, and step length
- **Visualizations**: Interactive plots for reward and step distributions
- **Example List**: Paginated list with search functionality
- **Quick Preview**: Problem statement and reasoning preview

### Example Detail View
- **Problem Statement**: Both informal and formal representations
- **Step-by-step Breakdown**: Each reasoning step with:
  - Step content and word count
  - Reward value with color coding
  - Step numbering and organization
- **Summary Statistics**: Comprehensive metrics for the example

### Color Coding
- 🟢 High reward (≥0.8): Green
- 🟡 Medium reward (≥0.6): Yellow  
- 🟠 Low reward (≥0.4): Orange
- 🔴 Very low reward (<0.4): Red

## Dependencies

- **Flask**: Web framework for the application
- **Plotly**: Interactive visualizations
- **NumPy**: Statistical computations
- **Bootstrap**: UI styling (loaded via CDN)

## Troubleshooting

### ImportError for flask/plotly/numpy
Run the auto-installer: `python launch_visualizer.py --data_file your_file.json`

### Port already in use
Use a different port: `--port 8080`

### Large dataset loading slowly
The visualizer loads the entire dataset into memory. For very large datasets, consider:
- Using pagination (automatically handled)
- Sampling a subset of your data
- Increasing available system memory

### File not found
Ensure your data file path is correct and the file exists:
```bash
ls -la your_dataset.json
python visualizer.py --data_file $(pwd)/your_dataset.json
```

## Example Screenshots

The visualizer provides:
1. **Dashboard Overview**: Statistics and distribution plots
2. **Example Browser**: Search and pagination interface  
3. **Detailed Analysis**: Step-by-step reasoning with rewards
4. **Interactive Elements**: Hover effects and responsive design

Perfect for analyzing your DreamPRM training data and understanding model reasoning patterns!