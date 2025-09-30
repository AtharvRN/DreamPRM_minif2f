#!/usr/bin/env python3
"""
DreamPRM MinIF2F Dataset Visualizer

An interactive web application to explore and analyze the MinIF2F dataset
with chain-of-thought reasoning, step-wise rewards, and key metrics.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

# Web framework
from flask import Flask, render_template, jsonify, request, send_from_directory
import plotly.graph_objs as go
import plotly.utils
import numpy as np

app = Flask(__name__)

# Global variables for data
dataset = []
dataset_stats = {}
current_file = ""

STEP_HEADER_RE = re.compile(r"^###\s*Step\s+(\d+)\s*:", re.IGNORECASE)

def extract_steps_from_cot_response(cot_response: str) -> List[str]:
    """Extract reasoning steps from a chain-of-thought response."""
    if not isinstance(cot_response, str):
        return []
    
    lines = cot_response.splitlines()
    steps = []
    current_step = []
    in_step = False
    
    for line in lines:
        if STEP_HEADER_RE.match(line.strip()):
            if in_step and current_step:
                steps.append(current_step)
                current_step = []
            in_step = True
            current_step.append(line)  # Include the step header
        elif in_step:
            current_step.append(line)
    
    if in_step and current_step:
        steps.append(current_step)
    
    return ["\n".join(step_lines).strip() for step_lines in steps if step_lines]

def extract_rewards_from_cot_steps(cot_steps) -> List[float]:
    """Extract reward values from chain-of-thought step annotations."""
    rewards = []
    
    if isinstance(cot_steps, dict):
        cot_steps = [cot_steps[k] for k in sorted(cot_steps.keys())]
    elif isinstance(cot_steps, list):
        pass
    else:
        return rewards
    
    for item in cot_steps:
        if isinstance(item, dict) and "pi" in item:
            try:
                rewards.append(float(item["pi"]))
            except (ValueError, TypeError):
                continue
    
    return rewards

def analyze_dataset(data: List[Dict]) -> Dict[str, Any]:
    """Analyze the dataset and compute statistics."""
    stats = {
        "total_examples": len(data),
        "avg_steps": 0,
        "avg_rewards": 0,
        "meta_label_distribution": {"0": 0, "1": 0},
        "reward_distribution": [],
        "step_count_distribution": {},
        "avg_step_length": 0,
        "problem_types": {},
        "quality_metrics": {}
    }
    
    if not data:
        return stats
    
    total_steps = 0
    total_rewards = 0
    total_step_length = 0
    step_count = 0
    all_rewards = []
    
    for item in data:
        # Extract steps and rewards
        steps = extract_steps_from_cot_response(item.get("cot_response", ""))
        rewards = extract_rewards_from_cot_steps(item.get("cot_steps", []))
        
        # Count steps
        num_steps = len(steps)
        total_steps += num_steps
        
        if num_steps in stats["step_count_distribution"]:
            stats["step_count_distribution"][num_steps] += 1
        else:
            stats["step_count_distribution"][num_steps] = 1
        
        # Analyze rewards
        if rewards:
            total_rewards += len(rewards)
            all_rewards.extend(rewards)
        
        # Step length analysis
        for step in steps:
            step_count += 1
            total_step_length += len(step.split())
        
        # Meta label distribution
        meta_label = str(int(item.get("meta_label", 0)))
        stats["meta_label_distribution"][meta_label] = stats["meta_label_distribution"].get(meta_label, 0) + 1
        
        # Problem type analysis (basic)
        formal_statement = item.get("formal_statement", "")
        informal_prefix = item.get("informal_prefix", "")
        
        if "prove" in (formal_statement + informal_prefix).lower():
            stats["problem_types"]["proof"] = stats["problem_types"].get("proof", 0) + 1
        elif "show" in (formal_statement + informal_prefix).lower():
            stats["problem_types"]["show"] = stats["problem_types"].get("show", 0) + 1
        elif "find" in (formal_statement + informal_prefix).lower():
            stats["problem_types"]["find"] = stats["problem_types"].get("find", 0) + 1
        else:
            stats["problem_types"]["other"] = stats["problem_types"].get("other", 0) + 1
    
    # Compute averages
    if len(data) > 0:
        stats["avg_steps"] = total_steps / len(data)
        stats["avg_rewards"] = total_rewards / len(data) if total_rewards > 0 else 0
        stats["avg_step_length"] = total_step_length / step_count if step_count > 0 else 0
    
    # Reward statistics
    if all_rewards:
        stats["reward_distribution"] = all_rewards
        stats["quality_metrics"] = {
            "min_reward": min(all_rewards),
            "max_reward": max(all_rewards),
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "median_reward": np.median(all_rewards)
        }
    
    return stats

def load_dataset(file_path: str, filter_steps: Optional[int] = None, min_rewards: Optional[int] = None, skip_zero_rewards: bool = True) -> List[Dict]:
    """Load dataset from JSON or JSONL file, optionally filtering by step count and reward count."""
    data = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return data
    
    try:
        # Try to load as JSONL first (each line is a JSON object)
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                print("Empty file")
                return data
            
            # Reset file pointer
            f.seek(0)
            
            try:
                # Try loading the first line as JSON
                json.loads(first_line)
                # If successful, treat as JSONL
                print("Detected JSONL format")
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            
                            # Apply filters if specified
                            if filter_steps is not None or min_rewards is not None:
                                steps = extract_steps_from_cot_response(item.get("cot_response", ""))
                                rewards = extract_rewards_from_cot_steps(item.get("cot_steps", []))
                                
                                # Filter by step count
                                if filter_steps is not None and len(steps) != filter_steps:
                                    continue
                                
                                # Filter by minimum reward count
                                if min_rewards is not None and len(rewards) < min_rewards:
                                    continue
                                
                                                                # Skip examples where all rewards are 0 (poor quality)\n                                if skip_zero_rewards and rewards and all(reward == 0.0 for reward in rewards):\n                                    continue
                            
                            data.append(item)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num}: {e}")
                            continue
            except json.JSONDecodeError:
                # If first line fails, try loading entire file as JSON
                print("Trying standard JSON format")
                f.seek(0)
                try:
                    raw_data = json.load(f)
                    
                    # Apply filters if specified
                    if filter_steps is not None or min_rewards is not None:
                        filter_desc = []
                        if filter_steps is not None:
                            filter_desc.append(f"exactly {filter_steps} steps")
                        if min_rewards is not None:
                            filter_desc.append(f"at least {min_rewards} rewards")
                        print(f"Filtering for examples with {' and '.join(filter_desc)}...")
                        
                        for item in raw_data:
                            steps = extract_steps_from_cot_response(item.get("cot_response", ""))
                            rewards = extract_rewards_from_cot_steps(item.get("cot_steps", []))
                            
                            # Apply filters
                            if filter_steps is not None and len(steps) != filter_steps:
                                continue
                            if min_rewards is not None and len(rewards) < min_rewards:
                                continue
                            
                            # Skip examples where all rewards are 0 (poor quality)
                            if skip_zero_rewards and rewards and all(reward == 0.0 for reward in rewards):
                                continue
                                
                            data.append(item)
                    else:
                        data = raw_data
                        
                except json.JSONDecodeError as e:
                    print(f"File is neither valid JSON nor JSONL: {e}")
                    return data
        
        # Print loading summary
        filter_desc = []
        if filter_steps is not None:
            filter_desc.append(f"{filter_steps} steps")
        if min_rewards is not None:
            filter_desc.append(f"≥{min_rewards} rewards")
            
        if filter_desc:
            print(f"Loaded {len(data)} examples with {' and '.join(filter_desc)} from {file_path}")
        else:
            print(f"Loaded {len(data)} examples from {file_path}")
        return data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

# Flask Routes

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html', 
                         stats=dataset_stats, 
                         total_examples=len(dataset),
                         current_file=current_file)

@app.route('/api/stats')
def get_stats():
    """API endpoint for dataset statistics."""
    return jsonify(dataset_stats)

@app.route('/api/examples')
def get_examples():
    """API endpoint to get examples with pagination."""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    search = request.args.get('search', '')
    
    filtered_data = dataset
    
    # Simple search functionality
    if search:
        filtered_data = [
            item for item in dataset 
            if search.lower() in (item.get('informal_prefix', '') + 
                                 item.get('formal_statement', '') + 
                                 item.get('cot_response', '')).lower()
        ]
    
    # Pagination
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_data = filtered_data[start:end]
    
    # Process each example for display
    processed_examples = []
    for i, example in enumerate(paginated_data, start=start):
        steps = extract_steps_from_cot_response(example.get("cot_response", ""))
        rewards = extract_rewards_from_cot_steps(example.get("cot_steps", []))
        
        processed_example = {
            "id": i,
            "informal_prefix": example.get("informal_prefix", ""),
            "formal_statement": example.get("formal_statement", ""),
            "cot_response": example.get("cot_response", ""),
            "steps": steps,
            "rewards": rewards,
            "meta_label": example.get("meta_label", 0),
            "num_steps": len(steps),
            "avg_reward": np.mean(rewards) if rewards else 0,
            "min_reward": min(rewards) if rewards else 0,
            "max_reward": max(rewards) if rewards else 0
        }
        processed_examples.append(processed_example)
    
    return jsonify({
        "examples": processed_examples,
        "total": len(filtered_data),
        "page": page,
        "per_page": per_page,
        "total_pages": (len(filtered_data) + per_page - 1) // per_page
    })

@app.route('/api/example/<int:example_id>')
def get_example_detail(example_id):
    """API endpoint to get detailed view of a specific example."""
    if 0 <= example_id < len(dataset):
        example = dataset[example_id]
        steps = extract_steps_from_cot_response(example.get("cot_response", ""))
        rewards = extract_rewards_from_cot_steps(example.get("cot_steps", []))
        
        # Create step-by-step breakdown
        step_breakdown = []
        for i, (step, reward) in enumerate(zip(steps, rewards + [None] * len(steps))):
            step_breakdown.append({
                "step_number": i + 1,
                "content": step,
                "reward": reward,
                "word_count": len(step.split()),
                "char_count": len(step)
            })
        
        detailed_example = {
            "id": example_id,
            "informal_prefix": example.get("informal_prefix", ""),
            "formal_statement": example.get("formal_statement", ""),
            "cot_response": example.get("cot_response", ""),
            "meta_label": example.get("meta_label", 0),
            "step_breakdown": step_breakdown,
            "summary": {
                "total_steps": len(steps),
                "total_rewards": len(rewards),
                "avg_reward": np.mean(rewards) if rewards else 0,
                "reward_variance": np.var(rewards) if rewards else 0,
                "total_words": sum(len(step.split()) for step in steps),
                "avg_words_per_step": np.mean([len(step.split()) for step in steps]) if steps else 0
            }
        }
        
        return jsonify(detailed_example)
    else:
        return jsonify({"error": "Example not found"}), 404

@app.route('/api/visualizations/reward_distribution')
def reward_distribution_plot():
    """Generate reward distribution visualization."""
    all_rewards = dataset_stats.get("reward_distribution", [])
    
    if not all_rewards:
        return jsonify({"error": "No reward data available"})
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=all_rewards,
        nbinsx=30,
        name="Reward Distribution",
        marker_color='skyblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Distribution of Step Rewards",
        xaxis_title="Reward Value",
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )
    
    return jsonify(plotly.utils.PlotlyJSONEncoder().encode(fig))

@app.route('/api/visualizations/step_distribution')
def step_distribution_plot():
    """Generate step count distribution visualization."""
    step_dist = dataset_stats.get("step_count_distribution", {})
    
    if not step_dist:
        return jsonify({"error": "No step data available"})
    
    steps = list(step_dist.keys())
    counts = list(step_dist.values())
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=steps,
        y=counts,
        name="Step Count Distribution",
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title="Distribution of Step Counts",
        xaxis_title="Number of Steps",
        yaxis_title="Number of Examples",
        showlegend=False,
        height=400
    )
    
    return jsonify(plotly.utils.PlotlyJSONEncoder().encode(fig))

@app.route('/example/<int:example_id>')
def example_detail(example_id):
    """Detailed view page for a specific example."""
    return render_template('example_detail.html', example_id=example_id)

def create_templates():
    """Create HTML templates for the web application."""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Main index template
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DreamPRM MinIF2F Dataset Visualizer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .example-card { margin-bottom: 20px; }
        .step-content { background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; }
        .reward-badge { margin-left: 10px; }
        .search-container { margin: 20px 0; }
        .stats-card { margin-bottom: 20px; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container">
            <span class="navbar-brand">DreamPRM MinIF2F Dataset Visualizer</span>
            <span class="navbar-text">{{ total_examples }} examples loaded from {{ current_file }}</span>
        </div>
    </nav>

    <div class="container mt-4">
        <!-- Statistics Dashboard -->
        <div class="row">
            <div class="col-md-3">
                <div class="card stats-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Total Examples</h5>
                        <h2 class="text-primary">{{ stats.total_examples }}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stats-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Avg Steps</h5>
                        <h2 class="text-success">{{ "%.1f"|format(stats.avg_steps) }}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stats-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Avg Reward</h5>
                        <h2 class="text-info">{{ "%.3f"|format(stats.quality_metrics.mean_reward if stats.quality_metrics else 0) }}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stats-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Avg Step Length</h5>
                        <h2 class="text-warning">{{ "%.1f"|format(stats.avg_step_length) }} words</h2>
                    </div>
                </div>
            </div>
        </div>

        <!-- Visualizations -->
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Reward Distribution</h5>
                    </div>
                    <div class="card-body">
                        <div id="reward-plot"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Step Count Distribution</h5>
                    </div>
                    <div class="card-body">
                        <div id="step-plot"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Search and Examples -->
        <div class="search-container">
            <div class="input-group">
                <input type="text" class="form-control" id="searchInput" placeholder="Search examples...">
                <button class="btn btn-outline-secondary" type="button" id="searchBtn">Search</button>
            </div>
        </div>

        <div id="examples-container">
            <!-- Examples will be loaded here -->
        </div>

        <!-- Pagination -->
        <nav aria-label="Examples pagination">
            <ul class="pagination justify-content-center" id="pagination">
                <!-- Pagination will be loaded here -->
            </ul>
        </nav>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentPage = 1;
        const perPage = 10;

        // Load visualizations
        fetch('/api/visualizations/reward_distribution')
            .then(response => response.json())
            .then(data => {
                Plotly.newPlot('reward-plot', data.data, data.layout);
            });

        fetch('/api/visualizations/step_distribution')
            .then(response => response.json())
            .then(data => {
                Plotly.newPlot('step-plot', data.data, data.layout);
            });

        // Load examples
        function loadExamples(page = 1, search = '') {
            fetch(`/api/examples?page=${page}&per_page=${perPage}&search=${encodeURIComponent(search)}`)
                .then(response => response.json())
                .then(data => {
                    displayExamples(data.examples);
                    displayPagination(data.page, data.total_pages);
                    currentPage = data.page;
                });
        }

        function displayExamples(examples) {
            const container = document.getElementById('examples-container');
            container.innerHTML = '';

            examples.forEach(example => {
                const card = document.createElement('div');
                card.className = 'card example-card';
                card.innerHTML = `
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h6 class="mb-0">Example ${example.id}</h6>
                        <div>
                            <span class="badge bg-primary">${example.num_steps} steps</span>
                            <span class="badge bg-success">Avg Reward: ${example.avg_reward.toFixed(3)}</span>
                            <span class="badge bg-${example.meta_label ? 'success' : 'secondary'}">Meta: ${example.meta_label}</span>
                        </div>
                    </div>
                    <div class="card-body">
                        <h6>Problem:</h6>
                        <p class="text-muted">${example.informal_prefix || example.formal_statement}</p>
                        <h6>Reasoning Preview:</h6>
                        <p class="text-truncate">${example.cot_response.substring(0, 200)}...</p>
                        <a href="/example/${example.id}" class="btn btn-sm btn-outline-primary">View Details</a>
                    </div>
                `;
                container.appendChild(card);
            });
        }

        function displayPagination(currentPage, totalPages) {
            const pagination = document.getElementById('pagination');
            pagination.innerHTML = '';

            for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
                const li = document.createElement('li');
                li.className = `page-item ${i === currentPage ? 'active' : ''}`;
                li.innerHTML = `<a class="page-link" href="#" onclick="loadExamples(${i})">${i}</a>`;
                pagination.appendChild(li);
            }
        }

        // Search functionality
        document.getElementById('searchBtn').addEventListener('click', () => {
            const search = document.getElementById('searchInput').value;
            loadExamples(1, search);
        });

        document.getElementById('searchInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const search = document.getElementById('searchInput').value;
                loadExamples(1, search);
            }
        });

        // Initial load
        loadExamples();
    </script>
</body>
</html>'''

    # Example detail template
    example_detail_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Example {{ example_id }} - DreamPRM MinIF2F</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .step-card { margin-bottom: 15px; }
        .step-content { white-space: pre-wrap; }
        .reward-circle {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">← Back to Dataset</a>
            <span class="navbar-text">Example {{ example_id }}</span>
        </div>
    </nav>

    <div class="container mt-4">
        <div id="example-detail">
            <!-- Example details will be loaded here -->
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function getRewardColor(reward) {
            if (reward >= 0.8) return '#28a745';
            if (reward >= 0.6) return '#ffc107';
            if (reward >= 0.4) return '#fd7e14';
            return '#dc3545';
        }

        fetch(`/api/example/{{ example_id }}`)
            .then(response => response.json())
            .then(example => {
                const container = document.getElementById('example-detail');
                
                let html = `
                    <div class="row">
                        <div class="col-md-8">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Problem Statement</h5>
                                </div>
                                <div class="card-body">
                                    <h6>Informal:</h6>
                                    <p>${example.informal_prefix}</p>
                                    <h6>Formal:</h6>
                                    <p class="font-monospace">${example.formal_statement}</p>
                                </div>
                            </div>

                            <div class="card mt-3">
                                <div class="card-header">
                                    <h5>Chain of Thought Reasoning</h5>
                                </div>
                                <div class="card-body">
                `;

                example.step_breakdown.forEach((step, index) => {
                    const rewardColor = step.reward ? getRewardColor(step.reward) : '#6c757d';
                    html += `
                        <div class="card step-card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h6 class="mb-0">Step ${step.step_number}</h6>
                                <div class="d-flex align-items-center">
                                    <small class="text-muted me-2">${step.word_count} words</small>
                                    ${step.reward ? `<div class="reward-circle" style="background-color: ${rewardColor}">${step.reward.toFixed(2)}</div>` : ''}
                                </div>
                            </div>
                            <div class="card-body">
                                <div class="step-content">${step.content}</div>
                            </div>
                        </div>
                    `;
                });

                html += `
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Summary Statistics</h5>
                                </div>
                                <div class="card-body">
                                    <p><strong>Total Steps:</strong> ${example.summary.total_steps}</p>
                                    <p><strong>Total Rewards:</strong> ${example.summary.total_rewards}</p>
                                    <p><strong>Average Reward:</strong> ${example.summary.avg_reward.toFixed(3)}</p>
                                    <p><strong>Reward Variance:</strong> ${example.summary.reward_variance.toFixed(3)}</p>
                                    <p><strong>Total Words:</strong> ${example.summary.total_words}</p>
                                    <p><strong>Avg Words/Step:</strong> ${example.summary.avg_words_per_step.toFixed(1)}</p>
                                    <p><strong>Meta Label:</strong> <span class="badge bg-${example.meta_label ? 'success' : 'secondary'}">${example.meta_label}</span></p>
                                </div>
                            </div>
                        </div>
                    </div>
                `;

                container.innerHTML = html;
            })
            .catch(error => {
                document.getElementById('example-detail').innerHTML = 
                    '<div class="alert alert-danger">Error loading example details.</div>';
            });
    </script>
</body>
</html>'''

    # Write templates
    with open(templates_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    
    with open(templates_dir / "example_detail.html", "w", encoding="utf-8") as f:
        f.write(example_detail_html)

def main():
    """Main function to run the visualizer."""
    parser = argparse.ArgumentParser(description="DreamPRM MinIF2F Dataset Visualizer")
    parser.add_argument("--data_file", type=str, required=True, 
                       help="Path to the dataset file (JSON or JSONL)")
    parser.add_argument("--filter_steps", type=int, default=None,
                       help="Filter examples to only show those with exactly N steps")
    parser.add_argument("--min_rewards", type=int, default=None,
                       help="Filter examples to only show those with at least N rewards")
    parser.add_argument("--include_zero_rewards", action="store_true",
                       help="Include examples where all rewards are 0 (by default they are skipped)")
    parser.add_argument("--host", type=str, default="localhost", 
                       help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, 
                       help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", 
                       help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Load dataset
    global dataset, dataset_stats, current_file
    current_file = Path(args.data_file).name
    
    # Add filtering information to current_file display
    filter_info = []
    if args.filter_steps is not None:
        filter_info.append(f"{args.filter_steps} steps")
    if args.min_rewards is not None:
        filter_info.append(f"≥{args.min_rewards} rewards")
    
    if filter_info:
        current_file = f"{current_file} (filtered: {' + '.join(filter_info)})"
    
    dataset = load_dataset(args.data_file, filter_steps=args.filter_steps, min_rewards=args.min_rewards, skip_zero_rewards=not args.include_zero_rewards)
    
    if not dataset:
        print("No data loaded. Please check your file path and format.")
        if args.filter_steps is not None or args.min_rewards is not None:
            filters = []
            if args.filter_steps is not None:
                filters.append(f"exactly {args.filter_steps} steps")
            if args.min_rewards is not None:
                filters.append(f"at least {args.min_rewards} rewards")
            print(f"Note: No examples found with {' and '.join(filters)}.")
        return
    
    # Analyze dataset
    print("Analyzing dataset...")
    dataset_stats = analyze_dataset(dataset)
    
    # Create templates
    print("Creating web templates...")
    create_templates()
    
    # Print statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    if args.filter_steps is not None or args.min_rewards is not None:
        filter_desc = []
        if args.filter_steps is not None:
            filter_desc.append(f"{args.filter_steps} steps")
        if args.min_rewards is not None:
            filter_desc.append(f"≥{args.min_rewards} rewards")
        print(f"(FILTERED: {' + '.join(filter_desc)})")
    print("="*50)
    print(f"Total examples: {dataset_stats['total_examples']}")
    print(f"Average steps per example: {dataset_stats['avg_steps']:.1f}")
    print(f"Average rewards per example: {dataset_stats['avg_rewards']:.1f}")
    print(f"Average step length: {dataset_stats['avg_step_length']:.1f} words")
    
    if dataset_stats['quality_metrics']:
        print(f"Reward statistics:")
        print(f"  Mean: {dataset_stats['quality_metrics']['mean_reward']:.3f}")
        print(f"  Std:  {dataset_stats['quality_metrics']['std_reward']:.3f}")
        print(f"  Min:  {dataset_stats['quality_metrics']['min_reward']:.3f}")
        print(f"  Max:  {dataset_stats['quality_metrics']['max_reward']:.3f}")
    
    print(f"Meta label distribution: {dataset_stats['meta_label_distribution']}")
    print("="*50)
    
    # Start web server
    print(f"\nStarting web server...")
    if args.filter_steps is not None or args.min_rewards is not None:
        filter_desc = []
        if args.filter_steps is not None:
            filter_desc.append(f"{args.filter_steps} steps")
        if args.min_rewards is not None:
            filter_desc.append(f"≥{args.min_rewards} rewards")
        print(f"Showing only examples with {' and '.join(filter_desc)}")
    print(f"Open your browser and go to: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()