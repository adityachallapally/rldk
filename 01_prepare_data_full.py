#!/usr/bin/env python3
"""
01_prepare_data_full.py - Generate comprehensive synthetic preference data for RLHF training

This script creates:
- 1000+ synthetic preference pairs for reward model training
- 200+ validation pairs
- 20+ PPO prompts for policy training
- 8+ fixed probe prompts for evaluation

All data is designed to be realistic and diverse, with proper preference labeling
based on helpfulness, harmlessness, and honesty criteria.
"""

import json
import random
import hashlib
from typing import List, Dict, Any
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Prompt templates for different types of interactions
PROMPT_TEMPLATES = [
    # General knowledge questions
    "Explain the concept of {topic} in simple terms.",
    "What are the main benefits and drawbacks of {topic}?",
    "How does {topic} work?",
    "What is the history of {topic}?",
    "Compare {topic} with {topic2}.",
    
    # Creative writing
    "Write a short story about {topic}.",
    "Create a poem about {topic}.",
    "Describe a day in the life of {topic}.",
    "Write a dialogue between {topic} and {topic2}.",
    
    # Problem solving
    "How would you solve this problem: {problem}?",
    "What steps would you take to {action}?",
    "Analyze this situation: {situation}",
    "What are the potential solutions to {problem}?",
    
    # Opinion and advice
    "What is your opinion on {topic}?",
    "Give advice about {topic}.",
    "What would you recommend for {situation}?",
    "How should someone handle {situation}?",
    
    # Technical explanations
    "Explain how to {technical_action}.",
    "What are the technical requirements for {topic}?",
    "Describe the process of {technical_process}.",
    "What tools are needed for {technical_task}?",
    
    # Ethical and social questions
    "What are the ethical implications of {topic}?",
    "How does {topic} affect society?",
    "What are the social consequences of {action}?",
    "Discuss the moral aspects of {topic}.",
    
    # Educational content
    "Teach me about {topic}.",
    "What should I know about {topic}?",
    "Explain {topic} like I'm 10 years old.",
    "What are the key concepts in {topic}?",
    
    # Practical applications
    "How can I use {topic} in my daily life?",
    "What are practical applications of {topic}?",
    "Give me examples of {topic} in action.",
    "How is {topic} used in industry?",
    
    # Comparative analysis
    "What's the difference between {topic} and {topic2}?",
    "Which is better: {topic} or {topic2}?",
    "Compare the advantages of {topic} vs {topic2}.",
    "What are the similarities between {topic} and {topic2}?",
    
    # Future and trends
    "What is the future of {topic}?",
    "What trends do you see in {topic}?",
    "How will {topic} evolve?",
    "What changes do you expect in {topic}?",
]

# Topics for filling in templates
TOPICS = [
    "artificial intelligence", "machine learning", "renewable energy", "climate change",
    "space exploration", "quantum computing", "blockchain technology", "virtual reality",
    "sustainable living", "mental health", "education", "healthcare", "democracy",
    "social media", "privacy", "cybersecurity", "robotics", "biotechnology",
    "urban planning", "economic inequality", "cultural diversity", "environmental protection",
    "scientific research", "innovation", "creativity", "leadership", "teamwork",
    "communication", "problem solving", "critical thinking", "emotional intelligence",
    "time management", "productivity", "work-life balance", "personal development",
    "financial planning", "investment", "entrepreneurship", "technology adoption",
    "digital transformation", "remote work", "online learning", "social networking",
    "data privacy", "algorithmic bias", "automation", "human-computer interaction",
    "user experience", "design thinking", "project management", "quality assurance"
]

# Problems and situations for templates
PROBLEMS = [
    "reducing carbon footprint", "improving team communication", "managing stress",
    "learning a new skill", "solving conflicts", "increasing productivity",
    "building better relationships", "making difficult decisions", "overcoming obstacles",
    "adapting to change", "balancing priorities", "achieving goals",
    "handling criticism", "building confidence", "developing creativity",
    "improving focus", "managing time effectively", "building trust",
    "resolving disputes", "encouraging innovation", "fostering collaboration",
    "promoting diversity", "ensuring fairness", "maintaining quality",
    "scaling operations", "reducing costs", "improving efficiency",
    "enhancing security", "protecting privacy", "ensuring compliance"
]

# Technical actions and processes
TECHNICAL_ACTIONS = [
    "implement a machine learning model", "design a user interface", "optimize database performance",
    "deploy a web application", "set up a development environment", "configure a server",
    "debug code issues", "write unit tests", "perform code review",
    "manage version control", "automate testing", "monitor system performance",
    "backup data", "restore from backup", "scale infrastructure",
    "implement security measures", "conduct security audit", "manage user permissions",
    "integrate APIs", "handle errors gracefully", "optimize algorithms",
    "profile application performance", "refactor legacy code", "document code",
    "create technical specifications", "plan system architecture", "conduct load testing"
]

def generate_prompt() -> str:
    """Generate a random prompt using templates and topics."""
    template = random.choice(PROMPT_TEMPLATES)
    
    # Fill in topic placeholders
    if "{topic}" in template and "{topic2}" in template:
        topic1 = random.choice(TOPICS)
        topic2 = random.choice([t for t in TOPICS if t != topic1])
        return template.format(topic=topic1, topic2=topic2)
    elif "{topic}" in template:
        topic = random.choice(TOPICS)
        return template.format(topic=topic)
    elif "{problem}" in template:
        problem = random.choice(PROBLEMS)
        return template.format(problem=problem)
    elif "{situation}" in template:
        situation = random.choice(PROBLEMS)
        return template.format(situation=situation)
    elif "{action}" in template:
        action = random.choice(PROBLEMS)
        return template.format(action=action)
    elif "{technical_action}" in template:
        tech_action = random.choice(TECHNICAL_ACTIONS)
        return template.format(technical_action=tech_action)
    elif "{technical_process}" in template:
        tech_process = random.choice(TECHNICAL_ACTIONS)
        return template.format(technical_process=tech_process)
    elif "{technical_task}" in template:
        tech_task = random.choice(TECHNICAL_ACTIONS)
        return template.format(technical_task=tech_task)
    else:
        return template

def generate_response(prompt: str, quality: str = "good") -> str:
    """Generate a response of specified quality."""
    if quality == "good":
        return generate_good_response(prompt)
    elif quality == "bad":
        return generate_bad_response(prompt)
    else:
        return generate_mediocre_response(prompt)

def generate_good_response(prompt: str) -> str:
    """Generate a high-quality, helpful response."""
    responses = {
        "explain": "I'd be happy to explain that concept. Let me break it down into clear, understandable parts...",
        "compare": "That's an excellent question. Let me provide a comprehensive comparison...",
        "advice": "Based on my understanding, here are some practical recommendations...",
        "problem": "I can help you work through this step by step. Here's a systematic approach...",
        "creative": "What an interesting creative challenge! Here's my take on that...",
        "technical": "From a technical perspective, here's how you would approach this...",
        "ethical": "This raises important ethical considerations. Let me explore the key issues...",
        "educational": "Great question! Let me teach you about this in a clear, engaging way...",
        "practical": "Here are some practical ways you can apply this in real-world situations...",
        "future": "Looking ahead, I see several important trends and developments..."
    }
    
    # Determine response type based on prompt keywords
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ["explain", "what is", "how does"]):
        response_type = "explain"
    elif any(word in prompt_lower for word in ["compare", "difference", "better"]):
        response_type = "compare"
    elif any(word in prompt_lower for word in ["advice", "recommend", "should"]):
        response_type = "advice"
    elif any(word in prompt_lower for word in ["solve", "problem", "how would"]):
        response_type = "problem"
    elif any(word in prompt_lower for word in ["write", "story", "poem", "creative"]):
        response_type = "creative"
    elif any(word in prompt_lower for word in ["technical", "implement", "how to"]):
        response_type = "technical"
    elif any(word in prompt_lower for word in ["ethical", "moral", "implications"]):
        response_type = "ethical"
    elif any(word in prompt_lower for word in ["teach", "learn", "understand"]):
        response_type = "educational"
    elif any(word in prompt_lower for word in ["practical", "use", "apply"]):
        response_type = "practical"
    elif any(word in prompt_lower for word in ["future", "trend", "evolve"]):
        response_type = "future"
    else:
        response_type = "explain"
    
    base_response = responses[response_type]
    
    # Add detailed content based on the prompt
    if "artificial intelligence" in prompt.lower():
        return base_response + " Artificial intelligence is a rapidly evolving field that combines computer science, mathematics, and cognitive science to create systems that can perform tasks typically requiring human intelligence. The key components include machine learning algorithms, neural networks, and natural language processing. Recent advances in deep learning have enabled breakthroughs in areas like computer vision, speech recognition, and autonomous systems. However, it's important to consider the ethical implications, including bias in algorithms, job displacement, and the need for responsible AI development."
    elif "climate change" in prompt.lower():
        return base_response + " Climate change represents one of the most pressing challenges of our time. It's caused primarily by greenhouse gas emissions from human activities, particularly the burning of fossil fuels. The effects include rising global temperatures, sea level rise, extreme weather events, and ecosystem disruption. Solutions require both mitigation (reducing emissions) and adaptation (preparing for changes). Key strategies include transitioning to renewable energy, improving energy efficiency, protecting natural carbon sinks, and developing climate-resilient infrastructure."
    elif "machine learning" in prompt.lower():
        return base_response + " Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It works by identifying patterns in data and using those patterns to make predictions or decisions. The main types include supervised learning (learning from labeled examples), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through trial and error). Popular algorithms include linear regression, decision trees, neural networks, and support vector machines."
    else:
        return base_response + " This is a complex topic that requires careful consideration of multiple factors. The key points to understand are: 1) The fundamental concepts and principles involved, 2) The practical applications and real-world examples, 3) The potential benefits and limitations, 4) The current state of research and development, and 5) Future prospects and challenges. By examining these aspects systematically, we can develop a comprehensive understanding of the subject matter."

def generate_bad_response(prompt: str) -> str:
    """Generate a low-quality, unhelpful response."""
    bad_responses = [
        "I don't know.",
        "That's a stupid question.",
        "Google it.",
        "I can't help with that.",
        "This is too complicated for me.",
        "Ask someone else.",
        "I'm not sure what you mean.",
        "That doesn't make sense.",
        "I don't have time for this.",
        "You should figure it out yourself.",
        "This is boring.",
        "I don't care about this topic.",
        "Why are you asking me?",
        "I'm not qualified to answer.",
        "This is above my pay grade.",
        "I don't understand the question.",
        "Can you rephrase that?",
        "I'm confused.",
        "This is too hard.",
        "I give up."
    ]
    
    # Sometimes give a longer but still bad response
    if random.random() < 0.3:
        return random.choice(bad_responses) + " " + random.choice([
            "Maybe try asking someone who actually knows what they're talking about.",
            "I really don't see the point of this question.",
            "There are better things to worry about.",
            "This seems like a waste of time.",
            "I'm not going to waste my time on this.",
            "You're asking the wrong person.",
            "This is not my area of expertise.",
            "I don't have the energy for this.",
            "Can we talk about something more interesting?",
            "I'm not in the mood to help."
        ])
    
    return random.choice(bad_responses)

def generate_mediocre_response(prompt: str) -> str:
    """Generate a mediocre, partially helpful response."""
    mediocre_responses = [
        "That's an interesting question. I think it might be related to...",
        "I'm not entirely sure, but I believe...",
        "This is a complex topic. From what I understand...",
        "I don't have all the details, but generally speaking...",
        "It's hard to say for certain, but my guess is...",
        "I'm not an expert, but I think...",
        "This is outside my area of expertise, but...",
        "I'm not completely confident about this, but...",
        "I might be wrong, but I think...",
        "I don't have the full picture, but..."
    ]
    
    base = random.choice(mediocre_responses)
    
    # Add some content but make it vague or incomplete
    if "artificial intelligence" in prompt.lower():
        return base + " artificial intelligence is about computers doing smart things. There are different types like machine learning and neural networks. It's used in many applications today."
    elif "climate change" in prompt.lower():
        return base + " climate change is about the Earth getting warmer. It's caused by pollution and greenhouse gases. We need to reduce emissions and use cleaner energy."
    elif "machine learning" in prompt.lower():
        return base + " machine learning is when computers learn from data. There are different algorithms and methods. It's used in many applications like recommendation systems."
    else:
        return base + " this involves several factors and considerations. There are different approaches and methods. It's an important topic that affects many areas."

def create_preference_pair() -> Dict[str, Any]:
    """Create a single preference pair with prompt and two responses."""
    prompt = generate_prompt()
    
    # Generate responses of different qualities
    # 70% chance of good vs bad, 20% chance of good vs mediocre, 10% chance of mediocre vs bad
    rand = random.random()
    if rand < 0.7:
        chosen = generate_response(prompt, "good")
        rejected = generate_response(prompt, "bad")
    elif rand < 0.9:
        chosen = generate_response(prompt, "good")
        rejected = generate_response(prompt, "mediocre")
    else:
        chosen = generate_response(prompt, "mediocre")
        rejected = generate_response(prompt, "bad")
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "generated": True,
            "quality_control": "synthetic_preference",
            "preference_criteria": ["helpfulness", "harmlessness", "honesty"]
        }
    }

def create_ppo_prompt() -> Dict[str, Any]:
    """Create a prompt for PPO training."""
    prompt = generate_prompt()
    return {
        "prompt": prompt,
        "metadata": {
            "type": "ppo_training",
            "generated": True,
            "category": "synthetic"
        }
    }

def create_probe_prompt() -> Dict[str, Any]:
    """Create a fixed probe prompt for evaluation."""
    probe_prompts = [
        "Explain the concept of artificial intelligence in simple terms.",
        "What are the main benefits and drawbacks of renewable energy?",
        "How would you solve the problem of reducing carbon footprint?",
        "What is your opinion on the future of work?",
        "Give advice about managing stress in a busy work environment.",
        "Compare the advantages of remote work vs office work.",
        "What are the ethical implications of social media?",
        "How can I use machine learning in my daily life?"
    ]
    
    prompt = random.choice(probe_prompts)
    return {
        "prompt": prompt,
        "metadata": {
            "type": "evaluation_probe",
            "generated": True,
            "category": "fixed_evaluation"
        }
    }

def save_jsonl(data: List[Dict[str, Any]], filename: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} items to {filename}")

def calculate_data_hash(data: List[Dict[str, Any]]) -> str:
    """Calculate hash of data for reproducibility tracking."""
    data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

def main():
    """Generate all required datasets."""
    print("Generating comprehensive synthetic preference data for RLHF training...")
    print(f"Using random seed: {RANDOM_SEED}")
    
    # Generate reward model training data (1000+ pairs)
    print("\nGenerating reward model training pairs...")
    rm_train_pairs = []
    for i in range(1200):  # Generate 1200 pairs
        if i % 100 == 0:
            print(f"Generated {i} pairs...")
        rm_train_pairs.append(create_preference_pair())
    
    print(f"Generated {len(rm_train_pairs)} training pairs")
    
    # Generate reward model validation data (200+ pairs)
    print("\nGenerating reward model validation pairs...")
    rm_val_pairs = []
    for i in range(250):  # Generate 250 pairs
        if i % 50 == 0:
            print(f"Generated {i} pairs...")
        rm_val_pairs.append(create_preference_pair())
    
    print(f"Generated {len(rm_val_pairs)} validation pairs")
    
    # Generate PPO prompts (20+ prompts)
    print("\nGenerating PPO training prompts...")
    ppo_prompts = []
    for i in range(25):  # Generate 25 prompts
        ppo_prompts.append(create_ppo_prompt())
    
    print(f"Generated {len(ppo_prompts)} PPO prompts")
    
    # Generate fixed probe prompts (8+ prompts)
    print("\nGenerating fixed probe prompts...")
    probe_prompts = []
    for i in range(10):  # Generate 10 probe prompts
        probe_prompts.append(create_probe_prompt())
    
    print(f"Generated {len(probe_prompts)} probe prompts")
    
    # Save all datasets
    print("\nSaving datasets...")
    save_jsonl(rm_train_pairs, "./rldk_demos/rm_pairs_train.jsonl")
    save_jsonl(rm_val_pairs, "./rldk_demos/rm_pairs_val.jsonl")
    save_jsonl(ppo_prompts, "./rldk_demos/ppo_prompts.jsonl")
    save_jsonl(probe_prompts, "./rldk_demos/probes.jsonl")
    
    # Calculate and save data hashes for reproducibility
    data_hashes = {
        "rm_pairs_train": calculate_data_hash(rm_train_pairs),
        "rm_pairs_val": calculate_data_hash(rm_val_pairs),
        "ppo_prompts": calculate_data_hash(ppo_prompts),
        "probes": calculate_data_hash(probe_prompts),
        "random_seed": RANDOM_SEED
    }
    
    with open("./rldk_demos/data_hashes.json", 'w') as f:
        json.dump(data_hashes, f, indent=2)
    
    print("\nData generation complete!")
    print(f"Training pairs: {len(rm_train_pairs)}")
    print(f"Validation pairs: {len(rm_val_pairs)}")
    print(f"PPO prompts: {len(ppo_prompts)}")
    print(f"Probe prompts: {len(probe_prompts)}")
    print(f"Data hashes saved to: ./rldk_demos/data_hashes.json")
    
    # Show sample data
    print("\nSample training pair:")
    sample = rm_train_pairs[0]
    print(f"Prompt: {sample['prompt']}")
    print(f"Chosen: {sample['chosen'][:100]}...")
    print(f"Rejected: {sample['rejected'][:100]}...")

if __name__ == "__main__":
    main()