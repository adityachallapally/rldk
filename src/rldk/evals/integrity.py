"""Evaluation integrity checks for detecting prompt contamination and answer leakage."""

import re
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_prompt_contamination(
    data: pd.DataFrame, seed: int = 42, **kwargs
) -> Dict[str, Any]:
    """
    Evaluate for prompt contamination in evaluation data.

    Detects if evaluation prompts contain information that could bias the model
    or if there's contamination between train/validation/test splits.

    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        Dictionary with contamination score and details
    """
    np.random.seed(seed)

    contamination_metrics = []
    contamination_score = 0.0

    # Check for prompt-related columns
    prompt_cols = [col for col in data.columns if any(keyword in col.lower()
                                                    for keyword in ['prompt', 'query', 'input', 'text'])]

    if not prompt_cols:
        return {
            "score": 0.5,  # Neutral score when no prompt data available
            "details": "No prompt data available for contamination analysis",
            "method": "no_prompt_data",
            "metrics": [],
            "sample_size": len(data),
        }

    # 1. Check for duplicate prompts (potential contamination)
    for col in prompt_cols:
        if col in data.columns:
            duplicates = data[col].duplicated().sum()
            duplicate_ratio = duplicates / len(data)
            contamination_metrics.append(("duplicate_prompts", duplicate_ratio))

            if duplicate_ratio > 0.1:  # More than 10% duplicates
                contamination_score += 0.3

    # 2. Check for prompt length anomalies (very short/long prompts might indicate issues)
    for col in prompt_cols:
        if col in data.columns:
            prompt_lengths = data[col].astype(str).str.len()
            length_std = prompt_lengths.std()
            length_mean = prompt_lengths.mean()

            # Check for suspicious length patterns
            if length_std < 5:  # Very uniform lengths
                contamination_metrics.append(("uniform_prompt_lengths", length_std))
                contamination_score += 0.2

            if length_mean < 10:  # Very short prompts
                contamination_metrics.append(("short_prompts", length_mean))
                contamination_score += 0.2

    # 3. Check for suspicious patterns in prompts
    for col in prompt_cols:
        if col in data.columns:
            # Check for test-like patterns
            test_patterns = [
                r'test\s+question',
                r'answer\s+the\s+following',
                r'choose\s+the\s+correct',
                r'multiple\s+choice',
                r'[a-d]\)',  # Multiple choice options
            ]

            pattern_matches = 0
            for pattern in test_patterns:
                matches = data[col].astype(str).str.contains(pattern, case=False, regex=True).sum()
                pattern_matches += matches

            pattern_ratio = pattern_matches / len(data)
            contamination_metrics.append(("test_patterns", pattern_ratio))

            if pattern_ratio > 0.3:  # More than 30% have test patterns
                contamination_score += 0.3

    # 4. Check for metadata leakage in prompts
    metadata_cols = ['epoch', 'step', 'batch_idx', 'run_id', 'timestamp']
    for col in prompt_cols:
        if col in data.columns:
            for meta_col in metadata_cols:
                if meta_col in data.columns:
                    # Check if metadata values appear in prompts
                    leakage_count = 0
                    for idx, prompt in enumerate(data[col].astype(str)):
                        meta_value = str(data.iloc[idx][meta_col])
                        if meta_value in prompt and len(meta_value) > 3:  # Avoid short matches
                            leakage_count += 1

                    leakage_ratio = leakage_count / len(data)
                    if leakage_ratio > 0.05:  # More than 5% have metadata leakage
                        contamination_metrics.append((f"{meta_col}_leakage", leakage_ratio))
                        contamination_score += 0.4

    # Normalize score to [0, 1] range
    contamination_score = min(contamination_score, 1.0)

    return {
        "score": float(1.0 - contamination_score),  # Higher score = less contamination
        "details": f"Prompt contamination evaluation based on {len(contamination_metrics)} metrics",
        "method": "pattern_and_metadata_analysis",
        "metrics": contamination_metrics,
        "sample_size": len(data),
    }


def evaluate_answer_leakage(
    data: pd.DataFrame, seed: int = 42, **kwargs
) -> Dict[str, Any]:
    """
    Evaluate for answer leakage in evaluation data.

    Detects if the expected answers or solutions are inadvertently present
    in the prompts or if there's information that could give away the answer.

    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        Dictionary with leakage score and details
    """
    np.random.seed(seed)

    leakage_metrics = []
    leakage_score = 0.0

    # Check for response/answer-related columns
    response_cols = [col for col in data.columns if any(keyword in col.lower()
                                                       for keyword in ['response', 'answer', 'output', 'target'])]
    prompt_cols = [col for col in data.columns if any(keyword in col.lower()
                                                     for keyword in ['prompt', 'query', 'input', 'text'])]

    if not response_cols or not prompt_cols:
        return {
            "score": 0.5,  # Neutral score when no response/prompt data available
            "details": "No response/prompt data available for leakage analysis",
            "method": "no_response_data",
            "metrics": [],
            "sample_size": len(data),
        }

    # 1. Check for direct answer leakage (answer appears in prompt)
    for prompt_col in prompt_cols:
        for response_col in response_cols:
            if prompt_col in data.columns and response_col in data.columns:
                direct_leakage_count = 0

                for idx in range(len(data)):
                    prompt = str(data.iloc[idx][prompt_col]).lower()
                    response = str(data.iloc[idx][response_col]).lower()

                    # Check if significant parts of the response appear in the prompt
                    response_words = set(response.split())
                    prompt_words = set(prompt.split())

                    # Calculate word overlap
                    if len(response_words) > 0:
                        overlap_ratio = len(response_words.intersection(prompt_words)) / len(response_words)
                        if overlap_ratio > 0.3:  # More than 30% overlap
                            direct_leakage_count += 1

                direct_leakage_ratio = direct_leakage_count / len(data)
                leakage_metrics.append(("direct_answer_leakage", direct_leakage_ratio))

                if direct_leakage_ratio > 0.1:  # More than 10% have direct leakage
                    leakage_score += 0.5

    # 2. Check for partial answer leakage (key parts of answer in prompt)
    for prompt_col in prompt_cols:
        for response_col in response_cols:
            if prompt_col in data.columns and response_col in data.columns:
                partial_leakage_count = 0

                for idx in range(len(data)):
                    prompt = str(data.iloc[idx][prompt_col]).lower()
                    response = str(data.iloc[idx][response_col]).lower()

                    # Look for key phrases that might give away the answer
                    key_phrases = [
                        'the answer is',
                        'correct answer',
                        'solution:',
                        'result:',
                        'therefore',
                        'thus',
                        'conclusion:',
                    ]

                    for phrase in key_phrases:
                        if phrase in prompt and any(word in response for word in phrase.split()):
                            partial_leakage_count += 1
                            break

                partial_leakage_ratio = partial_leakage_count / len(data)
                leakage_metrics.append(("partial_answer_leakage", partial_leakage_ratio))

                if partial_leakage_ratio > 0.2:  # More than 20% have partial leakage
                    leakage_score += 0.3

    # 3. Check for numerical answer leakage
    for prompt_col in prompt_cols:
        for response_col in response_cols:
            if prompt_col in data.columns and response_col in data.columns:
                numerical_leakage_count = 0

                for idx in range(len(data)):
                    prompt = str(data.iloc[idx][prompt_col])
                    response = str(data.iloc[idx][response_col])

                    # Extract numbers from both prompt and response
                    prompt_numbers = set(re.findall(r'\d+\.?\d*', prompt))
                    response_numbers = set(re.findall(r'\d+\.?\d*', response))

                    # Check for number overlap (excluding common numbers like years, etc.)
                    if len(response_numbers) > 0:
                        overlap = prompt_numbers.intersection(response_numbers)
                        # Filter out common numbers (years, small numbers, etc.)
                        significant_overlap = [n for n in overlap
                                             if len(n) > 2 or (len(n) <= 2 and int(n) > 20)]

                        if len(significant_overlap) > 0:
                            numerical_leakage_count += 1

                numerical_leakage_ratio = numerical_leakage_count / len(data)
                leakage_metrics.append(("numerical_answer_leakage", numerical_leakage_ratio))

                if numerical_leakage_ratio > 0.15:  # More than 15% have numerical leakage
                    leakage_score += 0.2

    # 4. Check for semantic similarity between prompts and responses
    if len(data) > 10:  # Need sufficient data for meaningful similarity analysis
        for prompt_col in prompt_cols:
            for response_col in response_cols:
                if prompt_col in data.columns and response_col in data.columns:
                    try:
                        # Use TF-IDF for semantic similarity
                        combined_texts = []
                        for idx in range(len(data)):
                            prompt = str(data.iloc[idx][prompt_col])
                            response = str(data.iloc[idx][response_col])
                            combined_texts.append(f"{prompt} {response}")

                        if len(combined_texts) > 0:
                            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                            tfidf_matrix = vectorizer.fit_transform(combined_texts)

                            # Calculate cosine similarities
                            similarities = cosine_similarity(tfidf_matrix)

                            # Check for high similarities (potential leakage)
                            high_similarity_count = 0
                            for i in range(len(similarities)):
                                for j in range(i+1, len(similarities)):
                                    if similarities[i][j] > 0.8:  # Very high similarity
                                        high_similarity_count += 1

                            similarity_ratio = high_similarity_count / (len(similarities) * (len(similarities) - 1) / 2)
                            leakage_metrics.append(("semantic_similarity_leakage", similarity_ratio))

                            if similarity_ratio > 0.1:  # More than 10% have high similarity
                                leakage_score += 0.2
                    except Exception:
                        # Skip if vectorization fails
                        continue

    # Normalize score to [0, 1] range
    leakage_score = min(leakage_score, 1.0)

    return {
        "score": float(1.0 - leakage_score),  # Higher score = less leakage
        "details": f"Answer leakage evaluation based on {len(leakage_metrics)} metrics",
        "method": "direct_partial_and_semantic_analysis",
        "metrics": leakage_metrics,
        "sample_size": len(data),
    }


def evaluate_data_split_integrity(
    data: pd.DataFrame, seed: int = 42, **kwargs
) -> Dict[str, Any]:
    """
    Evaluate the integrity of data splits (train/validation/test).

    Detects contamination between different data splits and ensures
    proper separation of evaluation data.

    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        Dictionary with split integrity score and details
    """
    np.random.seed(seed)

    integrity_metrics = []
    integrity_score = 0.0

    # Check for split-related columns
    split_cols = [col for col in data.columns if any(keyword in col.lower()
                                                    for keyword in ['split', 'fold', 'partition', 'dataset'])]

    if not split_cols:
        return {
            "score": 0.5,  # Neutral score when no split data available
            "details": "No split data available for integrity analysis",
            "method": "no_split_data",
            "metrics": [],
            "sample_size": len(data),
        }

    # 1. Check for proper split distribution
    for split_col in split_cols:
        if split_col in data.columns:
            split_counts = data[split_col].value_counts()
            total_samples = len(data)

            # Check if splits are reasonably balanced
            min_split_ratio = split_counts.min() / total_samples
            max_split_ratio = split_counts.max() / total_samples

            balance_score = min_split_ratio / max_split_ratio if max_split_ratio > 0 else 1.0
            integrity_metrics.append(("split_balance", balance_score))

            if balance_score < 0.1:  # Very unbalanced splits
                integrity_score += 0.2

    # 2. Check for duplicate content across splits
    content_cols = [col for col in data.columns if any(keyword in col.lower()
                                                      for keyword in ['prompt', 'query', 'input', 'text', 'response', 'answer'])]

    for content_col in content_cols:
        if content_col in data.columns:
            # Group by split and check for duplicates
            split_duplicates = 0
            total_duplicates = 0

            for split_name in data[split_cols[0]].unique():
                split_data = data[data[split_cols[0]] == split_name]
                duplicates = split_data[content_col].duplicated().sum()
                split_duplicates += duplicates

            # Check for cross-split duplicates
            all_content = data[content_col].astype(str)
            total_duplicates = all_content.duplicated().sum()

            cross_split_duplicates = total_duplicates - split_duplicates
            cross_split_ratio = cross_split_duplicates / len(data)

            integrity_metrics.append(("cross_split_duplicates", cross_split_ratio))

            if cross_split_ratio > 0.05:  # More than 5% cross-split duplicates
                integrity_score += 0.4

    # 3. Check for temporal leakage (if timestamps are available)
    time_cols = [col for col in data.columns if any(keyword in col.lower()
                                                   for keyword in ['timestamp', 'time', 'date', 'created'])]

    for time_col in time_cols:
        if time_col in data.columns and len(split_cols) > 0:
            try:
                # Convert to datetime if possible
                data_copy = data.copy()
                data_copy[time_col] = pd.to_datetime(data_copy[time_col], errors='coerce')

                # Check for temporal ordering violations
                violations = 0
                for split_name in data_copy[split_cols[0]].unique():
                    split_data = data_copy[data_copy[split_cols[0]] == split_name]
                    if len(split_data) > 1:
                        # Check if timestamps are ordered within split
                        original_times = split_data[time_col].values
                        sorted_times = split_data[time_col].sort_values().values
                        if not np.array_equal(original_times, sorted_times):
                            violations += 1

                violation_ratio = violations / len(data_copy[split_cols[0]].unique())
                integrity_metrics.append(("temporal_violations", violation_ratio))

                if violation_ratio > 0.3:  # More than 30% of splits have violations
                    integrity_score += 0.2
            except Exception:
                # Skip if datetime conversion fails
                continue

    # Normalize score to [0, 1] range
    integrity_score = min(integrity_score, 1.0)

    return {
        "score": float(1.0 - integrity_score),  # Higher score = better integrity
        "details": f"Data split integrity evaluation based on {len(integrity_metrics)} metrics",
        "method": "balance_duplicates_and_temporal_analysis",
        "metrics": integrity_metrics,
        "sample_size": len(data),
    }


def evaluate_evaluation_robustness(
    data: pd.DataFrame, seed: int = 42, **kwargs
) -> Dict[str, Any]:
    """
    Evaluate the robustness of evaluation metrics.

    Checks for potential issues that could make evaluations unreliable,
    such as insufficient sample sizes, high variance, or systematic biases.

    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        Dictionary with robustness score and details
    """
    np.random.seed(seed)

    robustness_metrics = []
    robustness_score = 0.0

    # 1. Check sample size adequacy
    sample_size = len(data)
    if sample_size < 10:
        robustness_score += 0.3
        robustness_metrics.append(("small_sample_size", sample_size))
    elif sample_size < 50:
        robustness_score += 0.1
        robustness_metrics.append(("moderate_sample_size", sample_size))
    else:
        robustness_metrics.append(("adequate_sample_size", sample_size))

    # 2. Check for high variance in key metrics
    metric_cols = [col for col in data.columns if any(keyword in col.lower()
                                                     for keyword in ['reward', 'score', 'accuracy', 'loss'])]

    for col in metric_cols:
        if col in data.columns:
            values = pd.to_numeric(data[col], errors='coerce')
            if len(values.dropna()) > 5:
                cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                robustness_metrics.append((f"{col}_coefficient_of_variation", cv))

                if cv > 1.0:  # Very high variance
                    robustness_score += 0.2

    # 3. Check for systematic biases
    # Check for correlation with metadata that shouldn't affect results
    metadata_cols = ['step', 'epoch', 'batch_idx']
    for meta_col in metadata_cols:
        for metric_col in metric_cols:
            if meta_col in data.columns and metric_col in data.columns:
                try:
                    meta_values = pd.to_numeric(data[meta_col], errors='coerce')
                    metric_values = pd.to_numeric(data[metric_col], errors='coerce')

                    # Remove NaN values
                    valid_mask = ~(meta_values.isna() | metric_values.isna())
                    if valid_mask.sum() > 10:
                        correlation = np.corrcoef(meta_values[valid_mask], metric_values[valid_mask])[0, 1]

                        if abs(correlation) > 0.7:  # Strong correlation
                            robustness_metrics.append((f"{meta_col}_{metric_col}_correlation", correlation))
                            robustness_score += 0.3
                except Exception:
                    continue

    # 4. Check for outliers that might skew results
    for col in metric_cols:
        if col in data.columns:
            values = pd.to_numeric(data[col], errors='coerce')
            if len(values.dropna()) > 10:
                # Use IQR method for outlier detection
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1

                outliers = ((values < (Q1 - 1.5 * IQR)) | (values > (Q3 + 1.5 * IQR))).sum()
                outlier_ratio = outliers / len(values.dropna())

                robustness_metrics.append((f"{col}_outlier_ratio", outlier_ratio))

                if outlier_ratio > 0.1:  # More than 10% outliers
                    robustness_score += 0.2

    # Normalize score to [0, 1] range
    robustness_score = min(robustness_score, 1.0)

    return {
        "score": float(1.0 - robustness_score),  # Higher score = more robust
        "details": f"Evaluation robustness analysis based on {len(robustness_metrics)} metrics",
        "method": "sample_size_variance_and_bias_analysis",
        "metrics": robustness_metrics,
        "sample_size": len(data),
    }
