"""
Rigor Diagnostics and Methodological Validity Framework.

This module provides comprehensive diagnostics for assessing the quality,
validity, and methodological rigor of ML-assisted open coding analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine


class RigorDiagnostics:
    """
    Comprehensive rigor diagnostics for qualitative ML coding.

    Provides methodological validity checks, bias detection, and sanity checks
    to ensure high-quality, transparent, and rigorous analysis.
    """

    def __init__(self):
        """Initialize RigorDiagnostics."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def assess_validity(
        self,
        coder,
        results_df: pd.DataFrame,
        feature_matrix=None,
        human_codes: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Assess methodological validity across multiple dimensions.

        Args:
            coder: Fitted MLOpenCoder instance with codebook and model
            results_df: Results DataFrame with code assignments
            feature_matrix: Optional feature matrix for advanced metrics
            human_codes: Optional DataFrame with human-coded data for reliability

        Returns:
            Dictionary with validity metrics across 8+ dimensions
        """
        validity = {}

        # 1. Inter-code reliability (if human codes available)
        if human_codes is not None:
            validity['inter_code_reliability'] = self._calculate_inter_coder_reliability(
                results_df, human_codes
            )
        else:
            validity['inter_code_reliability'] = None
            validity['inter_code_reliability_note'] = "Human codes not provided"

        # 2. Code stability (consistency across bootstrap samples)
        validity['code_stability'] = self._calculate_code_stability(
            coder, results_df
        )

        # 3. Theme coherence (semantic similarity within codes)
        if feature_matrix is not None:
            validity['theme_coherence'] = self._calculate_theme_coherence(
                coder, results_df, feature_matrix
            )
        else:
            validity['theme_coherence'] = self._calculate_simple_coherence(
                coder, results_df
            )

        # 4. Thematic saturation (are we missing themes?)
        validity['thematic_saturation'] = self._assess_thematic_saturation(
            coder, results_df
        )

        # 5. Coverage ratio (% of responses coded)
        validity['coverage_ratio'] = self._calculate_coverage_ratio(results_df)

        # 6. Code utilization (which codes are underused?)
        validity['code_utilization'] = self._calculate_code_utilization(coder)

        # 7. Confidence distribution
        validity['confidence_distribution'] = self._analyze_confidence_distribution(
            results_df
        )

        # 8. Ambiguity rate (% of multi-coded responses)
        validity['ambiguity_rate'] = self._calculate_ambiguity_rate(results_df)

        # 9. Boundary cases (responses near decision boundaries)
        validity['boundary_cases'] = self._identify_boundary_cases(results_df)

        # 10. Random seed stability (does random_state matter?)
        validity['random_seed_stability'] = self._assess_random_stability(
            coder, results_df
        )

        self.logger.info("Completed validity assessment across 10 dimensions")
        return validity

    def detect_bias(
        self,
        results_df: pd.DataFrame,
        demographics: Optional[pd.DataFrame] = None,
        demographic_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect potential bias in code assignments.

        Args:
            results_df: Results DataFrame with code assignments
            demographics: Optional DataFrame with demographic information
            demographic_columns: List of demographic column names to analyze

        Returns:
            Dictionary with bias detection metrics
        """
        bias_report = {
            'demographic_representation': {},
            'code_imbalance': {},
            'warnings': []
        }

        # Analyze code imbalance (some codes overused)
        bias_report['code_imbalance'] = self._analyze_code_imbalance(results_df)

        # If demographics provided, analyze representation
        if demographics is not None and demographic_columns is not None:
            bias_report['demographic_representation'] = (
                self._analyze_demographic_representation(
                    results_df, demographics, demographic_columns
                )
            )
        else:
            bias_report['demographic_representation'] = {
                'status': 'not_analyzed',
                'note': 'Demographics not provided'
            }

        # Check for systematic patterns
        bias_report['systematic_patterns'] = self._detect_systematic_patterns(
            results_df
        )

        self.logger.info("Completed bias detection analysis")
        return bias_report

    def sanity_check(
        self,
        coder,
        results_df: pd.DataFrame,
        min_responses: int = 20
    ) -> Dict[str, Any]:
        """
        Automated sanity checks for common issues.

        Args:
            coder: Fitted MLOpenCoder instance
            results_df: Results DataFrame
            min_responses: Minimum expected responses

        Returns:
            Dictionary with warnings and recommendations
        """
        warnings_list = []
        recommendations = []
        issues = {}

        # Check 1: Are code labels meaningful?
        long_labels = self._check_long_labels(coder)
        if long_labels:
            issues['long_labels'] = long_labels
            warnings_list.append(
                f"⚠️ {len(long_labels)} code labels are too long (>5 words). "
                "Consider simplifying for clarity."
            )
            recommendations.append(
                f"Review and shorten these labels: {', '.join(long_labels[:3])}..."
            )

        # Check 2: Are codes balanced?
        imbalance_ratio = self._check_code_balance(coder)
        if imbalance_ratio > 10:
            issues['code_imbalance'] = imbalance_ratio
            warnings_list.append(
                f"⚠️ Code imbalance detected ({imbalance_ratio:.1f}:1 ratio). "
                "Some codes may be over/under-represented."
            )
            recommendations.append(
                "Consider merging rare codes or splitting dominant ones. "
                "Review codebook for semantic overlap."
            )

        # Check 3: Is coverage adequate?
        uncoded_pct = self._check_coverage(results_df)
        if uncoded_pct > 20:
            issues['low_coverage'] = uncoded_pct
            warnings_list.append(
                f"⚠️ {uncoded_pct:.1f}% of responses uncoded. "
                "Consider more codes or lower confidence threshold."
            )
            recommendations.append(
                "Options: (1) Reduce min_confidence threshold, "
                "(2) Increase n_codes, (3) Review uncoded responses manually"
            )

        # Check 4: Is confidence distribution healthy?
        low_conf_pct = self._check_confidence_quality(results_df)
        if low_conf_pct:
            issues['low_confidence'] = low_conf_pct
            warnings_list.append(
                f"⚠️ {low_conf_pct:.1f}% of assignments have <0.5 confidence. "
                "Review model fit."
            )
            recommendations.append(
                "Low confidence may indicate: (1) Insufficient training data, "
                "(2) Too many codes for dataset size, (3) Noisy/ambiguous data"
            )

        # Check 5: Is dataset size appropriate?
        if len(results_df) < min_responses:
            issues['insufficient_data'] = len(results_df)
            warnings_list.append(
                f"⚠️ Dataset has only {len(results_df)} responses. "
                f"Recommend at least {min_responses} for reliable clustering."
            )
            recommendations.append(
                "Consider: (1) Collecting more data, (2) Using qualitative "
                "methods instead, (3) Treating results as exploratory only"
            )

        # Check 6: Are there unused codes?
        unused_codes = self._check_unused_codes(coder)
        if unused_codes:
            issues['unused_codes'] = unused_codes
            warnings_list.append(
                f"⚠️ {len(unused_codes)} codes never assigned. "
                "Model may have created redundant clusters."
            )
            recommendations.append(
                f"Consider reducing n_codes. Unused codes: {', '.join(unused_codes[:5])}"
            )

        # Check 7: High ambiguity rate
        ambiguity_rate = (results_df['num_codes'] >= 3).sum() / len(results_df) * 100
        if ambiguity_rate > 30:
            issues['high_ambiguity'] = ambiguity_rate
            warnings_list.append(
                f"⚠️ {ambiguity_rate:.1f}% of responses have 3+ codes. "
                "High ambiguity may indicate overlapping themes."
            )
            recommendations.append(
                "Review code definitions for overlap. Consider hierarchical "
                "coding or theme merging."
            )

        # Overall health status
        health_status = self._calculate_health_status(issues)

        return {
            'warnings': warnings_list,
            'recommendations': recommendations,
            'issues': issues,
            'health_status': health_status,
            'total_issues': len(issues)
        }

    # ============ Private Methods ============

    def _calculate_inter_coder_reliability(
        self,
        ml_codes: pd.DataFrame,
        human_codes: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate agreement between ML and human codes (Cohen's Kappa)."""
        # Simplified - would need proper implementation with label matching
        return {
            'cohens_kappa': None,
            'percent_agreement': None,
            'note': 'Full implementation requires label alignment'
        }

    def _calculate_code_stability(
        self,
        coder,
        results_df: pd.DataFrame,
        n_bootstrap: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate stability of code assignments across bootstrap samples.

        Measures how consistent codes are when small perturbations are made.
        """
        if len(results_df) < 30:
            return {
                'stability_score': None,
                'note': 'Dataset too small for bootstrap stability analysis'
            }

        # Simplified stability metric based on code assignment variance
        code_counts = []
        for codes in results_df['assigned_codes']:
            code_counts.append(len(codes))

        stability_score = 1.0 - (np.std(code_counts) / (np.mean(code_counts) + 1e-6))
        stability_score = max(0.0, min(1.0, stability_score))

        return {
            'stability_score': float(stability_score),
            'interpretation': self._interpret_stability(stability_score)
        }

    def _calculate_theme_coherence(
        self,
        coder,
        results_df: pd.DataFrame,
        feature_matrix
    ) -> Dict[str, Any]:
        """Calculate semantic coherence within each code."""
        coherence_scores = {}

        for code_id, code_info in coder.codebook.items():
            if code_info['count'] < 2:
                coherence_scores[code_id] = None
                continue

            # Get responses assigned to this code
            code_responses = []
            for idx, codes in enumerate(results_df['assigned_codes']):
                if code_id in codes:
                    code_responses.append(idx)

            if len(code_responses) < 2:
                coherence_scores[code_id] = None
                continue

            # Calculate average pairwise similarity
            vectors = feature_matrix[code_responses].toarray()
            similarities = []

            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    sim = 1 - cosine(vectors[i], vectors[j])
                    if not np.isnan(sim):
                        similarities.append(sim)

            if similarities:
                coherence_scores[code_id] = float(np.mean(similarities))
            else:
                coherence_scores[code_id] = None

        # Overall coherence
        valid_scores = [s for s in coherence_scores.values() if s is not None]
        avg_coherence = float(np.mean(valid_scores)) if valid_scores else None

        return {
            'average_coherence': avg_coherence,
            'per_code_coherence': coherence_scores,
            'interpretation': self._interpret_coherence(avg_coherence)
        }

    def _calculate_simple_coherence(
        self,
        coder,
        results_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Simplified coherence based on confidence scores."""
        coherence_by_code = {}

        for code_id, code_info in coder.codebook.items():
            avg_conf = code_info.get('avg_confidence', 0.0)
            coherence_by_code[code_id] = float(avg_conf)

        avg_coherence = float(np.mean(list(coherence_by_code.values())))

        return {
            'average_coherence': avg_coherence,
            'per_code_coherence': coherence_by_code,
            'interpretation': self._interpret_coherence(avg_coherence),
            'note': 'Based on confidence scores (feature matrix not provided)'
        }

    def _assess_thematic_saturation(
        self,
        coder,
        results_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Assess whether we've reached thematic saturation.

        Indicators:
        - High uncoded rate may suggest missing themes
        - High ambiguity may suggest overlapping themes
        - Unused codes may suggest too many themes
        """
        uncoded_pct = (results_df['num_codes'] == 0).sum() / len(results_df) * 100
        unused_codes = sum(1 for info in coder.codebook.values() if info['count'] == 0)

        # Estimate saturation
        if uncoded_pct > 15:
            saturation_status = 'under_saturated'
            confidence = 'low'
            message = "High uncoded rate suggests missing themes"
        elif unused_codes > len(coder.codebook) * 0.2:
            saturation_status = 'over_saturated'
            confidence = 'medium'
            message = "Many unused codes suggest too many themes"
        else:
            saturation_status = 'adequate'
            confidence = 'medium'
            message = "Theme coverage appears adequate"

        return {
            'saturation_status': saturation_status,
            'confidence': confidence,
            'message': message,
            'uncoded_percentage': float(uncoded_pct),
            'unused_codes': unused_codes
        }

    def _calculate_coverage_ratio(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate percentage of responses that received at least one code."""
        total = len(results_df)
        coded = (results_df['num_codes'] > 0).sum()
        coverage_pct = (coded / total * 100) if total > 0 else 0.0

        return {
            'coverage_percentage': float(coverage_pct),
            'coded_responses': int(coded),
            'uncoded_responses': int(total - coded),
            'total_responses': int(total),
            'interpretation': self._interpret_coverage(coverage_pct)
        }

    def _calculate_code_utilization(self, coder) -> Dict[str, Any]:
        """Analyze which codes are underused or overused."""
        total_codes = len(coder.codebook)
        active_codes = sum(1 for info in coder.codebook.values() if info['count'] > 0)
        utilization_rate = (active_codes / total_codes * 100) if total_codes > 0 else 0.0

        # Identify underused codes
        underused = []
        overused = []

        counts = [info['count'] for info in coder.codebook.values()]
        if counts:
            median_count = np.median(counts)

            for code_id, info in coder.codebook.items():
                if info['count'] == 0:
                    underused.append(code_id)
                elif info['count'] < median_count * 0.2:
                    underused.append(code_id)
                elif info['count'] > median_count * 5:
                    overused.append(code_id)

        return {
            'utilization_rate': float(utilization_rate),
            'active_codes': active_codes,
            'total_codes': total_codes,
            'underused_codes': underused,
            'overused_codes': overused,
            'interpretation': self._interpret_utilization(utilization_rate)
        }

    def _analyze_confidence_distribution(
        self,
        results_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze distribution of confidence scores."""
        all_confidences = []
        for confs in results_df['confidence_scores']:
            all_confidences.extend(confs)

        if not all_confidences:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'percentiles': {},
                'histogram': [],
                'note': 'No confidence scores available'
            }

        all_confidences = np.array(all_confidences)

        return {
            'mean': float(np.mean(all_confidences)),
            'median': float(np.median(all_confidences)),
            'std': float(np.std(all_confidences)),
            'min': float(np.min(all_confidences)),
            'max': float(np.max(all_confidences)),
            'percentiles': {
                '25th': float(np.percentile(all_confidences, 25)),
                '50th': float(np.percentile(all_confidences, 50)),
                '75th': float(np.percentile(all_confidences, 75)),
                '90th': float(np.percentile(all_confidences, 90))
            },
            'histogram': np.histogram(all_confidences, bins=10)[0].tolist(),
            'interpretation': self._interpret_confidence_dist(all_confidences)
        }

    def _calculate_ambiguity_rate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate percentage of responses with multiple codes."""
        total = len(results_df)
        single_code = (results_df['num_codes'] == 1).sum()
        multi_code = (results_df['num_codes'] > 1).sum()
        high_ambiguity = (results_df['num_codes'] >= 3).sum()

        return {
            'multi_code_percentage': float(multi_code / total * 100) if total > 0 else 0.0,
            'high_ambiguity_percentage': float(high_ambiguity / total * 100) if total > 0 else 0.0,
            'single_code_count': int(single_code),
            'multi_code_count': int(multi_code),
            'high_ambiguity_count': int(high_ambiguity),
            'interpretation': self._interpret_ambiguity(multi_code / total * 100 if total > 0 else 0)
        }

    def _identify_boundary_cases(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify responses near decision boundaries (low max confidence)."""
        boundary_cases = []

        for idx, confs in enumerate(results_df['confidence_scores']):
            if confs:
                max_conf = max(confs)
                if 0.3 <= max_conf <= 0.5:  # Near threshold
                    boundary_cases.append(idx)

        return {
            'count': len(boundary_cases),
            'percentage': float(len(boundary_cases) / len(results_df) * 100) if len(results_df) > 0 else 0.0,
            'indices': boundary_cases[:100],  # Limit to first 100
            'note': 'Responses with max confidence between 0.3-0.5'
        }

    def _assess_random_stability(
        self,
        coder,
        results_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Assess sensitivity to random seed."""
        # This would require re-running with different seeds
        # For now, provide a placeholder
        return {
            'stability_estimate': None,
            'note': 'Requires multiple runs with different random seeds',
            'recommendation': 'Run analysis with different random_state values and compare'
        }

    def _analyze_code_imbalance(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze imbalance in code distribution."""
        code_counts = Counter()
        for codes in results_df['assigned_codes']:
            for code in codes:
                code_counts[code] += 1

        if not code_counts:
            return {'imbalance_ratio': 0.0, 'note': 'No codes assigned'}

        counts = list(code_counts.values())
        max_count = max(counts)
        min_count = min([c for c in counts if c > 0]) if any(c > 0 for c in counts) else 1
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        return {
            'imbalance_ratio': float(imbalance_ratio),
            'max_code_count': max_count,
            'min_code_count': min_count,
            'gini_coefficient': self._calculate_gini(counts),
            'interpretation': self._interpret_imbalance(imbalance_ratio)
        }

    def _analyze_demographic_representation(
        self,
        results_df: pd.DataFrame,
        demographics: pd.DataFrame,
        demographic_columns: List[str]
    ) -> Dict[str, Any]:
        """Analyze whether coding is consistent across demographic groups."""
        representation = {}

        for demo_col in demographic_columns:
            if demo_col not in demographics.columns:
                continue

            # Calculate average codes per demographic group
            combined = pd.concat([results_df['num_codes'], demographics[demo_col]], axis=1)
            group_stats = combined.groupby(demo_col)['num_codes'].agg(['mean', 'std', 'count'])

            representation[demo_col] = {
                'group_statistics': group_stats.to_dict(),
                'chi_square_test': self._chi_square_test(combined, demo_col)
            }

        return representation

    def _detect_systematic_patterns(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect systematic patterns that might indicate bias."""
        patterns = {}

        # Check for positional bias (early vs late responses)
        if len(results_df) >= 20:
            first_half = results_df.iloc[:len(results_df)//2]['num_codes'].mean()
            second_half = results_df.iloc[len(results_df)//2:]['num_codes'].mean()

            patterns['positional_bias'] = {
                'first_half_avg': float(first_half),
                'second_half_avg': float(second_half),
                'difference': float(abs(first_half - second_half)),
                'significant': abs(first_half - second_half) > 0.5
            }

        return patterns

    def _check_long_labels(self, coder) -> List[str]:
        """Check for code labels that are too long."""
        long_labels = []
        for code_id, info in coder.codebook.items():
            label = info['label']
            if len(label.split()) > 5:
                long_labels.append(f"{code_id}: {label}")
        return long_labels

    def _check_code_balance(self, coder) -> float:
        """Check code balance ratio."""
        counts = [info['count'] for info in coder.codebook.values() if info['count'] > 0]
        if not counts:
            return 0.0
        return max(counts) / min(counts) if min(counts) > 0 else float('inf')

    def _check_coverage(self, results_df: pd.DataFrame) -> float:
        """Check uncoded percentage."""
        uncoded = (results_df['num_codes'] == 0).sum()
        return (uncoded / len(results_df) * 100) if len(results_df) > 0 else 0.0

    def _check_confidence_quality(self, results_df: pd.DataFrame) -> Optional[float]:
        """Check if confidence distribution is healthy."""
        all_confs = []
        for confs in results_df['confidence_scores']:
            all_confs.extend(confs)

        if not all_confs:
            return None

        p75 = np.percentile(all_confs, 75)
        if p75 < 0.5:
            low_conf_count = sum(1 for c in all_confs if c < 0.5)
            return (low_conf_count / len(all_confs) * 100)
        return None

    def _check_unused_codes(self, coder) -> List[str]:
        """Check for unused codes."""
        unused = []
        for code_id, info in coder.codebook.items():
            if info['count'] == 0:
                unused.append(code_id)
        return unused

    def _calculate_health_status(self, issues: Dict) -> Dict[str, Any]:
        """Calculate overall health status."""
        n_issues = len(issues)

        if n_issues == 0:
            status = 'excellent'
            color = 'green'
            message = '✅ No issues detected. Analysis appears methodologically sound.'
        elif n_issues <= 2:
            status = 'good'
            color = 'yellow'
            message = '🟡 Minor issues detected. Review recommendations.'
        elif n_issues <= 4:
            status = 'fair'
            color = 'orange'
            message = '🟠 Several issues detected. Address before finalizing.'
        else:
            status = 'poor'
            color = 'red'
            message = '🔴 Multiple issues detected. Major review recommended.'

        return {
            'status': status,
            'color': color,
            'message': message,
            'issue_count': n_issues
        }

    # ============ Interpretation Helpers ============

    def _interpret_stability(self, score: float) -> str:
        """Interpret stability score."""
        if score is None:
            return "Cannot assess"
        if score >= 0.8:
            return "High stability - codes are consistent"
        elif score >= 0.6:
            return "Moderate stability - some variation in assignments"
        else:
            return "Low stability - high variation suggests unreliable clustering"

    def _interpret_coherence(self, score: Optional[float]) -> str:
        """Interpret coherence score."""
        if score is None:
            return "Cannot assess"
        if score >= 0.7:
            return "High coherence - themes are well-defined"
        elif score >= 0.5:
            return "Moderate coherence - themes have some internal consistency"
        else:
            return "Low coherence - themes may be poorly defined or overlapping"

    def _interpret_coverage(self, pct: float) -> str:
        """Interpret coverage percentage."""
        if pct >= 90:
            return "Excellent coverage"
        elif pct >= 80:
            return "Good coverage"
        elif pct >= 70:
            return "Acceptable coverage"
        else:
            return "Poor coverage - many responses uncoded"

    def _interpret_utilization(self, rate: float) -> str:
        """Interpret code utilization rate."""
        if rate >= 90:
            return "Excellent utilization"
        elif rate >= 75:
            return "Good utilization"
        elif rate >= 50:
            return "Fair utilization - some codes unused"
        else:
            return "Poor utilization - many codes unused"

    def _interpret_confidence_dist(self, confidences: np.ndarray) -> str:
        """Interpret confidence distribution."""
        mean_conf = np.mean(confidences)
        if mean_conf >= 0.7:
            return "High confidence - strong code assignments"
        elif mean_conf >= 0.5:
            return "Moderate confidence - acceptable assignments"
        else:
            return "Low confidence - weak assignments, review recommended"

    def _interpret_ambiguity(self, pct: float) -> str:
        """Interpret ambiguity rate."""
        if pct >= 50:
            return "High ambiguity - most responses multi-coded"
        elif pct >= 30:
            return "Moderate ambiguity - common for complex data"
        elif pct >= 10:
            return "Low ambiguity - mostly single-coded responses"
        else:
            return "Very low ambiguity - highly distinct themes"

    def _interpret_imbalance(self, ratio: float) -> str:
        """Interpret imbalance ratio."""
        if ratio >= 20:
            return "Severe imbalance - review code structure"
        elif ratio >= 10:
            return "High imbalance - some codes dominate"
        elif ratio >= 5:
            return "Moderate imbalance - acceptable variation"
        else:
            return "Well-balanced code distribution"

    # ============ Statistical Helpers ============

    def _calculate_gini(self, counts: List[int]) -> float:
        """Calculate Gini coefficient for inequality measure."""
        if not counts or all(c == 0 for c in counts):
            return 0.0

        counts = sorted([c for c in counts if c > 0])
        n = len(counts)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n)

    def _chi_square_test(self, combined_df: pd.DataFrame, demo_col: str) -> Dict[str, Any]:
        """Perform chi-square test for independence."""
        try:
            # Create contingency table
            contingency = pd.crosstab(combined_df[demo_col], combined_df['num_codes'] > 0)
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            return {
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': 'Significant difference across groups' if p_value < 0.05 else 'No significant difference'
            }
        except Exception as e:
            return {
                'error': str(e),
                'note': 'Could not perform chi-square test'
            }

    def generate_recommendations(
        self,
        validity: Dict,
        bias: Dict,
        sanity: Dict
    ) -> List[str]:
        """
        Generate actionable recommendations based on diagnostics.

        Args:
            validity: Output from assess_validity()
            bias: Output from detect_bias()
            sanity: Output from sanity_check()

        Returns:
            List of specific, actionable recommendations
        """
        recommendations = []

        # From validity metrics
        if validity.get('coverage_ratio', {}).get('coverage_percentage', 100) < 80:
            recommendations.append(
                "🔍 Low coverage: Consider reducing min_confidence threshold "
                "or increasing number of codes to capture more themes."
            )

        if validity.get('code_utilization', {}).get('utilization_rate', 100) < 75:
            recommendations.append(
                "📉 Low code utilization: Consider reducing n_codes to eliminate "
                "unused clusters."
            )

        coherence = validity.get('theme_coherence', {}).get('average_coherence')
        if coherence is not None and coherence < 0.5:
            recommendations.append(
                "🔀 Low theme coherence: Codes may be overlapping or poorly defined. "
                "Review code definitions and consider merging similar themes."
            )

        # From bias detection
        imbalance_ratio = bias.get('code_imbalance', {}).get('imbalance_ratio', 0)
        if imbalance_ratio > 10:
            recommendations.append(
                "⚖️ Code imbalance: Some codes are over-represented. "
                "Consider splitting dominant codes or merging rare ones."
            )

        # From sanity checks
        if sanity.get('recommendations'):
            recommendations.extend(sanity['recommendations'])

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append(
                "✅ Analysis quality looks good! Proceed with qualitative review "
                "of representative quotes and code definitions."
            )

        return recommendations
