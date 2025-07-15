"""
Advanced Prompt Management System for Translation Service

This module provides comprehensive prompt template management with:
- Template versioning and lifecycle management
- Performance analytics and optimization
- A/B testing framework for prompts
- Domain-specific prompt collections
- Real-time prompt effectiveness tracking
"""

import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata and performance tracking."""
    id: str
    name: str
    description: str
    template: str
    system_message: Optional[str] = None
    language_pairs: List[str] = None
    category: str = 'general'
    version: str = '1.0'
    is_active: bool = True
    is_default: bool = False
    metadata: Dict[str, Any] = None
    performance_metrics: Dict[str, Any] = None
    test_results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.language_pairs is None:
            self.language_pairs = ['*']
        if self.metadata is None:
            self.metadata = {
                'created_at': time.time(),
                'updated_at': time.time(),
                'created_by': 'system',
                'tags': []
            }
        if self.performance_metrics is None:
            self.performance_metrics = {
                'avg_quality': 0.0,
                'avg_speed': 0.0,
                'avg_confidence': 0.0,
                'usage_count': 0,
                'success_rate': 0.0,
                'last_used': None
            }

@dataclass
class PromptPerformanceMetric:
    """Performance metric for a single prompt execution."""
    prompt_id: str
    quality_score: float
    processing_time: float
    confidence_score: float
    success: bool
    timestamp: float
    language_pair: str
    text_length: int
    error_message: Optional[str] = None

class PromptOptimizer:
    """Analyzes prompt performance and suggests optimizations."""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> Dict[str, Dict]:
        """Load optimization rules for different scenarios."""
        return {
            'low_quality': {
                'threshold': 0.7,
                'suggestions': [
                    'Add more context to the prompt',
                    'Specify target audience or style',
                    'Include examples of desired output',
                    'Break down complex instructions'
                ]
            },
            'slow_processing': {
                'threshold': 1000,  # ms
                'suggestions': [
                    'Simplify prompt structure',
                    'Reduce unnecessary context',
                    'Use more direct instructions',
                    'Consider shorter system messages'
                ]
            },
            'low_confidence': {
                'threshold': 0.75,
                'suggestions': [
                    'Add confidence keywords',
                    'Specify uncertainty handling',
                    'Include validation instructions',
                    'Add error handling guidance'
                ]
            }
        }
    
    def analyze_prompt_performance(self, prompt_id: str, metrics: List[PromptPerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance metrics and provide optimization suggestions."""
        if not metrics:
            return {'status': 'insufficient_data', 'suggestions': []}
        
        # Calculate aggregate metrics
        avg_quality = sum(m.quality_score for m in metrics) / len(metrics)
        avg_speed = sum(m.processing_time for m in metrics) / len(metrics)
        avg_confidence = sum(m.confidence_score for m in metrics) / len(metrics)
        success_rate = sum(1 for m in metrics if m.success) / len(metrics)
        
        suggestions = []
        issues = []
        
        # Analyze quality
        if avg_quality < self.optimization_rules['low_quality']['threshold']:
            issues.append('low_quality')
            suggestions.extend(self.optimization_rules['low_quality']['suggestions'])
        
        # Analyze speed
        if avg_speed > self.optimization_rules['slow_processing']['threshold']:
            issues.append('slow_processing')
            suggestions.extend(self.optimization_rules['slow_processing']['suggestions'])
        
        # Analyze confidence
        if avg_confidence < self.optimization_rules['low_confidence']['threshold']:
            issues.append('low_confidence')
            suggestions.extend(self.optimization_rules['low_confidence']['suggestions'])
        
        # Performance trend analysis
        recent_metrics = sorted(metrics, key=lambda x: x.timestamp)[-10:]
        if len(recent_metrics) >= 5:
            recent_quality = sum(m.quality_score for m in recent_metrics) / len(recent_metrics)
            if recent_quality < avg_quality * 0.9:
                issues.append('degrading_performance')
                suggestions.append('Review recent changes to prompt or model')
        
        return {
            'status': 'analyzed',
            'aggregate_metrics': {
                'avg_quality': avg_quality,
                'avg_speed': avg_speed,
                'avg_confidence': avg_confidence,
                'success_rate': success_rate,
                'sample_size': len(metrics)
            },
            'issues': issues,
            'suggestions': list(set(suggestions)),  # Remove duplicates
            'recommendation': self._generate_recommendation(issues, avg_quality, avg_speed)
        }
    
    def _generate_recommendation(self, issues: List[str], quality: float, speed: float) -> str:
        """Generate an overall recommendation based on analysis."""
        if not issues:
            return "Prompt is performing well. Continue monitoring."
        
        priority_issues = []
        if 'low_quality' in issues:
            priority_issues.append("quality improvement")
        if 'slow_processing' in issues:
            priority_issues.append("speed optimization")
        if 'degrading_performance' in issues:
            priority_issues.append("performance regression investigation")
        
        if len(priority_issues) == 1:
            return f"Focus on {priority_issues[0]}."
        elif len(priority_issues) > 1:
            return f"Multi-faceted optimization needed: {', '.join(priority_issues)}."
        else:
            return "Monitor performance and consider minor adjustments."

class PromptManager:
    """
    Comprehensive prompt management system with versioning, analytics, and optimization.
    """
    
    def __init__(self, storage_path: str = "prompts", enable_analytics: bool = True):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.enable_analytics = enable_analytics
        
        # In-memory storage
        self.prompts: Dict[str, PromptTemplate] = {}
        self.performance_metrics: List[PromptPerformanceMetric] = []
        self.active_tests: Dict[str, Dict] = {}
        
        # Components
        self.optimizer = PromptOptimizer()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load existing prompts
        self._load_prompts()
        self._load_default_prompts()
        
        logger.info(f"PromptManager initialized with {len(self.prompts)} prompts")
    
    def _load_prompts(self):
        """Load prompts from storage."""
        prompts_file = self.storage_path / "prompts.json"
        if prompts_file.exists():
            try:
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for prompt_data in data.get('prompts', []):
                        prompt = PromptTemplate(**prompt_data)
                        self.prompts[prompt.id] = prompt
                logger.info(f"Loaded {len(self.prompts)} prompts from storage")
            except Exception as e:
                logger.error(f"Failed to load prompts: {e}")
        
        # Load performance metrics
        metrics_file = self.storage_path / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for metric_data in data.get('metrics', []):
                        metric = PromptPerformanceMetric(**metric_data)
                        self.performance_metrics.append(metric)
                logger.info(f"Loaded {len(self.performance_metrics)} performance metrics")
            except Exception as e:
                logger.error(f"Failed to load metrics: {e}")
    
    def _load_default_prompts(self):
        """Load default prompt templates."""
        default_prompts = [
            PromptTemplate(
                id='default',
                name='Default Translation',
                description='Standard translation prompt for general use',
                template='Translate the following text from {source_language} to {target_language}. Provide only the translation without any explanation:\n\n{text}',
                language_pairs=['*'],
                category='general',
                version='1.0',
                is_active=True,
                is_default=True,
                metadata={
                    'created_at': time.time(),
                    'updated_at': time.time(),
                    'created_by': 'system',
                    'tags': ['basic', 'general']
                }
            ),
            PromptTemplate(
                id='conversational',
                name='Conversational Style',
                description='Natural conversational translation with context awareness',
                template='Translate this conversational text naturally, maintaining the tone and style from {source_language} to {target_language}:\n\n{text}\n\nKeep the natural flow and cultural context appropriate for casual conversation.',
                system_message='You are a skilled translator specializing in natural, conversational language. Maintain cultural context and informal tone.',
                language_pairs=['en-es', 'en-fr', 'zh-en', 'ja-en'],
                category='conversational',
                version='1.2',
                is_active=True,
                metadata={
                    'created_at': time.time(),
                    'updated_at': time.time(),
                    'created_by': 'system',
                    'tags': ['conversational', 'natural', 'context-aware']
                }
            ),
            PromptTemplate(
                id='technical',
                name='Technical Documentation',
                description='Specialized translation for technical documentation and manuals',
                template='Translate this technical documentation from {source_language} to {target_language}, maintaining technical accuracy and terminology:\n\n{text}\n\nPreserve technical terms, maintain precision, and ensure clarity for technical audience.',
                system_message='You are a technical translator with expertise in software, engineering, and scientific documentation. Prioritize accuracy and consistency of technical terminology.',
                language_pairs=['en-de', 'en-ja', 'en-zh'],
                category='technical',
                version='2.0',
                is_active=True,
                metadata={
                    'created_at': time.time(),
                    'updated_at': time.time(),
                    'created_by': 'system',
                    'tags': ['technical', 'documentation', 'precision']
                }
            ),
            PromptTemplate(
                id='formal',
                name='Formal Communication',
                description='Professional and formal translation for business communications',
                template='Translate this formal text from {source_language} to {target_language}, maintaining professional tone and appropriate formality:\n\n{text}\n\nEnsure the translation reflects proper business etiquette and cultural formality expectations.',
                system_message='You are a professional translator specializing in formal business communications. Maintain appropriate levels of formality and cultural sensitivity.',
                language_pairs=['*'],
                category='formal',
                version='1.1',
                is_active=True,
                metadata={
                    'created_at': time.time(),
                    'updated_at': time.time(),
                    'created_by': 'system',
                    'tags': ['formal', 'business', 'professional']
                }
            ),
            PromptTemplate(
                id='creative',
                name='Creative Translation',
                description='Creative and artistic translation for literature and creative content',
                template='Translate this creative text from {source_language} to {target_language}, preserving artistic intent and emotional resonance:\n\n{text}\n\nMaintain the creative spirit, metaphors, and emotional impact of the original while adapting to the target culture.',
                system_message='You are a literary translator skilled in preserving artistic intent and emotional nuance. Prioritize creative adaptation over literal accuracy.',
                language_pairs=['en-fr', 'en-es', 'fr-en', 'es-en'],
                category='creative',
                version='1.3',
                is_active=True,
                metadata={
                    'created_at': time.time(),
                    'updated_at': time.time(),
                    'created_by': 'system',
                    'tags': ['creative', 'literary', 'artistic']
                }
            )
        ]
        
        # Add default prompts if they don't exist
        for prompt in default_prompts:
            if prompt.id not in self.prompts:
                self.prompts[prompt.id] = prompt
                logger.info(f"Added default prompt: {prompt.name}")
    
    def save_prompts(self):
        """Save prompts and metrics to storage."""
        with self.lock:
            try:
                # Save prompts
                prompts_data = {
                    'prompts': [asdict(prompt) for prompt in self.prompts.values()],
                    'saved_at': time.time()
                }
                prompts_file = self.storage_path / "prompts.json"
                with open(prompts_file, 'w', encoding='utf-8') as f:
                    json.dump(prompts_data, f, indent=2, ensure_ascii=False)
                
                # Save metrics (keep only recent ones to avoid bloat)
                recent_cutoff = time.time() - (30 * 24 * 3600)  # 30 days
                recent_metrics = [m for m in self.performance_metrics if m.timestamp > recent_cutoff]
                
                metrics_data = {
                    'metrics': [asdict(metric) for metric in recent_metrics],
                    'saved_at': time.time()
                }
                metrics_file = self.storage_path / "metrics.json"
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved {len(self.prompts)} prompts and {len(recent_metrics)} metrics")
                
            except Exception as e:
                logger.error(f"Failed to save prompts: {e}")
                raise
    
    def create_prompt(self, prompt: PromptTemplate) -> bool:
        """Create a new prompt template."""
        with self.lock:
            if prompt.id in self.prompts:
                return False
            
            prompt.metadata['created_at'] = time.time()
            prompt.metadata['updated_at'] = time.time()
            self.prompts[prompt.id] = prompt
            
            self.save_prompts()
            logger.info(f"Created prompt: {prompt.name} ({prompt.id})")
            return True
    
    def update_prompt(self, prompt_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing prompt template."""
        with self.lock:
            if prompt_id not in self.prompts:
                return False
            
            prompt = self.prompts[prompt_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(prompt, key):
                    setattr(prompt, key, value)
            
            prompt.metadata['updated_at'] = time.time()
            
            self.save_prompts()
            logger.info(f"Updated prompt: {prompt.name} ({prompt_id})")
            return True
    
    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt template."""
        with self.lock:
            if prompt_id not in self.prompts or self.prompts[prompt_id].is_default:
                return False
            
            prompt_name = self.prompts[prompt_id].name
            del self.prompts[prompt_id]
            
            self.save_prompts()
            logger.info(f"Deleted prompt: {prompt_name} ({prompt_id})")
            return True
    
    def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        """Get a specific prompt template."""
        return self.prompts.get(prompt_id)
    
    def get_active_prompts(self) -> List[PromptTemplate]:
        """Get all active prompt templates."""
        return [p for p in self.prompts.values() if p.is_active]
    
    def get_prompts_by_category(self, category: str) -> List[PromptTemplate]:
        """Get prompts by category."""
        return [p for p in self.prompts.values() if p.category == category]
    
    def get_prompts_for_language_pair(self, source_lang: str, target_lang: str) -> List[PromptTemplate]:
        """Get prompts suitable for a specific language pair."""
        pair = f"{source_lang}-{target_lang}"
        suitable_prompts = []
        
        for prompt in self.prompts.values():
            if not prompt.is_active:
                continue
            
            if '*' in prompt.language_pairs:
                suitable_prompts.append(prompt)
            elif pair in prompt.language_pairs:
                suitable_prompts.append(prompt)
            elif f"{target_lang}-{source_lang}" in prompt.language_pairs:
                suitable_prompts.append(prompt)
        
        return suitable_prompts
    
    def build_prompt(self, prompt_id: str, variables: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Build a complete prompt with variable substitution."""
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            return None, None
        
        try:
            # Substitute variables in template
            filled_template = prompt.template.format(**variables)
            
            # Return template and system message
            return filled_template, prompt.system_message
            
        except KeyError as e:
            logger.error(f"Missing variable {e} for prompt {prompt_id}")
            return None, None
        except Exception as e:
            logger.error(f"Error building prompt {prompt_id}: {e}")
            return None, None
    
    def record_performance(self, metric: PromptPerformanceMetric):
        """Record a performance metric for analytics."""
        if not self.enable_analytics:
            return
        
        with self.lock:
            self.performance_metrics.append(metric)
            
            # Update prompt performance metrics
            if metric.prompt_id in self.prompts:
                prompt = self.prompts[metric.prompt_id]
                metrics = prompt.performance_metrics
                
                # Update usage count
                metrics['usage_count'] = metrics.get('usage_count', 0) + 1
                metrics['last_used'] = metric.timestamp
                
                # Calculate rolling averages (last 100 uses)
                recent_metrics = [m for m in self.performance_metrics 
                                if m.prompt_id == metric.prompt_id and m.timestamp > time.time() - 86400][-100:]
                
                if recent_metrics:
                    metrics['avg_quality'] = sum(m.quality_score for m in recent_metrics) / len(recent_metrics)
                    metrics['avg_speed'] = sum(m.processing_time for m in recent_metrics) / len(recent_metrics)
                    metrics['avg_confidence'] = sum(m.confidence_score for m in recent_metrics) / len(recent_metrics)
                    metrics['success_rate'] = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
                
                # Save periodically
                if metrics['usage_count'] % 10 == 0:
                    self.save_prompts()
    
    def get_performance_analysis(self, prompt_id: str) -> Dict[str, Any]:
        """Get comprehensive performance analysis for a prompt."""
        if prompt_id not in self.prompts:
            return {'error': 'Prompt not found'}
        
        # Get metrics for this prompt
        prompt_metrics = [m for m in self.performance_metrics if m.prompt_id == prompt_id]
        
        if not prompt_metrics:
            return {'status': 'no_data', 'message': 'No performance data available'}
        
        # Basic analysis
        analysis = self.optimizer.analyze_prompt_performance(prompt_id, prompt_metrics)
        
        # Add trend analysis
        if len(prompt_metrics) >= 10:
            sorted_metrics = sorted(prompt_metrics, key=lambda x: x.timestamp)
            
            # Split into first half and second half for trend analysis
            mid_point = len(sorted_metrics) // 2
            first_half = sorted_metrics[:mid_point]
            second_half = sorted_metrics[mid_point:]
            
            first_avg_quality = sum(m.quality_score for m in first_half) / len(first_half)
            second_avg_quality = sum(m.quality_score for m in second_half) / len(second_half)
            
            quality_trend = (second_avg_quality - first_avg_quality) / first_avg_quality * 100
            
            analysis['trend_analysis'] = {
                'quality_change_percent': quality_trend,
                'trend_direction': 'improving' if quality_trend > 5 else 'degrading' if quality_trend < -5 else 'stable'
            }
        
        return analysis
    
    def compare_prompts(self, prompt_ids: List[str], days: int = 7) -> Dict[str, Any]:
        """Compare performance of multiple prompts."""
        cutoff_time = time.time() - (days * 24 * 3600)
        comparison_data = {}
        
        for prompt_id in prompt_ids:
            if prompt_id not in self.prompts:
                continue
            
            # Get recent metrics
            recent_metrics = [m for m in self.performance_metrics 
                            if m.prompt_id == prompt_id and m.timestamp > cutoff_time]
            
            if recent_metrics:
                comparison_data[prompt_id] = {
                    'prompt_name': self.prompts[prompt_id].name,
                    'sample_size': len(recent_metrics),
                    'avg_quality': sum(m.quality_score for m in recent_metrics) / len(recent_metrics),
                    'avg_speed': sum(m.processing_time for m in recent_metrics) / len(recent_metrics),
                    'avg_confidence': sum(m.confidence_score for m in recent_metrics) / len(recent_metrics),
                    'success_rate': sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
                }
        
        # Rank prompts
        if comparison_data:
            # Sort by composite score (quality * 0.4 + confidence * 0.3 + success_rate * 0.3)
            ranked_prompts = sorted(
                comparison_data.items(),
                key=lambda x: (x[1]['avg_quality'] * 0.4 + 
                              x[1]['avg_confidence'] * 0.3 + 
                              x[1]['success_rate'] * 0.3),
                reverse=True
            )
            
            return {
                'comparison_period_days': days,
                'prompts_compared': len(comparison_data),
                'detailed_comparison': comparison_data,
                'ranking': [(prompt_id, data['prompt_name']) for prompt_id, data in ranked_prompts],
                'best_performer': ranked_prompts[0][0] if ranked_prompts else None,
                'recommendation': self._generate_comparison_recommendation(ranked_prompts)
            }
        
        return {'error': 'No data available for comparison'}
    
    def _generate_comparison_recommendation(self, ranked_prompts: List[Tuple[str, Dict]]) -> str:
        """Generate recommendation based on prompt comparison."""
        if not ranked_prompts:
            return "No prompts to compare."
        
        best_prompt_id, best_data = ranked_prompts[0]
        best_name = best_data['prompt_name']
        
        if len(ranked_prompts) == 1:
            return f"Only one prompt ({best_name}) has sufficient data."
        
        # Compare with second best
        second_prompt_id, second_data = ranked_prompts[1]
        second_name = second_data['prompt_name']
        
        quality_diff = (best_data['avg_quality'] - second_data['avg_quality']) * 100
        
        if quality_diff > 10:
            return f"Strong recommendation: Use '{best_name}' - significantly outperforms alternatives ({quality_diff:.1f}% better quality)."
        elif quality_diff > 5:
            return f"Moderate recommendation: '{best_name}' shows better performance than '{second_name}' ({quality_diff:.1f}% better quality)."
        else:
            return f"Close performance: '{best_name}' and '{second_name}' perform similarly. Consider other factors like speed or specific use cases."
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get overall prompt management statistics."""
        total_prompts = len(self.prompts)
        active_prompts = len(self.get_active_prompts())
        
        categories = defaultdict(int)
        for prompt in self.prompts.values():
            categories[prompt.category] += 1
        
        total_usage = sum(p.performance_metrics.get('usage_count', 0) for p in self.prompts.values())
        
        # Recent activity (last 7 days)
        week_ago = time.time() - (7 * 24 * 3600)
        recent_metrics = [m for m in self.performance_metrics if m.timestamp > week_ago]
        
        return {
            'total_prompts': total_prompts,
            'active_prompts': active_prompts,
            'categories': dict(categories),
            'total_usage': total_usage,
            'recent_activity': {
                'metrics_count': len(recent_metrics),
                'unique_prompts_used': len(set(m.prompt_id for m in recent_metrics)),
                'avg_quality': sum(m.quality_score for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                'avg_speed': sum(m.processing_time for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
            },
            'last_updated': max((p.metadata.get('updated_at', 0) for p in self.prompts.values()), default=0)
        }
    
    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Clean up old performance metrics to manage storage."""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        with self.lock:
            original_count = len(self.performance_metrics)
            self.performance_metrics = [m for m in self.performance_metrics if m.timestamp > cutoff_time]
            removed_count = original_count - len(self.performance_metrics)
            
            if removed_count > 0:
                self.save_prompts()
                logger.info(f"Cleaned up {removed_count} old performance metrics")
        
        return removed_count

# Example usage and testing
if __name__ == "__main__":
    # Initialize prompt manager
    manager = PromptManager(storage_path="./test_prompts")
    
    # Test prompt creation
    custom_prompt = PromptTemplate(
        id='test_prompt',
        name='Test Prompt',
        description='A test prompt for development',
        template='Test translation from {source_language} to {target_language}: {text}',
        category='test',
        version='1.0'
    )
    
    # Create and test prompt
    success = manager.create_prompt(custom_prompt)
    print(f"Prompt creation: {success}")
    
    # Build prompt
    built_prompt, system_msg = manager.build_prompt('test_prompt', {
        'source_language': 'English',
        'target_language': 'Spanish', 
        'text': 'Hello world'
    })
    print(f"Built prompt: {built_prompt}")
    
    # Simulate performance metrics
    for i in range(10):
        metric = PromptPerformanceMetric(
            prompt_id='test_prompt',
            quality_score=0.8 + (i * 0.02),
            processing_time=500 + (i * 10),
            confidence_score=0.85 + (i * 0.01),
            success=True,
            timestamp=time.time(),
            language_pair='en-es',
            text_length=20
        )
        manager.record_performance(metric)
    
    # Get analysis
    analysis = manager.get_performance_analysis('test_prompt')
    print(f"Performance analysis: {analysis}")
    
    # Get statistics
    stats = manager.get_prompt_statistics()
    print(f"Statistics: {stats}")
    
    print("Prompt Manager test completed successfully!")