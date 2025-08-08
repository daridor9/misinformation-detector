import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import re

class ThreatLevel(Enum):
    GREEN = "green"
    YELLOW = "yellow" 
    ORANGE = "orange"
    RED = "red"

@dataclass
class ContentItem:
    id: str
    text: str
    source: str
    timestamp: datetime
    platform: str
    engagement_metrics: Dict
    metadata: Dict = None

@dataclass
class ThreatAssessment:
    content_id: str
    risk_score: float
    threat_level: ThreatLevel
    indicators: Dict
    recommended_actions: List[str]
    timestamp: datetime

class DefensiveDetectionSystem:
    """Comprehensive defensive system for misinformation detection and response"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=10000)
        self.pattern_library = self._initialize_patterns()
        self.vulnerability_profiles = {}
        self.response_protocols = self._initialize_protocols()
        
    def _initialize_patterns(self) -> Dict:
        """Initialize known misinformation patterns"""
        return {
            'linguistic_patterns': {
                'fear_appeals': ['urgent', 'breaking', 'shocking', 'exclusive', 'they don\'t want you to know'],
                'false_urgency': ['share before deleted', 'act now', 'time is running out'],
                'conspiracy_markers': ['wake up', 'hidden truth', 'mainstream media won\'t tell'],
                'emotional_triggers': ['horrific', 'outrageous', 'unbelievable', 'disgusting']
            },
            'behavioral_patterns': {
                'rapid_sharing': {'threshold': 100, 'window_minutes': 60},
                'coordinated_posting': {'similarity_threshold': 0.8, 'time_window_seconds': 300},
                'artificial_amplification': {'bot_likelihood_threshold': 0.7}
            },
            'content_patterns': {
                'manipulated_media': ['deepfake', 'doctored', 'out_of_context'],
                'false_attribution': ['fake_quote', 'misattributed_source'],
                'statistical_manipulation': ['cherry_picked', 'misleading_graph']
            }
        }
    
    def _initialize_protocols(self) -> Dict:
        """Initialize response protocols for different threat levels"""
        return {
            ThreatLevel.GREEN: {
                'actions': ['log_for_analysis'],
                'notification': [],
                'resources': 'minimal'
            },
            ThreatLevel.YELLOW: {
                'actions': ['enhanced_monitoring', 'fact_check_queue'],
                'notification': ['fact_check_team'],
                'resources': 'standard'
            },
            ThreatLevel.ORANGE: {
                'actions': ['priority_fact_check', 'prebunk_activation', 'community_alerts'],
                'notification': ['fact_check_team', 'platform_liaisons', 'community_leaders'],
                'resources': 'elevated'
            },
            ThreatLevel.RED: {
                'actions': ['crisis_response', 'rapid_prebunking', 'stakeholder_coordination'],
                'notification': ['all_teams', 'platform_partners', 'government_liaison'],
                'resources': 'maximum'
            }
        }

class ContentAnalyzer:
    """Analyzes content for misinformation indicators"""
    
    def __init__(self):
        self.linguistic_analyzer = LinguisticAnalyzer()
        self.network_analyzer = NetworkBehaviorAnalyzer()
        self.media_analyzer = MediaVerifier()
        self.fact_checker = FactCheckInterface()
        
    def analyze_content(self, content: ContentItem) -> Dict:
        """Comprehensive content analysis"""
        
        # Linguistic analysis
        linguistic_score = self.linguistic_analyzer.analyze(content.text)
        
        # Network behavior analysis
        network_score = self.network_analyzer.analyze_propagation(
            content.engagement_metrics,
            content.timestamp
        )
        
        # Media verification (if applicable)
        media_score = 0.0
        if content.metadata and 'media_urls' in content.metadata:
            media_score = self.media_analyzer.verify_media(content.metadata['media_urls'])
        
        # Fact-checking integration
        fact_check_result = self.fact_checker.check_claims(content.text)
        
        # Composite risk assessment
        risk_components = {
            'linguistic_risk': linguistic_score,
            'network_risk': network_score,
            'media_risk': media_score,
            'fact_check_score': fact_check_result.get('credibility_score', 0.5)
        }
        
        # Weighted risk calculation
        weights = {'linguistic_risk': 0.3, 'network_risk': 0.3, 'media_risk': 0.2, 'fact_check_score': 0.2}
        composite_risk = sum(risk_components[k] * weights[k] for k in weights)
        
        return {
            'composite_risk': composite_risk,
            'risk_components': risk_components,
            'detailed_analysis': {
                'linguistic_details': self.linguistic_analyzer.get_details(),
                'network_details': self.network_analyzer.get_details(),
                'fact_check_details': fact_check_result
            }
        }

class LinguisticAnalyzer:
    """Analyzes linguistic patterns associated with misinformation"""
    
    def __init__(self):
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.credibility_markers = self._load_credibility_markers()
        self.manipulation_patterns = self._load_manipulation_patterns()
        self.analysis_cache = {}
        
    def _load_emotion_lexicon(self) -> Dict:
        """Load emotion-triggering words and phrases"""
        return {
            'fear': ['terrifying', 'scary', 'threat', 'danger', 'attack'],
            'anger': ['outrageous', 'disgusting', 'corrupt', 'evil', 'betrayal'],
            'disgust': ['revolting', 'sickening', 'vile', 'repulsive'],
            'moral_outrage': ['injustice', 'unfair', 'wrong', 'immoral', 'unethical']
        }
    
    def _load_credibility_markers(self) -> Dict:
        """Load markers of credible vs non-credible content"""
        return {
            'credible': ['according to', 'research shows', 'data indicates', 'expert opinion'],
            'non_credible': ['everyone knows', 'obviously', 'they say', 'rumors suggest']
        }
    
    def _load_manipulation_patterns(self) -> List:
        """Load known manipulation patterns"""
        return [
            r'[A-Z]{3,}!+',  # Excessive caps and exclamation
            r'share\s+before\s+(deleted|removed|banned)',  # False urgency
            r'(they|media)\s+don\'t\s+want\s+you\s+to',  # Conspiracy framing
            r'(BREAKING|URGENT|EXCLUSIVE):?\s*[A-Z]',  # Sensational headers
        ]
    
    def analyze(self, text: str) -> float:
        """Analyze text for misinformation linguistic patterns"""
        
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.analysis_cache:
            return self.analysis_cache[text_hash]
        
        # Emotion density analysis
        emotion_score = self._calculate_emotion_density(text)
        
        # Credibility marker analysis
        credibility_score = self._assess_credibility_markers(text)
        
        # Manipulation pattern detection
        manipulation_score = self._detect_manipulation_patterns(text)
        
        # Composite linguistic risk score
        risk_score = (
            emotion_score * 0.4 +
            (1 - credibility_score) * 0.3 +
            manipulation_score * 0.3
        )
        
        self.analysis_cache[text_hash] = risk_score
        self.last_analysis = {
            'emotion_score': emotion_score,
            'credibility_score': credibility_score,
            'manipulation_score': manipulation_score
        }
        
        return risk_score
    
    def _calculate_emotion_density(self, text: str) -> float:
        """Calculate density of emotional language"""
        words = text.lower().split()
        if not words:
            return 0.0
            
        emotion_count = 0
        for emotion_type, emotion_words in self.emotion_lexicon.items():
            emotion_count += sum(1 for word in words if word in emotion_words)
        
        return min(emotion_count / len(words), 1.0)
    
    def _assess_credibility_markers(self, text: str) -> float:
        """Assess presence of credibility markers"""
        text_lower = text.lower()
        
        credible_count = sum(1 for marker in self.credibility_markers['credible'] 
                           if marker in text_lower)
        non_credible_count = sum(1 for marker in self.credibility_markers['non_credible'] 
                               if marker in text_lower)
        
        if credible_count + non_credible_count == 0:
            return 0.5
        
        return credible_count / (credible_count + non_credible_count)
    
    def _detect_manipulation_patterns(self, text: str) -> float:
        """Detect manipulation patterns using regex"""
        pattern_matches = 0
        for pattern in self.manipulation_patterns:
            if re.search(pattern, text):
                pattern_matches += 1
        
        return min(pattern_matches / len(self.manipulation_patterns), 1.0)
    
    def get_details(self) -> Dict:
        """Get detailed analysis results"""
        return getattr(self, 'last_analysis', {})

class NetworkBehaviorAnalyzer:
    """Analyzes network propagation patterns"""
    
    def __init__(self):
        self.suspicious_thresholds = {
            'velocity': {'shares_per_minute': 50, 'time_window': 60},
            'amplification': {'bot_percentage': 0.3, 'new_account_percentage': 0.5},
            'coordination': {'temporal_clustering': 0.7, 'content_similarity': 0.8}
        }
        self.analysis_history = deque(maxlen=1000)
        
    def analyze_propagation(self, engagement_metrics: Dict, timestamp: datetime) -> float:
        """Analyze propagation patterns for suspicious behavior"""
        
        # Velocity analysis
        velocity_score = self._analyze_velocity(engagement_metrics)
        
        # Amplification analysis
        amplification_score = self._analyze_amplification(engagement_metrics)
        
        # Coordination analysis
        coordination_score = self._detect_coordination(engagement_metrics, timestamp)
        
        # Composite network risk
        network_risk = (
            velocity_score * 0.4 +
            amplification_score * 0.3 +
            coordination_score * 0.3
        )
        
        self.last_analysis = {
            'velocity_score': velocity_score,
            'amplification_score': amplification_score,
            'coordination_score': coordination_score
        }
        
        return network_risk
    
    def _analyze_velocity(self, metrics: Dict) -> float:
        """Analyze sharing velocity"""
        shares = metrics.get('shares', 0)
        time_elapsed = metrics.get('time_elapsed_minutes', 1)
        
        shares_per_minute = shares / max(time_elapsed, 1)
        threshold = self.suspicious_thresholds['velocity']['shares_per_minute']
        
        return min(shares_per_minute / threshold, 1.0)
    
    def _analyze_amplification(self, metrics: Dict) -> float:
        """Analyze artificial amplification indicators"""
        total_shares = metrics.get('shares', 1)
        
        # Bot-like behavior indicators
        bot_shares = metrics.get('bot_like_shares', 0)
        new_account_shares = metrics.get('new_account_shares', 0)
        
        bot_percentage = bot_shares / total_shares
        new_account_percentage = new_account_shares / total_shares
        
        bot_score = bot_percentage / self.suspicious_thresholds['amplification']['bot_percentage']
        new_account_score = new_account_percentage / self.suspicious_thresholds['amplification']['new_account_percentage']
        
        return min((bot_score + new_account_score) / 2, 1.0)
    
    def _detect_coordination(self, metrics: Dict, timestamp: datetime) -> float:
        """Detect coordinated behavior"""
        # Simplified coordination detection
        temporal_clustering = metrics.get('temporal_clustering_score', 0)
        content_similarity = metrics.get('similar_content_score', 0)
        
        coordination_score = (temporal_clustering + content_similarity) / 2
        return coordination_score
    
    def get_details(self) -> Dict:
        """Get detailed analysis results"""
        return getattr(self, 'last_analysis', {})

class MediaVerifier:
    """Verifies media authenticity"""
    
    def __init__(self):
        self.verification_methods = {
            'reverse_image_search': self._reverse_image_search,
            'metadata_analysis': self._analyze_metadata,
            'manipulation_detection': self._detect_manipulation
        }
        
    def verify_media(self, media_urls: List[str]) -> float:
        """Verify media authenticity"""
        if not media_urls:
            return 0.0
            
        verification_scores = []
        for url in media_urls:
            score = self._verify_single_media(url)
            verification_scores.append(score)
            
        return sum(verification_scores) / len(verification_scores)
    
    def _verify_single_media(self, url: str) -> float:
        """Verify a single media item"""
        # Simplified verification - in reality would use AI models
        # and reverse image search APIs
        
        # Check if URL is from known manipulated media database
        if self._is_known_fake(url):
            return 1.0
            
        # Check metadata inconsistencies
        metadata_risk = self._analyze_metadata(url)
        
        # Check for manipulation markers
        manipulation_risk = self._detect_manipulation(url)
        
        return (metadata_risk + manipulation_risk) / 2
    
    def _is_known_fake(self, url: str) -> bool:
        """Check against known fake media database"""
        # Simplified - would use actual database
        known_fakes = ['fake1.jpg', 'manipulated2.png']
        return any(fake in url for fake in known_fakes)
    
    def _reverse_image_search(self, url: str) -> float:
        """Perform reverse image search"""
        # Placeholder - would use actual reverse image search API
        return 0.0
    
    def _analyze_metadata(self, url: str) -> float:
        """Analyze media metadata for inconsistencies"""
        # Placeholder - would analyze EXIF data, timestamps, etc.
        return 0.0
    
    def _detect_manipulation(self, url: str) -> float:
        """Detect media manipulation"""
        # Placeholder - would use deepfake detection, manipulation detection models
        return 0.0

class FactCheckInterface:
    """Interface to fact-checking systems and databases"""
    
    def __init__(self):
        self.fact_check_sources = [
            'internal_database',
            'partner_fact_checkers',
            'academic_sources'
        ]
        self.claim_cache = {}
        
    def check_claims(self, text: str) -> Dict:
        """Check claims against fact-checking databases"""
        
        # Extract claims from text
        claims = self._extract_claims(text)
        
        if not claims:
            return {'credibility_score': 0.5, 'claims_checked': 0}
        
        # Check each claim
        claim_results = []
        for claim in claims:
            if claim in self.claim_cache:
                result = self.claim_cache[claim]
            else:
                result = self._check_single_claim(claim)
                self.claim_cache[claim] = result
            claim_results.append(result)
        
        # Aggregate results
        credibility_score = sum(r['credibility'] for r in claim_results) / len(claim_results)
        
        return {
            'credibility_score': credibility_score,
            'claims_checked': len(claims),
            'detailed_results': claim_results
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text"""
        # Simplified claim extraction
        # In reality would use NLP to identify factual claims
        sentences = text.split('.')
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims[:5]  # Limit to 5 claims for efficiency
    
    def _check_single_claim(self, claim: str) -> Dict:
        """Check a single claim"""
        # Simplified fact-checking
        # In reality would query multiple fact-checking databases
        
        # Simulate checking against database
        if any(keyword in claim.lower() for keyword in ['false', 'hoax', 'debunked']):
            return {'claim': claim, 'credibility': 0.0, 'source': 'fact_check_db'}
        elif any(keyword in claim.lower() for keyword in ['verified', 'confirmed', 'true']):
            return {'claim': claim, 'credibility': 1.0, 'source': 'fact_check_db'}
        else:
            return {'claim': claim, 'credibility': 0.5, 'source': 'unverified'}

class VulnerabilityAssessment:
    """Assess population vulnerability to misinformation"""
    
    def __init__(self):
        self.vulnerability_factors = {
            'demographic': {
                'age_65_plus': 0.8,
                'low_digital_literacy': 0.7,
                'high_partisan': 0.6,
                'isolated_community': 0.7
            },
            'behavioral': {
                'high_social_media_use': 0.5,
                'low_source_diversity': 0.7,
                'emotion_driven_sharing': 0.8,
                'low_fact_checking': 0.9
            },
            'contextual': {
                'election_period': 0.8,
                'crisis_situation': 0.9,
                'polarized_issue': 0.7,
                'local_relevance': 0.6
            }
        }
        
    def assess_population_vulnerability(self, population_profile: Dict, context: Dict) -> Dict:
        """Assess vulnerability of a population segment"""
        
        # Calculate demographic vulnerability
        demo_score = self._calculate_demographic_vulnerability(population_profile)
        
        # Calculate behavioral vulnerability
        behavioral_score = self._calculate_behavioral_vulnerability(population_profile)
        
        # Calculate contextual vulnerability
        contextual_score = self._calculate_contextual_vulnerability(context)
        
        # Composite vulnerability
        overall_vulnerability = (
            demo_score * 0.3 +
            behavioral_score * 0.4 +
            contextual_score * 0.3
        )
        
        # Generate protection recommendations
        recommendations = self._generate_protection_recommendations(
            overall_vulnerability,
            population_profile,
            context
        )
        
        return {
            'overall_vulnerability': overall_vulnerability,
            'component_scores': {
                'demographic': demo_score,
                'behavioral': behavioral_score,
                'contextual': contextual_score
            },
            'high_risk_factors': self._identify_high_risk_factors(population_profile, context),
            'recommendations': recommendations
        }
    
    def _calculate_demographic_vulnerability(self, profile: Dict) -> float:
        """Calculate demographic vulnerability score"""
        total_score = 0
        factor_count = 0
        
        for factor, weight in self.vulnerability_factors['demographic'].items():
            if factor in profile and profile[factor]:
                total_score += weight
                factor_count += 1
        
        return total_score / max(factor_count, 1)
    
    def _calculate_behavioral_vulnerability(self, profile: Dict) -> float:
        """Calculate behavioral vulnerability score"""
        total_score = 0
        factor_count = 0
        
        for factor, weight in self.vulnerability_factors['behavioral'].items():
            if factor in profile and profile[factor]:
                total_score += weight
                factor_count += 1
        
        return total_score / max(factor_count, 1)
    
    def _calculate_contextual_vulnerability(self, context: Dict) -> float:
        """Calculate contextual vulnerability score"""
        total_score = 0
        factor_count = 0
        
        for factor, weight in self.vulnerability_factors['contextual'].items():
            if factor in context and context[factor]:
                total_score += weight
                factor_count += 1
        
        return total_score / max(factor_count, 1)
    
    def _identify_high_risk_factors(self, profile: Dict, context: Dict) -> List[str]:
        """Identify the highest risk factors"""
        risk_factors = []
        
        # Check all factors
        all_factors = {
            **self.vulnerability_factors['demographic'],
            **self.vulnerability_factors['behavioral'],
            **self.vulnerability_factors['contextual']
        }
        
        for factor, weight in all_factors.items():
            if weight >= 0.7 and (factor in profile or factor in context):
                if profile.get(factor) or context.get(factor):
                    risk_factors.append(factor)
        
        return risk_factors
    
    def _generate_protection_recommendations(self, vulnerability: float, 
                                           profile: Dict, context: Dict) -> List[str]:
        """Generate specific protection recommendations"""
        recommendations = []
        
        if vulnerability > 0.7:
            recommendations.append("High priority for prebunking campaigns")
            recommendations.append("Deploy trusted messenger network")
            recommendations.append("Increase fact-checking resources")
        
        if profile.get('low_digital_literacy'):
            recommendations.append("Digital literacy training programs")
            recommendations.append("Simplified fact-checking tools")
        
        if profile.get('high_partisan'):
            recommendations.append("Cross-partisan fact-checking initiatives")
            recommendations.append("Neutral source promotion")
        
        if context.get('election_period'):
            recommendations.append("Election-specific misinformation monitoring")
            recommendations.append("Voter information campaigns")
        
        return recommendations

class ResponseCoordinator:
    """Coordinates multi-stakeholder response to threats"""
    
    def __init__(self):
        self.stakeholders = {
            'fact_checkers': FactCheckerNetwork(),
            'platforms': PlatformLiaison(),
            'community': CommunityLeaders(),
            'media': MediaPartners(),
            'academic': AcademicExperts()
        }
        self.response_history = []
        self.active_responses = {}
        
    def coordinate_response(self, threat_assessment: ThreatAssessment) -> Dict:
        """Coordinate response based on threat assessment"""
        
        response_id = self._generate_response_id()
        
        # Determine required stakeholders
        required_stakeholders = self._determine_stakeholders(threat_assessment.threat_level)
        
        # Create response plan
        response_plan = {
            'response_id': response_id,
            'threat_assessment': threat_assessment,
            'stakeholders': required_stakeholders,
            'actions': [],
            'timeline': self._create_timeline(threat_assessment.threat_level),
            'success_metrics': self._define_success_metrics(threat_assessment)
        }
        
        # Activate stakeholders
        for stakeholder_type in required_stakeholders:
            stakeholder = self.stakeholders[stakeholder_type]
            action = stakeholder.activate(threat_assessment)
            response_plan['actions'].append(action)
        
        # Store active response
        self.active_responses[response_id] = response_plan
        self.response_history.append(response_plan)
        
        return response_plan
    
    def _generate_response_id(self) -> str:
        """Generate unique response ID"""
        return f"RESP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.response_history)}"
    
    def _determine_stakeholders(self, threat_level: ThreatLevel) -> List[str]:
        """Determine which stakeholders to activate"""
        stakeholder_map = {
            ThreatLevel.GREEN: [],
            ThreatLevel.YELLOW: ['fact_checkers'],
            ThreatLevel.ORANGE: ['fact_checkers', 'platforms', 'community'],
            ThreatLevel.RED: ['fact_checkers', 'platforms', 'community', 'media', 'academic']
        }
        return stakeholder_map[threat_level]
    
    def _create_timeline(self, threat_level: ThreatLevel) -> Dict:
        """Create response timeline"""
        timelines = {
            ThreatLevel.GREEN: {'initial_response': '24 hours', 'follow_up': '1 week'},
            ThreatLevel.YELLOW: {'initial_response': '6 hours', 'follow_up': '48 hours'},
            ThreatLevel.ORANGE: {'initial_response': '2 hours', 'follow_up': '24 hours'},
            ThreatLevel.RED: {'initial_response': '30 minutes', 'follow_up': '6 hours'}
        }
        return timelines[threat_level]
    
    def _define_success_metrics(self, threat_assessment: ThreatAssessment) -> Dict:
        """Define success metrics for response"""
        return {
            'spread_reduction': 0.7,  # 70% reduction in spread
            'fact_check_reach': 0.5,  # Fact-check reaches 50% of exposed population
            'sentiment_shift': 0.3,   # 30% positive shift in sentiment
            'engagement_decline': 0.6  # 60% decline in engagement with misinformation
        }
    
    def monitor_response(self, response_id: str) -> Dict:
        """Monitor ongoing response effectiveness"""
        if response_id not in self.active_responses:
            return {'error': 'Response ID not found'}
        
        response = self.active_responses[response_id]
        
        # Collect metrics from stakeholders
        metrics = {}
        for stakeholder_type in response['stakeholders']:
            stakeholder = self.stakeholders[stakeholder_type]
            metrics[stakeholder_type] = stakeholder.get_metrics(response_id)
        
        # Assess effectiveness
        effectiveness = self._assess_effectiveness(metrics, response['success_metrics'])
        
        return {
            'response_id': response_id,
            'status': 'active',
            'metrics': metrics,
            'effectiveness': effectiveness,
            'recommendations': self._generate_recommendations(effectiveness)
        }
    
    def _assess_effectiveness(self, current_metrics: Dict, success_metrics: Dict) -> float:
        """Assess response effectiveness"""
        # Simplified effectiveness calculation
        # In reality would compare actual metrics to success criteria
        return 0.75  # Placeholder
    
    def _generate_recommendations(self, effectiveness: float) -> List[str]:
        """Generate recommendations based on effectiveness"""
        recommendations = []
        
        if effectiveness < 0.5:
            recommendations.append("Escalate response level")
            recommendations.append("Engage additional stakeholders")
        elif effectiveness < 0.7:
            recommendations.append("Adjust messaging strategy")
            recommendations.append("Increase fact-check visibility")
        else:
            recommendations.append("Maintain current approach")
            recommendations.append("Document successful tactics")
        
        return recommendations

# Stakeholder implementations
class FactCheckerNetwork:
    """Network of fact-checkers"""
    
    def activate(self, threat_assessment: ThreatAssessment) -> Dict:
        """Activate fact-checking response"""
        return {
            'stakeholder': 'fact_checkers',
            'action': 'fact_check_initiated',
            'assigned_checkers': 3,
            'expected_completion': '2 hours'
        }
    
    def get_metrics(self, response_id: str) -> Dict:
        """Get fact-checking metrics"""
        return {
            'claims_checked': 15,
            'false_claims_identified': 8,
            'fact_checks_published': 5,
            'reach': 50000
        }

class PlatformLiaison:
    """Platform coordination"""
    
    def activate(self, threat_assessment: ThreatAssessment) -> Dict:
        """Activate platform response"""
        return {
            'stakeholder': 'platforms',
            'action': 'content_review_initiated',
            'platforms_notified': ['Twitter', 'Facebook', 'TikTok'],
            'expected_action': 'labels_and_reduced_distribution'
        }
    
    def get_metrics(self, response_id: str) -> Dict:
        """Get platform metrics"""
        return {
            'content_labeled': 1200,
            'distribution_reduced': 0.6,
            'accounts_suspended': 15
        }

class CommunityLeaders:
    """Community leader network"""
    
    def activate(self, threat_assessment: ThreatAssessment) -> Dict:
        """Activate community response"""
        return {
            'stakeholder': 'community',
            'action': 'community_alerts_sent',
            'leaders_notified': 25,
            'communities_reached': 10
        }
    
    def get_metrics(self, response_id: str) -> Dict:
        """Get community metrics"""
        return {
            'prebunking_shared': 500,
            'community_discussions': 30,
            'positive_engagement': 0.7
        }

class MediaPartners:
    """Media partner network"""
    
    def activate(self, threat_assessment: ThreatAssessment) -> Dict:
        """Activate media response"""
        return {
            'stakeholder': 'media',
            'action': 'media_briefing_scheduled',
            'outlets_engaged': 5,
            'stories_planned': 3
        }
    
    def get_metrics(self, response_id: str) -> Dict:
        """Get media metrics"""
        return {
            'articles_published': 8,
            'broadcast_segments': 3,
            'estimated_reach': 1000000
        }

class AcademicExperts:
    """Academic expert network"""
    
    def activate(self, threat_assessment: ThreatAssessment) -> Dict:
        """Activate academic response"""
        return {
            'stakeholder': 'academic',
            'action': 'expert_analysis_requested',
            'experts_engaged': 3,
            'analysis_timeline': '4 hours'
        }
    
    def get_metrics(self, response_id: str) -> Dict:
        """Get academic metrics"""
        return {
            'analyses_provided': 2,
            'media_appearances': 5,
            'briefings_delivered': 3
        }

# Main system orchestrator
class MisinformationDefenseSystem:
    """Main system that orchestrates all components"""
    
    def __init__(self):
        self.detection_system = DefensiveDetectionSystem()
        self.content_analyzer = ContentAnalyzer()
        self.vulnerability_assessor = VulnerabilityAssessment()
        self.response_coordinator = ResponseCoordinator()
        self.monitoring_dashboard = MonitoringDashboard()
        
    def process_content(self, content: ContentItem) -> ThreatAssessment:
        """Process content through full pipeline"""
        
        # Analyze content
        analysis_result = self.content_analyzer.analyze_content(content)
        
        # Determine threat level
        threat_level = self._determine_threat_level(analysis_result['composite_risk'])
        
        # Assess population vulnerability
        population_profile = self._get_population_profile(content.platform)
        context = self._get_current_context()
        vulnerability = self.vulnerability_assessor.assess_population_vulnerability(
            population_profile, context
        )
        
        # Create threat assessment
        threat_assessment = ThreatAssessment(
            content_id=content.id,
            risk_score=analysis_result['composite_risk'],
            threat_level=threat_level,
            indicators={
                'content_analysis': analysis_result,
                'vulnerability_assessment': vulnerability
            },
            recommended_actions=self._generate_recommendations(
                threat_level, vulnerability
            ),
            timestamp=datetime.now()
        )
        
        # Log in detection system
        self.detection_system.detection_history.append(threat_assessment)
        
        # Coordinate response if needed
        if threat_level in [ThreatLevel.ORANGE, ThreatLevel.RED]:
            response_plan = self.response_coordinator.coordinate_response(threat_assessment)
            self.monitoring_dashboard.add_active_threat(threat_assessment, response_plan)
        
        return threat_assessment
    
    def _determine_threat_level(self, risk_score: float) -> ThreatLevel:
        """Determine threat level from risk score"""
        if risk_score >= 0.85:
            return ThreatLevel.RED
        elif risk_score >= 0.7:
            return ThreatLevel.ORANGE
        elif risk_score >= 0.5:
            return ThreatLevel.YELLOW
        else:
            return ThreatLevel.GREEN
    
    def _get_population_profile(self, platform: str) -> Dict:
        """Get population profile for platform"""
        # Simplified - would use actual demographic data
        platform_profiles = {
            'facebook': {
                'age_65_plus': True,
                'low_digital_literacy': False,
                'high_social_media_use': True
            },
            'twitter': {
                'high_partisan': True,
                'low_source_diversity': False,
                'emotion_driven_sharing': True
            },
            'tiktok': {
                'age_65_plus': False,
                'high_social_media_use': True,
                'low_fact_checking': True
            }
        }
        return platform_profiles.get(platform.lower(), {})
    
    def _get_current_context(self) -> Dict:
        """Get current contextual factors"""
        # Simplified - would use actual context data
        return {
            'election_period': False,
            'crisis_situation': False,
            'polarized_issue': True,
            'local_relevance': False
        }
    
    def _generate_recommendations(self, threat_level: ThreatLevel, 
                                vulnerability: Dict) -> List[str]:
        """Generate action recommendations"""
        recommendations = []
        
        # Threat level based recommendations
        if threat_level == ThreatLevel.RED:
            recommendations.extend([
                "Immediate crisis response activation",
                "Cross-platform coordination required",
                "Deploy rapid prebunking campaign"
            ])
        elif threat_level == ThreatLevel.ORANGE:
            recommendations.extend([
                "Enhanced monitoring activated",
                "Fact-checking priority queue",
                "Community leader notification"
            ])
        
        # Vulnerability based recommendations
        if vulnerability['overall_vulnerability'] > 0.7:
            recommendations.extend(vulnerability['recommendations'])
        
        return recommendations

class MonitoringDashboard:
    """Real-time monitoring dashboard"""
    
    def __init__(self):
        self.active_threats = {}
        self.metrics_history = deque(maxlen=10000)
        self.alert_subscribers = []
        
    def add_active_threat(self, threat: ThreatAssessment, response: Dict):
        """Add active threat to monitoring"""
        self.active_threats[threat.content_id] = {
            'threat': threat,
            'response': response,
            'start_time': datetime.now(),
            'status': 'active'
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get current dashboard data"""
        return {
            'active_threats': len(self.active_threats),
            'threat_breakdown': self._get_threat_breakdown(),
            'response_effectiveness': self._calculate_average_effectiveness(),
            'system_health': self._get_system_health(),
            'recent_alerts': self._get_recent_alerts()
        }
    
    def _get_threat_breakdown(self) -> Dict:
        """Get breakdown of active threats by level"""
        breakdown = {level: 0 for level in ThreatLevel}
        for threat_data in self.active_threats.values():
            level = threat_data['threat'].threat_level
            breakdown[level] += 1
        return breakdown
    
    def _calculate_average_effectiveness(self) -> float:
        """Calculate average response effectiveness"""
        # Placeholder - would calculate from actual metrics
        return 0.78
    
    def _get_system_health(self) -> Dict:
        """Get system health metrics"""
        return {
            'detection_latency': '28 minutes',
            'processing_capacity': '85%',
            'api_availability': '99.9%',
            'model_accuracy': '94.5%'
        }
    
    def _get_recent_alerts(self) -> List[Dict]:
        """Get recent high-priority alerts"""
        recent = []
        for threat_id, threat_data in self.active_threats.items():
            if threat_data['threat'].threat_level in [ThreatLevel.ORANGE, ThreatLevel.RED]:
                recent.append({
                    'threat_id': threat_id,
                    'level': threat_data['threat'].threat_level.value,
                    'time': threat_data['start_time'].isoformat(),
                    'risk_score': threat_data['threat'].risk_score
                })
        return sorted(recent, key=lambda x: x['time'], reverse=True)[:10]

# Example usage
def run_defensive_system_demo():
    """Demonstrate the defensive system"""
    
    # Initialize system
    defense_system = MisinformationDefenseSystem()
    
    # Simulate suspicious content
    suspicious_content = ContentItem(
        id="content_001",
        text="URGENT: They don't want you to know this shocking truth about the government!",
        source="unknown_user_123",
        timestamp=datetime.now(),
        platform="Twitter",
        engagement_metrics={
            'shares': 850,
            'time_elapsed_minutes': 15,
            'bot_like_shares': 200,
            'new_account_shares': 300,
            'temporal_clustering_score': 0.8,
            'similar_content_score': 0.7
        }
    )
    
    # Process through system
    print("Processing suspicious content...")
    threat_assessment = defense_system.process_content(suspicious_content)
    
    # Display results
    print(f"\nThreat Assessment:")
    print(f"- Content ID: {threat_assessment.content_id}")
    print(f"- Risk Score: {threat_assessment.risk_score:.2f}")
    print(f"- Threat Level: {threat_assessment.threat_level.value}")
    print(f"\nRecommended Actions:")
    for action in threat_assessment.recommended_actions:
        print(f"  • {action}")
    
    # Show dashboard
    dashboard_data = defense_system.monitoring_dashboard.get_dashboard_data()
    print(f"\nDashboard Status:")
    print(f"- Active Threats: {dashboard_data['active_threats']}")
    print(f"- System Health: {dashboard_data['system_health']}")
    print(f"- Response Effectiveness: {dashboard_data['response_effectiveness']:.1%}")

if __name__ == "__main__":
    run_defensive_system_demo()