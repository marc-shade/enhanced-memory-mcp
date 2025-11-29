"""
Reasoning Prioritizer for Enhanced Memory MCP

Implements 75/15 rule from AI vision research:
- 75% priority for reasoning-centric content (code, math, science)
- 15% priority for visual-descriptive content
- 10% priority for general content

Research finding: Reasoning-centric content provides foundation for
visual understanding and should be prioritized in memory storage/retrieval.
"""
import re
from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass


class ContentCategory(Enum):
    """Content categories based on 75/15 rule."""
    REASONING_CENTRIC = "reasoning_centric"  # Code, math, science, logic
    VISUAL_CENTRIC = "visual_centric"        # Visual descriptions
    GENERAL = "general"                      # General text


@dataclass
class PriorityScore:
    """Priority scoring for memory content."""
    category: ContentCategory
    weight: float  # 0.75, 0.15, or 0.10
    reasoning_score: float  # 0-1
    visual_score: float     # 0-1
    confidence: float       # 0-1


class ReasoningPrioritizer:
    """
    Classifies and prioritizes memory content based on 75/15 rule.

    Priority weights:
    - Reasoning (code/math/science): 0.75
    - Visual (descriptions): 0.15
    - General (other): 0.10
    """

    def __init__(self):
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize regex patterns for content detection."""
        # Code patterns
        self.code_patterns = [
            r'def\s+\w+\s*\(',
            r'class\s+\w+',
            r'function\s+\w+\s*\(',
            r'import\s+\w+',
            r'=>',
            r'for\s*\(',
            r'if\s*\(',
            r'\{[^}]*\}',
            r'async\s+def',
            r'return\s+',
            r'lambda\s+',
        ]

        # Math patterns - EXPANDED
        self.math_patterns = [
            r'\$.*?\$',
            r'theorem|lemma|proof|corollary|axiom',
            r'\\frac|\\sum|\\int|\\prod',
            r'\d+\s*[+\-*/]\s*\d+',
            r'∀|∃|∈|∉|⊂|⊃|∪|∩',
            r'O\([nlogN²³]+\)',  # Big-O notation
            r'Ω\([nlogN²³]+\)',  # Big-Omega notation
            r'Θ\([nlogN²³]+\)',  # Big-Theta notation
            r'bijection|isomorphism|homomorphism',
            r'matrix|vector|tensor|eigenvalue',
            r'derivative|integral|limit|convergence',
            r'probability|expectation|variance',
            r'gradient|jacobian|hessian',
            r'algebra|topology|geometry|calculus',
            r'induction|recursion|iteration',
            r'∇|∂|∫|∑|∏',  # Math operators
        ]

        # Algorithm/CS Theory patterns - NEW
        self.algorithm_patterns = [
            r'algorithm|algorithmic',
            r'complexity|time complexity|space complexity',
            r'O\(n\)|O\(log n\)|O\(n\^2\)|O\(1\)',
            r'divide[- ]and[- ]conquer',
            r'dynamic programming|greedy|backtracking',
            r'breadth[- ]first|depth[- ]first|BFS|DFS',
            r'graph|tree|heap|hash|stack|queue',
            r'sorting|searching|traversal',
            r'optimization|optimize|optimal',
            r'data structure|linked list|binary tree',
            r'partition|pivot|merge|quicksort|mergesort',
            r'NP[- ]complete|NP[- ]hard|polynomial time',
            r'recursion|recursive|base case',
            r'invariant|precondition|postcondition',
            r'amortized|worst[- ]case|average[- ]case|best[- ]case',
        ]

        # Theoretical CS patterns - NEW
        self.theory_patterns = [
            r'automata|turing machine|finite state',
            r'decidable|undecidable|halting problem',
            r'P vs NP|computational complexity',
            r'type theory|type system|polymorphism',
            r'lambda calculus|functional programming',
            r'formal verification|model checking',
            r'logic programming|prolog',
            r'compiler|parser|lexer|AST',
            r'semantics|syntax|grammar',
            r'category theory|monad|functor',
        ]

        # Science patterns
        self.science_patterns = [
            r'experiment|hypothesis|methodology',
            r'Figure\s+\d+',
            r'p\s*[<>=]\s*0\.\d+',
            r'DNA|RNA|protein',
            r'velocity|acceleration|force',
            r'statistical significance|confidence interval',
            r'control group|experimental group',
            r'correlation|causation|causal',
        ]

        # Logic patterns - EXPANDED
        self.logic_patterns = [
            r'if\s+.*\s+then',
            r'therefore|thus|hence|consequently',
            r'assume|suppose|given|let\s+',
            r'necessary|sufficient',
            r'implies|implication|entails',
            r'contradiction|contrapositive',
            r'∧|∨|¬|→|↔',  # Logic operators
            r'forall|exists|quantifier',
            r'syllogism|modus ponens|modus tollens',
            r'deduction|induction|abduction',
            r'inference|reasoning|rational',
        ]

        # Numerical/Optimization patterns - NEW
        self.numerical_patterns = [
            r'gradient descent|backpropagation',
            r'neural network|deep learning|CNN|RNN|LSTM',
            r'optimization problem|minimize|maximize',
            r'convex|concave|saddle point',
            r'learning rate|momentum|adam optimizer',
            r'loss function|cost function|objective',
            r'regularization|overfitting|underfitting',
            r'cross[- ]validation|train[- ]test split',
            r'precision|recall|F1[- ]score|accuracy',
            r'regression|classification|clustering',
        ]

        # Visual patterns (expanded for comprehensive visual/spatial detection)
        self.visual_patterns = [
            # Basic visual terms
            r'color|colour|red|blue|green|yellow|orange|purple|pink|hue|saturation|brightness',
            r'shape|circle|square|triangle|rectangle|polygon|ellipse|curve',
            r'image|picture|photo|photograph|screenshot|thumbnail|icon|illustration',
            r'looks like|appears|visible|visual|sight|seen|view|display',
            r'left|right|top|bottom|center|middle|corner|edge|border|margin|padding',

            # Spatial reasoning
            r'spatial|position|location|coordinate|x-axis|y-axis|z-axis|dimension',
            r'rotate|rotation|translate|scale|transform|perspective|projection',
            r'distance|proximity|overlap|intersect|adjacent|parallel|perpendicular',
            r'width|height|depth|size|area|volume|aspect ratio',
            r'3D|2D|three-dimensional|two-dimensional|isometric|orthographic',

            # UI/UX design
            r'layout|grid|flexbox|responsive|breakpoint|viewport|mobile|desktop',
            r'typography|font|typeface|serif|sans-serif|kerning|leading|baseline',
            r'whitespace|negative space|visual hierarchy|contrast|alignment',
            r'button|input|form|card|modal|dropdown|navigation|menu|tab',
            r'hover|focus|active|disabled|animation|transition|easing',

            # Data visualization
            r'chart|graph|plot|diagram|infographic|dashboard|visualization',
            r'bar chart|line chart|pie chart|scatter|histogram|heatmap|treemap',
            r'axis|legend|label|tooltip|annotation|gridline|tick mark',
            r'sparkline|gauge|KPI|metric|trend|comparison',

            # Graphics and rendering
            r'pixel|resolution|DPI|PPI|retina|vector|raster|bitmap|SVG|PNG|JPEG',
            r'render|draw|paint|stroke|fill|gradient|shadow|blur|opacity|alpha',
            r'layer|z-index|composite|blend mode|mask|clip|filter',
            r'vertex|fragment|shader|texture|mesh|polygon|primitive',
            r'lighting|ambient|diffuse|specular|reflection|refraction',

            # Computer vision
            r'edge detection|segmentation|feature|keypoint|descriptor|SIFT|SURF',
            r'convolution|kernel|filter|pooling|CNN|object detection|classification',
            r'bounding box|mask|contour|region|blob|corner|Harris',
            r'optical flow|tracking|motion|stereo|depth map|point cloud',

            # Cartography/GIS
            r'map|geographic|latitude|longitude|projection|Mercator|GIS|tile',
            r'choropleth|marker|polygon|polyline|geofence|geocode',

            # Composition and design
            r'composition|rule of thirds|golden ratio|focal point|balance|symmetry',
            r'foreground|background|midground|depth|framing|leading line',
            r'gestalt|proximity|similarity|continuity|closure|figure-ground',

            # Accessibility
            r'WCAG|contrast ratio|colorblind|accessibility|alt text|screen reader',
        ]

    def classify_content(self, content: str) -> PriorityScore:
        """
        Classify content and assign priority score.

        Args:
            content: Text content to classify

        Returns:
            PriorityScore with category and weights
        """
        # Calculate scores for each content type
        code_score = self._calculate_pattern_score(content, self.code_patterns)
        math_score = self._calculate_pattern_score(content, self.math_patterns)
        algorithm_score = self._calculate_pattern_score(content, self.algorithm_patterns)
        theory_score = self._calculate_pattern_score(content, self.theory_patterns)
        science_score = self._calculate_pattern_score(content, self.science_patterns)
        logic_score = self._calculate_pattern_score(content, self.logic_patterns)
        numerical_score = self._calculate_pattern_score(content, self.numerical_patterns)
        visual_score = self._calculate_pattern_score(content, self.visual_patterns)

        # Overall reasoning score (max of all reasoning-related scores)
        reasoning_score = max(
            code_score,
            math_score,
            algorithm_score,
            theory_score,
            science_score,
            logic_score,
            numerical_score
        )

        # Determine category and weight
        # Lower threshold to 0.3 to catch more reasoning content
        if reasoning_score > 0.3:
            category = ContentCategory.REASONING_CENTRIC
            weight = 0.75
        elif visual_score > 0.5:
            category = ContentCategory.VISUAL_CENTRIC
            weight = 0.15
        else:
            category = ContentCategory.GENERAL
            weight = 0.10

        return PriorityScore(
            category=category,
            weight=weight,
            reasoning_score=reasoning_score,
            visual_score=visual_score,
            confidence=max(reasoning_score, visual_score, 0.3)
        )

    def _calculate_pattern_score(self, content: str, patterns: List[str]) -> float:
        """Calculate score based on pattern matches."""
        matches = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in patterns
        )

        # Normalize by content length
        words = len(content.split())
        if words == 0:
            return 0.0

        score = min(1.0, matches / max(1, words // 10))
        return score

    def rank_memories(
        self,
        memories: List[Dict],
        query: str,
        boost_reasoning: bool = True
    ) -> List[Tuple[Dict, float]]:
        """
        Rank memories with reasoning prioritization.

        Args:
            memories: List of memory dictionaries
            query: Search query
            boost_reasoning: Whether to boost reasoning-centric results

        Returns:
            List of (memory, score) tuples, sorted by score
        """
        ranked = []
        query_priority = self.classify_content(query)

        for memory in memories:
            # Get memory content
            content = str(memory.get('observations', ''))

            # Classify memory
            memory_priority = self.classify_content(content)

            # Base relevance score (simplified - should use semantic similarity)
            base_score = self._calculate_relevance(query, content)

            # Apply priority boost
            priority_multiplier = 1.0
            if boost_reasoning:
                # Boost if memory matches query category
                if memory_priority.category == query_priority.category:
                    priority_multiplier = 1.2

                # Always boost reasoning-centric content
                if memory_priority.category == ContentCategory.REASONING_CENTRIC:
                    priority_multiplier *= 1.3  # 75% boost

            final_score = base_score * priority_multiplier * memory_priority.weight

            ranked.append((memory, final_score))

        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate basic relevance score (simplified)."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        intersection = query_words & content_words
        return len(intersection) / len(query_words)

    def get_compression_level(self, content: str) -> int:
        """
        Get optimal compression level based on content category.

        Reasoning-centric content (75%): Lower compression (level 6) - preserve detail
        Visual content (15%): Higher compression (level 9) - less critical
        General content (10%): Highest compression (level 9) - least critical

        Args:
            content: Content to compress

        Returns:
            Compression level (1-9)
        """
        priority = self.classify_content(content)

        if priority.category == ContentCategory.REASONING_CENTRIC:
            return 6  # Moderate compression, preserve reasoning details
        elif priority.category == ContentCategory.VISUAL_CENTRIC:
            return 8  # Higher compression, less critical
        else:
            return 9  # Maximum compression for general content

    def should_prioritize_storage(self, content: str) -> Tuple[bool, str]:
        """
        Determine if content should be prioritized for storage.

        Args:
            content: Content to evaluate

        Returns:
            Tuple of (should_prioritize, reason)
        """
        priority = self.classify_content(content)

        if priority.category == ContentCategory.REASONING_CENTRIC:
            return True, f"Reasoning-centric content (weight: {priority.weight})"
        elif priority.category == ContentCategory.VISUAL_CENTRIC:
            if priority.visual_score > 0.7:
                return True, "High-quality visual description"
            return False, "Visual content below quality threshold"
        else:
            return False, "General content (lowest priority)"

    def calculate_tier_priority(self, content: str, access_count: int) -> str:
        """
        Calculate optimal tier for content based on category and access.

        Tiers:
        - core: Essential reasoning patterns, frequently accessed
        - working: Active reasoning content, moderate access
        - reference: Archived reasoning, visual, and general content

        Args:
            content: Content to tier
            access_count: Number of times accessed

        Returns:
            Tier name (core, working, or reference)
        """
        priority = self.classify_content(content)

        # Reasoning-centric content gets priority tiers
        if priority.category == ContentCategory.REASONING_CENTRIC:
            if access_count > 10 or priority.reasoning_score > 0.8:
                return "core"  # Hot, frequently accessed reasoning
            else:
                return "working"  # Active reasoning

        # Visual content gets working tier to maintain 15% balance (75/15/10 rule)
        elif priority.category == ContentCategory.VISUAL_CENTRIC:
            if access_count > 10 or priority.visual_score > 0.8:
                return "core"  # High-value visual patterns
            else:
                return "working"  # Active visual content (not demoted to reference)

        else:  # General content - lowest priority
            return "reference"


# Singleton instance
_prioritizer = None

def get_prioritizer() -> ReasoningPrioritizer:
    """Get singleton prioritizer instance."""
    global _prioritizer
    if _prioritizer is None:
        _prioritizer = ReasoningPrioritizer()
    return _prioritizer
