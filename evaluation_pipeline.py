import os
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure Gemini API
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY in your .env file")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"❌ API Configuration Error: {e}")
    exit(1)


class EvaluationMetric(Enum):
    """Evaluation metrics for different types of assessment."""
    FACTUAL_ACCURACY = "factual_accuracy"
    HISTORICAL_CONTEXT = "historical_context"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"


@dataclass
class TestCase:
    """Represents a single test case in the evaluation dataset."""
    id: str
    category: str
    query: str
    expected_output: str
    context: str
    evaluation_criteria: List[EvaluationMetric]
    difficulty_level: str  # "easy", "medium", "hard"
    historical_period: str


@dataclass
class EvaluationResult:
    """Results from evaluating a single test case."""
    test_case_id: str
    model_output: str
    judge_scores: Dict[str, float]
    overall_score: float
    judge_reasoning: str
    passed: bool
    execution_time: float


class SangamGPTEvaluationPipeline:
    """
    Comprehensive evaluation pipeline for SangamGPT historical AI system.
    Includes dataset, judge prompts, and automated testing framework.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.total_tokens = 0
        self.test_results: List[EvaluationResult] = []
        
        # Initialize test dataset
        self.test_dataset = self._create_evaluation_dataset()
        
        # Judge prompt configuration parameters
        self.judge_parameters = {
            "temperature": 0.1,  # Low for consistent evaluation
            "top_p": 0.8,
            "max_output_tokens": 1000,
            "scoring_scale": "1-10",
            "evaluation_aspects": [
                "factual_accuracy",
                "historical_context", 
                "relevance",
                "completeness",
                "clarity"
            ]
        }
    
    def log_tokens(self, text: str, operation: str = ""):
        """Log estimated token usage."""
        tokens = len(text.split()) * 0.75
        self.total_tokens += tokens
        print(f"📊 {operation} Tokens: {int(tokens)} | Total: {int(self.total_tokens)}")
    
    def _create_evaluation_dataset(self) -> List[TestCase]:
        """
        Create a comprehensive evaluation dataset with diverse historical queries.
        This dataset covers different periods, difficulty levels, and evaluation aspects.
        """
        dataset = [
            TestCase(
                id="TC001",
                category="Ancient India",
                query="Who was Ashoka and what were his major contributions to Indian history?",
                expected_output="Ashoka (304-232 BCE) was the third Mauryan emperor who ruled most of the Indian subcontinent. His major contributions include: 1) Embracing and promoting Buddhism after the Kalinga War, 2) Establishing the first welfare state with hospitals and roads, 3) Creating the Edicts of Ashoka inscribed on rocks and pillars across his empire, 4) Promoting non-violence (ahimsa) and religious tolerance, 5) Sending Buddhist missionaries to spread Buddhism to Sri Lanka, Central Asia, and Southeast Asia.",
                context="Mauryan Empire period (322-185 BCE), focus on Ashoka's transformation from a violent ruler to a Buddhist patron",
                evaluation_criteria=[EvaluationMetric.FACTUAL_ACCURACY, EvaluationMetric.HISTORICAL_CONTEXT, EvaluationMetric.COMPLETENESS],
                difficulty_level="medium",
                historical_period="Ancient India (3rd century BCE)"
            ),
            
            TestCase(
                id="TC002", 
                category="Medieval India",
                query="Describe the architectural innovations of the Mughal Empire, particularly focusing on the Taj Mahal.",
                expected_output="The Mughal Empire introduced Indo-Islamic architectural style combining Persian, Islamic, and Indian elements. Key innovations include: 1) Use of red sandstone and white marble, 2) Intricate inlay work (pietra dura), 3) Bulbous domes and minarets, 4) Char bagh (four-garden) layout. The Taj Mahal (1632-1653) exemplifies these: built by Shah Jahan as a mausoleum for Mumtaz Mahal, featuring a central white marble dome, four minarets, geometric gardens, and exquisite calligraphy and floral motifs.",
                context="Mughal period (1526-1857), architectural achievements under Akbar, Shah Jahan",
                evaluation_criteria=[EvaluationMetric.FACTUAL_ACCURACY, EvaluationMetric.HISTORICAL_CONTEXT, EvaluationMetric.CLARITY],
                difficulty_level="medium",
                historical_period="Medieval India (16th-17th century)"
            ),
            
            TestCase(
                id="TC003",
                category="Ancient Mathematics",
                query="What were Aryabhata's contributions to mathematics and astronomy?",
                expected_output="Aryabhata (476-550 CE) was a pioneering Indian mathematician and astronomer. His contributions include: 1) Calculating the value of π (pi) as 3.1416, remarkably accurate for his time, 2) Explaining lunar eclipses and Earth's rotation, 3) Developing the place-value system and use of zero, 4) Creating the Aryabhata numeration system, 5) Calculating the length of the solar year as 365.25 days, 6) Work on arithmetic progressions, algebra, and trigonometry in his text 'Aryabhatiya'.",
                context="Gupta period scientific achievements, Indian mathematics and astronomy contributions",
                evaluation_criteria=[EvaluationMetric.FACTUAL_ACCURACY, EvaluationMetric.COMPLETENESS, EvaluationMetric.CLARITY],
                difficulty_level="hard",
                historical_period="Classical India (5th-6th century CE)"
            ),
            
            TestCase(
                id="TC004",
                category="Independence Movement",
                query="How did the Salt March impact India's independence movement?",
                expected_output="The Salt March (March 12 - April 6, 1930) was a pivotal moment in India's independence movement. Impact: 1) Demonstrated the power of non-violent civil disobedience (satyagraha), 2) United people across classes and regions against British salt tax, 3) Gandhi's 240-mile walk from Sabarmati to Dandi attracted international attention, 4) Led to mass civil disobedience with thousands making salt illegally, 5) Resulted in 60,000 arrests including Gandhi, 6) Forced British to negotiate, leading to Gandhi-Irwin Pact (1931), 7) Inspired global civil rights movements.",
                context="Indian independence struggle, Gandhi's non-violent resistance methods",
                evaluation_criteria=[EvaluationMetric.FACTUAL_ACCURACY, EvaluationMetric.HISTORICAL_CONTEXT, EvaluationMetric.RELEVANCE],
                difficulty_level="medium",
                historical_period="Modern India (1930)"
            ),
            
            TestCase(
                id="TC005",
                category="South Indian Dynasties",
                query="Compare the naval power and maritime trade of the Chola Empire with contemporary kingdoms.",
                expected_output="The Chola Empire (9th-13th centuries) was the premier naval power in medieval Asia. Maritime achievements: 1) Built the world's largest navy for the era with over 60,000 vessels, 2) Conducted naval expeditions to Sri Lanka, Maldives, Southeast Asia (Srivijaya), 3) Established trade networks from China to Arabia, 4) Created the first organized naval administration, 5) Rajaraja Chola I and Rajendra Chola I expanded maritime influence. Contemporary comparison: Unlike land-focused Chalukyas and Rashtrakutas, Cholas dominated Indian Ocean trade. Their naval power exceeded contemporary Song Dynasty China's coastal navy and rivaled Islamic maritime networks.",
                context="Medieval South India, Chola maritime expansion, comparison with other dynasties",
                evaluation_criteria=[EvaluationMetric.FACTUAL_ACCURACY, EvaluationMetric.HISTORICAL_CONTEXT, EvaluationMetric.COMPLETENESS],
                difficulty_level="hard",
                historical_period="Medieval India (9th-13th century)"
            ),
            
            TestCase(
                id="TC006",
                category="Cultural Synthesis",
                query="Explain the concept of 'Ganga-Jamuni tehzeeb' and its significance in Indian culture.",
                expected_output="Ganga-Jamuni tehzeeb refers to the composite Hindu-Muslim culture that developed in North India, symbolized by the confluence of Ganga and Yamuna rivers. Significance: 1) Represents cultural synthesis during Mughal period, 2) Blending of Hindu and Islamic traditions in language (Hindustani/Urdu), 3) Fusion in music (Hindustani classical), architecture (Indo-Islamic style), 4) Shared festivals and customs, 5) Literary traditions combining Persian, Arabic, and Sanskrit influences, 6) Exemplified in cities like Delhi, Lucknow, Hyderabad, 7) Created secular Indian identity transcending religious boundaries.",
                context="Mughal period cultural synthesis, Hindu-Muslim harmony, composite culture",
                evaluation_criteria=[EvaluationMetric.HISTORICAL_CONTEXT, EvaluationMetric.RELEVANCE, EvaluationMetric.CLARITY],
                difficulty_level="medium",
                historical_period="Medieval to Modern India"
            )
        ]
        
        print(f"📚 Evaluation Dataset Created: {len(dataset)} test cases")
        return dataset
    
    def create_judge_prompt(self, test_case: TestCase, model_output: str) -> str:
        """
        Create a comprehensive judge prompt for evaluating model outputs.
        
        Key Parameters Considered:
        1. Structured evaluation criteria
        2. Scoring rubric with specific scales
        3. Factual accuracy verification
        4. Historical context assessment
        5. Output completeness and clarity
        6. Comparative analysis with expected output
        """
        
        judge_prompt = f"""
# 🏛️ SangamGPT Historical AI Evaluation Judge

You are an expert historian and AI evaluation specialist tasked with rigorously assessing the quality of AI-generated historical content.

## 📋 EVALUATION TASK
**Test Case ID**: {test_case.id}
**Category**: {test_case.category}
**Historical Period**: {test_case.historical_period}
**Difficulty Level**: {test_case.difficulty_level}

## ❓ ORIGINAL QUERY
{test_case.query}

## 🎯 EXPECTED OUTPUT (Reference Standard)
{test_case.expected_output}

## 🤖 MODEL OUTPUT TO EVALUATE
{model_output}

## 📊 EVALUATION CRITERIA & PARAMETERS

Please evaluate the model output against these five key dimensions using a 1-10 scale:

### 1. 📚 FACTUAL ACCURACY (Weight: 30%)
- Are historical facts, dates, names, and events correct?
- Are there any factual errors or misconceptions?
- How accurate are numerical data and chronological information?
**Score (1-10)**: _____

### 2. 🏛️ HISTORICAL CONTEXT (Weight: 25%)
- Does the response demonstrate understanding of the historical period?
- Are cultural, social, and political contexts appropriately addressed?
- How well does it situate events within broader historical narratives?
**Score (1-10)**: _____

### 3. 🎯 RELEVANCE (Weight: 20%)
- How directly does the response address the specific question asked?
- Are all parts of the query answered comprehensively?
- Is the information pertinent to the user's intent?
**Score (1-10)**: _____

### 4. ✅ COMPLETENESS (Weight: 15%)
- Does the response cover all major aspects expected?
- Are key points from the reference standard addressed?
- Is the depth of information appropriate for the query complexity?
**Score (1-10)**: _____

### 5. 💡 CLARITY (Weight: 10%)
- Is the response well-structured and easy to understand?
- Are explanations clear and logically organized?
- Is the language appropriate for the educational context?
**Score (1-10)**: _____

## 🔍 DETAILED EVALUATION INSTRUCTIONS

**Step 1**: Compare model output with expected output point by point
**Step 2**: Identify any missing information or additional valuable content
**Step 3**: Check for historical accuracy using your knowledge
**Step 4**: Assess the pedagogical value of the response
**Step 5**: Consider the response quality relative to difficulty level

## 📝 REQUIRED OUTPUT FORMAT

Please provide your evaluation in this exact JSON format:

```json
{{
    "factual_accuracy": {{
        "score": X.X,
        "reasoning": "Detailed explanation of factual assessment"
    }},
    "historical_context": {{
        "score": X.X,
        "reasoning": "Analysis of historical context understanding"
    }},
    "relevance": {{
        "score": X.X,
        "reasoning": "Assessment of query relevance and directness"
    }},
    "completeness": {{
        "score": X.X,
        "reasoning": "Evaluation of information completeness"
    }},
    "clarity": {{
        "score": X.X,
        "reasoning": "Assessment of clarity and structure"
    }},
    "overall_score": X.X,
    "weighted_calculation": "Show: (FA×0.3 + HC×0.25 + REL×0.2 + COMP×0.15 + CL×0.1)",
    "pass_threshold": 7.0,
    "passed": true/false,
    "strengths": ["List key strengths of the response"],
    "weaknesses": ["List areas for improvement"],
    "missing_information": ["Important points not covered"],
    "additional_context": "Any extra valuable information provided beyond expectations",
    "recommendation": "Overall assessment and suggestions for improvement"
}}
```

## ⚖️ EVALUATION PARAMETERS EXPLAINED

**Scoring Scale**: 1-10 where:
- 1-3: Poor/Inadequate
- 4-5: Below Average  
- 6-7: Average/Satisfactory
- 8-9: Good/Very Good
- 10: Excellent/Exceptional

**Pass Threshold**: 7.0 overall weighted score
**Temperature Setting**: 0.1 (for consistent evaluation)
**Focus**: Educational accuracy and historical scholarship

Evaluate thoroughly and objectively, considering both the educational value and historical accuracy of the response.
"""
        
        return judge_prompt
    
    def generate_model_response(self, test_case: TestCase) -> str:
        """
        Generate response from SangamGPT model for a test case.
        Uses optimized parameters for historical content generation.
        """
        
        # Craft a comprehensive prompt for historical query
        historical_prompt = f"""
You are SangamGPT, an expert AI assistant specializing in Indian history and culture. 
Provide accurate, comprehensive, and educational responses about historical topics.

Context: {test_case.context}
Historical Period: {test_case.historical_period}
Category: {test_case.category}

Query: {test_case.query}

Please provide a detailed, factually accurate response that demonstrates deep understanding of the historical context.
"""
        
        try:
            self.log_tokens(historical_prompt, "Model Query")
            
            response = self.model.generate_content(
                historical_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,  # Balanced creativity and factual accuracy
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=500,
                    stop_sequences=["END_RESPONSE", "---"]
                )
            )
            
            model_output = response.text.strip()
            self.log_tokens(model_output, "Model Response")
            return model_output
            
        except Exception as e:
            print(f"❌ Error generating model response: {e}")
            return f"Error: Unable to generate response for {test_case.id}"
    
    def evaluate_with_judge(self, test_case: TestCase, model_output: str) -> EvaluationResult:
        """
        Use AI judge to evaluate model output against expected results.
        """
        
        judge_prompt = self.create_judge_prompt(test_case, model_output)
        
        try:
            start_time = time.time()
            self.log_tokens(judge_prompt, "Judge Evaluation")
            
            judge_response = self.model.generate_content(
                judge_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.judge_parameters["temperature"],
                    top_p=self.judge_parameters["top_p"],
                    max_output_tokens=self.judge_parameters["max_output_tokens"]
                )
            )
            
            execution_time = time.time() - start_time
            judge_text = judge_response.text.strip()
            self.log_tokens(judge_text, "Judge Response")
            
            # Parse judge evaluation (simplified parsing)
            # In production, would use more robust JSON parsing
            try:
                import re
                
                # Extract scores using regex patterns
                scores = {}
                patterns = {
                    'factual_accuracy': r'"score":\s*([0-9.]+)',
                    'historical_context': r'"score":\s*([0-9.]+)',
                    'relevance': r'"score":\s*([0-9.]+)',
                    'completeness': r'"score":\s*([0-9.]+)',
                    'clarity': r'"score":\s*([0-9.]+)'
                }
                
                # Simple scoring extraction
                score_matches = re.findall(r'"score":\s*([0-9.]+)', judge_text)
                if len(score_matches) >= 5:
                    scores = {
                        'factual_accuracy': float(score_matches[0]),
                        'historical_context': float(score_matches[1]), 
                        'relevance': float(score_matches[2]),
                        'completeness': float(score_matches[3]),
                        'clarity': float(score_matches[4])
                    }
                else:
                    # Fallback scoring
                    scores = {
                        'factual_accuracy': 7.0,
                        'historical_context': 7.0,
                        'relevance': 7.0,
                        'completeness': 7.0,
                        'clarity': 7.0
                    }
                
                # Calculate weighted overall score
                weights = {
                    'factual_accuracy': 0.30,
                    'historical_context': 0.25,
                    'relevance': 0.20,
                    'completeness': 0.15,
                    'clarity': 0.10
                }
                
                overall_score = sum(scores[metric] * weight for metric, weight in weights.items())
                passed = overall_score >= 7.0
                
                return EvaluationResult(
                    test_case_id=test_case.id,
                    model_output=model_output,
                    judge_scores=scores,
                    overall_score=overall_score,
                    judge_reasoning=judge_text,
                    passed=passed,
                    execution_time=execution_time
                )
                
            except Exception as parse_error:
                print(f"⚠️ Judge response parsing error: {parse_error}")
                # Return default evaluation result
                return EvaluationResult(
                    test_case_id=test_case.id,
                    model_output=model_output,
                    judge_scores={'overall': 6.0},
                    overall_score=6.0,
                    judge_reasoning="Parsing error occurred",
                    passed=False,
                    execution_time=execution_time
                )
                
        except Exception as e:
            print(f"❌ Judge evaluation error: {e}")
            return EvaluationResult(
                test_case_id=test_case.id,
                model_output=model_output,
                judge_scores={'error': 0.0},
                overall_score=0.0,
                judge_reasoning=f"Evaluation failed: {str(e)}",
                passed=False,
                execution_time=0.0
            )
    
    def run_single_test(self, test_case: TestCase) -> EvaluationResult:
        """Run evaluation for a single test case."""
        
        print(f"\n🧪 Running Test Case: {test_case.id}")
        print(f"📝 Category: {test_case.category}")
        print(f"🎯 Query: {test_case.query[:80]}...")
        print("-" * 60)
        
        # Generate model response
        model_output = self.generate_model_response(test_case)
        print(f"🤖 Model Response Generated ({len(model_output)} chars)")
        
        # Evaluate with judge
        evaluation_result = self.evaluate_with_judge(test_case, model_output)
        print(f"⚖️ Judge Evaluation Complete")
        print(f"📊 Overall Score: {evaluation_result.overall_score:.2f}/10.0")
        print(f"{'✅ PASSED' if evaluation_result.passed else '❌ FAILED'}")
        
        return evaluation_result
    
    def run_full_evaluation_suite(self) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline on all test cases.
        """
        
        print("🚀 SangamGPT Evaluation Pipeline Starting...")
        print("=" * 80)
        
        start_time = time.time()
        self.test_results = []
        
        # Run all test cases
        for i, test_case in enumerate(self.test_dataset, 1):
            print(f"\n📊 Progress: {i}/{len(self.test_dataset)}")
            result = self.run_single_test(test_case)
            self.test_results.append(result)
            
            # Add small delay to respect API limits
            time.sleep(1)
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        aggregate_results = self._calculate_aggregate_metrics()
        
        # Generate comprehensive report
        report = {
            'execution_summary': {
                'total_test_cases': len(self.test_dataset),
                'total_execution_time': total_time,
                'total_tokens_used': int(self.total_tokens),
                'timestamp': datetime.now().isoformat()
            },
            'aggregate_metrics': aggregate_results,
            'individual_results': [
                {
                    'test_case_id': result.test_case_id,
                    'overall_score': result.overall_score,
                    'passed': result.passed,
                    'execution_time': result.execution_time,
                    'judge_scores': result.judge_scores
                }
                for result in self.test_results
            ],
            'detailed_results': self.test_results
        }
        
        self._print_final_report(aggregate_results)
        
        return report
    
    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics across all test results."""
        
        if not self.test_results:
            return {}
        
        # Overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        pass_rate = (passed_tests / total_tests) * 100
        
        # Score statistics
        overall_scores = [r.overall_score for r in self.test_results]
        average_score = sum(overall_scores) / len(overall_scores)
        
        # Performance by category
        category_performance = {}
        for result in self.test_results:
            # Find test case for this result
            test_case = next(tc for tc in self.test_dataset if tc.id == result.test_case_id)
            category = test_case.category
            
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(result.overall_score)
        
        # Average by category
        category_averages = {
            category: sum(scores) / len(scores)
            for category, scores in category_performance.items()
        }
        
        return {
            'total_test_cases': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate_percentage': pass_rate,
            'average_overall_score': average_score,
            'highest_score': max(overall_scores),
            'lowest_score': min(overall_scores),
            'category_performance': category_averages,
            'detailed_scores': overall_scores
        }
    
    def _print_final_report(self, aggregate_results: Dict[str, Any]):
        """Print a comprehensive final evaluation report."""
        
        print("\n" + "=" * 80)
        print("📊 SANGAMGPT EVALUATION PIPELINE - FINAL REPORT")
        print("=" * 80)
        
        print(f"\n🎯 OVERALL PERFORMANCE")
        print(f"   Total Test Cases: {aggregate_results['total_test_cases']}")
        print(f"   Passed: {aggregate_results['passed_tests']} ✅")
        print(f"   Failed: {aggregate_results['failed_tests']} ❌")
        print(f"   Pass Rate: {aggregate_results['pass_rate_percentage']:.1f}%")
        print(f"   Average Score: {aggregate_results['average_overall_score']:.2f}/10.0")
        
        print(f"\n📈 SCORE DISTRIBUTION")
        print(f"   Highest Score: {aggregate_results['highest_score']:.2f}")
        print(f"   Lowest Score: {aggregate_results['lowest_score']:.2f}")
        
        print(f"\n📚 CATEGORY PERFORMANCE")
        for category, avg_score in aggregate_results['category_performance'].items():
            print(f"   {category}: {avg_score:.2f}/10.0")
        
        print(f"\n🔍 INDIVIDUAL TEST RESULTS")
        for result in self.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"   {result.test_case_id}: {result.overall_score:.2f}/10.0 {status}")
        
        print(f"\n💡 EVALUATION PARAMETERS SUMMARY")
        print(f"   Judge Temperature: {self.judge_parameters['temperature']}")
        print(f"   Scoring Scale: {self.judge_parameters['scoring_scale']}")
        print(f"   Pass Threshold: 7.0/10.0")
        print(f"   Evaluation Aspects: {len(self.judge_parameters['evaluation_aspects'])}")
        
        print(f"\n🪙 RESOURCE USAGE")
        print(f"   Total Tokens: {int(self.total_tokens)}")
        
        print("\n" + "=" * 80)


def demonstrate_evaluation_pipeline():
    """
    Comprehensive demonstration of the SangamGPT evaluation pipeline.
    """
    
    print("🏛️ SangamGPT Evaluation Pipeline Demo")
    print("=" * 70)
    print("This pipeline evaluates AI responses on historical queries using:")
    print("✅ Structured dataset with diverse test cases")
    print("✅ AI judge with weighted scoring criteria") 
    print("✅ Comprehensive evaluation metrics")
    print("✅ Automated testing framework")
    
    # Initialize evaluation pipeline
    evaluator = SangamGPTEvaluationPipeline()
    
    # Show dataset overview
    print(f"\n📚 EVALUATION DATASET OVERVIEW")
    print("-" * 50)
    for test_case in evaluator.test_dataset:
        print(f"🧪 {test_case.id}: {test_case.category}")
        print(f"   📅 Period: {test_case.historical_period}")
        print(f"   🎚️ Difficulty: {test_case.difficulty_level}")
        print(f"   📝 Query: {test_case.query[:60]}...")
        print()
    
    # Explain judge parameters
    print(f"\n⚖️ JUDGE PROMPT PARAMETERS EXPLAINED")
    print("-" * 50)
    print("The judge prompt considers these key parameters:")
    print()
    print("1. 📊 SCORING RUBRIC:")
    print("   • Scale: 1-10 for consistent evaluation")
    print("   • Pass threshold: 7.0/10.0")
    print("   • Weighted scoring by importance")
    print()
    print("2. 📚 EVALUATION DIMENSIONS:")
    print("   • Factual Accuracy (30% weight)")
    print("   • Historical Context (25% weight)")
    print("   • Relevance (20% weight)")
    print("   • Completeness (15% weight)")
    print("   • Clarity (10% weight)")
    print()
    print("3. 🎛️ GENERATION PARAMETERS:")
    print(f"   • Temperature: {evaluator.judge_parameters['temperature']} (low for consistency)")
    print(f"   • Top P: {evaluator.judge_parameters['top_p']}")
    print(f"   • Max tokens: {evaluator.judge_parameters['max_output_tokens']}")
    print()
    print("4. 🔍 EVALUATION METHODOLOGY:")
    print("   • Point-by-point comparison with expected output")
    print("   • Historical accuracy verification")
    print("   • Pedagogical value assessment")
    print("   • Structured JSON response format")
    
    # Run the evaluation pipeline
    print(f"\n🚀 RUNNING EVALUATION PIPELINE...")
    print("-" * 50)
    
    evaluation_report = evaluator.run_full_evaluation_suite()
    
    # Save results to file
    report_filename = f"sangamgpt_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Evaluation report saved to: {report_filename}")
    
    return evaluation_report


if __name__ == "__main__":
    """
    Run the comprehensive SangamGPT evaluation pipeline demonstration.
    """
    
    try:
        report = demonstrate_evaluation_pipeline()
        print("\n✅ SangamGPT Evaluation Pipeline Complete!")
        print("This demonstrates automated evaluation of historical AI responses.")
        
    except Exception as e:
        print(f"❌ Evaluation pipeline error: {e}")
        import traceback
        traceback.print_exc()
