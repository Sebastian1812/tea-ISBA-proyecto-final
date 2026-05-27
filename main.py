"""
Main orchestrator for TEA Clinical Agent System
"""
import sys
from pathlib import Path
from mcp_server import MCPServer
from graph_agent import ClinicalAgent
import json
from datetime import datetime

class ClinicalOrchestrator:
    """Main orchestrator for the clinical agent system"""
    
    def __init__(self):
        self.mcp_server = MCPServer()
        self.agent = ClinicalAgent(self.mcp_server)
        self.analysis_history = {}
        
    def run_clinical_analysis(self, patient_id: str) -> dict:
        """
        Run complete clinical analysis for a patient
        Complexity: O(n) where n is number of sessions
        """
        print(f"\n{'='*60}")
        print(f"🏥 Starting Clinical Analysis for Patient: {patient_id}")
        print(f"{'='*60}\n")
        
        # Run agent
        result = self.agent.run(patient_id, thread_id=patient_id)
        
        # Store in history
        self.analysis_history[patient_id] = {
            'timestamp': datetime.now().isoformat(),
            'result': result
        }
        
        # Display results
        self._display_results(result)
        
        return result
    
    def _display_results(self, result: dict):
        """Display analysis results in a readable format"""
        
        if result.get('report_generated') and result.get('memory_context'):
            # Find the latest report
            for key, value in result['memory_context'].items():
                if key.startswith('report_'):
                    print("\n📋 CLINICAL REPORT")
                    print("-" * 40)
                    print(value)
                    print("-" * 40)
                    break
        
        if result.get('diagnostic_suggestion', {}).get('requires_review'):
            print("\n⚠️ DIAGNOSTIC REVIEW SUGGESTED")
            print("-" * 40)
            suggestions = result['diagnostic_suggestion'].get('suggestions', [])
            for suggestion in suggestions:
                print(f"• {suggestion['message']}")
                print(f"  Recommendation: {suggestion['recommendation']}\n")
            print("-" * 40)
        
        if not result.get('should_report') and not result.get('should_suggest_diagnostic'):
            total_sessions = result.get('metrics', {}).get('total_sessions', 0)
            print(f"\n⚠️ Insufficient data for report generation")
            print(f"   Current sessions: {total_sessions}")
            print(f"   Minimum required: 3")
            print(f"   Please collect more session data.\n")
        
        # Display metrics summary
        metrics = result.get('metrics', {})
        if metrics:
            print("\n📊 METRICS SUMMARY")
            print("-" * 40)
            print(f"Total Sessions: {metrics.get('total_sessions', 0)}")
            print(f"Average Progress Score: {metrics.get('average_progress_score', 0):.2f}")
            print(f"Overall Advancement: {metrics.get('overall_advancement_percentage', 0):.1f}%")
            print(f"Progress Trend: {metrics.get('progress_trend', 'N/A')}")
            print("-" * 40)
    
    def get_available_patients(self) -> list:
        """Get list of available patients"""
        patients = self.mcp_server.get_resource("patients://list")
        return patients
    
    def demonstrate_tools(self):
        """Demonstrate MCP tools functionality"""
        print("\n🔧 DEMONSTRATING MCP TOOLS")
        print("="*60)
        
        # Test outlier detection
        sample_scores = [65, 68, 70, 72, 68, 95, 70, 69, 71, 200]
        print("\n1. Testing Statistical Outlier Detection")
        outliers = self.mcp_server.filter_statistical_outliers(sample_scores, threshold=2.0)
        print(f"   Original scores: {sample_scores}")
        print(f"   Outliers found: {outliers['outliers_count']}")
        if outliers['outliers']:
            print(f"   Outlier values: {[o['value'] for o in outliers['outliers']]}")
        
        # Test intervention simulation
        print("\n2. Testing Intervention Simulation")
        simulation = self.mcp_server.simulate_intervention_outcome("ABA", 7.5, 24)
        print(f"   Intervention: {simulation['intervention_type']}")
        print(f"   Baseline severity: {simulation['baseline_severity']}")
        print(f"   Expected improvement: {simulation['improvement_percentage']}%")
        print(f"   Final severity: {simulation['final_severity']}")

def main():
    """Main entry point"""
    print("🚀 Starting TEA Clinical Agent System")
    print("="*60)
    
    orchestrator = ClinicalOrchestrator()
    
    # Demonstrate tools
    orchestrator.demonstrate_tools()
    
    # Show available patients
    patients = orchestrator.get_available_patients()
    print("\n📋 AVAILABLE PATIENTS")
    print("-"*40)
    for patient in patients:
        print(f"ID: {patient['id']} | Name: {patient['name']} | Age: {patient['age']}")
    print("-"*40)
    
    # Run analysis for each patient
    for patient in patients:
        orchestrator.run_clinical_analysis(patient['id'])
        print("\n" + "="*60 + "\n")
    
    print("✅ Clinical analysis complete!")

if __name__ == "__main__":
    main()