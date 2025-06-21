#!/usr/bin/env python3
"""
Opus Supervisor example

This example demonstrates using an expensive model (Opus) as a supervisor
for a cheaper model (Sonnet) to maximize quality while minimizing cost.
"""

import subprocess
import json
from pathlib import Path
from cadence import CheckpointSupervisor


class OpusSupervisor(CheckpointSupervisor):
    """
    Supervisor that uses Opus for analysis and Sonnet for execution
    
    This pattern allows using expensive, high-capability models for
    strategic oversight while cheaper models do the tactical work.
    """
    
    def __init__(self, supervisor_model: str = "claude-3-opus-20240229", **kwargs):
        # Default to Sonnet for execution
        kwargs.setdefault("model", "claude-3-5-sonnet-20241022")
        super().__init__(**kwargs)
        self.supervisor_model = supervisor_model
        
    def analyze_checkpoint(self, result, checkpoint_num):
        """Use Opus for deep analysis of checkpoint results"""
        
        # Prepare analysis prompt for Opus
        output_summary = "\n".join(result.output_lines[-50:])  # Last 50 lines
        
        analysis_prompt = f"""You are a senior software architect reviewing junior developer work.

Checkpoint {checkpoint_num} has completed with {result.turns_used} turns used.

Output summary (last 50 lines):
{output_summary}

Errors encountered: {result.errors or 'None'}

Analyze this checkpoint and provide:
1. Is the agent on track? (yes/no)
2. Is the task complete? (yes/no)  
3. What issues were detected?
4. Specific guidance for the next checkpoint

Format your response as JSON:
{{
    "on_track": true/false,
    "task_complete": true/false,
    "issues": ["issue1", "issue2"],
    "guidance": "specific guidance text",
    "confidence": 0.0-1.0
}}"""

        try:
            # Use Opus for analysis
            cmd = [
                "claude", "-p", analysis_prompt,
                "--model", self.supervisor_model,
                "--max-turns", "1"
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                # Extract JSON from response
                response = process.stdout
                
                # Find JSON in response (handle markdown code blocks)
                import re
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                
                if json_match:
                    analysis_data = json.loads(json_match.group())
                    
                    from cadence.supervisor import SupervisorAnalysis
                    return SupervisorAnalysis(
                        on_track=analysis_data.get("on_track", True),
                        task_complete=analysis_data.get("task_complete", False),
                        issues_detected=analysis_data.get("issues", []),
                        guidance=analysis_data.get("guidance", "Continue with the task."),
                        confidence=analysis_data.get("confidence", 0.8)
                    )
                    
        except Exception as e:
            print(f"⚠️  Opus analysis failed: {e}")
            
        # Fall back to base analysis
        return super().analyze_checkpoint(result, checkpoint_num)
        
    def _generate_enhanced_guidance(self, checkpoint_num: int, 
                                   initial_prompt: str,
                                   checkpoint_history: list) -> str:
        """Generate enhanced guidance using Opus's strategic capabilities"""
        
        history_summary = []
        for i, checkpoint in enumerate(checkpoint_history):
            history_summary.append(
                f"Checkpoint {i+1}: {checkpoint['result'].turns_used} turns, "
                f"{'succeeded' if checkpoint['result'].success else 'failed'}"
            )
            
        guidance_prompt = f"""As a senior architect, provide strategic guidance.

Original task: {initial_prompt}

Progress so far:
{chr(10).join(history_summary)}

The agent has {self.max_checkpoints - checkpoint_num} checkpoints remaining.

Provide specific, actionable guidance to keep the agent focused and efficient.
Focus on architectural decisions and avoiding common pitfalls."""

        try:
            cmd = [
                "claude", "-p", guidance_prompt,
                "--model", self.supervisor_model,
                "--max-turns", "1"
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                return process.stdout.strip()
                
        except Exception:
            pass
            
        return "Continue with the implementation following best practices."


def main():
    """Demonstrate Opus supervision of Sonnet execution"""
    
    print("🎭 Opus Supervisor Pattern Demo")
    print("📊 Using Opus for analysis, Sonnet for execution")
    
    supervisor = OpusSupervisor(
        supervisor_model="claude-3-opus-20240229",  # Expensive, smart
        model="claude-3-5-sonnet-20241022",         # Cheaper, capable
        checkpoint_turns=15,
        max_checkpoints=3,
        verbose=True
    )
    
    task_prompt = """
    Implement a robust retry mechanism for API calls with:
    1. Exponential backoff
    2. Jitter to prevent thundering herd
    3. Circuit breaker pattern
    4. Comprehensive logging
    5. Full test coverage
    
    The implementation should be production-ready and follow best practices.
    """
    
    success, cost = supervisor.run_supervised_task(task_prompt)
    
    # Calculate cost breakdown (rough estimates)
    # Assume 3 checkpoints for this example
    opus_cost = 3 * 0.10  # Opus analysis cost
    sonnet_cost = cost - opus_cost
    
    print(f"\n💰 Cost Analysis:")
    print(f"   - Sonnet execution: ${sonnet_cost:.4f}")
    print(f"   - Opus supervision: ${opus_cost:.4f}")
    print(f"   - Total cost: ${cost:.4f}")
    print(f"   - Savings vs pure Opus: ${(cost * 5):.4f} → ${cost:.4f} (80% reduction)")


if __name__ == "__main__":
    # Note: This example requires having the claude CLI configured
    # with access to both Opus and Sonnet models
    main()