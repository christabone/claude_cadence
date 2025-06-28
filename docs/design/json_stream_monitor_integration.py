"""
JSONStreamMonitor Integration Examples
This file demonstrates how JSONStreamMonitor will integrate with existing components
"""

from typing import Dict, List, Optional, Any
import asyncio
from json_stream_monitor import JSONStreamMonitor, JSONEvent, MessageType


class OrchestratorIntegration:
    """Example integration with SupervisorOrchestrator"""

    async def run_claude_with_realtime_output_new(
        self,
        cmd: List[str],
        cwd: str,
        process_name: str
    ) -> tuple[int, List[str]]:
        """Enhanced version using JSONStreamMonitor"""

        # Set up environment
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        # Choose color based on process name
        color = self._get_process_color(process_name)

        print(f"\n{color}{process_name} working...{Colors.RESET}")
        print(f"{Colors.WHITE}{'-' * 50}{Colors.RESET}")

        # Create JSONStreamMonitor
        monitor = JSONStreamMonitor(
            source=process_name.lower(),
            logger=self.logger
        )

        # Collect all output for return
        all_output = []

        # Register event handlers
        async def on_json_complete(event: JSONEvent):
            """Handle complete JSON objects"""
            all_output.append(event.raw_json)

            if event.message_type == MessageType.SYSTEM:
                subtype = event.json_data.get('subtype', '')
                if subtype == 'init':
                    model = event.json_data.get('model', 'unknown')
                    print(f"{color}[{process_name}]{Colors.RESET} {Colors.BOLD}Model: {model}{Colors.RESET}")

            elif event.message_type == MessageType.ASSISTANT:
                message = event.json_data.get('message', {})
                content = message.get('content', [])

                for item in content:
                    if item.get('type') == 'text':
                        text = item.get('text', '').strip()
                        if text:
                            print(f"{color}[{process_name}]{Colors.RESET} {text}")
                    elif item.get('type') == 'tool_use':
                        tool_name = item.get('name', 'unknown')
                        tool_input = item.get('input', {})
                        command = tool_input.get('command', '')
                        print(f"{color}[{process_name}]{Colors.RESET} 🛠️  {Colors.YELLOW}{tool_name}{Colors.RESET}: {command}")

            elif event.message_type == MessageType.USER:
                # Tool results
                message = event.json_data.get('message', {})
                content = message.get('content', [])
                for item in content:
                    if item.get('type') == 'tool_result':
                        result_content = str(item.get('content', ''))
                        if len(result_content) > 200:
                            result_content = result_content[:200] + "..."
                        print(f"{color}[{process_name}]{Colors.RESET} 📋 {Colors.CYAN}Result:{Colors.RESET} {result_content}")

            elif event.message_type == MessageType.RESULT:
                # Final result
                duration = event.json_data.get('duration_ms', 0)
                print(f"{color}[{process_name}]{Colors.RESET} ✅ {Colors.BOLD_GREEN}Completed{Colors.RESET} in {duration}ms")

        async def on_parse_error(error):
            """Handle parsing errors"""
            self.logger.warning(f"Parse error in {process_name}: {error.error}")
            # Still append the partial data
            all_output.append(error.partial_data)

        async def on_non_json_line(line: str):
            """Handle non-JSON lines"""
            all_output.append(line)
            print(f"{color}[{process_name}]{Colors.RESET} {line}")

        # Register handlers
        monitor.on('json_complete', on_json_complete)
        monitor.on('parse_error', on_parse_error)

        # Start subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env
        )

        # Process the stream with custom line handler for non-JSON
        async def process_with_fallback():
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line_str = line.decode('utf-8', errors='replace').strip()
                if not line_str:
                    continue

                # Try to process as JSON first
                original_buffer = monitor._buffer
                await monitor.process_line(line_str)

                # If buffer didn't change and we're not in JSON, it's plain text
                if monitor._buffer == original_buffer and monitor._state == ParseState.SEEKING_JSON:
                    await on_non_json_line(line_str)

        await process_with_fallback()

        # Wait for process to complete
        await process.wait()

        print(f"{Colors.WHITE}{'-' * 50}{Colors.RESET}")
        print(f"{color}{process_name} completed with return code {process.returncode}{Colors.RESET}")

        return process.returncode, all_output


class SupervisorDecisionExtractor:
    """Extract supervisor decisions using JSONStreamMonitor"""

    def __init__(self, logger):
        self.logger = logger
        self.monitor = JSONStreamMonitor(source="supervisor", logger=logger)
        self.decision_candidates = []
        self.all_output = []

        # Register handlers
        self.monitor.on('json_complete', self._on_json_complete)
        self.monitor.on('parse_error', self._on_parse_error)

    async def _on_json_complete(self, event: JSONEvent):
        """Collect assistant messages that might contain decisions"""
        self.all_output.append(event.raw_json)

        if event.message_type == MessageType.ASSISTANT:
            content = event.json_data.get('message', {}).get('content', [])

            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text = item.get('text', '')

                    # Look for JSON with "action" field
                    if '"action"' in text:
                        # Extract JSON from text
                        import re
                        json_pattern = re.compile(r'\{[^{}]*"action"[^{}]*\}', re.DOTALL)
                        matches = json_pattern.findall(text)

                        for match in matches:
                            try:
                                decision_data = json.loads(match)
                                if 'action' in decision_data:
                                    self.decision_candidates.append(decision_data)
                                    self.logger.debug(f"Found decision candidate: {decision_data.get('action')}")
                            except json.JSONDecodeError:
                                continue

    async def _on_parse_error(self, error):
        """Handle parse errors"""
        self.all_output.append(error.partial_data)
        self.logger.warning(f"Parse error while extracting decision: {error.error}")

    async def process_output(self, output_lines: List[str]):
        """Process supervisor output lines"""
        for line in output_lines:
            await self.monitor.process_line(line)

    def get_final_decision(self) -> Optional[Dict[str, Any]]:
        """Get the last valid decision found"""
        if not self.decision_candidates:
            return None
        return self.decision_candidates[-1]

    def get_all_output(self) -> List[str]:
        """Get all collected output"""
        return self.all_output


class CodeReviewTriggerDetector:
    """Detect code review triggers in agent output"""

    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self.monitor = JSONStreamMonitor(source="agent", logger=logger)

        # Trigger patterns from config
        self.completion_signals = [
            "ALL TASKS COMPLETE",
            "WORK COMPLETE",
            "Implementation complete"
        ]

        # State tracking
        self.trigger_detected = False
        self.trigger_event = None

        # Register handler
        self.monitor.on('json_complete', self._check_for_triggers)

    async def _check_for_triggers(self, event: JSONEvent):
        """Check if the JSON contains code review triggers"""
        if event.message_type == MessageType.ASSISTANT:
            content = event.json_data.get('message', {}).get('content', [])

            for item in content:
                if item.get('type') == 'text':
                    text = item.get('text', '').upper()

                    # Check for completion signals
                    for signal in self.completion_signals:
                        if signal.upper() in text:
                            self.logger.info(f"Code review trigger detected: {signal}")
                            self.trigger_detected = True
                            self.trigger_event = event
                            break

    async def process_stream(self, stream):
        """Process a stream looking for triggers"""
        await self.monitor.process_stream(stream)

    def should_trigger_review(self) -> bool:
        """Check if code review should be triggered"""
        return self.trigger_detected

    def get_trigger_context(self) -> Optional[JSONEvent]:
        """Get the event that triggered the review"""
        return self.trigger_event


# Example usage in orchestrator's run_supervisor_analysis method
async def enhanced_supervisor_json_extraction(self, all_output: List[str]) -> Dict[str, Any]:
    """Enhanced JSON extraction using JSONStreamMonitor"""

    extractor = SupervisorDecisionExtractor(self.logger)

    # Process all output lines
    await extractor.process_output(all_output)

    # Get the final decision
    decision = extractor.get_final_decision()

    if not decision:
        raise ValueError("No decision JSON found in supervisor output")

    # Validate required fields
    required_fields = self._get_required_fields(decision.get('action'))
    missing_fields = [field for field in required_fields if field not in decision]

    if missing_fields:
        raise ValueError(f"Decision JSON missing required fields: {missing_fields}")

    return decision
