# Backup of the original run_supervisor_analysis method before refactoring
# This file can be deleted after successful refactoring and testing

def run_supervisor_analysis(self, session_id: str, use_continue: bool, iteration: int) -> SupervisorDecision:
    """Run supervisor in its directory to analyze state"""
    original_dir = os.getcwd()

    try:
        # Change to supervisor directory
        os.chdir(self.supervisor_dir)

        json_retry_count = 0
        max_json_retries = self.config.get("retry_behavior", {}).get("max_json_retries", 3)

        while json_retry_count < max_json_retries:
            # Build supervisor command using Claude CLI for MCP access
            supervisor_script = self.project_root / "cadence" / "supervisor_cli.py"
            config_file = self.project_root / "config.yaml"

            # Get code review config
            zen_config = self.config.get("zen_integration", {})
            code_review_frequency = zen_config.get("code_review_frequency", "task")

            # Check if we have previous agent results (only after first iteration)
            previous_agent_result = None
            if iteration > 1:
                agent_results_file = self.supervisor_dir / self.cadence_config.file_patterns.agent_result_file.format(session_id=session_id)
                logger.debug(f"Checking for agent results file: {agent_results_file}")
                if agent_results_file.exists():
                    logger.debug(f"Agent results file exists, loading...")
                    try:
                        with open(agent_results_file, 'r') as f:
                            previous_agent_result = json.load(f)
                        logger.debug(f"Successfully loaded agent results: task_id={previous_agent_result.get('task_id', 'unknown')}")
                    except Exception as e:
                        logger.warning(f"Failed to load previous agent results: {e}")
                else:
                    logger.debug(f"No agent results file found at {agent_results_file}")
            else:
                logger.debug(f"First iteration - not checking for agent results")

            # Log agent result status
            if previous_agent_result is not None:
                logger.info(f"Previous agent result: Found (task_id={previous_agent_result.get('task_id', 'unknown')})")
            else:
                logger.info("Previous agent result: None")

            # Build prompt using YAML templates
            context = {
                "project_path": str(self.project_root),  # Unified project path
                "session_id": session_id,
                "iteration": iteration,
                "is_first_iteration": iteration == 1,
                "has_previous_agent_result": previous_agent_result is not None,
                "agent_success": previous_agent_result.get("success", False) if previous_agent_result else False,
                "agent_completed_normally": previous_agent_result.get("completed_normally", False) if previous_agent_result else False,
                "agent_todos": previous_agent_result.get("todos", []) if previous_agent_result else [],
                "agent_task_id": previous_agent_result.get("task_id", "") if previous_agent_result else "",
                "max_turns": self.config.get("execution", {}).get("max_agent_turns", 120)  # Agent's turn limit for supervisor context
            }

            # Get base prompt
            logger.debug(f"Getting supervisor prompt template with context: has_previous_agent_result={context['has_previous_agent_result']}")
            base_prompt = self.prompt_loader.get_template("supervisor_prompts.orchestrator_taskmaster.base_prompt")
            base_prompt = self.prompt_loader.format_template(base_prompt, context)

            # Log a preview of the TASK section to verify correct template
            if "Process the agent's completed work" in base_prompt:
                logger.debug("Supervisor prompt includes agent work processing (iteration 2+)")
            elif "Analyze the current task state and decide what the agent should work on first" in base_prompt:
                logger.debug("Supervisor prompt is for first iteration (no agent work)")
            else:
                logger.warning("Supervisor prompt TASK section unclear - check template rendering")

            # Get code review section based on config
            # When code_review_frequency is "task", we should ALSO include project review
            # instructions so the supervisor knows to run final review before completion
            if code_review_frequency == "task":
                # Include BOTH task-level and project-level review instructions
                task_review_key = "supervisor_prompts.orchestrator_taskmaster.code_review_sections.task"
                task_review_section = self.prompt_loader.get_template(task_review_key)
                task_review_section = self.prompt_loader.format_template(task_review_section, context)

                project_review_key = "supervisor_prompts.orchestrator_taskmaster.code_review_sections.project"
                project_review_section = self.prompt_loader.get_template(project_review_key)
                project_review_section = self.prompt_loader.format_template(project_review_section, context)

                code_review_section = task_review_section + project_review_section
            else:
                # For "project" or "none" frequency, use only the specified section
                code_review_key = f"supervisor_prompts.orchestrator_taskmaster.code_review_sections.{code_review_frequency}"
                code_review_section = self.prompt_loader.get_template(code_review_key)
                code_review_section = self.prompt_loader.format_template(code_review_section, context)

            # Get zen guidance
            zen_guidance = self.prompt_loader.get_template("supervisor_prompts.orchestrator_taskmaster.zen_guidance")

            # Get output format
            output_format = self.prompt_loader.get_template("supervisor_prompts.orchestrator_taskmaster.output_format")
            output_format = self.prompt_loader.format_template(output_format, context)

            # If this is a retry, use a shorter prompt to avoid token limits
            if json_retry_count > 0:
                # Create a minimal retry prompt to avoid token limits
                minimal_retry_prompt = f"""You are the Task Supervisor. Project root: {context['project_path']}

CRITICAL: Your previous output had invalid JSON formatting.
Please analyze the current task state and output ONLY a valid JSON object.

REQUIRED OUTPUT FORMAT:
For "execute" action:
{{
    "action": "execute",
    "task_id": "X",
    "task_title": "Title",
    "subtasks": [{{"id": "X.Y", "title": "...", "description": "..."}}],
    "project_root": "{context['project_path']}",
    "guidance": "Brief guidance",
    "session_id": "{context['session_id']}",
    "reason": "Why this action"
}}

For "skip" or "complete" actions:
{{
    "action": "skip|complete",
    "session_id": "{context['session_id']}",
    "reason": "Why this action"
}}

DO NOT include any explanatory text. Start with {{ and end with }}.
Retry attempt {json_retry_count + 1} of {max_json_retries}."""
                supervisor_prompt = minimal_retry_prompt
            else:
                # Combine all sections normally
                supervisor_prompt = f"{base_prompt}{code_review_section}{zen_guidance}{output_format}"

            # Save supervisor prompt to file for debugging
            prompt_debug_file = self.supervisor_dir / f"supervisor_prompt_{session_id}.txt"
            try:
                with open(prompt_debug_file, 'w') as f:
                    f.write(supervisor_prompt)
                logger.debug(f"Saved supervisor prompt to {prompt_debug_file}")
                logger.debug(f"Supervisor prompt length: {len(supervisor_prompt)} characters")
            except Exception as e:
                logger.warning(f"Failed to save supervisor prompt: {e}")

            # Build allowed tools from config
            basic_tools = self.config.get("supervisor", {}).get("tools", [
                "bash", "read", "write", "edit", "grep", "glob", "search", "WebFetch"
            ])
            mcp_servers = self.config.get("integrations", {}).get("mcp", {}).get("supervisor_servers", [
                "taskmaster-ai", "zen", "serena", "Context7"
            ])
            # Add mcp__ prefix and * suffix to each MCP server
            mcp_tools = [f"mcp__{server}__*" for server in mcp_servers]
            all_tools = basic_tools + mcp_tools

            # Get supervisor model from config
            supervisor_config = self.config.get("supervisor")
            if not supervisor_config:
                raise ValueError("Supervisor configuration not found in config")

            supervisor_model = supervisor_config.get("model")
            if not supervisor_model:
                raise ValueError("Supervisor model not specified in config")


            cmd = [
                "claude",
                "-p", supervisor_prompt,
                "--model", supervisor_model,
                "--allowedTools", ",".join(all_tools),
                "--max-turns", "80",
                "--output-format", "stream-json",
                "--verbose",
                "--dangerously-skip-permissions"  # Skip permission prompts
            ]

            # Check config for supervisor continuation setting
            supervisor_use_continue = supervisor_config.get("use_continue", True)  # Default to True for backward compatibility

            # Add --continue flag based on config and conditions
            if supervisor_use_continue and (use_continue or json_retry_count > 0):
                cmd.append("--continue")
                if json_retry_count > 0:
                    logger.warning(f"Retrying supervisor with --continue due to JSON parsing error (attempt {json_retry_count + 1})")
                else:
                    logger.debug("Running supervisor with --continue flag")
            else:
                if not supervisor_use_continue:
                    logger.debug("Running supervisor without --continue flag (disabled in config)")
                else:
                    logger.debug("Running supervisor (first run)")


            # Run supervisor
            logger.debug(f"Command: {' '.join(cmd)}")
            logger.info("-" * 60)
            if json_retry_count > 0:
                logger.info(f"SUPERVISOR STARTING... (JSON RETRY {json_retry_count + 1}/{max_json_retries})")
            else:
                logger.info("SUPERVISOR STARTING...")
            logger.info("-" * 60)

            # Run supervisor with real-time output
            supervisor_start_time = time.time()
            supervisor_end_time = None
            supervisor_duration = None
            try:
                logger.debug(f"Starting supervisor subprocess with {len(cmd)} args")
                # Run supervisor async for real-time output
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    returncode, all_output = loop.run_until_complete(
                        self.run_claude_with_realtime_output(cmd, self.supervisor_dir, "SUPERVISOR")
                    )
                    supervisor_end_time = time.time()
                    supervisor_duration = (supervisor_end_time - supervisor_start_time) * 1000
                    logger.info(f"Supervisor process completed in {supervisor_duration:.0f}ms with return code {returncode}")
                finally:
                    loop.close()

                if returncode != 0:
                    logger.error(f"Supervisor failed with code {returncode}")
                    logger.error(f"Output lines collected: {len(all_output)}")
                    if all_output:
                        logger.error(f"Last few output lines: {all_output[-5:]}")
                    raise RuntimeError(f"Supervisor failed with exit code {returncode}")

            except asyncio.TimeoutError as e:
                supervisor_end_time = time.time()
                supervisor_duration = (supervisor_end_time - supervisor_start_time) * 1000
                logger.error(f"Supervisor timed out after {supervisor_duration:.0f}ms: {e}")
                raise RuntimeError(f"Supervisor execution timed out after {supervisor_duration:.0f}ms")
            except Exception as e:
                supervisor_end_time = time.time()
                supervisor_duration = (supervisor_end_time - supervisor_start_time) * 1000
                logger.error(f"Supervisor execution error after {supervisor_duration:.0f}ms: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Supervisor execution failed: {e}")

            logger.info("-" * 60)
            logger.info("SUPERVISOR COMPLETED")
            logger.info("-" * 60)

            # Parse supervisor JSON output using SimpleJSONStreamMonitor
            try:
                logger.debug(f"Parsing supervisor output from {len(all_output)} lines...")
                logger.debug(f"First 3 output lines: {all_output[:3] if all_output else 'None'}")
                logger.debug(f"Last 3 output lines: {all_output[-3:] if all_output else 'None'}")

                # Look for the decision JSON in assistant messages
                json_str = None
                found_jsons = []

                # Create JSON stream monitor
                json_monitor = SimpleJSONStreamMonitor()

                # Process all output lines to find JSON objects
                for line_str in all_output:
                    try:
                        # Parse the stream-json line
                        data = json.loads(line_str)
                        if data.get('type') == 'assistant':
                            # Extract content from assistant message
                            content = data.get('message', {}).get('content', [])
                            # Handle content as a list of items
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        text = item.get('text', '')
                                        # Process each line of the text with the JSON monitor
                                        for text_line in text.split('\n'):
                                            result = json_monitor.process_line(text_line)
                                            if result and isinstance(result, dict) and 'action' in result:
                                                # Convert back to string for compatibility
                                                found_jsons.append(json.dumps(result))
                                        # Reset monitor after each text block
                                        json_monitor.reset()
                    except json.JSONDecodeError:
                        # This line wasn't JSON, skip it
                        continue

                # Take the LAST valid JSON found (the final decision)
                if found_jsons:
                    json_str = found_jsons[-1]

                if not json_str:
                    logger.error("No decision JSON found in supervisor output")
                    logger.error(f"Found {len(found_jsons)} candidate JSONs using SimpleJSONStreamMonitor")
                    logger.error(f"Total output lines processed: {len(all_output)}")
                    # Log a sample of assistant messages for debugging
                    assistant_lines = []
                    for line_str in all_output[:10]:  # Just first 10 to avoid spam
                        try:
                            data = json.loads(line_str)
                            if data.get('type') == 'assistant':
                                assistant_lines.append(line_str[:200])  # Truncate for readability
                        except:
                            pass
                    logger.error(f"Sample assistant message lines: {assistant_lines}")
                    raise ValueError("No decision JSON found in supervisor output")

                logger.debug(f"Found decision JSON: {json_str[:100]}...")
                decision_data = json.loads(json_str)

                # Validate required fields based on action
                if decision_data.get('action') == 'execute':
                    # For execute action, check for new subtasks structure
                    if 'subtasks' in decision_data:
                        required_fields = ['action', 'task_id', 'task_title', 'subtasks', 'project_root', 'session_id', 'reason']  # Note: project_root is kept for backward compatibility in JSON
                    else:
                        # Backward compatibility with todos
                        required_fields = ['action', 'todos', 'guidance', 'task_id', 'session_id', 'reason']
                else:
                    # For skip/complete actions
                    required_fields = ['action', 'session_id', 'reason']

                missing_fields = [field for field in required_fields if field not in decision_data]
                if missing_fields:
                    raise ValueError(f"Decision JSON missing required fields: {missing_fields}")

                # Success! Reset retry counter and break out of retry loop
                json_retry_count = 0
                break

            except (json.JSONDecodeError, ValueError) as e:
                json_retry_count += 1
                logger.error(f"Failed to parse supervisor output as JSON: {e}")

                if json_retry_count >= max_json_retries:
                    logger.error(f"Failed to get valid JSON after {max_json_retries} attempts. Giving up.")
                    # Show last few lines of output for debugging
                    if all_output:
                        logger.error("Last 5 lines of output:")
                        for line in all_output[-5:]:
                            logger.error(f"  {line.strip()}")
                    raise RuntimeError(f"Supervisor failed to produce valid JSON after {max_json_retries} attempts")

                logger.warning(f"Will retry supervisor to get valid JSON output...")
                # Continue to next iteration of retry loop
                continue

        # Calculate execution time
        supervisor_execution_time = time.time() - supervisor_start_time
        quick_quit_threshold = self.config.get("orchestration", {}).get("quick_quit_seconds", 10.0)

        # Create decision object
        # Map project_root from JSON to project_path for internal use
        if 'project_root' in decision_data and 'project_path' not in decision_data:
            decision_data['project_path'] = decision_data.pop('project_root')

        decision = SupervisorDecision(**decision_data)
        decision.execution_time = supervisor_execution_time
        decision.quit_too_quickly = supervisor_execution_time < quick_quit_threshold

        logger.info(f"Supervisor decision: {decision.action}")
        if decision.reason:
            logger.info(f"   Reason: {decision.reason}")

        return decision

    finally:
        # Always return to original directory
        os.chdir(original_dir)
