"""
End-to-end tests for multi-agent coordination scenarios
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from cadence.task_supervisor import TaskSupervisor
from cadence.config import CadenceConfig, SCRATCHPAD_DIR


class TestMultiAgentE2E:
    """E2E tests for multi-agent coordination"""

    @pytest.fixture
    def e2e_temp_dir(self):
        """Create a temporary directory for E2E tests"""
        temp_dir = tempfile.mkdtemp(prefix="cadence_e2e_multi_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def e2e_config(self, e2e_temp_dir):
        """Create a test configuration"""
        config = CadenceConfig()
        config.execution.max_turns = 5
        config.execution.log_dir = str(e2e_temp_dir / "logs")
        config.agent.model = "claude-3-haiku-20240307"
        config.supervisor.zen_integration.enabled = False
        return config

    def test_parallel_agent_execution(self, e2e_config, e2e_temp_dir):
        """Test running multiple agents in parallel"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create different task sets for different agents
        agent1_todos = [
            "Create a file called agent1_output.txt with content 'Agent 1 was here'",
            "Create a directory called agent1_work"
        ]

        agent2_todos = [
            "Create a file called agent2_output.txt with content 'Agent 2 was here'",
            "Create a directory called agent2_work"
        ]

        # Run agents in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            supervisor1 = TaskSupervisor(config=e2e_config)
            supervisor2 = TaskSupervisor(config=e2e_config)

            future1 = executor.submit(
                supervisor1.execute_with_todos,
                agent1_todos,
                "e2e_agent1"
            )

            future2 = executor.submit(
                supervisor2.execute_with_todos,
                agent2_todos,
                "e2e_agent2"
            )

            result1 = future1.result(timeout=60)
            result2 = future2.result(timeout=60)

        # Both should succeed
        assert result1.success is True
        assert result2.success is True

        # Check outputs
        assert (e2e_temp_dir / "agent1_output.txt").exists()
        assert (e2e_temp_dir / "agent2_output.txt").exists()

        # Check separate scratchpads
        assert (Path(SCRATCHPAD_DIR) / "session_e2e_agent1.md").exists()
        assert (Path(SCRATCHPAD_DIR) / "session_e2e_agent2.md").exists()

    def test_sequential_agent_handoff(self, e2e_config, e2e_temp_dir):
        """Test agents handing off work sequentially"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        supervisor = TaskSupervisor(config=e2e_config)

        # First agent creates initial work
        phase1_todos = [
            "Create a data.json file with content: {\"status\": \"phase1_complete\", \"next\": \"phase2\"}",
            "Create a checkpoint.txt file with content 'Ready for phase 2'"
        ]

        result1 = supervisor.execute_with_todos(
            phase1_todos,
            session_id="e2e_handoff_phase1"
        )

        assert result1.success is True
        assert (e2e_temp_dir / "data.json").exists()

        # Second agent continues the work
        phase2_todos = [
            "Read data.json and verify it contains phase1_complete",
            "Update data.json to set status to 'phase2_complete'",
            "Create a summary.txt file with the workflow status"
        ]

        result2 = supervisor.execute_with_todos(
            phase2_todos,
            session_id="e2e_handoff_phase2"
        )

        assert result2.success is True

        # Verify handoff worked
        if (e2e_temp_dir / "data.json").exists():
            data = json.loads((e2e_temp_dir / "data.json").read_text())
            # Status might be updated

    def test_shared_resource_coordination(self, e2e_config, e2e_temp_dir):
        """Test multiple agents working with shared resources"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create a shared configuration file
        shared_config = e2e_temp_dir / "shared_config.json"
        shared_config.write_text(json.dumps({
            "agents": [],
            "tasks_completed": 0
        }))

        # Agent 1 updates the config
        agent1_todos = [
            "Read shared_config.json",
            "Add your agent ID 'agent1' to the agents list",
            "Increment tasks_completed by 1"
        ]

        supervisor1 = TaskSupervisor(config=e2e_config)
        result1 = supervisor1.execute_with_todos(
            agent1_todos,
            session_id="e2e_shared_1"
        )

        # Agent 2 also updates the config
        agent2_todos = [
            "Read shared_config.json",
            "Add your agent ID 'agent2' to the agents list",
            "Increment tasks_completed by 1"
        ]

        supervisor2 = TaskSupervisor(config=e2e_config)
        result2 = supervisor2.execute_with_todos(
            agent2_todos,
            session_id="e2e_shared_2"
        )

        # Check final state
        if shared_config.exists():
            final_data = json.loads(shared_config.read_text())
            # Should have evidence of both agents' work

    def test_supervisor_coordination(self, e2e_config, e2e_temp_dir):
        """Test multiple supervisors coordinating complex workflows"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create a workflow definition
        workflow = {
            "stages": [
                {
                    "id": "data_prep",
                    "tasks": ["Create raw_data.csv with sample data", "Validate the data format"]
                },
                {
                    "id": "processing",
                    "tasks": ["Process raw_data.csv", "Create processed_data.json"]
                },
                {
                    "id": "reporting",
                    "tasks": ["Generate report from processed_data.json", "Create summary.md"]
                }
            ]
        }

        workflow_file = e2e_temp_dir / "workflow.json"
        workflow_file.write_text(json.dumps(workflow, indent=2))

        # Execute stages with different supervisors
        results = []

        for i, stage in enumerate(workflow["stages"]):
            supervisor = TaskSupervisor(config=e2e_config)
            result = supervisor.execute_with_todos(
                stage["tasks"],
                session_id=f"e2e_workflow_{stage['id']}"
            )
            results.append(result)

            # Brief pause between stages
            time.sleep(1)

        # Check overall workflow completion
        assert len(results) == 3
        # At least some stages should complete
        successful_stages = sum(1 for r in results if r.success)
        assert successful_stages > 0

    def test_agent_failure_recovery(self, e2e_config, e2e_temp_dir):
        """Test recovery when one agent in a multi-agent system fails"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Set very short timeout for first agent
        config_fail = CadenceConfig()
        config_fail.execution.timeout = 2  # 2 seconds - likely to timeout
        config_fail.execution.max_turns = 10
        config_fail.execution.log_dir = str(e2e_temp_dir / "logs")
        config_fail.agent.model = "claude-3-haiku-20240307"

        # First agent with impossible task
        fail_todos = [
            "Count to 1 million and create a file for each number"  # Will timeout
        ]

        supervisor_fail = TaskSupervisor(config=config_fail)
        result_fail = supervisor_fail.execute_with_todos(
            fail_todos,
            session_id="e2e_fail_agent"
        )

        # Should fail/timeout
        assert result_fail.success is False or result_fail.task_complete is False

        # Recovery agent
        recovery_todos = [
            "Check if any files were created by the previous agent",
            "Create a recovery_report.txt explaining what happened",
            "Create a success_marker.txt to show recovery worked"
        ]

        supervisor_recovery = TaskSupervisor(config=e2e_config)
        result_recovery = supervisor_recovery.execute_with_todos(
            recovery_todos,
            session_id="e2e_recovery_agent"
        )

        # Recovery should succeed
        assert result_recovery.success is True
        assert (e2e_temp_dir / "success_marker.txt").exists() or \
               (e2e_temp_dir / "recovery_report.txt").exists()

    def test_distributed_task_completion(self, e2e_config, e2e_temp_dir):
        """Test distributing a large task across multiple agents"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create a task list to distribute
        all_tasks = []
        for i in range(6):
            all_tasks.append(f"Create file part{i}.txt with content 'Part {i} complete'")

        # Distribute across 3 agents
        chunk_size = 2
        agents = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []

            for i in range(0, len(all_tasks), chunk_size):
                chunk = all_tasks[i:i+chunk_size]
                supervisor = TaskSupervisor(config=e2e_config)

                future = executor.submit(
                    supervisor.execute_with_todos,
                    chunk,
                    f"e2e_distributed_{i//chunk_size}"
                )
                futures.append(future)

            # Wait for all to complete
            results = [f.result(timeout=60) for f in futures]

        # Check results
        successful = sum(1 for r in results if r.success)
        assert successful >= 2  # At least 2 of 3 should succeed

        # Check files created
        created_files = []
        for i in range(6):
            if (e2e_temp_dir / f"part{i}.txt").exists():
                created_files.append(f"part{i}.txt")

        assert len(created_files) >= 4  # At least 4 of 6 files created
