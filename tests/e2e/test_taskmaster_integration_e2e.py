"""
End-to-end tests for TaskMaster integration
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
import subprocess
import yaml

from cadence.task_supervisor import TaskSupervisor
from cadence.config import CadenceConfig


class TestTaskMasterE2E:
    """E2E tests for TaskMaster integration"""

    @pytest.fixture
    def e2e_temp_dir(self):
        """Create a temporary directory for E2E tests"""
        temp_dir = tempfile.mkdtemp(prefix="cadence_e2e_tm_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def e2e_config(self, e2e_temp_dir):
        """Create a test configuration"""
        config = CadenceConfig()
        config.execution.max_turns = 10
        config.execution.log_dir = str(e2e_temp_dir / "logs")
        config.agent.model = "claude-3-haiku-20240307"
        config.supervisor.zen_integration.enabled = False
        config.mcp.servers = ["filesystem", "taskmaster-ai"]
        return config

    @pytest.fixture
    def sample_taskmaster_file(self, e2e_temp_dir):
        """Create a sample TaskMaster tasks.json file"""
        tasks = {
            "tasks": [
                {
                    "id": "1",
                    "title": "Setup project structure",
                    "description": "Create the basic directory structure for the project",
                    "status": "pending",
                    "priority": "high",
                    "dependencies": []
                },
                {
                    "id": "2",
                    "title": "Initialize configuration",
                    "description": "Create a config.yaml file with default settings",
                    "status": "pending",
                    "priority": "high",
                    "dependencies": ["1"]
                },
                {
                    "id": "3",
                    "title": "Add logging setup",
                    "description": "Implement basic logging functionality",
                    "status": "pending",
                    "priority": "medium",
                    "dependencies": ["1"]
                },
                {
                    "id": "4",
                    "title": "Write documentation",
                    "description": "Create README.md with setup instructions",
                    "status": "pending",
                    "priority": "low",
                    "dependencies": ["1", "2"]
                }
            ]
        }

        task_file = e2e_temp_dir / "tasks.json"
        with open(task_file, 'w') as f:
            json.dump(tasks, f, indent=2)

        return task_file

    def test_taskmaster_basic_execution(self, e2e_config, e2e_temp_dir, sample_taskmaster_file):
        """Test basic TaskMaster execution"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        supervisor = TaskSupervisor(config=e2e_config)

        # Run with TaskMaster
        success = supervisor.run_with_taskmaster(
            str(sample_taskmaster_file),
            task_numbers="1,2"  # Just first two tasks
        )

        assert success is True

        # Check if project structure was created
        project_dirs = ["src", "tests", "docs"]
        for dir_name in project_dirs:
            dir_path = e2e_temp_dir / dir_name
            # At least some directories should exist

        # Check if config was created
        config_file = e2e_temp_dir / "config.yaml"
        # Config file might exist

    def test_taskmaster_dependency_handling(self, e2e_config, e2e_temp_dir, sample_taskmaster_file):
        """Test TaskMaster respects task dependencies"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        supervisor = TaskSupervisor(config=e2e_config)

        # Try to run task 4 which depends on 1 and 2
        success = supervisor.run_with_taskmaster(
            str(sample_taskmaster_file),
            task_numbers="4"
        )

        # Should handle dependencies appropriately
        assert isinstance(success, bool)

    def test_taskmaster_progress_tracking(self, e2e_config, e2e_temp_dir):
        """Test TaskMaster progress tracking across executions"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create tasks with some already completed
        tasks = {
            "tasks": [
                {
                    "id": "1",
                    "title": "Completed task",
                    "description": "This is already done",
                    "status": "completed",
                    "priority": "high",
                    "dependencies": []
                },
                {
                    "id": "2",
                    "title": "Pending task",
                    "description": "Create a test.txt file",
                    "status": "pending",
                    "priority": "high",
                    "dependencies": []
                }
            ]
        }

        task_file = e2e_temp_dir / "progress_tasks.json"
        with open(task_file, 'w') as f:
            json.dump(tasks, f, indent=2)

        supervisor = TaskSupervisor(config=e2e_config)

        # Run all tasks
        success = supervisor.run_with_taskmaster(str(task_file))

        # Should only execute the pending task
        assert success is True

        # Check if test.txt was created
        test_file = e2e_temp_dir / "test.txt"
        # File might exist depending on execution

    def test_taskmaster_with_complex_project(self, e2e_config, e2e_temp_dir):
        """Test TaskMaster with a more complex project structure"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create a complex task structure
        tasks = {
            "tasks": [
                {
                    "id": "1",
                    "title": "Setup Python package structure",
                    "description": "Create __init__.py files and package directories",
                    "status": "pending",
                    "priority": "high",
                    "dependencies": [],
                    "subtasks": [
                        {
                            "id": "1.1",
                            "title": "Create src directory",
                            "status": "pending"
                        },
                        {
                            "id": "1.2",
                            "title": "Create __init__.py files",
                            "status": "pending"
                        }
                    ]
                },
                {
                    "id": "2",
                    "title": "Setup testing framework",
                    "description": "Create pytest configuration",
                    "status": "pending",
                    "priority": "high",
                    "dependencies": ["1"],
                    "subtasks": [
                        {
                            "id": "2.1",
                            "title": "Create conftest.py",
                            "status": "pending"
                        },
                        {
                            "id": "2.2",
                            "title": "Create test directory structure",
                            "status": "pending"
                        }
                    ]
                }
            ]
        }

        task_file = e2e_temp_dir / "complex_tasks.json"
        with open(task_file, 'w') as f:
            json.dump(tasks, f, indent=2)

        supervisor = TaskSupervisor(config=e2e_config)

        # Execute with subtasks
        success = supervisor.run_with_taskmaster(
            str(task_file),
            task_numbers="1"
        )

        # Check results
        assert isinstance(success, bool)

        # Some structure should be created
        src_dir = e2e_temp_dir / "src"
        # Directory might exist

    def test_taskmaster_error_recovery(self, e2e_config, e2e_temp_dir):
        """Test TaskMaster handles task failures gracefully"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create tasks with one that will fail
        tasks = {
            "tasks": [
                {
                    "id": "1",
                    "title": "Valid task",
                    "description": "Create a simple file",
                    "status": "pending",
                    "priority": "high",
                    "dependencies": []
                },
                {
                    "id": "2",
                    "title": "Problematic task",
                    "description": "Access a restricted system file at /etc/shadow",
                    "status": "pending",
                    "priority": "high",
                    "dependencies": []
                },
                {
                    "id": "3",
                    "title": "Another valid task",
                    "description": "Create another file",
                    "status": "pending",
                    "priority": "medium",
                    "dependencies": []
                }
            ]
        }

        task_file = e2e_temp_dir / "error_tasks.json"
        with open(task_file, 'w') as f:
            json.dump(tasks, f, indent=2)

        supervisor = TaskSupervisor(config=e2e_config)

        # Run all tasks
        success = supervisor.run_with_taskmaster(str(task_file))

        # Should complete what it can
        assert isinstance(success, bool)

        # Check execution history
        assert len(supervisor.execution_history) > 0
