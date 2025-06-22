"""
Unit tests for TaskSupervisor analysis mode
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from cadence.task_supervisor import TaskSupervisor
from cadence.task_manager import Task


class TestSupervisorAnalysisMode:
    """Test TaskSupervisor analysis mode functionality"""
    
    @pytest.fixture
    def mock_task_manager(self):
        """Create mock task manager with tasks and subtasks"""
        task_manager = Mock()
        
        # Create mock tasks with subtasks
        task1 = Mock(spec=Task)
        task1.id = "1"
        task1.title = "Main Task 1"
        task1.status = "pending"
        
        # Subtasks for task1
        subtask1_1 = Mock()
        subtask1_1.id = "1.1"
        subtask1_1.title = "Subtask 1.1"
        subtask1_1.status = "done"
        subtask1_1.description = "Already completed"
        
        subtask1_2 = Mock()
        subtask1_2.id = "1.2"
        subtask1_2.title = "Subtask 1.2"
        subtask1_2.status = "pending"
        subtask1_2.description = "Need to implement this"
        subtask1_2.details = "Implementation details here"
        subtask1_2.testStrategy = "Unit test the function"
        
        subtask1_3 = Mock()
        subtask1_3.id = "1.3"
        subtask1_3.title = "Subtask 1.3"
        subtask1_3.status = "pending"
        subtask1_3.description = "Another pending task"
        
        task1.subtasks = [subtask1_1, subtask1_2, subtask1_3]
        
        # Task 2 with all subtasks complete
        task2 = Mock(spec=Task)
        task2.id = "2"
        task2.title = "Main Task 2"
        task2.status = "done"
        
        subtask2_1 = Mock()
        subtask2_1.id = "2.1"
        subtask2_1.status = "done"
        
        task2.subtasks = [subtask2_1]
        
        task_manager.tasks = [task1, task2]
        
        return task_manager
    
    def test_analyze_and_decide_basic(self, supervisor, mock_task_manager, temp_dir):
        """Test basic analyze_and_decide functionality"""
        task_file = temp_dir / "tasks.json"
        task_file.write_text("{}")
        
        # Mock the AI analysis to avoid actual API calls
        mock_ai_analysis = {
            "should_execute": True,
            "guidance": "Test guidance",
            "reasoning": "Test reasoning",
            "needs_assistance": False
        }
        
        with patch.object(supervisor, 'load_previous_results', return_value=None):
            with patch.object(supervisor, 'supervisor_ai_analysis', return_value=mock_ai_analysis):
                with patch('cadence.task_supervisor.TaskManager') as MockTaskManager:
                    MockTaskManager.return_value = mock_task_manager
                    mock_task_manager.load_tasks.return_value = True
                    
                    decision = supervisor.analyze_and_decide(
                        task_file=str(task_file),
                        session_id="test_session",
                        continue_from_previous=False
                    )
                    
                    # Should execute task 1 with 2 pending subtasks
                    assert decision["action"] == "execute"
                    assert decision["task_id"] == "1"
                    assert len(decision["todos"]) == 2
                    assert "Task 1.2: Subtask 1.2" in decision["todos"][0]
                    assert "Task 1.3: Subtask 1.3" in decision["todos"][1]
                
    def test_analyze_and_decide_all_complete(self, supervisor, temp_dir):
        """Test when all tasks and subtasks are complete"""
        task_file = temp_dir / "tasks.json"
        task_file.write_text("{}")
        
        # Mock task manager with all tasks complete
        task_manager = Mock()
        task_manager.tasks = []
        task_manager.load_tasks.return_value = True
        
        with patch('cadence.task_supervisor.TaskManager', return_value=task_manager):
            decision = supervisor.analyze_and_decide(
                task_file=str(task_file),
                session_id="test_session"
            )
            
            assert decision["action"] == "complete"
            assert "All tasks and subtasks are complete" in decision["reason"]
            
    def test_get_next_task_with_subtasks(self, supervisor, mock_task_manager):
        """Test finding next task with incomplete subtasks"""
        task = supervisor.get_next_task_with_subtasks(mock_task_manager)
        
        assert task is not None
        assert task.id == "1"  # First task has incomplete subtasks
        
    def test_extract_subtask_todos(self, supervisor, mock_task_manager):
        """Test extracting subtasks as TODOs"""
        task = mock_task_manager.tasks[0]  # Task 1
        todos = supervisor.extract_subtask_todos(task)
        
        assert len(todos) == 2  # Only pending subtasks
        
        # Check first TODO formatting
        assert "Task 1.2: Subtask 1.2" in todos[0]
        assert "Description: Need to implement this" in todos[0]
        assert "Details: Implementation details here" in todos[0]
        assert "Test Strategy: Unit test the function" in todos[0]
        
        # Check second TODO
        assert "Task 1.3: Subtask 1.3" in todos[1]
        assert "Description: Another pending task" in todos[1]
        
    def test_load_previous_results(self, supervisor, temp_dir):
        """Test loading previous agent results"""
        session_id = "test_session"
        
        # Create mock previous results
        result_data = {
            "success": True,
            "execution_time": 15.5,
            "completed_normally": False,
            "requested_help": True,
            "errors": ["Error 1", "Error 2"]
        }
        
        result_file = temp_dir / f"agent_result_{session_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f)
            
        # Change to temp directory for test
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_dir)
            
            results = supervisor.load_previous_results(session_id)
            
            assert results is not None
            assert results["success"] is True
            assert results["requested_help"] is True
            assert len(results["errors"]) == 2
        finally:
            os.chdir(original_cwd)
            
    def test_supervisor_ai_analysis_mock(self, supervisor, mock_task_manager):
        """Test supervisor AI analysis with mock"""
        task = mock_task_manager.tasks[0]
        todos = ["TODO 1", "TODO 2"]
        
        # Mock subprocess to avoid actual Claude API calls
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "content": json.dumps({
                "should_execute": True,
                "guidance": "Test guidance",
                "reasoning": "Test reasoning",
                "needs_assistance": False
            })
        })
        
        with patch('subprocess.run', return_value=mock_result):
            # Test without previous results
            analysis = supervisor.supervisor_ai_analysis(
                current_task=task,
                todos=todos,
                previous_results=None,
                session_id="test"
            )
            
            assert analysis["should_execute"] is True
            assert "guidance" in analysis
            assert "reasoning" in analysis
            assert analysis["needs_assistance"] is False
        
        # Test with failed previous results
        mock_result.stdout = json.dumps({
            "content": json.dumps({
                "should_execute": True,
                "guidance": "Previous execution encountered errors. Test guidance",
                "reasoning": "Test reasoning with errors",
                "needs_assistance": True
            })
        })
        
        with patch('subprocess.run', return_value=mock_result):
            previous_results = {"success": False, "errors": ["Error"]}
            analysis = supervisor.supervisor_ai_analysis(
                current_task=task,
                todos=todos,
                previous_results=previous_results,
                session_id="test"
            )
            
            assert analysis["needs_assistance"] is True
            assert "Previous execution encountered errors" in analysis["guidance"]
        
    def test_supervisor_enforces_ai_model(self, mock_config):
        """Test that supervisor refuses heuristic mode"""
        # Try to set heuristic mode
        mock_config.supervisor.model = "heuristic"
        
        supervisor = TaskSupervisor(config=mock_config)
        
        # Should raise error when trying to do AI analysis
        with pytest.raises(ValueError, match="Supervisor MUST use AI model"):
            supervisor.supervisor_ai_analysis(
                current_task=Mock(),
                todos=[],
                previous_results=None,
                session_id="test"
            )