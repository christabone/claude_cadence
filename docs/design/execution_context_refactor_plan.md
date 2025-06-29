# ExecutionContext Standardization Refactor Plan

## Current Problem Summary

### Runtime Bug
- **Issue**: Orchestrator crashes on second agent run with `ExecutionContext.__init__() got an unexpected keyword argument 'session_id'`
- **Root Cause**: Trying to pass metadata parameters (`session_id`, `task_numbers`, `project_root`) to ExecutionContext dataclass
- **Location**: `cadence/orchestrator.py:1014-1020`

### API Inconsistency
Currently 3 different patterns for prompt generation:

1. **`generate_initial_todo_prompt()`**: Individual parameters
   ```python
   def generate_initial_todo_prompt(self, todos: List[str], max_turns: int,
                                   session_id: str, task_numbers: str, project_root: str)
   ```

2. **`generate_continuation_prompt()`**: ExecutionContext + metadata
   ```python
   def generate_continuation_prompt(self, context: ExecutionContext,
                                   analysis_guidance: str, supervisor_analysis: Dict)
   ```

3. **`generate_supervisor_analysis_prompt()`**: ExecutionContext + metadata
   ```python
   def generate_supervisor_analysis_prompt(self, execution_output: str,
                                          context: ExecutionContext, previous_executions: List)
   ```

## ExecutionContext Design

### Valid Fields (dataclass definition)
```python
@dataclass
class ExecutionContext:
    todos: List[str]
    max_turns: int
    completed_todos: List[str] = field(default_factory=list)
    remaining_todos: List[str] = field(default_factory=list)
    issues_encountered: List[str] = field(default_factory=list)
    previous_guidance: List[str] = field(default_factory=list)
    continuation_context: Optional[str] = None
```

### Metadata (NOT part of ExecutionContext)
- `session_id` - Runtime session identifier
- `task_numbers` - Task ID string
- `project_root` - Working directory path

## Good Pattern Reference

**TodoPromptManager** shows the correct approach:

```python
class TodoPromptManager:
    def __init__(self, todos: List[str], max_turns: int):
        # ✅ ExecutionContext with only valid fields
        self.context = ExecutionContext(
            todos=todos,
            max_turns=max_turns,
            remaining_todos=todos.copy()
        )
        # ✅ Metadata stored separately
        self.session_id = "unknown"
        self.task_numbers = ""

    def get_initial_prompt(self) -> str:
        # ✅ Pass context data + metadata separately
        return self.generator.generate_initial_todo_prompt(
            todos=self.context.todos,
            max_turns=self.context.max_turns,
            session_id=self.session_id,
            task_numbers=self.task_numbers
        )
```

## Refactor Strategy

### Option 1: Standardize on ExecutionContext (Recommended)
Update `generate_initial_todo_prompt()` to match other methods:

```python
def generate_initial_todo_prompt(
    self,
    context: ExecutionContext,
    session_id: str = "unknown",
    task_numbers: str = "",
    project_root: str = None
) -> str:
```

### Option 2: Remove ExecutionContext (Not Recommended)
Would require updating 3 methods and losing state management benefits.

## Implementation Steps

### 1. Fix Immediate Bug
Update `cadence/orchestrator.py:1014-1020`:
```python
# ✅ Create ExecutionContext properly
context = ExecutionContext(
    todos=todos,
    max_turns=max_turns
)

# ✅ Pass metadata separately
base_prompt = prompt_generator.generate_continuation_prompt(
    context=context,
    analysis_guidance=guidance,
    supervisor_analysis={
        'session_id': session_id,
        'previous_session_id': getattr(self, 'previous_session_id', 'unknown'),
        'completed_normally': False,
        'has_issues': False
    }
)
```

### 2. Standardize API
Update `generate_initial_todo_prompt()` signature:
```python
def generate_initial_todo_prompt(
    self,
    context: ExecutionContext,
    session_id: str = "unknown",
    task_numbers: str = "",
    project_root: str = None
) -> str:
```

### 3. Update Orchestrator Call
Update the `else` branch in `build_agent_prompt()`:
```python
else:
    # ✅ Create ExecutionContext first
    context = ExecutionContext(
        todos=todos,
        max_turns=max_turns
    )

    # ✅ Pass to standardized method
    base_prompt = prompt_generator.generate_initial_todo_prompt(
        context=context,
        session_id=session_id,
        task_numbers=str(task_id) if task_id else "",
        project_root=project_root or str(self.project_root)
    )
```

### 4. Update TodoPromptManager
Minimal changes needed since it already follows the pattern:
```python
def get_initial_prompt(self) -> str:
    return self.generator.generate_initial_todo_prompt(
        context=self.context,  # ✅ Pass ExecutionContext
        session_id=self.session_id,
        task_numbers=self.task_numbers
    )
```

## Code Smells Addressed

1. **API Inconsistency**: All prompt methods use ExecutionContext
2. **Parameter Explosion**: Reduced from 5+ parameters to context object + metadata
3. **Runtime Crashes**: Proper ExecutionContext instantiation
4. **Coupling**: Clear separation between execution state and metadata
5. **Maintainability**: Consistent patterns across codebase

## Benefits

1. **Consistency**: All prompt generation follows same pattern
2. **State Management**: ExecutionContext properly tracks execution state
3. **Extensibility**: Easy to add new execution state fields
4. **Debugging**: Clear separation of concerns
5. **Testing**: Easier to mock and test with consistent interfaces

## Files to Modify

1. `cadence/prompts.py` - Update `generate_initial_todo_prompt()` signature
2. `cadence/orchestrator.py` - Fix ExecutionContext creation and usage
3. `cadence/prompts.py` - Update TodoPromptManager if needed

## Testing Strategy

1. Test both initial and continuation prompt generation
2. Verify no regression in TodoPromptManager
3. Test orchestrator with multiple iterations (where crash occurred)
4. Validate all prompt methods produce expected output
