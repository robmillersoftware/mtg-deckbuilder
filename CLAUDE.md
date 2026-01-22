# Cahoots Project Implementation Guide

This project uses a **Cahoots spec YAML file** (`*-sync.yaml`) as the single source of truth for implementation. Claude Code must follow this file exactly when implementing features.

## Spec YAML Structure

The spec file contains:
- **`description`**: Overall project description with tech stack and requirements
- **`epics`**: High-level feature groupings with priorities (1 = highest)
- **`stories`**: User stories with acceptance criteria, linked to epics via `epic_id`
- **`tasks`**: Technical implementation tasks with:
  - `id`: Unique task identifier
  - `title`: What to implement
  - `story_id` / `epic_id`: Parent linkage
  - `implementation_details`: Specific technical guidance
  - `status`: `pending`, `in_progress`, or `completed`
  - `story_points`: Complexity estimate
  - `depends_on`: List of task IDs that must be completed first

## Implementation Rules

### 1. Task Selection Order

**ALWAYS** implement tasks in dependency order:
1. Read the spec YAML file
2. Find all tasks with `status: pending`
3. Filter to tasks where ALL `depends_on` tasks have `status: completed`
4. Among eligible tasks, prioritize by:
   - Epic priority (lower number = higher priority)
   - Story priority (`must_have` > `should_have` > `could_have`)
   - Fewer story points (simpler tasks first)

### 2. Before Starting Any Task

1. **Read the full task** including `title`, `implementation_details`, and linked story's `acceptance_criteria`
2. **Review dependencies** to understand what code/models already exist
3. **Check the tech stack** from the project description (FastAPI, React, PostgreSQL, Redis, etc.)
4. **Use the TodoWrite tool** to plan subtasks for the implementation

### 3. During Implementation

- Follow `implementation_details` exactly - they specify models, endpoints, patterns, and integrations
- Match the acceptance criteria from the parent story
- Use consistent patterns with previously implemented tasks
- Create proper tests for all new functionality
- Update database migrations as needed (Alembic)

### 4. After Completing a Task

1. Run all tests to verify nothing is broken
2. **Update the spec YAML** - change the task's `status` from `pending` to `completed`
3. Move to the next eligible task

## Commands

### Start Implementation
When asked to implement from the spec:
1. Read the spec YAML file
2. Identify the next pending task (respecting dependencies)
3. Implement it fully
4. Mark as completed in the YAML
5. Ask whether to continue to the next task

### Status Check
When asked for status:
- Count completed vs pending tasks
- List currently blocked tasks and what they're waiting on
- Identify the next implementable task

## Tech Stack Reference

From the spec description, this project uses:
- **Backend**: FastAPI (Python), PostgreSQL, Redis, RQ (Redis Queue)
- **Frontend**: React with TypeScript
- **Auth**: Auth0 integration
- **Payments**: Stripe
- **Email**: SendGrid
- **Migrations**: Alembic
- **Container**: Docker & docker-compose

## Example Workflow

```
User: Implement the next task from the spec

Claude:
1. Reads 6f77cd4a-...-sync.yaml
2. Finds task with status: pending and no unmet dependencies
3. Creates TodoWrite items for subtasks
4. Implements the task following implementation_details
5. Runs tests
6. Updates YAML: status: completed
7. Reports completion and asks about next task
```

## Important Notes

- **Never skip dependencies** - if a task depends on another, the dependency MUST be completed first
- **Follow implementation_details literally** - they contain specific model names, endpoint paths, and integration patterns
- **Match acceptance criteria** - each story has specific criteria that the implementation must satisfy
- **Preserve IDs** - when updating the YAML, keep all IDs unchanged
- **One task at a time** - complete each task fully before moving to the next
