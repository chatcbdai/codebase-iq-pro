# CodebaseIQ Pro Configuration for Claude

## 🚨 MANDATORY: Read This Before ANY Code Changes

This codebase uses CodebaseIQ Pro for comprehensive analysis. You MUST follow these steps:

### 1. Initial Context Loading (REQUIRED)
```
# First command in EVERY new conversation:
get_codebase_context
```

This loads essential knowledge about:
- Danger zones (files you should NOT modify)
- Impact analysis (what breaks when you change things)
- Business logic understanding
- Safe modification guidelines

### 2. Verification Before Implementation

**RED FLAG SYSTEM**: Before implementing ANY code changes, you MUST:

1. Run `check_understanding` with your implementation plan
2. Receive EXPLICIT approval (score >= 8/10)
3. Only then proceed with implementation

Example:
```
check_understanding "I plan to modify auth_service.py to add a new login method"
```

### 3. File Modification Rules

**CRITICAL FILES** (Never modify without explicit permission):
- Any file marked as "CRITICAL" in danger zones
- Authentication/security files
- Payment processing files
- Core configuration files

**Before modifying ANY file**:
```
get_modification_guidance <file_path>
```

### 4. Testing Requirements

After ANY code changes:
1. Run tests for the modified file
2. Run tests for ALL files in the impact zone
3. Verify no regression in critical paths

### 5. Performance Considerations

This codebase has 300+ files. Initial analysis takes 4-5 minutes.
- Use `get_codebase_context` for quick access to pre-analyzed data
- Don't run `analyze_codebase` unless specifically requested
- Context is cached and refreshed every 24 hours

## Secondary Rules (CANNOT BE OVERLOOKED)

### Rule 1: Understanding Verification
Before ANY new code implementation, the AI assistant MUST re-evaluate its understanding:
- What are the dependencies of this change?
- What business rules might be affected?
- What tests need to be updated?
- What is the risk level of this modification?

### Rule 2: Cutting-Edge Tech Warning
This codebase uses technologies that may not be in your training data:
- Always verify your understanding against the actual code
- Don't assume standard patterns apply
- Check the codebase's specific implementation before suggesting changes

### Rule 3: Zero Assumptions Policy
- NEVER assume a file's purpose from its name alone
- ALWAYS read the actual implementation
- VERIFY your understanding through the tools provided

## Quick Reference Commands

1. **Start of conversation**: `get_codebase_context`
2. **Before ANY modification**: `get_modification_guidance <file>`
3. **Verify understanding**: `check_understanding "<your plan>"`
4. **Check specific impact**: `get_impact_analysis <file>`
5. **Business context**: `get_business_context`

## Remember

You're working with a sophisticated, interdependent codebase. Every change has consequences. When in doubt, ASK before implementing.

**Success Metrics**:
- Zero breaking changes
- All tests passing
- Business logic preserved
- No unintended side effects