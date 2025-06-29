# Setting Up CodebaseIQ Pro as an MCP Tool for Claude Code

## 📋 Prerequisites

1. **Claude Desktop App** - Must be installed and configured
2. **Python 3.9+** - Check with `python3 --version`
3. **OpenAI API Key** - Required for embeddings

## 🚀 Setup Steps

### Step 1: Configure Your API Key

CodebaseIQ Pro now automatically loads configuration from the `.env` file:

1. **Edit the .env file**:
   ```bash
   cd /Users/chrisryviss/codebase_iq_pro
   nano .env  # or use your favorite editor
   ```

2. **Add your OpenAI API key**:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

3. **That's it!** No need to edit any JSON configs or set environment variables elsewhere.

### Step 2: Install Dependencies

```bash
cd /Users/chrisryviss/codebase_iq_pro
python3 setup.py
```

### Step 3: Restart Claude Desktop

**Important**: You must restart the Claude Desktop app for the new MCP server to be loaded.

1. Quit Claude Desktop completely (Cmd+Q)
2. Reopen Claude Desktop
3. Open VS Code through Claude

### Step 4: Verify Installation

In a new Claude Code session in VS Code, the tools should be available. You can check by typing:

```
analyze_codebase
```

If you see the tool suggestion, it's working!

## 🎯 Using CodebaseIQ Pro in Claude Code

Once set up, you can use these commands:

### Analyze a Codebase
```
analyze_codebase path: "/path/to/project"
```

### Search Code Semantically
```
semantic_code_search query: "authentication logic"
```

### Find Similar Code
```
find_similar_code entity_path: "src/auth/login.py"
```

### Get Analysis Summary
```
get_analysis_summary
```

### Check Security Issues
```
get_danger_zones
```

### View Dependencies
```
get_dependencies
```

## 🔧 Troubleshooting

### "Tool not found" Error
- Make sure you restarted Claude Desktop after adding the configuration
- Check that the path in the config matches your actual installation path
- Verify Python 3 is installed: `which python3`

### "OPENAI_API_KEY is required" Error
- Edit the claude_desktop_config.json file
- Add your actual OpenAI API key in the env section
- Restart Claude Desktop

### Python Module Errors
- Run `python3 setup.py` to install dependencies
- Make sure you're using Python 3.9 or higher
- Check that all files were properly cloned/copied

### Server Won't Start
- Check the logs in Claude Desktop (View → Toggle Developer Tools → Console)
- Verify the path exists: `ls /Users/chrisryviss/codebase_iq_pro/codebaseiq_pro.py`
- Test manually: `cd /Users/chrisryviss/codebase_iq_pro && python3 codebaseiq_pro.py`

## 📦 GitHub Repository (Optional)

The GitHub repository you created (`https://github.com/chatcbdai/codebase-iq-pro.git`) is useful for:

1. **Version Control** - Track changes and updates
2. **Collaboration** - Share with others
3. **Distribution** - Others can clone and use your tool
4. **Updates** - Easy to pull latest changes

To push your code:

```bash
cd /Users/chrisryviss/codebase_iq_pro
git init
git add .
git commit -m "Initial commit of CodebaseIQ Pro"
git branch -M main
git remote add origin https://github.com/chatcbdai/codebase-iq-pro.git
git push -u origin main
```

## 🎉 Success!

Once everything is set up and Claude Desktop is restarted, CodebaseIQ Pro will be available as a tool in all your Claude Code sessions in VS Code!

## 📝 Alternative: Direct GitHub Installation

If you push to GitHub, others can install directly:

```json
{
  "codebase-iq-pro": {
    "command": "npx",
    "args": ["-y", "github:chatcbdai/codebase-iq-pro"],
    "env": {
      "OPENAI_API_KEY": "sk-..."
    }
  }
}
```

But this requires additional npm packaging setup.