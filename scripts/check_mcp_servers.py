#!/usr/bin/env python3
"""
Check and ensure required MCP servers are installed and running

This script verifies that the required MCP servers (zen and taskmaster-ai)
are properly installed and accessible via the Claude CLI.
"""

import subprocess
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MCPChecker:
    """Check and validate MCP server installation and status"""
    
    REQUIRED_SERVERS = ["zen", "taskmaster-ai"]
    OPTIONAL_SERVERS = ["github", "filesystem", "mcp"]
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def check_claude_cli(self) -> bool:
        """Check if Claude CLI is installed and accessible"""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                if self.verbose:
                    print(f"✅ Claude CLI found: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ Claude CLI not working properly: {result.stderr}")
                return False
        except FileNotFoundError:
            print("❌ Claude CLI not found. Please install: npm install -g @anthropic-ai/claude-code")
            return False
        except Exception as e:
            print(f"❌ Error checking Claude CLI: {e}")
            return False
            
    def get_mcp_status(self) -> Dict[str, any]:
        """Get MCP server status using Claude CLI"""
        try:
            # Try different commands that might show MCP status
            # Note: The actual command may vary - check Context7 docs
            
            # Try listing MCP tools
            result = subprocess.run(
                ["claude", "mcp", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return self._parse_mcp_list(result.stdout)
                
            # Alternative: Try getting MCP status via tool listing
            result = subprocess.run(
                ["claude", "--tool", "mcp", "-p", "List available MCP servers"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return self._parse_mcp_output(result.stdout)
                
            # If no specific command works, try to infer from available tools
            return self._check_via_tools()
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not get MCP status directly: {e}")
            return {}
            
    def _parse_mcp_list(self, output: str) -> Dict[str, any]:
        """Parse output from mcp list command"""
        servers = {}
        lines = output.strip().split('\n')
        
        for line in lines:
            # Parse server info (format may vary)
            if any(server in line.lower() for server in self.REQUIRED_SERVERS):
                # Extract server name and status
                parts = line.split()
                if parts:
                    server_name = parts[0].lower()
                    servers[server_name] = {"status": "found", "line": line}
                    
        return servers
        
    def _parse_mcp_output(self, output: str) -> Dict[str, any]:
        """Parse general MCP output"""
        servers = {}
        
        # Look for server mentions in output
        for server in self.REQUIRED_SERVERS + self.OPTIONAL_SERVERS:
            if server in output.lower():
                servers[server] = {"status": "mentioned", "output": output}
                
        return servers
        
    def _check_via_tools(self) -> Dict[str, any]:
        """Check MCP availability by testing specific tools"""
        servers = {}
        
        # Test zen
        if self._test_mcp_tool("zen__version"):
            servers["zen"] = {"status": "working", "method": "tool_test"}
            
        # Test taskmaster
        if self._test_mcp_tool("taskmaster-ai__initialize_project", {"projectRoot": "/tmp/test"}):
            servers["taskmaster-ai"] = {"status": "working", "method": "tool_test"}
            
        return servers
        
    def _test_mcp_tool(self, tool_name: str, params: Optional[Dict] = None) -> bool:
        """Test if a specific MCP tool is accessible"""
        try:
            cmd = [
                "claude", 
                "-p", f"Test MCP tool: {tool_name}",
                "--tool", "mcp",
                "--max-turns", "1"
            ]
            
            # Add tool-specific parameters if needed
            if params:
                cmd.extend(["--mcp-tool", tool_name])
                
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return result.returncode == 0 and tool_name in result.stdout
            
        except:
            return False
            
    def check_required_servers(self) -> Tuple[List[str], List[str]]:
        """
        Check which required servers are available
        
        Returns:
            Tuple of (working_servers, missing_servers)
        """
        if self.verbose:
            print("\n🔍 Checking MCP servers...")
            
        mcp_status = self.get_mcp_status()
        
        working = []
        missing = []
        
        for server in self.REQUIRED_SERVERS:
            if server in mcp_status:
                status = mcp_status[server].get("status", "unknown")
                if status in ["working", "found", "mentioned"]:
                    working.append(server)
                    if self.verbose:
                        print(f"  ✅ {server}: {status}")
                else:
                    missing.append(server)
                    if self.verbose:
                        print(f"  ❌ {server}: {status}")
            else:
                missing.append(server)
                if self.verbose:
                    print(f"  ❌ {server}: not found")
                    
        # Check optional servers
        if self.verbose:
            print("\n📦 Optional MCP servers:")
            for server in self.OPTIONAL_SERVERS:
                if server in mcp_status:
                    print(f"  ✅ {server}: found")
                else:
                    print(f"  ⚪ {server}: not found")
                    
        return working, missing
        
    def suggest_fixes(self, missing_servers: List[str]):
        """Suggest how to install/configure missing servers"""
        if not missing_servers:
            return
            
        print("\n💡 To install missing MCP servers:")
        
        for server in missing_servers:
            if server == "zen":
                print(f"""
  For {server}:
    1. Check your ~/.claude.json or ~/.config/claude/claude.json
    2. Ensure zen MCP server is configured:
       {{
         "mcpServers": {{
           "zen": {{
             "command": "zen-mcp",
             "args": [],
             "env": {{}}
           }}
         }}
       }}
    3. Install if needed: npm install -g zen-mcp (or appropriate command)
""")
            elif server == "taskmaster-ai":
                print(f"""
  For {server}:
    1. Check your MCP configuration file
    2. Ensure taskmaster-ai is configured
    3. Install if needed: npm install -g @taskmaster/mcp-server
    4. Or check the Task Master documentation for installation
""")
                
    def create_config_template(self):
        """Create a template MCP configuration if needed"""
        config_path = Path.home() / ".claude.json"
        
        if config_path.exists():
            print(f"\n📄 MCP config exists at: {config_path}")
            return
            
        print(f"\n📝 Creating MCP config template at: {config_path}")
        
        template = {
            "mcpServers": {
                "zen": {
                    "command": "zen-mcp",
                    "args": [],
                    "env": {}
                },
                "taskmaster-ai": {
                    "command": "taskmaster-mcp", 
                    "args": ["--config", "~/.taskmaster/config.json"],
                    "env": {}
                }
            }
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(template, f, indent=2)
            print("✅ Template created. Please update with correct paths and commands.")
        except Exception as e:
            print(f"❌ Could not create template: {e}")
            

def main():
    """Main entry point"""
    print("🚀 Claude Cadence MCP Server Check")
    print("=" * 50)
    
    checker = MCPChecker(verbose=True)
    
    # Check Claude CLI
    if not checker.check_claude_cli():
        print("\n⚠️  Please install Claude CLI first")
        return 1
        
    # Check required servers
    working, missing = checker.check_required_servers()
    
    if missing:
        print(f"\n⚠️  Missing required MCP servers: {', '.join(missing)}")
        checker.suggest_fixes(missing)
        
        # Offer to create config template
        response = input("\nCreate MCP config template? (y/n): ")
        if response.lower() == 'y':
            checker.create_config_template()
            
        return 1
    else:
        print("\n✅ All required MCP servers are available!")
        print("\nClaude Cadence is ready to use with:")
        print(f"  - Zen MCP for enhanced assistance")
        print(f"  - Task Master for task management")
        return 0
        

if __name__ == "__main__":
    sys.exit(main())