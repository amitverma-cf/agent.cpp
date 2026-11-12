# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it privately. **Do not disclose it as a public issue.** This allows us to work on a fix before public exposure.

Report it privately with GitHub's private security advisory form: https://github.com/amitverma-cf/agent.cpp/security/advisories/new

Include the version or commit, the platform, the engine and a way to reproduce the problem. Do not open a public issue or pull request for it.

## Covered Topics

Vulnerabilities within the following areas are considered high priority:
- **Core Logic**: Memory safety issues in the Arena allocator, turn executor or graph runtime.
- **Hardware Abstraction**: Vulnerabilities in the abstraction layer that could lead to privilege escalation or unauthorized system access.
- **Tool Execution**: Insecure handling of tool parameters that could lead to command injection.
- **Input Sanitization**: Failures to properly sanitize or validate outputs before action execution.

## Using the Framework Securely

### Untrusted Models & Inputs
Autonomous agents are inherently susceptible to **Prompt Injection**. This project provides the orchestration layer, but it is the developer's responsibility to:
- **Sandbox Tool Execution**: Always execute the agent in an isolated environment (e.g., a container or VM) when tools have access to the file system or network.
- **Validate Observations**: Sanitize all data returned from tools before feeding them back into the context.
- **Restrict Tool Access**: Use the principle of least privilege when registering tools.
- **OS sandbox**: on Linux the engine host and `run_command` run under Landlock + seccomp + rlimits by default (`Config::engine_sandbox`, `Config::tool_sandbox`), MCP servers opt in per server. It narrows what a compromised child can touch but is not a substitute for a container or VM; on Windows, MCP servers and `run_command` run in an AppContainer (tested on Windows 11), and engine hosts get a Job Object only. The macOS sandbox profile has never been run. See [docs/security-sandbox.md](docs/security-sandbox.md).

### Data Privacy
Since the framework uses a contiguous Arena allocator, sensitive data from previous sessions could remain in memory if the arena is not explicitly zeroed out. While the default behavior is a pointer reset, production deployments should consider zero-filling the arena between sessions if handling high-sensitivity data.

### Multi-Tenant Environments
In multi-tenant environments, ensure strict isolation between memory arenas. Do not share arenas between different users or agents.
