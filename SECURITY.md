# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it privately. **Do not disclose it as a public issue.** This allows us to work on a fix before public exposure.

Please report vulnerabilities via the private security advisory feature or by contacting the maintainers directly.

## Covered Topics

Vulnerabilities within the following areas are considered high priority:
- **Core Logic**: Memory safety issues in the Arena allocator or FSM executor.
- **Hardware Abstraction**: Vulnerabilities in the abstraction layer that could lead to privilege escalation or unauthorized system access.
- **Tool Execution**: Insecure handling of tool parameters that could lead to command injection.
- **Input Sanitization**: Failures to properly sanitize or validate outputs before action execution.

## Using the Framework Securely

### Untrusted Models & Inputs
Autonomous agents are inherently susceptible to **Prompt Injection**. This project provides the orchestration layer, but it is the developer's responsibility to:
- **Sandbox Tool Execution**: Always execute the agent in an isolated environment (e.g., a container or VM) when tools have access to the file system or network.
- **Validate Observations**: Sanitize all data returned from tools before feeding them back into the context.
- **Restrict Tool Access**: Use the principle of least privilege when registering tools.

### Data Privacy
Since the framework uses a contiguous Arena allocator, sensitive data from previous sessions could remain in memory if the arena is not explicitly zeroed out. While the default behavior is a pointer reset, production deployments should consider zero-filling the arena between sessions if handling high-sensitivity data.

### Multi-Tenant Environments
In multi-tenant environments, ensure strict isolation between memory arenas. Do not share arenas between different users or agents.
