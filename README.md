# Agent

A modular, lightweight C/C++ AI agent library for edge devices and ultra-low latency

## Recent changes

- See [CHANGELOG.md](CHANGELOG.md) for the latest updates and project history.

## Description

The main goal of this project is to enable lightweight, ultra-low latency AI agent orchestration with minimal setup and zero-cost abstractions on a wide range of hardware — from edge devices to high-end servers.

- **Zero-Cost Abstractions**: Leveraging modern language features to provide a modular framework without performance penalties.
- **Zero Hot-Path Allocations**: Deterministic memory management via a session-based arena, preventing fragmentation and OOM on restricted hardware.
- **Universal Portability**: Self-contained core designed for everything from high-end servers to resource-constrained environments (WASM, RISC-V, ESP32, etc.).
- **Stable Binary Interface**: Flat C-ABI boundary for seamless integration with Python, Rust, Go, and other ecosystems.

## Contributing

- Contributions that align with the core philosophy of performance and portability are welcome!
- Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for technical mandates and coding standards.
- Refer to [AGENTS.md](AGENTS.md) for rules regarding the use of coding assistants in this repository.

## Security

- Security is a priority. Please report any vulnerabilities privately through the established channels.
- Review the [SECURITY.md](SECURITY.md) for details on the threat model and best practices for secure deployment.

## License

[MIT LICENSE](LICENSE)
