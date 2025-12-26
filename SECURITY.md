# Security Policy

## Supported Versions

VecStore is currently in **alpha** (0.0.x). Security updates are provided for the latest release only.

| Version | Supported          |
| ------- | ------------------ |
| 0.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of these methods:

1. **GitHub Security Advisories** (Preferred): Use the "Report a vulnerability" button on the [Security tab](https://github.com/PhilipJohnBasile/vecstore/security/advisories/new)

2. **Email**: Send details to the repository maintainer (see GitHub profile)

### What to Include

Please include as much of the following information as possible:

- Type of vulnerability (e.g., buffer overflow, SQL injection, XSS)
- Full paths of source file(s) related to the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue and how an attacker might exploit it

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Updates**: We will provide updates on the status within 7 days
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days
- **Credit**: We will credit reporters in the release notes (unless you prefer anonymity)

## Security Considerations

### Data at Rest

- VecStore stores vectors and metadata on disk in binary format
- No built-in encryption at rest (encrypt the storage directory if needed)
- File permissions follow system defaults

### Network (Server Mode)

When using the optional `server` feature:

- gRPC and HTTP endpoints have no built-in authentication
- Deploy behind a reverse proxy with TLS for production use
- Use network-level access controls (firewall, VPC)

### Dependencies

- We regularly update dependencies to patch known vulnerabilities
- Run `cargo audit` to check for known vulnerabilities in dependencies

## Security Best Practices

1. **Keep Updated**: Always use the latest version
2. **Access Control**: Restrict file system access to the data directory
3. **Network Security**: Never expose server endpoints to the public internet without authentication
4. **Backups**: Use snapshots for data recovery, not as a security measure

## License

This security policy is part of the VecStore project, licensed under Apache 2.0.
