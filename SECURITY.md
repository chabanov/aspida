# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in Aspida, please report it responsibly.

### How to Report

**Do NOT open a public issue.** Instead, please:

1. Email security reports to: [TBD - add security email]
2. Include "SECURITY" in the subject line
3. Provide:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Status Updates**: Every 7 days until resolved
- **Disclosure**: After fix is released (typically 90 days)

### Security Features

Aspida implements several security features:

- **End-to-End Encryption**: X25519 key exchange + ChaCha20-Poly1305 AEAD
- **SPARK Verification**: ChaCha20, SHA-256, HKDF & PBKDF2 proved to absence-of-runtime-errors + functional contracts (`make prove`); the rest of the crypto library is SPARK flow-analysed (`make prove-flow`). X25519/Poly1305/AEAD field-arithmetic proofs are tracked as future work.
- **Constant-Time Operations**: Timing-safe comparisons, secure memory wiping
- **Forward Secrecy**: Ephemeral key exchange per session
- **No Third-Party Crypto**: All cryptographic primitives implemented from scratch

### Known Security Considerations

1. **Model Files**: GGUF weights are not encrypted at rest. Set `ASPIDA_STORE_PASSWORD` to enable PBKDF2/ChaCha20-Poly1305 encryption of persisted *session history* (not weights); rely on host disk encryption (LUKS/FileVault) for weight confidentiality.

2. **Client Tokens**: `ASPIDA_CLIENT_TOKEN` should be kept secret. Do not commit to version control.

3. **GPU Memory**: GPU offload may expose model weights in memory. Consider threat model before enabling.

4. **Network**: Default configuration binds to `0.0.0.0`. Use `ASPIDA_BIND=127.0.0.1` for local-only access.

### Security Best Practices

1. **Authentication**: Always set `ASPIDA_CLIENT_TOKEN` in production
2. **TLS**: Use a reverse proxy (nginx, Caddy) for TLS termination
3. **Firewall**: Restrict access to trusted IPs
4. **Updates**: Keep up to date with security patches
5. **Logs**: Monitor for suspicious activity

### Cryptographic Primitives

All cryptographic implementations follow published standards:

| Primitive | Standard | Reference |
|-----------|----------|-----------|
| ChaCha20 | RFC 8439 | https://tools.ietf.org/html/rfc8439 |
| Poly1305 | RFC 8439 | https://tools.ietf.org/html/rfc8439 |
| X25519 | RFC 7748 | https://tools.ietf.org/html/rfc7748 |
| HKDF | RFC 5869 | https://tools.ietf.org/html/rfc5869 |
| PBKDF2 | RFC 8018 | https://tools.ietf.org/html/rfc8018 |
| SHA-256 | FIPS 180-4 | https://csrc.nist.gov/publications/detail/fips/180/4/final |

### Security Audits

Aspida has not yet undergone a formal security audit. We welcome security researchers to review the code.

### Disclosure Policy

- We follow responsible disclosure
- CVEs will be assigned for confirmed vulnerabilities
- Security advisories will be published on GitHub Security tab
- Credit will be given to reporters (unless anonymity requested)

---

Thank you for helping keep Aspida secure! 🔒
