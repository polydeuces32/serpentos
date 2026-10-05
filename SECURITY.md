# Security policy

SerpentOS is an embedded library. The host owns authentication, authorization,
action execution, resource limits, and storage permissions. See
[THREAT_MODEL.md](THREAT_MODEL.md) for boundaries and residual risks.

## Reporting

Do not place credentials, personal data, or exploitable details in public issues.
Use GitHub's private vulnerability reporting mechanism if enabled for this
repository. If it is unavailable, open a minimal public issue requesting a private
contact channel without disclosing exploit details or sensitive data. This policy
does not assert that private reporting is currently enabled.

Include affected versions, a minimal reproduction, impact, and proposed mitigation
through the private channel. No response-time guarantee or supported-version
security maintenance window has been established here.

## Engineering requirements

- Treat imported policies and audit files as untrusted data; preserve schema checks
  and the closed rule-operator set.
- Do not introduce `eval`, `exec`, `pickle`, or dynamic imports into serialized
  policy handling.
- Configure validators for permission-sensitive decisions. Missing validation
  intentionally accepts proposals and is not an authorization mechanism.
- Keep secrets out of contexts and metadata where possible; configure redaction
  and verify representative records before persisting them.
- Restrict audit/checkpoint paths, file permissions, retention, and disk usage
  in the host deployment. Do not assume logs are tamper-proof or encrypted.
- Review runtime dependencies and CI permission changes as security-sensitive work.
