# Privacy and security model

LCM is intentionally local infrastructure.

- It binds to `127.0.0.1` by default.
- Backend URLs are restricted to loopback addresses, RFC-private IP addresses, `localhost`, and `.local` hosts.
- `httpx` environment proxy discovery is disabled for inference requests, preventing an inherited corporate or system proxy from redirecting local-model traffic.
- There is no account system, analytics SDK, telemetry endpoint, update checker, CDN asset, web font, or cloud-model fallback.
- The dashboard Content Security Policy limits network connections to the same origin and local loopback hosts.
- LCM mutates only the transient payload sent to inference. It cannot rewrite a frontend's stored conversation.
- Backend API tokens, when used, remain in the local SQLite database and are never returned by dashboard APIs.

## Local data

The default database is `data/lcm.sqlite3`. The dashboard exposes session deletion and full local-data deletion. Request snapshots are bounded per session. Large data is not copied indefinitely.

## Threat boundary

LCM is not an authentication gateway. If the bind host is changed from `127.0.0.1`, the operator is responsible for host firewall rules and access control. Do not expose port 7331 to an untrusted network.

## Reporting

Please report security issues privately to the project maintainer before public disclosure.
