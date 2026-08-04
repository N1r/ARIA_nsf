# Security policy

## Supported version

PhonLab-DDSP is a `0.1.x` research preview. Security fixes target the latest
source revision; there is no long-term support branch yet.

## Reporting a vulnerability

Do not publish credentials, private recordings, cluster paths, or a working
exploit in a public issue. Before publishing this repository, replace this
paragraph with the maintainer's private security contact or enable GitHub
private vulnerability reporting. Include the affected revision, reproduction
steps, impact, and whether untrusted files or network access are required.

## WebUI threat model

The WebUI is a loopback-only, single-user research workbench. It has the file
permissions of its Unix user and can submit Slurm jobs after explicit
confirmation. It has no accounts, tenant isolation, TLS termination, or
protection suitable for direct Internet exposure.

- Bind it only to `127.0.0.1`; use an SSH tunnel remotely.
- Never expose it through `0.0.0.0` on a shared or Internet-facing host.
- Set `--workspace` to the repository or a narrower trusted directory.
- Treat third-party checkpoints and generated HTML as untrusted input.
- Never place private recordings or tokens in issues or release assets.

The server rejects workspace escapes and symlink-based serving, limits exports,
and refuses silent overwrites. These controls do not make it a public web app.
