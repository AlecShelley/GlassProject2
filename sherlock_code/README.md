# Sherlock Notes

## GitHub Push Workaround

Sherlock may not have a usable `ssh-agent`, and the system Git can fail to use
`GIT_SSH_COMMAND` reliably. If a normal `git push` fails but the project SSH key
authenticates with GitHub, use a temporary `GIT_SSH` wrapper.

Check the key:

```bash
ssh -i /home/users/ams01/.ssh/id_ed25519_glassproject2 -o IdentitiesOnly=yes -T git@github.com
```

Create the wrapper:

```bash
cat > /tmp/codex_glassproject2_ssh <<'EOF'
#!/bin/sh
exec ssh -i /home/users/ams01/.ssh/id_ed25519_glassproject2 -o IdentitiesOnly=yes "$@"
EOF
chmod 700 /tmp/codex_glassproject2_ssh
```

Push with the wrapper:

```bash
env GIT_SSH=/tmp/codex_glassproject2_ssh git push -u origin main
```

The wrapper contains only the key path and SSH options, not the private key.
It lives in `/tmp`, so recreate it when needed.
