# Hosted BYOK native qualification authority

The accepted production SDK source is commit
`39d51380f6a3c1e923f0f72dd94dcee3a176693c`, tree
`a69e8c0f727e7ca8986722fb1d5218b9cb7d2a60`. A qualification commit must have
that exact commit as its sole parent and must change exactly these five paths:

- `packages/percept-vision/Dockerfile.hosted-byok-native`
- `packages/percept-vision/HOSTED_BYOK_NATIVE_QUALIFICATION.md`
- `packages/percept-vision/tests/test_hosted_byok_native_authority.py`
- `packages/percept-vision/tools/hosted_byok_native_matrix.py`
- `packages/percept-vision/tools/qualify_hosted_byok_native.py`

The qualification commit does not modify the production SDK. It is not a
wheel, native result, or release authorization.

## Trust boundary

Code in the SDK repository cannot safely authenticate itself before it runs as
root. Therefore the checked-in controller is a reference payload for the
already reviewed Hosted release authority, not a developer-side root launcher.
The separate authority must review and pin the exact qualification commit/tree,
then perform all of these steps from protected configuration:

1. Fetch that exact object into the root-owned, non-mutable bare repository
   `/var/lib/percept-native-authority/source.git`.
2. Materialize the exact reviewed controller, recipe and runner at:
   `/opt/percept-native-authority/v1/qualify_hosted_byok_native.py`,
   `/opt/percept-native-authority/v1/Dockerfile.hosted-byok-native`, and
   `/opt/percept-native-authority/v1/hosted_byok_native_matrix.py`.
3. Install `/etc/percept/native-authority-v1.json` as a root-owned mode-0600
   policy. The policy pins the qualification object and all three payload
   digests, exact Git/Python/Docker/socket identities, exact daemon ID/version/
   root, and the protected runner identity.
4. Verify the payload and interpreter before execution, then invoke exactly:

   ```console
   /usr/bin/env -i \
     PATH=/usr/bin:/bin LC_ALL=C LANG=C HOME=/var/empty \
     TMPDIR=/var/lib/percept-native-authority/work \
     /usr/bin/python3 -I -S \
     /opt/percept-native-authority/v1/qualify_hosted_byok_native.py
   ```

5. Run on a dedicated Linux/amd64 authority with an ephemeral Docker daemon.
   Destroy that daemon and runner after every result, signal, process crash,
   `SIGKILL`, or host loss. A local or shared developer daemon is forbidden.

The controller accepts no arguments and no environment beyond the exact set
above. Source, result, tool paths, socket, daemon, temp root, image, tag,
container, and evidence path are not caller inputs. Direct execution from a
checkout exits 2 with `status=not_executed`; stdout is diagnostic and is never
qualification evidence.

## Root policy

The policy has the following closed schema. Placeholders are supplied by the
protected authority only after independent acceptance of an immutable
qualification object:

```json
{
  "schema": 1,
  "qualification": {
    "commit": "<40 lowercase hex>",
    "tree": "<40 lowercase hex>",
    "controller_sha256": "<64 lowercase hex>",
    "recipe_sha256": "<64 lowercase hex>",
    "runner_sha256": "<64 lowercase hex>"
  },
  "accepted_production": {
    "commit": "39d51380f6a3c1e923f0f72dd94dcee3a176693c",
    "tree": "a69e8c0f727e7ca8986722fb1d5218b9cb7d2a60"
  },
  "tools": {
    "git": {
      "path": "/usr/bin/git",
      "sha256": "<64 lowercase hex>",
      "uid": 0,
      "gid": 0,
      "mode": 493,
      "device": 1,
      "inode": 1
    },
    "python": {
      "path": "/usr/bin/python3",
      "sha256": "<64 lowercase hex>",
      "uid": 0,
      "gid": 0,
      "mode": 493,
      "device": 1,
      "inode": 1
    },
    "docker": {
      "path": "/usr/bin/docker",
      "sha256": "<64 lowercase hex>",
      "uid": 0,
      "gid": 0,
      "mode": 493,
      "device": 1,
      "inode": 1
    },
    "socket": {
      "path": "/run/docker.sock",
      "sha256": "",
      "uid": 0,
      "gid": 0,
      "mode": 384,
      "device": 1,
      "inode": 1
    }
  },
  "daemon": {
    "id": "<exact protected daemon ID>",
    "version": "<exact protected daemon version>",
    "docker_root_dir": "<exact protected root directory>"
  },
  "authority": {
    "runner_id": "<retained protected runner identity>",
    "ephemeral_daemon": true,
    "destroy_on_every_outcome": true
  }
}
```

Every path component is rejected if it is a symlink, not root-owned, or
group/world writable. Tool and socket device/inode/mode/digest identities are
rechecked around commands. A group-writable Docker socket is not accepted.

## Native proof

The authority constructs the context from exact Git blobs in the protected
bare repository. The recipe is byte-locked, uses a Python 3.12 Linux/amd64 base
by manifest digest, contains no `RUN` instruction, and builds with no network,
no cache and no pull.

The matrix runs exactly:

- Gemini: success, deadline, cancellation
- OpenRouter: success, deadline, cancellation

The container has a private PID namespace, no network, read-only root,
non-root UID/GID, no capabilities, `no-new-privileges`, and fixed
PID/memory/CPU/tmpfs limits. After every case, the runner:

- observes that every PID except itself has exited;
- waits 2.75 seconds, beyond the deliberate 2.25-second fake late effect;
- observes the PID namespace is still empty; and
- rejects every late-effect file.

Those observations—not constants—appear in the one-line bounded JSON result.
The controller independently rejects skipped, duplicate, reordered or forged
cases and verifies exact image labels/platform/entrypoint and container
resources.

Docker operations use a unique name and exact container ID. Catchable signals
terminate the client process group; all paths force-remove and then prove the
container and image absent. A successful evidence file is created mode 0600
with no-follow/exclusive creation and file/directory `fsync`, only after
terminal container inspection and joined cleanup. `SIGKILL` and daemon-side
build ambiguity are closed by destroying the dedicated ephemeral daemon; a
shared daemon cannot satisfy this contract.

The only controller-record path is
`/var/lib/percept-native-authority/pending/<qualification-commit>.json`. Its
status is `controller_passed_pending_authority_teardown`, deliberately not
`passed`. After copying and hashing that pending record, the external authority
must destroy the runner and daemon and emit its own protected final envelope
containing the exact pending-record SHA-256, qualification commit/tree, runner
identity, daemon identity, and authenticated destruction result. Only that
external envelope may use `status=passed`.

No final external envelope means no native qualification. The final wheel may
be stamped only after a different reviewer validates both records and binds
them to the reviewed qualification object and accepted production SDK.
