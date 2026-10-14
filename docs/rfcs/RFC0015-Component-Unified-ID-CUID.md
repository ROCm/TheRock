---
author: Gabriel Pham (gabrpham)
created: 2026-09-25
modified: 2026-09-28
status: draft
discussion: https://github.com/ROCm/rocm-systems/pull/12075
---

# RFC0015: Component Unified ID (CUID)

## Overview

Platform management tools (AMD-SMI, debuggers, profilers, schedulers, fleet
inventory) each identify a GPU, CPU, NIC or platform in their own
driver-specific way, which makes it hard to correlate a single physical
component across tool chains, reboots, or a live migration. The Component
Unified ID (CUID) is a 128-bit identifier, encoded as a UUIDv8, that gives
every architecturally identifiable component in a system one consistent name
regardless of which tool reads it.

CUID defines two related values for a component:

- **PrimaryID**: derived directly from hardware-provided properties (PCIe
  Device Serial Number, ACPI/SMBIOS UUID, vendor/device IDs, etc.). It is
  reversible to those properties and is therefore only exposed to privileged
  software.
- **DerivedID**: an HMAC-SHA-256 hash of the PrimaryID under a per-host
  secret key. It cannot be reversed to recover the PrimaryID and is the value
  every non-privileged consumer (schedulers, telemetry, `amd-smi`) actually
  uses.

This RFC documents the CUID design as of the 2026-09-23 design sync, which
replaced the earlier key model (a RAM-only shared seed with a public default)
with a driver-owned key sealed in a UEFI variable, and the current reference
implementation in the Linux `amdgpu` driver, `libamdcuid`, and `amd-smi`.

## Motivation / Goals

- Give every physical or hypervisor-projected component (GPU, GPU spatial
  partition, CPU package, NIC, platform) one stable identifier, usable
  identically on bare metal and inside a VM.
- Let the identifier persist across reboots, driver reloads, GPU
  unbind/rebind, and live migration, without requiring a central
  identity-tracking service.
- Hide the underlying hardware serial numbers from non-privileged consumers
  while still letting privileged tools recover the PrimaryID when needed.
- Support both OS environments where the full hardware property set is
  accessible and constrained environments (containers, limited-privilege
  guests) where it is not, via a clearly-marked fallback identifier.

## Non-Goals

- CUID is not a cryptographic attestation of a device's authenticity; that is
  the province of mechanisms such as DEVSEC.
- CUID does not replace existing driver-specific enumeration (PCIe BDF, DRM
  minor numbers, etc.); it augments it with a stable cross-tool name.
- The auxiliary/fallback identifier is explicitly **not** guaranteed unique
  across nodes; it is node-local only.

## Terminology

| Term | Meaning |
| --- | --- |
| PrimaryID | Reversible identifier built from hardware properties. Root/privileged access only. |
| DerivedID | HMAC-SHA-256 of the PrimaryID under the host key. Non-reversible, readable by anyone. |
| Constructed CUID | A CUID built from the 122-bit payload defined below and framed as UUIDv8 (version nibble `8`). Fully decodable. |
| Adopted CUID | A firmware-provided UUID (SMBIOS system UUID, ACPI device-object UUID) used verbatim, keeping its own version/variant bits. Opaque, not decoded as a CUID payload. |
| Canonical CUID | A PrimaryID/DerivedID pair whose serial comes from an architectural hardware source (DSN, PPIN, adopted UUID, etc.). Globally non-colliding. |
| Auxiliary (temp) CUID | A value computed from the OS machine identity and a routing ID when no architectural serial is available. Node-local only. Per the reference algorithm below it is an opaque truncated HMAC digest with only the UUID version, variant, and bit 117 patched in — it does not decode to a payload the way a canonical CUID does. |
| UnitID | 13-bit sub-component index (e.g., a GPU spatial partition) within a physical device. |
| Source | Which layer answered a lookup: `DRIVER` or `LIBRARY`. |

## Requirements

- **Compatibility**: 128-bit size, represented as UUIDv8 (custom UUID per
  RFC 9562), to minimize churn in existing UUID-based tool chains.
- **Generality**: usable for any component identifiable in software from its
  architectural hardware properties, on bare metal or as a hypervisor-projected
  device in a guest.
- **Consistency**: the same deterministic algorithm applies unchanged to bare
  metal and to a projected virtual function.
- **Portability**: implementable on any OS or system environment.
- **Non-reversibility**: the DerivedID cannot be used to recompute the
  PrimaryID.
- **Scalability**: a canonical CUID never collides with any other component in
  a system, network, or cloud environment. An auxiliary CUID carries no such
  guarantee — it is unique only within the node, and bit 117 lets a consumer
  tell the two apart before relying on either.
- **Persistence**: a DerivedID stays constant for as long as the host's key is
  unchanged. Re-keying is an explicit administrative operation that
  invalidates every DerivedID handed out so far; the producer is expected to
  keep a primary-to-derived association log so earlier values stay traceable
  (important for live migration).

## Design

### Payload format

Primary and Derived identifiers share a 122-bit payload, handled as 16
octets packed LSB-first (payload bit *b* is bit `b mod 8` of octet `b / 8`;
multi-octet fields are little-endian; payload bits 122:127 are always zero).
Those 16 octets are both what gets framed into the UUID and the HMAC message
used to compute a DerivedID.

The 122-bit payload is placed into a UUIDv8 by inserting (not overwriting)
the version (`8`) and variant (`10b`) bits, so every payload bit survives and
a constructed CUID decodes back to its inputs exactly:

```
U[0..5] = P[0..5]
U[6]    = 0x80 | (P[6] >> 4)                        # version 8
U[7]    = ((P[6] & 0x0f) << 4) | (P[7] >> 4)
U[8]    = 0x80 | ((P[7] & 0x0f) << 2) | (P[8] >> 6)  # variant 10
U[i]    = ((P[i-1] & 0x3f) << 2) | (P[i] >> 6)         # i = 9..14
U[15]   = ((P[14] & 0x3f) << 2) | (P[15] & 0x03)
```

`P[0..15]` are payload octets and `U[0..15]` are the rendered UUID octets in
printed order. Example: payload `d4abaad39b34c5060000a37302108000` renders as
`d4abaad3-9b34-8c50-9800-028dcc084200`.

An **adopted** CUID (a firmware-provided UUID) is used as-is: it is never
re-framed through this formula, since doing so would drop 6 of its bits.

### PrimaryID

| Bit | Name | Comments |
| --- | --- | --- |
| 0:63 | Serial number | PCIe DSN, platform/CPU serial, or SW fingerprint; zero-extended or truncated to 64 bits |
| 64:71 | UnitID (low 8 bits) | Sub-component index; 0 if the component doesn't have one |
| 72:79 | RevisionID | 0 if not supported |
| 80:95 | DeviceID | From PCIe config space / SMBIOS; CPU Family+Model for CPUs |
| 96:111 | VendorID | 16-bit PCIe vendor ID (e.g., `0x1002`/`0x1022` for AMD) |
| 112:116 | UnitID (high 5 bits) | Bits 8:12 of UnitID; 0 if UnitID < 256 |
| 117 | Auxiliary Value Identifier | 1 if the serial is a synthesized auxiliary value, 0 otherwise |
| 118:121 | Component Type | See table below |

UnitID is **13 bits total** (max `0x1FFF` = 8191), split across bits 64:71
and 112:116. A producer refuses a larger value rather than masking it onto
another component. Bit 117 belongs to the Auxiliary Value Identifier in both
the Primary and Derived layouts, so a large UnitID can never be mistaken for
the auxiliary flag.

Component types (selected):

| Type | Component | Notes |
| --- | --- | --- |
| 0 | Platform | VendorID = OEM/SI |
| 1 | CPU | UnitID 0 (per package, not per core); RevisionID = CPUID stepping; serial = PPIN or equivalent |
| 2 | GPU | UnitID 0 = whole device; spatial partitions get an XCC-range UnitID (see below) |
| 3 | NIC | Falls back to MAC address if no DSN |
| 4 | NPU | |
| 5 | Storage | |
| 6 | Memory | |
| 7 | GenPCIe | Any other PCIe/CXL device |
| 8 | GenC | Non-PCIe SW-visible component |
| 9 / 0xA | RackTray / Rack | |
| 0xF | Other | |

### DerivedID

`DerivedID = HMAC-SHA-256(key = host salt, message = the 16 PrimaryID payload octets)`.

The digest is folded into the DerivedID payload as follows:

| Bit | Name | Comments |
| --- | --- | --- |
| 0:63 | hash bits 0:63 | |
| 64:71 | Reserved | must be 0 |
| 72:116 | hash bits 64:108 (45 bits) | |
| 117 | Auxiliary Value Identifier | copied from the PrimaryID |
| 118:121 | Reserved | must be 0 |

The DerivedID therefore carries **109 hash bits**; the remaining 13 bits are
reserved or the auxiliary flag and add nothing to collision resistance. Two
DerivedIDs collide with probability $2^{-109}$, and by the birthday bound a
population of $n$ components collides with probability of order
$n^2/2^{110}$ — negligible at any realistic fleet size.

A provisioned key (the "salt") is **exactly 32 octets**; any other length is
refused and the previously effective key stays in use.

### Worked example

Radeon PRO W6800, serial `0x06C5349BD3AAABD4`, UnitID 0, Rev `0x00`, Device
`0x73A3`, Vendor `0x1002`, type GPU:

| | Payload octets | UUID |
| --- | --- | --- |
| Primary | `d4abaad39b34c5060000a37302108000` | `d4abaad3-9b34-8c50-9800-028dcc084200` |

The full set of normative conformance vectors (Primary, Derived, Auxiliary,
and serial-source vectors) is maintained as a single generated file shared
byte-for-byte between the Linux kernel selftests and `libamdcuid`; every
producer's build fails if its copy drifts from that file.

## Key management: `AmdCuidKey`

The DerivedID's key ("salt") lives in exactly one place per host: a UEFI
variable, not a daemon, key file, or public default.

| Field | Value |
| --- | --- |
| Name | `AmdCuidKey` |
| Vendor GUID | `e41c1f7f-63cb-46b9-bf27-36a55f92a06d` |
| Attributes | `NON_VOLATILE \| BOOTSERVICE_ACCESS \| RUNTIME_ACCESS`, never authenticated |
| Payload | 36 octets: version (1) · flags (bit 0 = administrator-provisioned) · 2 reserved octets (zero) · 32-octet key |

- The `amdgpu` kernel driver reads or creates this variable once per module
  lifetime, at first CUID registration. If absent, it generates 32 random
  octets and writes them back before using them (`unprovisioned` state).
- An administrator replaces the key via `amd-smi set --cuid-seed <file|->`,
  which writes `cuid_seed` through the driver (or directly to efivarfs if
  `amdgpu` isn't loaded), marking the key `provisioned`. The variable is
  written before the new key takes effect; a failed write changes nothing.
  Weak keys (all-equal bytes, or known public/test constants) are refused.
- Re-keying re-derives every DerivedID on the node, including for GPUs that
  bind later — this is an intentional, explicit administrative action, not a
  routine occurrence.
- Without a usable key (`efi=noruntime`, `CONFIG_EFI_DISABLE_RUNTIME`, a
  hypervisor without a runtime variable store, or non-UEFI boot), the driver
  publishes **no** CUID for that device; there is no public default to fall
  back to. In this case, only an Auxiliary CUID can be created by the Library
  which does not provide all the gaurantees of a full CUID. See
  [Auxiliary (temp) CUID](#fallback-auxiliary-temp-cuids) in the glossary.
- `efivarfs` currently exposes `RUNTIME_ACCESS` variables at `0644`, which
  would let any local user read the key. A kernel patch to create CUID-GUID
  entries at `0600` is drafted but not yet upstream; in the interim, a
  distro/kernel-shipped `tmpfiles.d` rule chmods the variable after boot.

This replaces an earlier design based on a RAM-only shared seed with a public
default (`AMD-CUID-DEFAULT-SEED-v1`); that default and any daemon/key-file
component are withdrawn.

## GPU spatial partition identity

Each active spatial partition (SPX/DPX/QPX/CPX) publishes its own CUID, using
the **whole device's serial** and a UnitID computed from its logical XCC
range rather than a lookup table:

```
UnitID = (hweight(m) << 6) | __ffs(m)
```

where `m` is the partition's contiguous logical XCC mask: bits 0:5 are the
first logical XCC (0..63), bits 6:11 are the XCC count (1..63), and bit 12 is
set only for SLC mode (SLC = the SPX range | `0x1000`). On 8 XCCs:

| Mode | Partitions | UnitID |
| --- | --- | --- |
| SPX | 1 | `0x200` |
| DPX | 2 | `0x100`, `0x104` |
| QPX | 4 | `0x080`, `0x082`, `0x084`, `0x086` |
| CPX | 8 | `0x040`..`0x047` |

The whole device keeps UnitID 0; a partition's UnitID is never 0 since the
count is always at least 1, so the two are always distinguishable. Memory
partition mode (NPS1/2/4) does not enter the identity. A non-contiguous XCC
mask is refused (no CUID for that partition, with a driver warning) rather
than silently accepted. The closed form above lets a tool outside the driver
compute the same UnitIDs without reading a kernel table; a decimal
`cuid_unit_id` sysfs file is also published for convenience. This design was
checked against an earlier "buddy table" proposal and kept in preference to
it because the buddy table cannot distinguish device-0/SPX, does not cover
SLC or 6-XCC/harvested parts, and is not computable without the driver.

A CPU, by contrast, is identified per package (UnitID 0) with the physical
package ID as its auxiliary routing ID — cores are not separate components.

## Partition selection and CUID propagation

Selecting a spatial partition and then obtaining its CUID spans several
software layers. This flow is described here using this RFC's
PrimaryID/DerivedID terminology in place of the "primary/secondary CUID"
naming used in the source proposal ("Unit ID Definition in CUID", 11/25/2025).
The only two authoritative sources of a CUID are the driver's sysfs files
(`cuid_primary`/`cuid_derived`/`cuid_unit_id`, per device and per partition;
see Reference implementation) and `libamdcuid`'s privileged cache file;
every consumer below ultimately reads one of those two.

- **KFD** maintains, per supported GPU model, a standard table mapping
  UnitID to a set of XCC partitions, exposed through a sysfs entry so every
  software layer can look up valid partitions for a device. It exposes an
  ioctl that takes a UnitID and a file handle and creates the corresponding
  partition and memory interleaving.
- **Partition selection** is driven by an environment variable: a user picks
  a UnitID from the sysfs table and sets it in the environment; **ROCr**
  reads that variable and forwards the partition request to KFD's ioctl.
- Once a partition exists, **ROCr** reads its DerivedID directly from the
  partition's sysfs `cuid_derived` file (or, when it needs every partition's
  identity ahead of time, from `libamdcuid`'s cache), reports it to tools
  that have registered for device DerivedIDs, and exposes its own query
  interface for the DerivedID of any device visible to it.
- **HIP** queries DerivedIDs through ROCr's interface; HIP's existing tools
  table returns a DerivedID as a 128-bit hex string and documents it as an
  opaque value (callers must not attempt to interpret its bits).
- **`libamdcuid`**, running with privilege, is the second authoritative
  source alongside sysfs: it precomputes the DerivedID for every UnitID a
  device supports (from the sysfs UnitID table) using the host key and the
  PrimaryID for that UnitID, and caches the results, alongside their UnitID,
  in an unprivileged-readable file (`/var/lib/amdcuid/cuid` in the original
  proposal), plus an API for unprivileged callers to look up a cached
  DerivedID by file handle and UnitID.
- **AMD-SMI** honors the same partition-selection environment variable ROCr
  reads, and reports device information for the selected partitions.
- **Debuggers** read DerivedIDs directly from sysfs or `libamdcuid`'s cache.
  An in-process **profiler** can use ROCr's interface; an attached
  (out-of-process) profiler instead queries KFD directly through an ioctl
  that returns the UnitID of a queue, then looks that UnitID up in sysfs or
  the cache.
- **Out-of-band (OOB) tooling** receives the host key through a UEFI
  protocol rather than through `libamdcuid`, then uses the same standard
  UnitID table to regenerate the PrimaryID, and from it the DerivedID, for
  every allowed partition of a device — matching the values already
  published through sysfs and the cache.

## Fallback: auxiliary (temp) CUIDs

When no architectural serial is available (no PCIe DSN, no vendor-specific
capability, or the OS/privilege context can't reach one), user-mode software
computes a **temporary derived CUID** instead. Unlike a canonical
Primary/DerivedID pair, no primary value is ever exposed for a temporary
identity — the input structure below is used purely as HMAC key material.

### AMD UUID namespace and CUID service application ID

Two fixed constants anchor every temporary derived CUID, independent of the
device:

```python
import uuid
amd_domain = "amd.com"
amd_uuid_namespace = uuid.uuid5(uuid.NAMESPACE_DNS, amd_domain.lower().encode("idna")).bytes
cuid_application_id = uuid.uuid5(amd_uuid_namespace, "com.amd.cuid.v1").bytes
```

- `amd_uuid_namespace`: a UUIDv5 ([RFC 9562 §5.5](https://www.rfc-editor.org/info/rfc9562/#section-5.5))
  built from the DNS namespace ([RFC 9562 §6.6](https://www.rfc-editor.org/info/rfc9562/#section-6.6))
  and the domain `amd.com`.
- `cuid_application_id`: a UUIDv5 built from `amd_uuid_namespace` and the
  reverse-DNS name `com.amd.cuid.v1`. This 16-byte value is the same for
  every device, on every host, for a given CUID service version.

### Temporary primary ID (HMAC key material, never exposed)

The temporary primary ID depends on the machine ID (from `/etc/machine-id`,
32 hex characters decoded to 16 bytes) and the device type. It is **never**
exposed by the CUID service — unlike the canonical PrimaryID, which
privileged callers can read as `cuid_primary` — it exists only to key the
HMAC below.

For a PCIe device:

| Bit | Name | Comments |
| --- | --- | --- |
| 0:16 | Format | Always 1 for a PCIe device |
| 17:143 | Machine ID | From `/etc/machine-id` |
| 144:175 | PCIe Address | Domain:Bus:Device:Function (BDF), from PCIe config space |
| 176:183 | Revision ID | From PCIe config space |
| 184:199 | Device ID | From PCIe config space |
| 200:215 | Vendor ID | From PCIe config space |
| 216:219 | Component type | 2 = GPU, 3 = NIC, 4 = NPU |
| 220:255 | Reserved | Must be 0 |

For a CPU:

| Bit | Name | Comments |
| --- | --- | --- |
| 0:16 | Format | Always 2 for a CPU device |
| 17:143 | Machine ID | From `/etc/machine-id` |
| 144:175 | Physical Package ID | The CPU's physical package ID (zero on socket 0), so two sockets of one host differ |
| 176:183 | Revision ID | From PCIe config space |
| 184:199 | Device ID | From PCIe config space |
| 200:215 | Vendor ID | From PCIe config space |
| 216:219 | Component type | Always 1 for CPU |
| 220:255 | Reserved | Must be 0 |

### Temporary derived CUID

```python
import hmac
import hashlib

digest = hmac.new(key=temporary_primary_id, msg=cuid_application_id, digestmod=hashlib.sha256).digest()

# Truncate to first 16 bytes.
b = bytearray(digest[:16])
b[6] = (b[6] & 0x0F) | 0x80    # UUID version 8
b[8] = (b[8] & 0x3F) | 0x80    # UUID variant 10b (RFC 9562)
b[14] = (b[14] & 0x20) | 0x20  # bit 117: temporary derived CUID

device_temporary_derived_cuid = bytes(b)
```

The HMAC key is the device's temporary primary ID (the PCIe or CPU structure
above); the message is the fixed `cuid_application_id` — the opposite of the
key/message roles used for a canonical DerivedID, where the key is the host
salt and the message is the device's PrimaryID payload. Differentiation
between devices therefore comes entirely from the key.

Unlike a canonical Primary/DerivedID, where the version and variant bits are
*inserted* around an otherwise-untouched 122-bit payload (see Payload format
above), a temporary derived CUID *overwrites* 3 bits of a truncated HMAC
digest in place ([UUIDv8, RFC 9562 §5.8](https://www.rfc-editor.org/info/rfc9562/#section-5.8)).
It does not decode back to any payload; only its version (`8`), variant
(`10b`), and bit 117 (set to `1` here; `0` for a non-temporary DerivedID) are
meaningful.

Where no machine identity is available, the producer does **not** emit a
temporary derived CUID; it reports an error. A zero machine ID would
otherwise give identically-configured hosts the same temporary CUID for
physically different parts. Temporary CUIDs are also node-local by
construction and unreliable in containers: a container with no machine-id
gets none at all, and one whose image bakes in a machine-id gets the same
temporary CUID in every container started from that image, on every host.
Containers should read the driver-published CUID through sysfs instead of
computing a temporary one.

## Component-specific identification sources

- **PCIe/CXL devices**: the PCIe Device Serial Number Extended Capability
  (two little-endian dwords, read as-is without byte-swapping). A DSN that
  reads as all-zero is treated as absent, not as a valid identity, since an
  unimplemented capability also reads as zero.
- **NIC**: falls back to the permanent MAC address (preferred over an
  administratively-assigned one) when no DSN is present; an all-zero MAC is
  absent.
- **CPU**: prefers an architectural serial (PPIN on AMD, enabled via SBIOS,
  gated by `CPUID Fn8000_0008.EBX[23]`); otherwise falls back to
  Family/Model/Stepping plus the SMBIOS platform ID as a proxy.
- **Platform**: the SMBIOS System UUID (Type 1, offset `08h`) is used
  directly as an **adopted** CUID, kept verbatim with its own version/variant
  bits; its DerivedID uses the UUID's 16 octets as the HMAC message.
- **Virtual Functions (SR-IOV)**: never publish a CUID from the driver. A
  hypervisor may project a VM-specific fingerprint into the guest, which the
  guest OS then treats as a PrimaryID exactly as it would on bare metal.

## Reference implementation

| Layer | Responsibility |
| --- | --- |
| `amdgpu` kernel driver | Loads/creates `AmdCuidKey` once per module lifetime; publishes, per GPU, `cuid_primary` (0400 + `CAP_SYS_ADMIN`), `cuid_derived` (0444), `cuid_seed` (0600 + `CAP_SYS_ADMIN`), `cuid_seed_state`, and `cuid_unit_id` (0444); the same set minus `cuid_seed` per spatial partition. Publishes nothing without a key, and nothing on SR-IOV VFs. |
| `libamdcuid` (ROCm `shared/cuid`) | Answers GPU lookups from the driver (source `DRIVER`); computes CPU/NIC/platform identities (source `LIBRARY`) using the key from `cuid_seed` or efivarfs when the caller is root, else an auxiliary CUID. Ships as a static library (`libamdcuid_static.a`) with a CMake package (`amdcuid::amdcuid`); no CLI of its own. `amdcuid_set_hash_key()` sets the key through the driver or directly through efivarfs. |
| `amd-smi` | Administrator surface: `set --cuid-seed <file\|->` to set the node key; `node --cuid` / `static --cuid` to list every component's CUID with its `source`, whether it is auxiliary, and key state, via `amdsmi_get_cuid_components()` / `amdsmi_get_gpu_cuid_info()`. |

### Verification status (as of 2026-09-24)

- **8x MI350X** (stock 6.8 kernel, compat/backport driver): key creation and
  persistence across reload verified; one GPU cycled through
  SPX→DPX→QPX→CPX→SPX produced 15 pairwise-distinct partition CUIDs with the
  expected UnitIDs; re-keying propagated to all 16 nodes; kernel sysfs
  selftest passed 343/343.
- **2x Radeon PRO W6800** (native driver, real UEFI firmware): create,
  reuse, provision, and delete of the key all passed; runtime sysfs selftest
  passed 78/78; `libamdcuid`'s CPU/NIC/platform CUIDs matched an independent
  HMAC computed with the driver's key; setting the key both with and without
  the driver loaded worked; weak/public keys were refused.
- **QEMU + OVMF**: confirmed the efivarfs confidentiality patch produces
  `0600` entries across remount/reboot, versus `0644` unpatched.
- Every producer (kernel, library, `amd-smi`) passes the same shared
  conformance vector file.

Design rationale is tracked as OpenSpec change `adopt-uefi-key-store`.

## Consumer integration options

Tools that want a component's CUID have three supported ways to get one:

1. **Link `libamdcuid_static.a` directly**, via the CMake package
   `amdcuid::amdcuid`, and call the library's C API. This is the richest
   option: the caller gets key-source resolution (root vs. non-root),
   auxiliary/temp CUID fallback, and CPU/NIC/platform identities in addition
   to GPU lookups, all in-process.
2. **Go through `amd-smi`'s APIs** — `amdsmi_get_cuid_components()` /
   `amdsmi_get_gpu_cuid_info()`, or the `amd-smi node --cuid` /
   `amd-smi static --cuid` CLI — instead of linking the library. This is the
   lowest-effort path for tools that already depend on AMD-SMI; it reports
   the DerivedID alongside its `source` (`DRIVER`/`LIBRARY`), whether it is
   auxiliary, and the key's provisioning state.
3. **Read the driver's sysfs files directly.** For a GPU or a spatial
   partition, `cuid_primary`, `cuid_derived`, and `cuid_unit_id` are ordinary
   sysfs attributes (see Reference implementation above); a tool that only
   needs a GPU's DerivedID, and none of the key management or non-GPU
   support, can read them with no dependency on `libamdcuid` or `amd-smi` at
   all. This path has no access to auxiliary/temp CUIDs: synthesizing one is
   exclusively a `libamdcuid` function (see Fallback: auxiliary (temp)
   CUIDs), so a device the driver publishes nothing for (no key, or an
   SR-IOV VF) is simply invisible to a sysfs-only consumer, rather than
   falling back to an auxiliary CUID the way `libamdcuid` and `amd-smi` do.

All three agree by construction rather than being independent
implementations: `amd-smi` and a direct sysfs read of a GPU's `cuid_derived`
bottom out in the same file, and `libamdcuid` reads that same file for GPU
lookups (source `DRIVER`) rather than recomputing it. A consumer should pick
the option matching how much surrounding context it needs (key management,
auxiliary CUIDs, non-GPU components), not mix more than one against the same
device.

## Security considerations

- The PrimaryID is reversible to hardware serials by design and is therefore
  gated to `CAP_SYS_ADMIN` / root at every layer (sysfs mode, library key
  resolution, `amd-smi` reporting).
- The DerivedID is a one-way HMAC; This design protects sensitive PrimaryID
  material from unprivileged access, as the nature of a one-way HMAC hash
  prevents derivation of the source material, so long as the key is protected
  separately.
- The key-store variable must not be world-readable. An exposed key would allow
  malicious users to potentially brute force sensitive PrimaryID material.
  `efivarfs`'s current `0644` default for `RUNTIME_ACCESS` variables is a known
  gap, mitigated today by a `tmpfiles.d` rule pending an upstream `efivarfs`
  fix.
- Weak or previously-public keys (all-equal bytes, retired defaults, or the
  shared conformance-vector constants) are explicitly refused by `amd-smi
  set --cuid-seed`.
- Auxiliary CUIDs are intentionally weaker (node-local only) and are always
  marked via bit 117 so a consumer cannot mistake one for a canonical,
  globally non-colliding identifier.

## Alternatives considered

- **UUIDv5 with an `amd.com` namespace for auxiliary CUIDs**: withdrawn. An
  HMAC-SHA-256 payload is not a conforming UUIDv5 construction, validating
  parsers may reject it, and making the UUID version depend on caller
  privilege was fragile. Replaced by a normal UUIDv8 distinguished only by
  bit 117.
- **RAM-only shared seed with a public default value**: withdrawn. It
  required a daemon and/or key file to persist and reconcile the seed, and a
  public default undermined non-reversibility. Replaced by the
  driver-owned, UEFI-backed `AmdCuidKey`.
- **Buddy table for GPU partition UnitID** ("Unit ID Definition in CUID",
  11/25/2025; `2N-1` entries in KFD sysfs for `N = 2^x` XCCs, recursively
  split into 1 partition of `N`, 2 of `N/2`, 4 of `N/4`, ..., `N` partitions
  of 1 XCC): rejected in favor of the closed-form XCC-range encoding, which
  additionally distinguishes device-0 from SPX, covers SLC and 6-XCC/
  harvested parts, and is computable by tools outside the driver without
  reading a kernel table.

## Open items

- **`efivarfs` confidentiality**: land the drafted `0600`-mode patch for
  CUID-GUID entries upstream.
- **SLC identification**: confirm where the driver learns SLC is active; the
  UnitID bit (12) is decided but not yet wired to a capability source.
- **Virtualization**: per-VM serial projection and whether a hypervisor
  should expose its own virtual `AmdCuidKey` are open, with full-passthrough
  guests currently minting their own per-variable-store key.
- **Non-contiguous XCC masks**: currently refused with a warning and no CUID
  for that partition; no further handling planned.
- **Upstreaming**: land the efivarfs patch, then the native and compat
  kernel series; take rocm-systems#12075 through CI and review after #10818;
  add an automated OVMF test for driver-created variables and the
  efivarfs-direct key-write path.
- Several ROCm components (`rocr-runtime`, `rocdbgapi`, `api-headers`) still
  reference an older "secondary CUID" naming that needs to be updated to
  match this design.

## Appendix: HMAC (Hash-based Message Authentication Code)

HMAC is defined in [RFC 2104](https://www.rfc-editor.org/info/rfc2104/) as:

$$HMAC(K, m) = H\big((K' \oplus \text{opad}) \parallel H((K' \oplus \text{ipad}) \parallel m)\big)$$

- $H$ — a cryptographic hash function (SHA2-256 throughout this RFC)
- $B$ / $S$ — the block size / output size of $H$
- $K$ — the secret key; $K'$ — $K$ hashed down to length $B$ if longer than
  $B$, or zero-padded to $B$ if shorter
- $m$ — the message; opad/ipad — `0x5c`/`0x36` repeated to length $B$
- $\oplus$ — XOR; $\parallel$ — concatenation

The two-pass structure prevents length-extension attacks that affect plain
`H(key || message)` constructions. With SHA2-256:

| Parameter | Size |
| --- | --- |
| Hash output size ($S$) | 256 bits (32 bytes) |
| Hash block size ($B$) | 512 bits (64 bytes) |
| $K'$ (derived key) | 512 bits (64 bytes) — hashed down if $K$ > 64 bytes, zero-padded if shorter |
| ipad / opad | 512 bits (64 bytes) each |
| HMAC output | 256 bits (32 bytes) |

The block size governs padding and the XOR operations; the output size is
what each hash invocation — and the final HMAC result — actually returns.

## References

- [rocm-systems PR #12075](https://github.com/ROCm/rocm-systems/pull/12075)
- [RFC 9562 — UUID](https://www.rfc-editor.org/rfc/rfc9562)
- [RFC 2104 — HMAC](https://www.rfc-editor.org/info/rfc2104/)
- [FIPS 180-4 — Secure Hash Standard](https://csrc.nist.gov/pubs/fips/180-4/upd1/final)
- [HMAC](https://en.wikipedia.org/wiki/HMAC), [SHA-2](https://en.wikipedia.org/wiki/SHA-2)
- [PCI Express Base Specification 6.2](https://members.pcisig.com/wg/PCI-SIG/document/20590?downloadRevision=active), §7.9.3 Device Serial Number Extended Capability
- [SMBIOS Specification (DMTF DSP0134)](https://www.dmtf.org/sites/default/files/standards/documents/DSP0134_3.4.0.pdf)
- [UEFI Specification 2.11](https://uefi.org/specs/UEFI/2.11/)
- [`machine-id(5)`](https://manpages.ubuntu.com/manpages/focal/man5/machine-id.5.html)
- "Unit ID Definition in CUID" (11/25/2025) — internal proposal for the
  buddy-table UnitID scheme and the KFD/ROCr/HIP/AMD-SMI/debugger/profiler/
  OOB CUID propagation flow

## Revision History

- 2026-09-25: Initial RFC, based on the CUID running doc (v20) and the
  2026-09-23 UEFI-key design sync.
- 2026-09-25: Replaced the auxiliary/temp CUID description with the
  citation-backed namespace/application-ID HMAC algorithm, added an HMAC
  appendix, and flagged open discrepancies with the running doc's
  provisional formula.
- 2026-09-25: Added a Partition selection and CUID propagation section
  summarizing the KFD/ROCr/HIP/AMD-SMI/debugger/profiler/OOB flow from the
  original Unit ID proposal, and flagged it for reconciliation against the
  current sysfs-based implementation.
- 2026-09-28: Resolved the CPU routing field discrepancy in favor of the
  physical package ID; updated the temporary primary ID table and removed
  the now-resolved discrepancy bullet.
- 2026-09-28: Confirmed `HMAC-SHA-256(temporary_primary_id,
  cuid_application_id)` as the authoritative temporary derived CUID
  construction; removed the key/message-roles discrepancy and its Open
  items entry.
- 2026-09-28: Established sysfs and `libamdcuid`'s cache file as the two
  authoritative CUID sources, reworded the partition selection/propagation
  flow accordingly, and removed the reconciliation question and its Open
  items entry.
- 2026-09-28: Added a Consumer integration options section describing the
  three supported ways to consume a CUID: linking `libamdcuid_static.a`,
  `amd-smi`'s APIs, or direct sysfs reads.
