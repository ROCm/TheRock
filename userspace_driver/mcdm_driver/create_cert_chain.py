"""Create a 2-tier certificate chain for kernel driver test signing.

Tier 1: Root CA cert (self-signed, CA=true)
Tier 2: Leaf code signing cert (signed by CA, CA=false, EKU=CodeSigning)

Outputs PFX files for both, and the leaf PFX is used with signtool.
"""

import datetime
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography.x509.oid import ExtendedKeyUsageOID

OUT_DIR = Path(r"D:\R\userspace_driver\kernel_driver")
PFX_PASSWORD = b"test"


def create_ca_cert():
    """Create self-signed root CA certificate."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "AmdGpuTestCA"),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=5 * 365)
        )
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    return key, cert


def create_leaf_cert(ca_key, ca_cert):
    """Create code signing leaf certificate signed by the CA."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([
        x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "AmdGpuTestSign"),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=5 * 365)
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CODE_SIGNING]),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(
                ca_key.public_key()
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    return key, cert


def main():
    print("Creating 2-tier certificate chain...")

    # Create CA
    ca_key, ca_cert = create_ca_cert()
    ca_cer_path = OUT_DIR / "AmdGpuTestCA.cer"
    ca_cer_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.DER))
    print(f"  CA cert: {ca_cer_path}")

    ca_pfx_path = OUT_DIR / "AmdGpuTestCA.pfx"
    ca_pfx_path.write_bytes(
        pkcs12.serialize_key_and_certificates(
            b"AmdGpuTestCA",
            ca_key,
            ca_cert,
            None,
            serialization.BestAvailableEncryption(PFX_PASSWORD),
        )
    )
    print(f"  CA PFX:  {ca_pfx_path}")

    # Create leaf
    leaf_key, leaf_cert = create_leaf_cert(ca_key, ca_cert)
    leaf_cer_path = OUT_DIR / "AmdGpuTestSign.cer"
    leaf_cer_path.write_bytes(
        leaf_cert.public_bytes(serialization.Encoding.DER)
    )
    print(f"  Leaf cert: {leaf_cer_path}")

    # PFX with full chain (leaf + CA)
    leaf_pfx_path = OUT_DIR / "AmdGpuTestSign.pfx"
    leaf_pfx_path.write_bytes(
        pkcs12.serialize_key_and_certificates(
            b"AmdGpuTestSign",
            leaf_key,
            leaf_cert,
            [ca_cert],
            serialization.BestAvailableEncryption(PFX_PASSWORD),
        )
    )
    print(f"  Leaf PFX: {leaf_pfx_path} (password: test)")

    print("\nNext steps:")
    print(f'  1. certutil -addstore Root "{ca_cer_path}"')
    print(f'  2. certutil -addstore TrustedPublisher "{ca_cer_path}"')
    print(
        f'  3. signtool sign /fd sha256 /f "{leaf_pfx_path}" /p test /v '
        r'"D:\R\userspace_driver\kernel_driver\x64\Release\amdgpu_mcdm.sys"'
    )
    print("  4. Copy signed .sys to driver store and restart device")


if __name__ == "__main__":
    main()
