import hashlib


def calculate_hash(file, hash_algorithm):
    with open(file, "rb") as f:
        try:
            digest = hashlib.file_digest(f, hash_algorithm)
        except AttributeError:  # file_digest() was added in Python 3.11.
            digest = hashlib.new(hash_algorithm)
            buffer = bytearray(2**16)
            view = memoryview(buffer)
            while True:
                size = f.readinto(buffer)
                if size == 0:
                    break
                digest.update(view[:size])

    return digest


def write_hash(hash_file, digest):
    with open(hash_file, "wt") as f:
        f.write(digest.hexdigest())
        f.write("\n")
