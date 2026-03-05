"""Convert a .co code object to a C header with an embedded byte array."""
import sys

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.co> <output.h>")
        sys.exit(1)

    co_path = sys.argv[1]
    header_path = sys.argv[2]

    data = open(co_path, "rb").read()

    with open(header_path, "w", encoding="ascii") as f:
        f.write("/* Auto-generated from %s */\n" % co_path)
        f.write("static const unsigned char vector_add_co[] = {\n")
        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            f.write("  " + ",".join("0x%02x" % b for b in chunk))
            if i + 16 < len(data):
                f.write(",")
            f.write("\n")
        f.write("};\n")
        f.write("static const size_t vector_add_co_size = %d;\n" % len(data))

    print("  %s -> %s (%d bytes)" % (co_path, header_path, len(data)))

if __name__ == "__main__":
    main()
