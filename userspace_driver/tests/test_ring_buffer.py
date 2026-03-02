"""Tests for ring buffer management."""

import ctypes

import pytest


class TestRingBufferLogic:
    """Test ring buffer wrap-around logic without real memory mapping."""

    def test_power_of_two_validation(self):
        from amd_gpu_driver.commands.ring import RingBuffer

        # Create a small buffer in regular memory for testing
        buf_size = 256
        buf = (ctypes.c_char * buf_size)()
        wp_mem = (ctypes.c_uint64 * 1)(0)
        rp_mem = (ctypes.c_uint64 * 1)(0)

        rb = RingBuffer(
            base_addr=ctypes.addressof(buf),
            size=buf_size,
            write_ptr_addr=ctypes.addressof(wp_mem),
            read_ptr_addr=ctypes.addressof(rp_mem),
        )
        assert rb.size == 256

    def test_non_power_of_two_raises(self):
        from amd_gpu_driver.commands.ring import RingBuffer

        buf = (ctypes.c_char * 300)()
        wp_mem = (ctypes.c_uint64 * 1)(0)
        rp_mem = (ctypes.c_uint64 * 1)(0)

        with pytest.raises(ValueError, match="power of 2"):
            RingBuffer(
                base_addr=ctypes.addressof(buf),
                size=300,
                write_ptr_addr=ctypes.addressof(wp_mem),
                read_ptr_addr=ctypes.addressof(rp_mem),
            )

    def test_write_and_read_ptr(self):
        from amd_gpu_driver.commands.ring import RingBuffer

        buf_size = 256
        buf = (ctypes.c_char * buf_size)()
        wp_mem = (ctypes.c_uint64 * 1)(0)
        rp_mem = (ctypes.c_uint64 * 1)(0)

        rb = RingBuffer(
            base_addr=ctypes.addressof(buf),
            size=buf_size,
            write_ptr_addr=ctypes.addressof(wp_mem),
            read_ptr_addr=ctypes.addressof(rp_mem),
        )

        assert rb.write_ptr == 0
        assert rb.read_ptr == 0

        rb.write_ptr = 10
        assert rb.write_ptr == 10

    def test_available_space(self):
        from amd_gpu_driver.commands.ring import RingBuffer

        buf_size = 256
        buf = (ctypes.c_char * buf_size)()
        wp_mem = (ctypes.c_uint64 * 1)(0)
        rp_mem = (ctypes.c_uint64 * 1)(0)

        rb = RingBuffer(
            base_addr=ctypes.addressof(buf),
            size=buf_size,
            write_ptr_addr=ctypes.addressof(wp_mem),
            read_ptr_addr=ctypes.addressof(rp_mem),
        )

        # Initially all space is available
        # 256 bytes / 4 bytes per dword = 64 dwords * 4 = 256 bytes
        assert rb.available_space() == 256

    def test_write_data(self):
        from amd_gpu_driver.commands.ring import RingBuffer

        buf_size = 256
        buf = (ctypes.c_char * buf_size)()
        wp_mem = (ctypes.c_uint64 * 1)(0)
        rp_mem = (ctypes.c_uint64 * 1)(0)

        rb = RingBuffer(
            base_addr=ctypes.addressof(buf),
            size=buf_size,
            write_ptr_addr=ctypes.addressof(wp_mem),
            read_ptr_addr=ctypes.addressof(rp_mem),
        )

        # Write 16 bytes (4 dwords)
        data = b"\x01\x02\x03\x04" * 4
        new_wp = rb.write(data)
        assert new_wp == 4  # 16 bytes = 4 dwords
        assert rb.write_ptr == 4

        # Verify data was written
        written = bytes(buf[:16])
        assert written == data

    def test_write_wraps_around(self):
        from amd_gpu_driver.commands.ring import RingBuffer

        buf_size = 64  # Small buffer to test wrap
        buf = (ctypes.c_char * buf_size)()
        wp_mem = (ctypes.c_uint64 * 1)(0)
        rp_mem = (ctypes.c_uint64 * 1)(0)

        rb = RingBuffer(
            base_addr=ctypes.addressof(buf),
            size=buf_size,
            write_ptr_addr=ctypes.addressof(wp_mem),
            read_ptr_addr=ctypes.addressof(rp_mem),
        )

        # Write to near the end
        first = b"\xAA" * 56  # 56 bytes = 14 dwords
        rb.write(first)
        assert rb.write_ptr == 14

        # Write 16 bytes that should wrap around
        wrap_data = b"\xBB" * 16  # 16 bytes = 4 dwords
        new_wp = rb.write(wrap_data)
        assert new_wp == 18

        # Check: 8 bytes at end + 8 bytes at start
        assert bytes(buf[56:64]) == b"\xBB" * 8
        assert bytes(buf[0:8]) == b"\xBB" * 8
