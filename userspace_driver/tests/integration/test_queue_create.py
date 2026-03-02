"""Integration tests for hardware queue creation."""

import pytest

from tests.integration.conftest import requires_gpu


@requires_gpu
class TestQueueCreation:
    """Test compute and SDMA queue creation."""

    def test_create_compute_queue(self, amd_device):
        backend = amd_device.backend
        queue = backend.create_compute_queue()
        assert queue.queue_id >= 0
        assert queue.doorbell_addr != 0
        assert queue.ring_buffer is not None
        backend.destroy_queue(queue)

    def test_create_sdma_queue(self, amd_device):
        backend = amd_device.backend
        queue = backend.create_sdma_queue()
        assert queue.queue_id >= 0
        backend.destroy_queue(queue)

    def test_multiple_compute_queues(self, amd_device):
        backend = amd_device.backend
        queues = []
        for _ in range(3):
            q = backend.create_compute_queue()
            assert q.queue_id >= 0
            queues.append(q)

        # All should have unique IDs
        ids = [q.queue_id for q in queues]
        assert len(set(ids)) == len(ids)

        for q in queues:
            backend.destroy_queue(q)
