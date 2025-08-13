"""
Tests for functions in utils/checkpointing_utils.py
"""

import os
import tempfile

from fms_fsdp.utils.checkpointing_utils import get_latest, get_oldest


def test_get_oldest():
    """
    Ensure that the get_oldest function returns the name of the file with the oldest
    timestamp (i.e. that was created first).
    """
    with tempfile.TemporaryDirectory() as tempdir:
        for i in range(3):
            filename = os.path.join(tempdir, f"file_{i}")
            print("random content", file=open(file=filename, mode="w"))

        oldest_filename = get_oldest(targdir=tempdir)
        assert oldest_filename.endswith("file_0")


def test_get_latest():
    """
    Ensure that the get_latest function returns the name of the file with the latest
    integer suffix (i.e. that was created last).
    """
    with tempfile.TemporaryDirectory() as tempdir:
        for i in range(3):
            filename = os.path.join(tempdir, f"file_{i}")
            print("random content", file=open(file=filename, mode="w"))

        latest_filename = get_latest(targdir=tempdir)
        assert latest_filename.endswith("file_2")
