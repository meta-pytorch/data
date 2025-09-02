"""Tests for FileLister class."""

import logging
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

from torchdata.nodes.io.file_list import FileLister

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def sample_local_files(tmp_path_factory):
    """Create temporary files with various extensions for testing."""
    # Create a temp directory with random name
    tmp_dir_name = f"file_list_test_{uuid.uuid4().hex[:8]}"
    tmp_path = tmp_path_factory.mktemp(tmp_dir_name)

    # Create various files
    file_paths = []

    # Text files
    for i in range(3):
        file_path = tmp_path / f"file{i}.txt"
        with open(file_path, "w") as f:
            f.write(f"Content of file{i}.txt")
        file_paths.append(str(file_path))

    # JSON files
    for i in range(2):
        file_path = tmp_path / f"data{i}.json"
        with open(file_path, "w") as f:
            f.write(f'{{"data": "value{i}"}}')
        file_paths.append(str(file_path))

    # CSV files
    for i in range(2):
        file_path = tmp_path / f"data{i}.csv"
        with open(file_path, "w") as f:
            f.write(f"id,value\n{i},{i * 10}")
        file_paths.append(str(file_path))

    # Create a subdirectory with files
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_file = subdir / "subfile.txt"
    with open(subdir_file, "w") as f:
        f.write("Content in subdirectory")
    file_paths.append(str(subdir_file))

    return str(tmp_path), file_paths


def test_local_file_list_all_files(sample_local_files):
    """Test listing all files in a directory."""
    tmp_dir, expected_files = sample_local_files

    # Create a FileLister to list all files
    node = FileLister(tmp_dir, ["**/*"])

    # Get all files
    file_items = list(node)
    file_paths = [item["data"] for item in file_items]

    # Verify the count (should match all files except subdirectory itself)
    assert len(file_paths) == len(expected_files)

    # Normalize expected files to forward slashes for cross-platform consistency
    normalized_expected_files = [expected_file.replace("\\", "/") for expected_file in expected_files]

    # Check if all expected files are in the result
    for expected_file in normalized_expected_files:
        assert expected_file in file_paths


def test_local_file_list_specific_pattern(sample_local_files):
    """Test listing files matching specific patterns."""
    tmp_dir, _ = sample_local_files

    # Create a FileLister to list only txt files
    node = FileLister(tmp_dir, ["*.txt"])

    # Get matching files
    file_items = list(node)
    file_paths = [item["data"] for item in file_items]

    # Verify we only got txt files
    assert len(file_paths) == 3  # 3 txt files in the root directory
    for file_path in file_paths:
        assert file_path.endswith(".txt")
        # Ensure paths use forward slashes
        assert "\\" not in file_path

    # Test with multiple patterns
    node = FileLister(tmp_dir, ["*.json", "*.csv"])
    file_items = list(node)
    file_paths = [item["data"] for item in file_items]

    # Verify we got json and csv files
    assert len(file_paths) == 4  # 2 json + 2 csv
    for file_path in file_paths:
        assert file_path.endswith(".json") or file_path.endswith(".csv")


def test_local_file_list_recursive(sample_local_files):
    """Test recursive file listing."""
    tmp_dir, expected_files = sample_local_files

    # Create a FileLister with recursive pattern
    node = FileLister(tmp_dir, ["**/*.txt"])

    # Get matching files
    file_items = list(node)
    file_paths = [item["data"] for item in file_items]

    # Verify we got all txt files including in subdirectory
    assert len(file_paths) == 4  # 3 in root + 1 in subdir

    # Check specific paths
    subdir_files = [p for p in file_paths if "subdir" in p]
    assert len(subdir_files) == 1


def test_local_file_list_metadata(sample_local_files):
    """Test that metadata is correctly included in the output."""
    tmp_dir, _ = sample_local_files

    # Create a FileLister
    node = FileLister(tmp_dir, ["*.txt"])

    # Get first file
    first_item = next(node)

    # Check metadata
    assert "metadata" in first_item
    assert first_item["metadata"]["source"] == "file"
    assert first_item["metadata"]["fs_protocol"] == "file"


def test_local_file_list_empty_result(tmp_path):
    """Test behavior with no matching files."""
    # Create a FileLister with pattern that won't match anything
    node = FileLister(str(tmp_path), ["*.nonexistent"])

    # List should be empty
    file_items = list(node)
    assert len(file_items) == 0


def test_local_file_list_state(sample_local_files):
    """Test state management and resumption."""
    tmp_dir, _ = sample_local_files

    # Create a FileLister
    node = FileLister(tmp_dir, ["*.txt"])

    # Get the first file
    first_item = next(node)

    # Save state
    state = node.get_state()

    # Create a new node and restore state
    new_node = FileLister(tmp_dir, ["*.txt"])
    new_node.reset(state)

    # Get next item from new node - should be the second file
    second_item = next(new_node)

    # Create another fresh node to verify
    verify_node = FileLister(tmp_dir, ["*.txt"])
    first_again = next(verify_node)
    second_again = next(verify_node)

    # First item from verify_node should match first_item
    assert first_again["data"] == first_item["data"]

    # Second item from new_node should match second_again
    assert second_item["data"] == second_again["data"]


@patch("fsspec.filesystem")
def test_s3_file_list(mock_filesystem, tmp_path):
    """Test S3 file listing with mocked S3 filesystem."""
    # Create mock S3 filesystem
    mock_fs = MagicMock()
    mock_filesystem.return_value = mock_fs

    # Configure mock to return some file paths
    s3_files = ["my-bucket/file1.txt", "my-bucket/file2.txt", "my-bucket/data.json"]
    mock_fs.glob.return_value = s3_files
    mock_fs.isfile.return_value = True

    # Create FileLister for S3
    node = FileLister("s3://my-bucket", ["*.txt", "*.json"])

    # Get files
    file_items = list(node)
    file_paths = [item["data"] for item in file_items]

    # Verify correct files returned
    assert len(file_paths) == 3
    assert "s3://my-bucket/file1.txt" in file_paths
    assert "s3://my-bucket/file2.txt" in file_paths
    assert "s3://my-bucket/data.json" in file_paths

    # Verify S3 metadata
    assert file_items[0]["metadata"]["source"] == "s3"
    assert file_items[0]["metadata"]["fs_protocol"] == "s3"


@patch("fsspec.filesystem")
def test_s3_file_list_custom_options(mock_filesystem):
    """Test S3 file listing with custom filesystem options."""
    # Create mock S3 filesystem
    mock_fs = MagicMock()
    mock_filesystem.return_value = mock_fs

    # Configure mock
    mock_fs.glob.return_value = ["my-bucket/file1.txt"]
    mock_fs.isfile.return_value = True

    # Create FileLister with custom options
    fs_options = {"anon": True, "connect_timeout": 10}
    node = FileLister("s3://my-bucket", ["*.txt"], fs_options=fs_options)

    # Get files (just to trigger initialization)
    next(node, None)

    # Verify fsspec.filesystem was called with the right options
    mock_filesystem.assert_called_with("s3", anon=True, connect_timeout=10)


def test_error_handling(tmp_path):
    """Test error handling for invalid paths."""
    # Create a FileLister with non-existent path
    nonexistent_path = os.path.join(str(tmp_path), "nonexistent")
    node = FileLister(nonexistent_path, ["*.txt"])

    # Should not raise exception, but log error and return empty list
    file_items = list(node)
    assert len(file_items) == 0


@patch("fsspec.filesystem")
def test_s3_file_list_error(mock_filesystem):
    """Test error handling for S3 file listing."""
    # Create mock S3 filesystem that raises exception
    mock_fs = MagicMock()
    mock_filesystem.return_value = mock_fs
    mock_fs.glob.side_effect = Exception("S3 Error")

    # Create FileLister for S3
    node = FileLister("s3://my-bucket", ["*.txt"])

    # Should not raise exception, but log error and return empty list
    file_items = list(node)
    assert len(file_items) == 0
