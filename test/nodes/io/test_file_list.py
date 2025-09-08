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


def test_file_list_state_management(sample_local_files):
    """Test state management in FileLister."""
    tmp_dir, _ = sample_local_files

    # Create a FileLister
    node = FileLister(tmp_dir, ["**/*"])

    # Read first file and store state
    first_item = next(iter(node))
    state = node.get_state()

    # Verify state contains expected keys
    assert "current_idx" in state
    assert "num_files" in state
    assert state["current_idx"] == 1  # After reading first item
    assert state["num_files"] > 0

    # Create new node and restore state
    new_node = FileLister(tmp_dir, ["**/*"])
    new_node.reset(state)

    # Should continue from second file
    second_item = next(iter(new_node))
    assert second_item["data"] != first_item["data"]


def test_file_list_state_validation_num_files_mismatch(sample_local_files):
    """Test that state validation fails when number of files changes."""
    tmp_dir, _ = sample_local_files

    # Create a FileLister and get initial state
    node = FileLister(tmp_dir, ["**/*"])
    initial_state = node.get_state()

    # Modify the state to have wrong number of files
    corrupted_state = initial_state.copy()
    corrupted_state["num_files"] = initial_state["num_files"] + 1

    # Create new node and try to restore corrupted state
    new_node = FileLister(tmp_dir, ["**/*"])

    with pytest.raises(ValueError, match="State validation failed: saved state has"):
        new_node.reset(corrupted_state)


def test_file_list_state_validation_index_out_of_bounds(sample_local_files):
    """Test that state validation fails when index is out of bounds."""
    tmp_dir, _ = sample_local_files

    # Create a FileLister and get initial state
    node = FileLister(tmp_dir, ["**/*"])
    # Populate file listing
    list(node)
    initial_state = node.get_state()

    # Modify the state to have invalid index but keep num_files matching
    corrupted_state = {**initial_state}
    corrupted_state["current_idx"] = initial_state["num_files"] + 10  # Way out of bounds

    # Create new node and try to restore corrupted state
    new_node = FileLister(tmp_dir, ["**/*"])

    with pytest.raises(ValueError, match="State validation failed: saved index"):
        new_node.reset(corrupted_state)


def test_file_list_protocol_handling():
    """Test that FileLister correctly handles different protocols."""
    # Test local filesystem
    with patch("fsspec.filesystem") as mock_fs_class:
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.glob.return_value = ["file1.txt", "file2.txt"]
        mock_fs.isfile.return_value = True

        node = FileLister("local/path", ["*"])
        node.reset()

        # Should not add protocol prefix for local files
        file_paths = [item["data"] for item in node]
        assert all(not path.startswith("file://") for path in file_paths)
        assert all("\\" not in path for path in file_paths)
        assert all("/" in path for path in file_paths)  # Should have forward slashes


def test_file_list_s3_protocol_handling():
    """Test that FileLister correctly handles S3 protocol."""
    with patch("fsspec.filesystem") as mock_fs_class:
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.glob.return_value = ["key1.txt", "key2.txt"]
        mock_fs.isfile.return_value = True

        node = FileLister("s3://bucket/prefix", ["*"])
        node.reset()

        # Should add S3 protocol prefix
        file_paths = [item["data"] for item in node]
        assert all(path.startswith("s3://") for path in file_paths)
        assert all("/" in path for path in file_paths)  # Should have forward slashes


def test_file_list_directory_filtering():
    """Test that FileLister correctly filters out directories."""
    with patch("fsspec.filesystem") as mock_fs_class:
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs

        # Mock glob to return both files and directories
        mock_fs.glob.return_value = ["file1.txt", "dir1", "file2.txt", "dir2"]
        mock_fs.isfile.side_effect = lambda x: x.endswith(".txt")  # Only .txt files are files

        node = FileLister("test/path", ["**/*"])
        node.reset()

        # Should only get files, not directories
        file_paths = [item["data"] for item in node]
        assert len(file_paths) == 2
        assert all(path.endswith(".txt") for path in file_paths)


def test_file_list_empty_directory():
    """Test FileLister behavior with empty directory."""
    with patch("fsspec.filesystem") as mock_fs_class:
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.glob.return_value = []

        node = FileLister("empty/path", ["*"])
        node.reset()

        # Should handle empty directory gracefully
        with pytest.raises(StopIteration):
            next(iter(node))


def test_file_list_glob_error_handling():
    """Test that FileLister handles glob errors gracefully."""
    with patch("fsspec.filesystem") as mock_fs_class:
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.glob.side_effect = Exception("Glob error")

        node = FileLister("error/path", ["*"])
        node.reset()

        # Should handle glob errors gracefully
        with pytest.raises(StopIteration):
            next(iter(node))


def test_file_list_multiple_patterns():
    """Test FileLister with multiple patterns."""
    with patch("fsspec.filesystem") as mock_fs_class:
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs

        # Mock different patterns returning different files
        def mock_glob(pattern):
            if "*.txt" in pattern:
                return ["file1.txt", "file2.txt"]
            elif "*.json" in pattern:
                return ["data1.json"]
            else:
                return []

        mock_fs.glob.side_effect = mock_glob
        mock_fs.isfile.return_value = True

        node = FileLister("test/path", ["*.txt", "*.json"])
        node.reset()

        # Should get files from all patterns
        file_paths = [item["data"] for item in node]
        assert len(file_paths) == 3
        assert any(path.endswith(".txt") for path in file_paths)
        assert any(path.endswith(".json") for path in file_paths)


def test_file_list_path_normalization():
    """Test that FileLister normalizes paths correctly."""
    with patch("fsspec.filesystem") as mock_fs_class:
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs

        # Mock paths with backslashes (Windows-style)
        mock_fs.glob.return_value = ["C:\\Users\\test\\file1.txt", "C:\\Users\\test\\file2.txt"]
        mock_fs.isfile.return_value = True

        node = FileLister("C:\\Users\\test", ["*"])
        node.reset()

        # Should normalize to forward slashes
        file_paths = [item["data"] for item in node]
        assert all("\\" not in path for path in file_paths)
        assert all("/" in path for path in file_paths)
        assert all(path.startswith("C:/") for path in file_paths)


def test_file_list_state_consistency(sample_local_files):
    """Test that state is consistent across multiple operations."""
    tmp_dir, _ = sample_local_files

    # Create a FileLister
    node = FileLister(tmp_dir, ["**/*"])

    # Initialize without consuming items so state has correct num_files
    node.reset()

    # Get initial state
    initial_state = node.get_state()
    initial_num_files = initial_state["num_files"]
    initial_idx = initial_state["current_idx"]

    # Read a few items
    for _ in range(3):
        next(iter(node))

    # Get updated state
    updated_state = node.get_state()
    updated_num_files = updated_state["num_files"]
    updated_idx = updated_state["current_idx"]

    # Number of files should remain the same
    assert updated_num_files == initial_num_files

    # Index should have increased
    assert updated_idx == initial_idx + 3


def test_file_list_reset_without_state(sample_local_files):
    """Test that reset without state works correctly."""
    tmp_dir, _ = sample_local_files

    # Create a FileLister
    node = FileLister(tmp_dir, ["**/*"])

    # Read some items
    for _ in range(2):
        next(iter(node))

    # Reset without state
    node.reset()

    # Should start from beginning
    first_item = next(iter(node))
    assert first_item["metadata"]["fs_protocol"] == "file"
