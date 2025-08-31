"""Tests for universal FileReader."""

import uuid
from typing import Any, Dict, List, Union
from unittest.mock import MagicMock, mock_open, patch

import boto3
import pytest
from moto import mock_aws
from torchdata.nodes import BaseNode

from torchdata.nodes.io.file_read import _fibonacci_backoff, FileReader


class MockSourceNode(BaseNode[Dict]):
    """Mock source node that provides file paths for testing."""

    def __init__(self, file_paths: List[str], metadata: Dict[str, Any] = None):
        super().__init__()
        self.file_paths = file_paths
        self.metadata = metadata or {}
        self._current_idx = 0

    def reset(self, initial_state=None):
        super().reset(initial_state)
        if initial_state is not None:
            self._current_idx = initial_state.get("idx", 0)
        else:
            self._current_idx = 0

    def next(self) -> Dict:
        if self._current_idx >= len(self.file_paths):
            raise StopIteration("No more files")

        path = self.file_paths[self._current_idx]
        self._current_idx += 1

        return {FileReader.DATA_KEY: path, FileReader.METADATA_KEY: dict(self.metadata)}

    def get_state(self):
        return {"idx": self._current_idx}


@pytest.fixture(scope="function")
def temp_dir(tmp_path_factory):
    """Create a temporary directory with test files."""
    tmp_dir_name = f"file_reader_test_{uuid.uuid4().hex[:8]}"
    tmp_path = tmp_path_factory.mktemp(tmp_dir_name)

    # Create test files in a deterministic order
    files = [
        ("file1.txt", "content1"),
        ("file2.txt", "content2"),
        ("file3.txt", "content3"),
        ("data.json", '{"key": "value"}'),
        ("image.png", b"fake png data"),
    ]

    for name, content in files:
        if isinstance(content, str):
            (tmp_path / name).write_text(content)
        else:
            (tmp_path / name).write_bytes(content)

    # Create a subdirectory with files
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file4.txt").write_text("content4")

    return tmp_path


@pytest.fixture(scope="function")
def mock_bucket():
    """Create a mock S3 bucket with test files."""
    with mock_aws():
        # Create S3 client and bucket with region
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket_name = "test-bucket"

        # Create bucket (no location constraint for us-east-1)
        s3.create_bucket(Bucket=bucket_name)

        # Create test files in a deterministic order
        files = [
            ("file1.txt", "content1"),
            ("file2.txt", "content2"),
            ("file3.txt", "content3"),
            ("data.json", '{"key": "value"}'),
            ("image.png", b"fake png data"),
            ("subdir/file4.txt", "content4"),
        ]

        for key, content in files:
            if isinstance(content, str):
                s3.put_object(Bucket=bucket_name, Key=key, Body=content.encode())
            else:
                s3.put_object(Bucket=bucket_name, Key=key, Body=content)

        yield bucket_name


def test_file_reader_local_text(temp_dir):
    """Test FileReader with local text files."""
    file_paths = [str(temp_dir / "file1.txt"), str(temp_dir / "file2.txt")]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node)

    # Read first file
    content1 = next(reader_node)
    assert content1[FileReader.DATA_KEY] == "content1"
    assert content1[FileReader.METADATA_KEY]["file_path"].endswith("file1.txt")

    # Read second file
    content2 = next(reader_node)
    assert content2[FileReader.DATA_KEY] == "content2"
    assert content2[FileReader.METADATA_KEY]["file_path"].endswith("file2.txt")


def test_file_reader_local_binary(temp_dir):
    """Test FileReader with local binary files."""
    file_paths = [str(temp_dir / "image.png")]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node, mode="rb")

    # Read binary file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == b"fake png data"
    assert content[FileReader.METADATA_KEY]["file_path"].endswith("image.png")


def test_file_reader_state_management(temp_dir):
    """Test state management in FileReader with local files."""
    file_paths = [str(temp_dir / "file1.txt"), str(temp_dir / "file2.txt"), str(temp_dir / "file3.txt")]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node)

    # Read first file
    content1 = next(reader_node)
    assert content1[FileReader.DATA_KEY] == "content1"

    # Save state
    state = reader_node.get_state()

    # Read second file
    content2 = next(reader_node)
    assert content2[FileReader.DATA_KEY] == "content2"

    # Create new reader and restore state
    new_source = MockSourceNode(file_paths)
    new_reader = FileReader(new_source)
    new_reader.reset(state)

    # Should continue from second file
    new_content2 = next(new_reader)
    assert new_content2[FileReader.DATA_KEY] == "content2"


@pytest.mark.aws
def test_file_reader_s3_text(mock_bucket):
    """Test FileReader with S3 text files."""
    file_paths = [f"s3://{mock_bucket}/file1.txt", f"s3://{mock_bucket}/file2.txt"]
    source_node = MockSourceNode(file_paths, {"source": "s3"})
    reader_node = FileReader(source_node)

    # Read first file
    content1 = next(reader_node)
    assert content1[FileReader.DATA_KEY] == "content1"
    assert content1[FileReader.METADATA_KEY]["file_path"] == f"s3://{mock_bucket}/file1.txt"
    assert content1[FileReader.METADATA_KEY]["source"] == "s3"

    # Read second file
    content2 = next(reader_node)
    assert content2[FileReader.DATA_KEY] == "content2"
    assert content2[FileReader.METADATA_KEY]["file_path"] == f"s3://{mock_bucket}/file2.txt"


@pytest.mark.aws
def test_file_reader_s3_binary(mock_bucket):
    """Test FileReader with S3 binary files."""
    file_paths = [f"s3://{mock_bucket}/image.png"]
    source_node = MockSourceNode(file_paths, {"source": "s3"})
    reader_node = FileReader(source_node, mode="rb")

    # Read binary file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == b"fake png data"
    assert content[FileReader.METADATA_KEY]["file_path"] == f"s3://{mock_bucket}/image.png"


@pytest.mark.aws
def test_file_reader_s3_state_management(mock_bucket):
    """Test state management in FileReader with S3 files."""
    file_paths = [f"s3://{mock_bucket}/file1.txt", f"s3://{mock_bucket}/file2.txt"]
    source_node = MockSourceNode(file_paths, {"source": "s3"})
    reader_node = FileReader(source_node)

    # Read first file
    content1 = next(reader_node)
    assert content1[FileReader.DATA_KEY] == "content1"

    # Save state after first file
    state = reader_node.get_state()

    # Read second file
    content2 = next(reader_node)
    assert content2[FileReader.DATA_KEY] == "content2"

    # Create new reader and restore state
    new_source = MockSourceNode(file_paths, {"source": "s3"})
    new_reader = FileReader(new_source)
    new_reader.reset(state)

    # Should continue from second file
    new_content2 = next(new_reader)
    assert new_content2[FileReader.DATA_KEY] == "content2"
    assert new_content2[FileReader.METADATA_KEY]["file_path"] == f"s3://{mock_bucket}/file2.txt"


@pytest.mark.azure
@patch("smart_open.open")
def test_file_reader_azure_text(mock_smart_open):
    """Test FileReader with Azure Blob Storage text files."""
    # Mock smart_open for Azure Blob Storage
    mock_smart_open.return_value.__enter__.return_value.read.return_value = "azure_content1"

    file_paths = ["abfs://container@account.dfs.core.windows.net/file1.txt"]
    source_node = MockSourceNode(file_paths, {"source": "abfs"})
    reader_node = FileReader(source_node, transport_params={"anon": False})

    # Read file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == "azure_content1"
    assert content[FileReader.METADATA_KEY]["file_path"] == "abfs://container@account.dfs.core.windows.net/file1.txt"
    assert content[FileReader.METADATA_KEY]["source"] == "abfs"

    # Verify smart_open was called with correct parameters
    mock_smart_open.assert_called_once_with(
        "abfs://container@account.dfs.core.windows.net/file1.txt",
        "r",
        encoding="utf-8",
        transport_params={"anon": False},
    )


@pytest.mark.azure
@patch("smart_open.open")
def test_file_reader_azure_binary(mock_smart_open):
    """Test FileReader with Azure Blob Storage binary files."""
    # Mock smart_open for Azure binary data
    mock_smart_open.return_value.__enter__.return_value.read.return_value = b"azure_binary_data"

    file_paths = ["abfs://container@account.dfs.core.windows.net/image.png"]
    source_node = MockSourceNode(file_paths, {"source": "abfs"})
    reader_node = FileReader(source_node, mode="rb")

    # Read binary file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == b"azure_binary_data"
    assert content[FileReader.METADATA_KEY]["file_path"] == "abfs://container@account.dfs.core.windows.net/image.png"

    # Verify smart_open was called with correct parameters for binary mode
    mock_smart_open.assert_called_once_with(
        "abfs://container@account.dfs.core.windows.net/image.png", "rb", encoding=None, transport_params={}
    )


@pytest.mark.gcs
@patch("smart_open.open")
def test_file_reader_gcs_text(mock_smart_open):
    """Test FileReader with Google Cloud Storage text files."""
    # Mock smart_open for GCS
    mock_smart_open.return_value.__enter__.return_value.read.return_value = "gcs_content1"

    file_paths = ["gs://my-bucket/file1.txt"]
    source_node = MockSourceNode(file_paths, {"source": "gs"})
    reader_node = FileReader(source_node, transport_params={"client": "mock_gcs_client"})

    # Read file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == "gcs_content1"
    assert content[FileReader.METADATA_KEY]["file_path"] == "gs://my-bucket/file1.txt"
    assert content[FileReader.METADATA_KEY]["source"] == "gs"

    # Verify smart_open was called with correct parameters
    mock_smart_open.assert_called_once_with(
        "gs://my-bucket/file1.txt", "r", encoding="utf-8", transport_params={"client": "mock_gcs_client"}
    )


@pytest.mark.gcs
@patch("smart_open.open")
def test_file_reader_gcs_binary(mock_smart_open):
    """Test FileReader with Google Cloud Storage binary files."""
    # Mock smart_open for GCS binary data
    mock_smart_open.return_value.__enter__.return_value.read.return_value = b"gcs_binary_data"

    file_paths = ["gs://my-bucket/data.parquet"]
    source_node = MockSourceNode(file_paths, {"source": "gs"})
    reader_node = FileReader(source_node, mode="rb")

    # Read binary file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == b"gcs_binary_data"
    assert content[FileReader.METADATA_KEY]["file_path"] == "gs://my-bucket/data.parquet"

    # Verify smart_open was called with correct parameters for binary mode
    mock_smart_open.assert_called_once_with("gs://my-bucket/data.parquet", "rb", encoding=None, transport_params={})


def test_file_reader_compression_handling():
    """Test FileReader with compressed files."""
    with patch("smart_open.open") as mock_smart_open:
        # Mock smart_open for compressed file
        mock_smart_open.return_value.__enter__.return_value.read.return_value = "decompressed_content"

        file_paths = ["s3://bucket/compressed.txt.gz"]
        source_node = MockSourceNode(file_paths, {"source": "s3"})
        reader_node = FileReader(source_node, transport_params={"compression": ".gz"})

        # Read compressed file
        content = next(reader_node)
        assert content[FileReader.DATA_KEY] == "decompressed_content"
        assert content[FileReader.METADATA_KEY]["file_path"] == "s3://bucket/compressed.txt.gz"

        # Verify smart_open was called with compression parameters
        mock_smart_open.assert_called_once_with(
            "s3://bucket/compressed.txt.gz", "r", encoding="utf-8", transport_params={"compression": ".gz"}
        )


def test_file_reader_error_handling():
    """Test FileReader error handling."""
    with patch("smart_open.open") as mock_smart_open:
        # Mock smart_open to raise an exception
        mock_smart_open.side_effect = IOError("File not found")

        file_paths = ["nonexistent://file.txt"]
        source_node = MockSourceNode(file_paths)
        reader_node = FileReader(source_node)

        # Should raise the original exception
        with pytest.raises(IOError, match="File not found"):
            next(reader_node)


def test_fibonacci_backoff():
    """Test Fibonacci backoff calculation."""
    # Test Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, ...
    expected_delays = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]

    for attempt, expected_delay in enumerate(expected_delays, 1):
        actual_delay = _fibonacci_backoff(attempt)
        assert actual_delay == expected_delay, f"Attempt {attempt}: expected {expected_delay}, got {actual_delay}"

    # Test with custom base delay
    assert _fibonacci_backoff(1, base_delay=2.0) == 2.0
    assert _fibonacci_backoff(2, base_delay=2.0) == 2.0
    assert _fibonacci_backoff(3, base_delay=2.0) == 4.0

    # Test edge cases
    assert _fibonacci_backoff(0) == 0.0
    assert _fibonacci_backoff(-1) == 0.0


@patch("smart_open.open")
@patch("time.sleep")
def test_retry_logic_success_after_failure(mock_sleep, mock_smart_open):
    """Test retry logic when it succeeds after initial failures."""
    # Mock smart_open to fail twice then succeed
    mock_smart_open.side_effect = [
        Exception("Connection timeout"),  # First attempt fails
        Exception("Network error"),  # Second attempt fails
        MagicMock(
            __enter__=MagicMock(return_value=MagicMock(read=lambda: "success_content")),
            __exit__=MagicMock(return_value=None),
        ),  # Third attempt succeeds
    ]

    file_paths = ["s3://bucket/test.txt"]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node, max_retries=3)

    # Should succeed on third attempt
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == "success_content"

    # Verify sleep was called twice with Fibonacci delays
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)  # First retry delay
    mock_sleep.assert_any_call(1.0)  # Second retry delay


@patch("smart_open.open")
@patch("time.sleep")
def test_retry_logic_max_retries_exceeded(mock_sleep, mock_smart_open):
    """Test retry logic when max retries are exceeded."""
    # Mock smart_open to always fail
    mock_smart_open.side_effect = Exception("Connection timeout")

    file_paths = ["s3://bucket/test.txt"]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node, max_retries=2)

    # Should raise exception after max retries
    with pytest.raises(Exception, match="Connection timeout"):
        next(reader_node)

    # Verify sleep was called twice
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)  # First retry delay
    mock_sleep.assert_any_call(1.0)  # Second retry delay


@patch("smart_open.open")
@patch("time.sleep")
def test_retry_logic_file_not_found(mock_sleep, mock_smart_open):
    """Test retry logic with file not found error (should retry anyway)."""
    # Mock smart_open to fail with file not found
    mock_smart_open.side_effect = Exception("File not found")

    file_paths = ["s3://bucket/test.txt"]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node, max_retries=2)

    # Should retry and then raise exception
    with pytest.raises(Exception, match="File not found"):
        next(reader_node)

    # Verify sleep was called twice (retries on any error)
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)  # First retry delay
    mock_sleep.assert_any_call(1.0)  # Second retry delay


def test_file_reader_custom_max_retries():
    """Test FileReader with custom max_retries parameter."""
    file_paths = ["test.txt"]
    source_node = MockSourceNode(file_paths)

    # Test default max_retries
    reader_default = FileReader(source_node)
    assert reader_default.max_retries == 3

    # Test custom max_retries
    reader_custom = FileReader(source_node, max_retries=5)
    assert reader_custom.max_retries == 5

    # Test zero retries
    reader_zero = FileReader(source_node, max_retries=0)
    assert reader_zero.max_retries == 0


@patch("smart_open.open")
@patch("time.sleep")
def test_retry_logic_zero_retries(mock_sleep, mock_smart_open):
    """Test retry logic with zero retries (should fail immediately)."""
    # Mock smart_open to fail
    mock_smart_open.side_effect = Exception("Connection timeout")

    file_paths = ["s3://bucket/test.txt"]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node, max_retries=0)

    # Should fail immediately without retrying
    with pytest.raises(Exception, match="Connection timeout"):
        next(reader_node)

    # Verify sleep was never called
    mock_sleep.assert_not_called()


@patch("smart_open.open")
@patch("time.sleep")
def test_retry_logic_break_on_success(mock_sleep, mock_smart_open):
    """Test that the loop breaks immediately on successful file opening."""
    # Mock smart_open to succeed on first attempt
    mock_smart_open.return_value.__enter__.return_value.read.return_value = "success_content"
    mock_smart_open.return_value.__exit__.return_value = None

    file_paths = ["s3://bucket/test.txt"]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node, max_retries=3)

    # Should succeed immediately
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == "success_content"

    # Verify sleep was never called (no retries needed)
    mock_sleep.assert_not_called()

    # Verify smart_open was called exactly once
    assert mock_smart_open.call_count == 1


@patch("smart_open.open")
@patch("time.sleep")
def test_retry_logic_reading_error_not_retried(mock_sleep, mock_smart_open):
    """Test that reading errors are not retried (only opening errors are retried)."""
    # Mock smart_open to succeed on opening but fail on reading
    mock_file = MagicMock()
    mock_file.read.side_effect = Exception("Reading error")
    mock_smart_open.return_value.__enter__.return_value = mock_file
    mock_smart_open.return_value.__exit__.return_value = None

    file_paths = ["s3://bucket/test.txt"]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node, max_retries=3)

    # Should fail immediately on reading error (no retries)
    with pytest.raises(Exception, match="Reading error"):
        next(reader_node)

    # Verify sleep was never called (no retries for reading errors)
    mock_sleep.assert_not_called()

    # Verify smart_open was called exactly once
    assert mock_smart_open.call_count == 1


def test_file_reader_metadata_preservation():
    """Test that metadata from source node is properly preserved."""
    file_paths = ["test.txt"]
    source_metadata = {"source": "test", "batch": 1, "custom_key": "custom_value"}
    source_node = MockSourceNode(file_paths, source_metadata)

    with patch("smart_open.open") as mock_smart_open:
        mock_smart_open.return_value.__enter__.return_value.read.return_value = "test content"
        mock_smart_open.return_value.__exit__.return_value = None

        reader_node = FileReader(source_node)
        result = next(reader_node)

        # Check that source metadata is preserved
        assert result[FileReader.METADATA_KEY]["source"] == "test"
        assert result[FileReader.METADATA_KEY]["batch"] == 1
        assert result[FileReader.METADATA_KEY]["custom_key"] == "custom_value"
        assert result[FileReader.METADATA_KEY]["file_path"] == "test.txt"
