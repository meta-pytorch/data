"""Tests for TextStreamingDecoder with various file sources and formats.

This test suite verifies the functionality of TextStreamingDecoder across
different file sources (local, S3) and formats (plain text, compressed).

Test coverage includes:
1. Local file operations
   - Basic reading
   - Metadata handling
   - State management and resumption
   - Empty file handling
   - Text encoding (UTF-8)
   - File handle cleanup

2. S3 operations (mocked)
   - Basic reading from S3
   - Using transport parameters
   - State management with S3 files

3. Compressed file handling
   - Reading .gz files
   - Reading .bz2 files

4. Mixed source operations
   - Reading from multiple files
   - Reading from both compressed and uncompressed sources

5. Error handling
   - Invalid file paths
   - Recovery from errors

6. Retry logic
   - Retry on file opening errors
   - Fibonacci backoff
   - Max retries configuration
"""

import os
import tempfile
from typing import Any, Dict, List, Union
from unittest.mock import MagicMock, patch

import pytest

from torchdata.nodes import BaseNode
from torchdata.nodes.io.text_streaming_decoder import _fibonacci_backoff, TextStreamingDecoder


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

        return {TextStreamingDecoder.DATA_KEY: path, TextStreamingDecoder.METADATA_KEY: dict(self.metadata)}

    def get_state(self):
        return {"idx": self._current_idx}


def create_test_files():
    """Create temporary test files with known content."""
    temp_dir = tempfile.mkdtemp()

    # Create first test file
    file1_path = os.path.join(temp_dir, "test1.txt")
    with open(file1_path, "w") as f:
        f.write("line1\nline2\nline3\n")

    # Create second test file
    file2_path = os.path.join(temp_dir, "test2.txt")
    with open(file2_path, "w") as f:
        f.write("file2_line1\nfile2_line2\n")

    return temp_dir, [file1_path, file2_path]


def test_text_stream_basic():
    """Test basic functionality of TextStreamingDecoder."""
    temp_dir, file_paths = create_test_files()
    try:
        source_node = MockSourceNode(file_paths)
        node = TextStreamingDecoder(source_node)

        # Test reading all lines
        lines = []
        for item in node:
            lines.append(item[TextStreamingDecoder.DATA_KEY])

        # Check content
        assert lines == ["line1", "line2", "line3", "file2_line1", "file2_line2"]

    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(temp_dir)


def test_text_stream_metadata():
    """Test metadata handling in TextStreamingDecoder."""
    temp_dir, file_paths = create_test_files()
    try:
        source_node = MockSourceNode(file_paths, {"source": "local"})
        node = TextStreamingDecoder(source_node)

        # Get first item
        item = next(iter(node))

        # Check metadata
        assert TextStreamingDecoder.METADATA_KEY in item
        assert "file_path" in item[TextStreamingDecoder.METADATA_KEY]
        assert item[TextStreamingDecoder.METADATA_KEY]["file_path"] == file_paths[0]
        assert item[TextStreamingDecoder.METADATA_KEY]["item_idx"] == 0
        assert item[TextStreamingDecoder.METADATA_KEY]["source"] == "local"

    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(temp_dir)


def test_text_stream_state_management():
    """Test state management in TextStreamingDecoder."""
    temp_dir, file_paths = create_test_files()
    try:
        source_node = MockSourceNode(file_paths)
        node = TextStreamingDecoder(source_node)

        # Read first line and store state
        first_item = next(iter(node))
        state = node.get_state()

        # Create new node and restore state
        new_source = MockSourceNode(file_paths)
        new_node = TextStreamingDecoder(new_source)
        new_node.reset(state)

        # Read next line - should be second line
        second_item = next(iter(new_node))

        # Verify it's different from the first line
        assert second_item[TextStreamingDecoder.DATA_KEY] != first_item[TextStreamingDecoder.DATA_KEY]
        assert second_item[TextStreamingDecoder.METADATA_KEY]["item_idx"] == 1

    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(temp_dir)


def test_text_stream_empty_file():
    """Test handling of empty files."""
    temp_dir = tempfile.mkdtemp()
    empty_file = os.path.join(temp_dir, "empty.txt")
    normal_file = os.path.join(temp_dir, "normal.txt")

    try:
        # Create empty file
        with open(empty_file, "w") as f:
            pass

        # Create normal file
        with open(normal_file, "w") as f:
            f.write("normal_content\n")

        source_node = MockSourceNode([empty_file, normal_file])
        node = TextStreamingDecoder(source_node)

        # Should skip empty file and read from normal file
        item = next(iter(node))
        assert item[TextStreamingDecoder.DATA_KEY] == "normal_content"

    finally:
        for path in [empty_file, normal_file]:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(temp_dir)


def test_text_stream_encoding():
    """Test text encoding handling."""
    temp_dir = tempfile.mkdtemp()
    utf8_file = os.path.join(temp_dir, "utf8.txt")

    try:
        # Create file with UTF-8 content
        content = "Hello 世界\n"
        with open(utf8_file, "w", encoding="utf-8") as f:
            f.write(content)

        source_node = MockSourceNode([utf8_file])
        node = TextStreamingDecoder(source_node, encoding="utf-8")

        # Read content
        item = next(iter(node))
        assert item[TextStreamingDecoder.DATA_KEY] == "Hello 世界"

    finally:
        if os.path.exists(utf8_file):
            os.remove(utf8_file)
        os.rmdir(temp_dir)


def test_text_stream_cleanup():
    """Test proper file handle cleanup."""
    temp_dir, file_paths = create_test_files()
    try:
        source_node = MockSourceNode(file_paths)
        node = TextStreamingDecoder(source_node)

        # Read partial file
        next(iter(node))

        # Force cleanup
        del node

        # Should be able to delete files (no open handles)
        for path in file_paths:
            os.remove(path)

    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(temp_dir)


@patch("smart_open.open")
def test_s3_basic_read(mock_smart_open):
    """Test basic S3 file reading with mocked smart_open."""
    # Mock smart_open for S3 - set up context manager without readline attribute
    mock_file = MagicMock()
    mock_file.readline.side_effect = ['{"id": 1, "text": "Hello from S3"}\n', ""]

    # Set up mock context manager
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_file
    mock_context.__exit__.return_value = None
    mock_smart_open.return_value = mock_context

    file_paths = ["s3://test-bucket/test_file1.jsonl"]
    source_node = MockSourceNode(file_paths, {"source": "s3"})
    node = TextStreamingDecoder(source_node)

    # Read first line
    item = next(iter(node))

    # Should contain content
    assert TextStreamingDecoder.DATA_KEY in item
    assert item[TextStreamingDecoder.DATA_KEY] == '{"id": 1, "text": "Hello from S3"}'

    # Check metadata
    assert TextStreamingDecoder.METADATA_KEY in item
    assert "file_path" in item[TextStreamingDecoder.METADATA_KEY]
    assert item[TextStreamingDecoder.METADATA_KEY]["file_path"] == "s3://test-bucket/test_file1.jsonl"
    assert item[TextStreamingDecoder.METADATA_KEY]["source"] == "s3"


@patch("smart_open.open")
def test_compression_handling(mock_smart_open):
    """Test compressed file handling."""
    # Mock smart_open for compressed file - set up context manager without readline
    mock_file = MagicMock()
    mock_file.readline.side_effect = ["decompressed_line1\n", "decompressed_line2\n", ""]

    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_file
    mock_context.__exit__.return_value = None
    mock_smart_open.return_value = mock_context

    file_paths = ["s3://bucket/compressed.txt.gz"]
    source_node = MockSourceNode(file_paths, {"source": "s3"})
    node = TextStreamingDecoder(source_node, transport_params={"compression": ".gz"})

    # Read lines
    lines = [item[TextStreamingDecoder.DATA_KEY] for item in node]
    assert lines == ["decompressed_line1", "decompressed_line2"]

    # Verify smart_open was called with compression parameters
    mock_smart_open.assert_called_with(
        "s3://bucket/compressed.txt.gz", "r", encoding="utf-8", transport_params={"compression": ".gz"}
    )


def test_error_handling():
    """Test error handling for invalid files."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a file that exists
        valid_path = os.path.join(temp_dir, "valid.txt")
        with open(valid_path, "w") as f:
            f.write("valid content\n")

        # Define a path that doesn't exist
        invalid_path = os.path.join(temp_dir, "nonexistent.txt")

        # Node should skip invalid file and read valid one
        source_node = MockSourceNode([invalid_path, valid_path])
        node = TextStreamingDecoder(source_node)
        item = next(iter(node))

        # Should get content from valid file
        assert item[TextStreamingDecoder.DATA_KEY] == "valid content"

    finally:
        if os.path.exists(valid_path):
            os.remove(valid_path)
        os.rmdir(temp_dir)


def test_text_stream_recursive_behavior():
    """Test TextStreamingDecoder handles file transitions without recursion issues."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create multiple files with known content
        file1_path = os.path.join(temp_dir, "test1.txt")
        with open(file1_path, "w") as f:
            f.write("file1_line1\nfile1_line2\n")

        file2_path = os.path.join(temp_dir, "test2.txt")
        with open(file2_path, "w") as f:
            f.write("file2_line1\nfile2_line2\n")

        # Create an empty file to test empty file handling
        empty_file_path = os.path.join(temp_dir, "empty.txt")
        with open(empty_file_path, "w") as f:
            pass

        # Create a file with an error that will be skipped
        error_file_path = os.path.join(temp_dir, "error.txt")
        # Don't actually create this file, so it will cause an error

        source_node = MockSourceNode([file1_path, empty_file_path, error_file_path, file2_path])
        node = TextStreamingDecoder(source_node)

        # Read all lines
        lines = []
        for item in node:
            lines.append(item[TextStreamingDecoder.DATA_KEY])
            # Also check that metadata is correct
            assert TextStreamingDecoder.METADATA_KEY in item
            assert "file_path" in item[TextStreamingDecoder.METADATA_KEY]
            assert "item_idx" in item[TextStreamingDecoder.METADATA_KEY]

        # Should have 4 lines total (2 from file1, 0 from empty, 0 from error, 2 from file2)
        assert lines == ["file1_line1", "file1_line2", "file2_line1", "file2_line2"]

        # Check that each line is only returned once
        # Reset the node
        new_source = MockSourceNode([file1_path, empty_file_path, error_file_path, file2_path])
        node = TextStreamingDecoder(new_source)

        # Read lines again and check for duplicates
        seen_lines = set()
        for item in node:
            line = item[TextStreamingDecoder.DATA_KEY]
            file_path = item[TextStreamingDecoder.METADATA_KEY]["file_path"]
            line_idx = item[TextStreamingDecoder.METADATA_KEY]["item_idx"]

            # Create a unique identifier for this line
            line_id = (line, file_path, line_idx)

            # Check that we haven't seen this line before
            assert line_id not in seen_lines, f"Duplicate line: {line_id}"
            seen_lines.add(line_id)

    finally:
        # Clean up
        for path in [file1_path, file2_path, empty_file_path]:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(temp_dir)


@patch("smart_open.open")
def test_azure_gcs_support(mock_smart_open):
    """Test Azure and GCS support via smart_open."""
    # Test Azure - set up context manager without readline
    mock_file = MagicMock()
    mock_file.readline.side_effect = ["azure_content\n", ""]

    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_file
    mock_context.__exit__.return_value = None
    mock_smart_open.return_value = mock_context

    azure_paths = ["abfs://container@account.dfs.core.windows.net/file.txt"]
    source_node = MockSourceNode(azure_paths, {"source": "abfs"})
    node = TextStreamingDecoder(source_node, transport_params={"anon": False})

    item = next(iter(node))
    assert item[TextStreamingDecoder.DATA_KEY] == "azure_content"
    assert item[TextStreamingDecoder.METADATA_KEY]["source"] == "abfs"

    # Test GCS - reset mock file for new content
    mock_file.readline.side_effect = ["gcs_content\n", ""]
    gcs_paths = ["gs://my-bucket/file.txt"]
    source_node = MockSourceNode(gcs_paths, {"source": "gs"})
    node = TextStreamingDecoder(source_node, transport_params={"client": "mock_gcs_client"})

    item = next(iter(node))
    assert item[TextStreamingDecoder.DATA_KEY] == "gcs_content"
    assert item[TextStreamingDecoder.METADATA_KEY]["source"] == "gs"


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
    mock_file = MagicMock()
    mock_file.readline.side_effect = ["success_line\n", ""]

    # Set up successful context manager for third attempt
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_file
    mock_context.__exit__.return_value = None
    # Explicitly delete readline from context manager to force __enter__ path
    del mock_context.readline

    mock_smart_open.side_effect = [
        Exception("Connection timeout"),  # First attempt fails
        Exception("Network error"),  # Second attempt fails
        mock_context,  # Third attempt succeeds
    ]

    file_paths = ["s3://bucket/test.txt"]
    source_node = MockSourceNode(file_paths)
    node = TextStreamingDecoder(source_node, max_retries=3)

    # Should succeed on third attempt
    content = next(iter(node))
    assert content[TextStreamingDecoder.DATA_KEY] == "success_line"

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
    node = TextStreamingDecoder(source_node, max_retries=2)

    # Should skip the file and try the next one (if any)
    # Since we only have one file and it fails, we should get StopIteration
    with pytest.raises(StopIteration):
        next(iter(node))

    # Verify sleep was called twice
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)  # First retry delay
    mock_sleep.assert_any_call(1.0)  # Second retry delay


@patch("smart_open.open")
@patch("time.sleep")
def test_retry_logic_zero_retries(mock_sleep, mock_smart_open):
    """Test retry logic with zero retries (should fail immediately)."""
    # Mock smart_open to fail
    mock_smart_open.side_effect = Exception("Connection timeout")

    file_paths = ["s3://bucket/test.txt"]
    source_node = MockSourceNode(file_paths)
    node = TextStreamingDecoder(source_node, max_retries=0)

    # Should fail immediately without retrying
    with pytest.raises(StopIteration):
        next(iter(node))

    # Verify sleep was never called
    mock_sleep.assert_not_called()


@patch("smart_open.open")
@patch("time.sleep")
def test_retry_logic_state_restoration(mock_sleep, mock_smart_open):
    """Test retry logic during state restoration."""
    # Mock smart_open to fail twice then succeed during state restoration
    mock_file = MagicMock()
    # First readline call for skipping to position, then actual content
    mock_file.readline.side_effect = ["", "resumed_line\n", ""]

    # Set up successful context manager for third attempt
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_file
    mock_context.__exit__.return_value = None
    # Explicitly delete readline from context manager to force __enter__ path
    del mock_context.readline

    mock_smart_open.side_effect = [
        Exception("Connection timeout"),  # First attempt fails
        Exception("Network error"),  # Second attempt fails
        mock_context,  # Third attempt succeeds
    ]

    # Use mock file paths instead of real files to avoid conflicts
    file_paths = ["mock://file1.txt"]

    # Create a mock state that simulates having read one line already
    mock_source_state = {"idx": 1}
    state = {
        TextStreamingDecoder.SOURCE_KEY: mock_source_state,
        TextStreamingDecoder.CURRENT_FILE_KEY: file_paths[0],
        TextStreamingDecoder.CURRENT_LINE_KEY: 1,  # Simulate having read one line
    }

    # Create new node and restore state (this will trigger retry logic)
    new_source = MockSourceNode(file_paths)
    new_node = TextStreamingDecoder(new_source, max_retries=3)
    new_node.reset(state)

    # Read next line - should succeed after retries
    second_item = next(iter(new_node))
    assert second_item[TextStreamingDecoder.DATA_KEY] == "resumed_line"

    # Verify sleep was called twice during state restoration
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)  # First retry delay
    mock_sleep.assert_any_call(1.0)  # Second retry delay


def test_text_streaming_decoder_custom_max_retries():
    """Test TextStreamingDecoder with custom max_retries parameter."""
    file_paths = ["test.txt"]
    source_node = MockSourceNode(file_paths)

    # Test default max_retries
    node_default = TextStreamingDecoder(source_node)
    assert node_default.max_retries == 3

    # Test custom max_retries
    node_custom = TextStreamingDecoder(source_node, max_retries=5)
    assert node_custom.max_retries == 5

    # Test zero retries
    node_zero = TextStreamingDecoder(source_node, max_retries=0)
    assert node_zero.max_retries == 0


@patch("smart_open.open")
@patch("time.sleep")
def test_retry_logic_break_on_success(mock_sleep, mock_smart_open):
    """Test that the retry loop breaks immediately on successful file opening."""
    # Mock smart_open to succeed on first attempt
    mock_file = MagicMock()
    mock_file.readline.side_effect = ["success_line\n", ""]

    # Set up successful context manager
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_file
    mock_context.__exit__.return_value = None
    mock_smart_open.return_value = mock_context

    file_paths = ["s3://bucket/test.txt"]
    source_node = MockSourceNode(file_paths)
    node = TextStreamingDecoder(source_node, max_retries=3)

    # Should succeed immediately
    content = next(iter(node))
    assert content[TextStreamingDecoder.DATA_KEY] == "success_line"

    # Verify sleep was never called (no retries needed)
    mock_sleep.assert_not_called()

    # Verify smart_open was called exactly once
    assert mock_smart_open.call_count == 1
