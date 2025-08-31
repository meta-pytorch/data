import logging
import time
from typing import Any, Dict, Optional, Union

import smart_open
from torchdata.nodes import BaseNode

logger = logging.getLogger(__name__)


def _fibonacci_backoff(attempt: int, base_delay: float = 1.0) -> float:
    """Calculate Fibonacci backoff delay for retry attempts.

    Args:
        attempt: Current attempt number (1-based)
        base_delay: Base delay in seconds

    Returns:
        float: Delay in seconds before next retry
    """
    if attempt <= 0:
        return 0.0

    # Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, ...
    fib_sequence = [1, 1]
    for i in range(2, attempt + 1):
        fib_sequence.append(fib_sequence[i - 1] + fib_sequence[i - 2])

    return base_delay * fib_sequence[attempt - 1]


class TextStreamingDecoder(BaseNode[Dict]):
    """Node that streams text files line by line from any source.

    This node combines functionality of file reading and line-by-line processing,
    supporting both local and remote (S3, GCS, Azure, HTTP, etc.) files via smart_open.

    Features:
    - Streams files line-by-line (memory efficient)
    - Supports any filesystem that smart_open supports (local, S3, GCS, Azure, HTTP, etc.)
    - Handles compressed files (.gz, .bz2) transparently
    - Maintains state for checkpointing and resumption
    - Preserves metadata from source nodes
    - Automatic retry with Fibonacci backoff for file opening errors

    Input format:
        {
            "data": "path/to/file.txt",  # File path (local) or URI (s3://, etc.)
            "metadata": {...}             # Optional metadata
        }
        or simply a string with the file path/URI

    Output format:
        {
            "data": "line content",       # Single line of text
            "metadata": {
                "file_path": "path/to/file.txt",
                "item_idx": 0,            # 0-based line index
                ...                       # Additional metadata from input
            }
        }

    Examples:
        >>> # Stream from local file
        >>> node = TextStreamingDecoder(source_node)
        >>>
        >>> # Stream from S3 with custom client and retry logic
        >>> node = TextStreamingDecoder(
        ...     source_node,
        ...     transport_params={'client': boto3.client('s3')},
        ...     max_retries=5
        ... )
        >>>
        >>> # Stream compressed files
        >>> node = TextStreamingDecoder(
        ...     source_node,
        ...     transport_params={'compression': '.gz'}
        ... )
    """

    SOURCE_KEY = "source"
    DATA_KEY = "data"
    METADATA_KEY = "metadata"
    CURRENT_FILE_KEY = "current_file"
    CURRENT_LINE_KEY = "current_line"

    def __init__(
        self,
        source_node: BaseNode[Union[str, Dict]],
        mode: str = "r",
        encoding: Optional[str] = "utf-8",
        transport_params: Optional[Dict] = None,
        max_retries: int = 3,
    ):
        """Initialize the TextStreamingDecoder.

        Args:
            source_node: Source node that yields dicts with file paths
            mode: File open mode ('r' for text, 'rb' for binary)
            encoding: Text encoding (None for binary mode)
            transport_params: Parameters for smart_open transport layer
                For S3:
                    {'client': boto3.client('s3')}  # Use specific client
                For compression:
                    {'compression': '.gz'}  # Force gzip compression
                    {'compression': '.bz2'}  # Force bz2 compression
                    {'compression': 'disable'}  # Disable compression
            max_retries: Maximum number of retry attempts for file opening errors (default: 3)
        """
        super().__init__()
        self.source = source_node
        self.mode = mode
        self.encoding = encoding
        self.transport_params = transport_params or {}
        self.max_retries = max_retries
        self._current_file = None
        self._current_line = 0
        self._file_handle = None
        self._source_metadata = {}

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """Reset must fully initialize the node's state.

        Args:
            initial_state: Optional state dictionary for resumption
        """
        super().reset(initial_state)

        # Close any open file
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

        if initial_state is None:
            # Full reset
            self.source.reset(None)
            self._current_file = None
            self._current_line = 0
            self._source_metadata = {}
        else:
            # Restore source state
            self.source.reset(initial_state[self.SOURCE_KEY])
            self._current_file = initial_state[self.CURRENT_FILE_KEY]
            self._current_line = initial_state.get(self.CURRENT_LINE_KEY, 0)
            self._source_metadata = initial_state.get(self.METADATA_KEY, {})

            # If we have a file to resume, open and seek to position
            if self._current_file is not None:
                # Retry logic for file opening during state restoration
                for attempt in range(1, self.max_retries + 1):
                    try:
                        self._file_handle = smart_open.open(
                            self._current_file,
                            self.mode,
                            encoding=self.encoding,
                            transport_params=self.transport_params,
                        )
                        # Skip lines to resume position
                        for _ in range(self._current_line):
                            next(self._file_handle)
                        break  # Successfully opened and positioned

                    except Exception as e:
                        if attempt < self.max_retries:
                            delay = _fibonacci_backoff(attempt)
                            logger.warning(
                                f"Error opening {self._current_file} during state restoration (attempt {attempt}/{self.max_retries}): {e}. Retrying in {delay:.2f}s..."
                            )
                            time.sleep(delay)
                        else:
                            # Max retries reached, log error and continue without file handle
                            logger.error(
                                f"Failed to open {self._current_file} during state restoration after {self.max_retries} attempts. Last error: {e}"
                            )
                            self._file_handle = None
                            break

    def __del__(self):
        """Ensure file is closed on deletion."""
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass  # Ignore errors during cleanup

    def _get_next_file(self) -> bool:
        """Get the next file and open it for reading.

        Returns:
            bool: True if a new file was successfully opened, False otherwise.
        """
        try:
            # Get next file from source
            file_data = self.source.next()

            # Extract file path from data
            if isinstance(file_data, dict) and self.DATA_KEY in file_data:
                self._current_file = file_data[self.DATA_KEY]
                # Copy metadata from source
                if self.METADATA_KEY in file_data:
                    self._source_metadata = file_data[self.METADATA_KEY]
            else:
                self._current_file = file_data
                self._source_metadata = {}

            # Retry logic for file opening
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Try to open the file
                    self._file_handle = smart_open.open(
                        self._current_file, self.mode, encoding=self.encoding, transport_params=self.transport_params
                    )
                    self._current_line = 0
                    return True

                except Exception as e:
                    if attempt < self.max_retries:
                        delay = _fibonacci_backoff(attempt)
                        logger.warning(
                            f"Error opening {self._current_file} (attempt {attempt}/{self.max_retries}): {e}. Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        # Max retries reached
                        logger.error(
                            f"Failed to open {self._current_file} after {self.max_retries} attempts. Last error: {e}"
                        )
                        self._file_handle = None
                        return False  # Failed to open file

        except StopIteration:
            # No more files
            raise

    def _get_next_line(self) -> Dict:
        """Read the next line from the current file.

        Returns:
            Dict: Dictionary with the line data and metadata.

        Raises:
            StopIteration: If end of file is reached and no more files are available.
        """
        try:
            line = self._file_handle.readline()

            # EOF or empty line at end of file
            if not line:
                self._file_handle.close()
                self._file_handle = None
                return None  # Signal end of file

            # Create output with metadata
            metadata = {"file_path": self._current_file, "item_idx": self._current_line}

            # Include metadata from source
            if self._source_metadata:
                metadata.update(self._source_metadata)

            self._current_line += 1

            return {self.DATA_KEY: line.rstrip("\n"), self.METADATA_KEY: metadata}

        except Exception as e:
            logger.error(f"Error reading from {self._current_file}: {e}")
            if self._file_handle:
                self._file_handle.close()
            self._file_handle = None
            return None  # Signal error

    def next(self) -> Dict:
        """Get the next line from current file or next available file."""
        # Loop until we get a valid line or run out of files
        while True:
            # If we don't have a file handle, get a new one
            while self._file_handle is None:
                if not self._get_next_file():
                    continue  # Try the next file if this one failed

            # Try to get the next line
            line_data = self._get_next_line()

            # If we reached the end of the file, try the next one
            if line_data is None:
                continue

            # We got a valid line, return it
            return line_data

    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing."""
        return {
            self.SOURCE_KEY: self.source.state_dict(),
            self.CURRENT_FILE_KEY: self._current_file,
            self.CURRENT_LINE_KEY: self._current_line,
        }

    def shutdown(self):
        """Shutdown the node."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
