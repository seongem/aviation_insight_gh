import logging
import os
from datetime import datetime

class DiagnosticFileHandler(logging.Handler):
    """
    Custom logging handler that mirrors your precise file logic.
    Accepts log records and writes them exclusively to a dedicated 
    timestamped file, keeping the main console completely unpolluted.
    """
    def __init__(self, issue_token: str, output_dir: str = "logs/diagnostics"):
        super().__init__()
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate clean, unified timestamp strings exactly like before
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.filename = f"{output_dir}/matrix_{issue_token.upper()}_{self.timestamp}.log"
        self.log_file = open(self.filename, "w", encoding="utf-8")
        print(f"📝 Logging session outputs directly to: {self.filename}")

    def emit(self, record):
        """Standard entrypoint called by the Python logging ecosystem."""
        try:
            message = self.format(record)
            self.log_file.write(message + '\n')
            self.log_file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        """Cleans up the file descriptor on engine shutdown."""
        self.log_file.close()
        super().close()