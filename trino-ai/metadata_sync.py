import threading
import time
from embeddings import embedding_service

class MetadataSync:
    def __init__(self, interval=3600):
        self.interval = interval
        self._stop_event = threading.Event()

    def start(self):
        thread = threading.Thread(target=self._run)
        thread.daemon = True  # Daemon thread will shut down with the main program
        thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                embedding_service.refresh_embeddings()
                print("Metadata sync completed successfully")
            except Exception as e:
                print(f"Error during metadata sync: {e}")
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()

# Start sync on module import
metadata_sync = MetadataSync()
metadata_sync.start() 