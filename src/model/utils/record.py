import csv
import os


class CSVLogSave:
    def __init__(self, root: str, key_name: str) -> None:
        self.HEADER_NAMES = ["key", "value", "timestamp", "step"]
        self.root = root
        self.key_name = key_name
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, self.key_name + ".csv")
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER_NAMES)
            f.close()

    def save(self, value: float, timestamp, step: int):
        append_dict = {
            self.HEADER_NAMES[0]: self.key_name,
            self.HEADER_NAMES[1]: value,
            self.HEADER_NAMES[2]: timestamp,
            self.HEADER_NAMES[3]: step,
        }
        with open(self.path, "a", newline="") as f:
            dictwriter_object = csv.DictWriter(f, fieldnames=self.HEADER_NAMES)
            dictwriter_object.writerow(append_dict)
            f.close()
