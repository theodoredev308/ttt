import wandb
import time

class WanDBInfo:
    def __init__(self, entity: str, project: str, run_id: str):
        self.api = wandb.Api()

        self.entity = entity
        self.project = project
        self.run_id = run_id

        self.data = {}

        self.run = self.api.run(f"{self.entity}/{self.project}/{self.run_id}")
        self.current = 0

    def sync(self):
        history = self.run.history()
        self.current = len(history)
        if len(history) > 0:
            self.data = {col: history[col].tolist() for col in history.columns}
            return self.data
        else:
            print("History is empty.")
            return None

    def get_gradient_score(self):
        result = {}
        for uid in range(0, 256):
            mask = f"latest/validator/gradient_scores/{uid}"
            if mask in self.data.keys():
                val = self.data[mask][-1]
                if val == 0.0 or val is None or (val != val):
                    continue
                result[uid] = val
        return result

    def get_gradient_score_uids(self, top_n: int = 5):
        result = self.get_gradient_score()
        sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        result = [uid for uid, _ in sorted_result]
        return result[:top_n]

    def get_ordinal(self):
        result = {}
        for uid in range(0, 256):
            mask = f"latest/validator/openskill/ordinal/{uid}"
            if mask in self.data.keys():
                for i in range(0, 10):
                    value = self.data[mask][-i-1]
                    if value == 0.0 or value is None or (value != value):
                        continue
                    result[uid] = value
                    break
        return result

    def get_ordinal_uids(self, top_n: int = 5):
        result = self.get_ordinal()
        sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        result = [uid for uid, _ in sorted_result]
        return result[:top_n]

    def get_final_score(self):
        result = {}
        for uid in range(0, 256):
            mask = f"latest/validator/final_scores/{uid}"
            if mask in self.data.keys():
                val = self.data[mask][-1]
                if val == 0.0 or val is None or (val != val):
                    continue
                result[uid] = val
        return result
    
    def get_sync_score(self):
        result = {}
        for uid in range(0, 256):
            mask = f"latest/validator/sync_score/{uid}"
            if mask in self.data.keys():
                val = self.data[mask][-1]
                if val == 0.0 or val is None or (val != val):
                    continue
                result[uid] = val
        return result

    def get_sync_score_uids(self, top_n: int = 5):
        result = self.get_sync_score()
        sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        result = [uid for uid, _ in sorted_result]
        return result[:top_n]

    def get_binary_moving_averages(self):
        result = {}
        for uid in range(0, 256):
            mask = f"latest/validator/binary_moving_averages/{uid}"
            if mask in self.data.keys():
                result[uid] = self.data[mask]
        return result
    
    def get_binary_moving_averages_uids(self, top_n: int = 5):
        result = self.get_binary_moving_averages()
        sorted_result = sorted(result.items(), key=lambda x: x[1][-1], reverse=True)
        result = [uid for uid, _ in sorted_result]
        return result[:top_n]

    def get_final_score(self):
        result = {}
        for uid in range(0, 256):
            mask = f"latest/validator/final_scores/{uid}"
            if mask in self.data.keys() and self.data[mask][-1] != 0.0:
                result[uid] = self.data[mask]
        return result

    def get_final_score_uids(self, top_n: int = 5):
        result = self.get_final_score()
        sorted_result = sorted(result.items(), key=lambda x: x[1][-1], reverse=True)
        result = [uid for uid, _ in sorted_result]
        return result[:top_n]

    def get_weights(self):
        result = {}
        for uid in range(0, 256):
            mask = f"latest/validator/weights/{uid}"
            if mask in self.data.keys():
                result[uid] = self.data[mask]
        return result

    def get_weights_uids(self, top_n: int = 5):
        result = self.get_weights()
        sorted_result = sorted(result.items(), key=lambda x: x[1][-1], reverse=True)
        result = [uid for uid, _ in sorted_result]
        return result[:top_n]