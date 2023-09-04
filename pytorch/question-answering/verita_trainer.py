from typing import Optional
from transformers import Trainer

class VeritaTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        print('VeritaTrainer: saving model to ', output_dir)
        self.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
        print('VeritaTrainer: saved the model?')
