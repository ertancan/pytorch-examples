from typing import Optional
from transformers import Trainer
import os
import boto3
import subprocess

s3 = boto3.resource('s3')
class VeritaTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        leaf_folder = output_dir.split('/')[-1]
        print('VeritaTrainer: saving model to ', output_dir)
        if output_dir.startswith("s3://"):
            temp_folder = '/root/.cache/'
            cache_path = os.path.join(temp_folder, leaf_folder)
            os.makedirs(cache_path, exist_ok=True)
            print('Saving the model to cache_path ', cache_path)
            self.model.save_pretrained(cache_path, safe_serialization=self.args.save_safetensors)
            upload_command = ['aws', 's3', 'cp', cache_path, output_dir, '--recursive']
            print(subprocess.run(upload_command, capture_output=True))
            #TODO: delete the cache file if needed
        else:
            self.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
        print('VeritaTrainer: saved the model?')
