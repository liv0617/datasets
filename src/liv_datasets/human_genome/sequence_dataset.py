import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from logging import getLogger

logger = getLogger(__name__)

class SequenceDataset(Dataset):

    def __init__(self, data_path, shuffle=False) -> None:
        self._data = open(data_path, 'r')
        self._data_path = data_path
        self.length = self._get_length()
        if shuffle:
            self._shuffle()

    def _shuffle(self, total_shards=100) -> None:
        """ Shuffle all lines in sequence file. """
        os.makedirs('shard_tmp', exist_ok=True)
        # write all lines from previous data file to a randomly chosen shard file
        shard_files = {f'shard_{i}': open(f'shard_tmp/shard_{i}.txt', 'w+') for i in range(total_shards)}
        line = self._data.readline()
        while line:
            shard_files[np.random.choice(shard_files.keys())].write(line)
            line = self._data.readline()
        
        logger.info(f'Sharded {self._data_path} (shard count: {total_shards}) to shard_tmp/')
        # shuffle each shard file
        shard_files_shuffled = {f'shard_{i}': open(f'shard_tmp/shard_{i}_shuffled.txt', 'w+') for i in range(total_shards)}
        for shard_name, shard_file in tqdm(shard_files.items(), desc=f'Sharding data in {self._data_path}...'):
            shuffled_lines = np.random.shuffle(shard_file.readlines())
            shard_files_shuffled[shard_name].writelines(shuffled_lines)
        
        logger.info(f'Shuffled all {total_shards} shards in shard_tmp/')
        
        # close the unshuffled shard files
        for shard_file in tqdm(shard_files.values(), desc="Shuffling shards..."):
            shard_file.close()

        # concatenate shuffled files
        new_data_path = f"{self._data_path.split('.')[0]}_shuffled.txt"
        data = open(new_data_path, 'w+')
        for shard_file in tqdm(shard_files_shuffled.values(), desc=f'Concatenating shards to {new_data_path}...'):
            data.writelines(shard_file.readlines())
            shard_file.close()

        logger.info(f'Concatenated shuffled shard files and saved to {new_data_path}')

        self.reset(data_path=new_data_path)

    def reset(self, data_path: str = None, reset_length: bool = False, rm_old_file: bool = False) -> None:
        """
        Reset data object to the start of the file.

        Parameters
        ----------
        - data_path [str]: if file path for data has been updated (i.e. in the case of a shuffle),
        update the data object to be this file.
        - reset_length [bool]: Whether to re-calculate the length of the file path. If file has just
        been shuffled, best to not recalculate.
        - rm_old_file [bool]: Whether to delete the file that previously held the data.
        Good for if shuffle is permanent and you want to save space. 
        """
        
        logger.info('Reseting data object...')
        
        if not data_path:
            self._data.seek(0)
        
        else:
            self._data.close()
            if rm_old_file:
                if self._data_path == data_path:
                    logger.exception(f'Cannot delete data file at path {self._data_path} as new data_path provided is identical.')
                else:
                    os.remove(self._data_path)
            self._data_path = data_path
            self._data = open(self._data_path, 'w+')
            if reset_length:
                self.length = self._get_length()
        
        logger.info('Successfully reset data object!')

    def _get_length(self) -> int:
        with tqdm(desc=f'Getting line count of {self._data_path}...') as pbar:
            line = self._data.readline()
            line_count = 0
            while line:
                line_count += 1
                pbar.update(1)
                line = self._data.readline()
            self.reset() # reset to start of file
            return line_count

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> str:
        """
        Retrieves the next unread sequence from the data object.

        Due to file size, don't make use of index and pre-shuffle so that the next line
        in the file can just be retrieved.

        If you want to reset to the beginning of the file object, call SequenceDataset.reset()
        """

        return self._data.readline().strip()